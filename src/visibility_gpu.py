"""
GPU-based visibility computation for camera placement optimization.

This module implements a CuPy RawKernel-based CUDA kernel that computes
per-cell visibility for multiple cameras viewing a DEM (Digital Elevation Model).

Method: Per-cell ray-marching with FOV gating and LOS occlusion sampling.
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# CUDA Kernel: Visibility computation for all cameras
# =============================================================================

VISIBILITY_KERNEL_CODE = r'''
extern "C" __global__ void compute_visibility(
    const float* __restrict__ dem,           // DEM heightfield [H, W]
    const float* __restrict__ cameras,       // Camera params [N, 8]: x,y,z,yaw,pitch,hfov,vfov,range
    unsigned char* __restrict__ visible_any, // Output: any camera sees this cell [H, W]
    int* __restrict__ vis_count,             // Output: count of cameras seeing this cell [H, W]
    int height,                              // DEM height (rows)
    int width,                               // DEM width (cols)
    int num_cameras,                         // Number of cameras
    float wall_threshold,                    // Height threshold for walls (e.g., 1e6)
    float occlusion_epsilon                  // Epsilon for occlusion test
) {
    // Thread index maps to DEM cell (row, col)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    int cell_idx = row * width + col;

    // Get DEM height at this cell
    float cell_z = dem[cell_idx];

    // If this cell is a wall, it cannot be "seen" (it blocks, not receives visibility)
    if (cell_z >= wall_threshold) {
        visible_any[cell_idx] = 0;
        vis_count[cell_idx] = 0;
        return;
    }

    // Cell center in world coordinates (x=col, y=row, z=cell_z)
    float cell_x = (float)col + 0.5f;
    float cell_y = (float)row + 0.5f;

    int count = 0;

    // Loop over all cameras
    for (int cam = 0; cam < num_cameras; cam++) {
        // Unpack camera parameters
        float cam_x = cameras[cam * 8 + 0];
        float cam_y = cameras[cam * 8 + 1];
        float cam_z = cameras[cam * 8 + 2];
        float cam_yaw = cameras[cam * 8 + 3];     // radians, 0 = +X axis, CCW positive
        float cam_pitch = cameras[cam * 8 + 4];   // radians, 0 = horizontal, negative = down
        float cam_hfov = cameras[cam * 8 + 5];    // horizontal FOV in radians
        float cam_vfov = cameras[cam * 8 + 6];    // vertical FOV in radians
        float cam_range = cameras[cam * 8 + 7];   // max range

        // Vector from camera to cell
        float dx = cell_x - cam_x;
        float dy = cell_y - cam_y;
        float dz = cell_z - cam_z;

        // Horizontal distance
        float dist_xy = sqrtf(dx * dx + dy * dy);
        float dist_3d = sqrtf(dx * dx + dy * dy + dz * dz);

        // =================================================================
        // Gate 1: Range check
        // =================================================================
        if (dist_xy > cam_range || dist_3d < 0.01f) {
            continue;  // Out of range or too close
        }

        // =================================================================
        // Gate 2: Horizontal FOV check
        // =================================================================
        // Angle from camera to cell in XY plane
        float angle_to_cell = atan2f(dy, dx);

        // Angular difference (wrapped to [-pi, pi])
        float h_diff = angle_to_cell - cam_yaw;
        // Normalize to [-pi, pi]
        while (h_diff > 3.14159265f) h_diff -= 2.0f * 3.14159265f;
        while (h_diff < -3.14159265f) h_diff += 2.0f * 3.14159265f;

        if (fabsf(h_diff) > cam_hfov * 0.5f) {
            continue;  // Outside horizontal FOV
        }

        // =================================================================
        // Gate 3: Vertical FOV check
        // =================================================================
        // Vertical angle: positive = up, negative = down
        // atan2(dz, dist_xy) gives angle from horizontal
        float v_angle_to_cell = atan2f(dz, dist_xy);

        // Angular difference from camera pitch
        float v_diff = v_angle_to_cell - cam_pitch;

        if (fabsf(v_diff) > cam_vfov * 0.5f) {
            continue;  // Outside vertical FOV
        }

        // =================================================================
        // Gate 4: Line-of-sight occlusion test via sampling
        // =================================================================
        // Sample K points along the ray from camera to cell
        // K = ceil(dist_xy / step_size), clamped to [8, 512]
        float step_size = 1.0f;  // ~1 cell per step
        int K = (int)ceilf(dist_xy / step_size);
        if (K < 8) K = 8;
        if (K > 512) K = 512;

        bool occluded = false;

        for (int s = 1; s < K; s++) {  // Start at s=1 to skip camera position
            float t = (float)s / (float)K;  // Interpolation parameter [0, 1]

            // Sample point along ray
            float sx = cam_x + t * dx;
            float sy = cam_y + t * dy;
            float sz = cam_z + t * dz;  // LOS height at sample point

            // Convert to grid coordinates (0-indexed)
            float gx = sx - 0.5f;  // Grid x
            float gy = sy - 0.5f;  // Grid y

            // Clamp to valid grid range for interpolation
            if (gx < 0.0f) gx = 0.0f;
            if (gy < 0.0f) gy = 0.0f;
            if (gx > (float)(width - 1) - 0.001f) gx = (float)(width - 1) - 0.001f;
            if (gy > (float)(height - 1) - 0.001f) gy = (float)(height - 1) - 0.001f;

            // Bilinear interpolation for DEM height at (gx, gy)
            int x0 = (int)floorf(gx);
            int y0 = (int)floorf(gy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Clamp indices
            if (x1 >= width) x1 = width - 1;
            if (y1 >= height) y1 = height - 1;

            float fx = gx - (float)x0;
            float fy = gy - (float)y0;

            // Sample DEM heights at corners
            float h00 = dem[y0 * width + x0];
            float h10 = dem[y0 * width + x1];
            float h01 = dem[y1 * width + x0];
            float h11 = dem[y1 * width + x1];

            // Bilinear interpolation
            float h0 = h00 * (1.0f - fx) + h10 * fx;
            float h1 = h01 * (1.0f - fx) + h11 * fx;
            float dem_height = h0 * (1.0f - fy) + h1 * fy;

            // Check occlusion: if terrain is above LOS + epsilon, ray is blocked
            if (dem_height > sz + occlusion_epsilon) {
                occluded = true;
                break;
            }
        }

        if (!occluded) {
            count++;
        }
    }

    // Write outputs
    visible_any[cell_idx] = (count > 0) ? 1 : 0;
    vis_count[cell_idx] = count;
}
'''

# Compile the CUDA kernel
_visibility_kernel = cp.RawKernel(VISIBILITY_KERNEL_CODE, 'compute_visibility')


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Camera:
    """
    Camera parameters for visibility computation.

    Attributes:
        x, y, z: Position in world coordinates (grid units)
        yaw: Horizontal viewing direction in radians (0 = +X, CCW positive)
        pitch: Vertical viewing angle in radians (0 = horizontal, negative = looking down)
        hfov: Horizontal field of view in radians
        vfov: Vertical field of view in radians
        max_range: Maximum viewing distance (horizontal)
    """
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    max_range: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, yaw, pitch, hfov, vfov, range]."""
        return np.array([
            self.x, self.y, self.z, self.yaw, self.pitch,
            self.hfov, self.vfov, self.max_range
        ], dtype=np.float32)


# =============================================================================
# Main Visibility Computation
# =============================================================================

def compute_visibility(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    occlusion_epsilon: float = 1e-3,
    block_size: Tuple[int, int] = (16, 16)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cell visibility from multiple cameras on GPU.

    Args:
        dem: 2D numpy array (H, W) of terrain heights (float32).
        cameras: List of Camera objects defining viewer positions and FOV.
        wall_threshold: Height value above which cells are treated as walls.
        occlusion_epsilon: Tolerance for occlusion test (prevents z-fighting).
        block_size: CUDA block dimensions (threads per block).

    Returns:
        visible_any: uint8 array (H, W), 1 if any camera sees the cell.
        vis_count: int32 array (H, W), count of cameras seeing the cell.
    """
    # Validate inputs
    assert dem.ndim == 2, f"DEM must be 2D, got shape {dem.shape}"
    assert len(cameras) > 0, "Must provide at least one camera"
    assert len(cameras) <= 64, f"Maximum 64 cameras supported, got {len(cameras)}"

    height, width = dem.shape
    num_cameras = len(cameras)

    # Pack camera parameters into array [N, 8]
    camera_params = np.stack([cam.to_array() for cam in cameras], axis=0)
    assert camera_params.shape == (num_cameras, 8)

    # Transfer to GPU
    dem_gpu = cp.asarray(dem.astype(np.float32), dtype=cp.float32)
    cameras_gpu = cp.asarray(camera_params, dtype=cp.float32)

    # Allocate output arrays
    visible_any_gpu = cp.zeros((height, width), dtype=cp.uint8)
    vis_count_gpu = cp.zeros((height, width), dtype=cp.int32)

    # Compute grid dimensions
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]

    # Launch kernel
    _visibility_kernel(
        (grid_x, grid_y),           # Grid dimensions
        block_size,                  # Block dimensions
        (
            dem_gpu,
            cameras_gpu,
            visible_any_gpu,
            vis_count_gpu,
            height,
            width,
            num_cameras,
            np.float32(wall_threshold),
            np.float32(occlusion_epsilon)
        )
    )

    # Synchronize and transfer back to CPU
    cp.cuda.Stream.null.synchronize()

    visible_any = cp.asnumpy(visible_any_gpu)
    vis_count = cp.asnumpy(vis_count_gpu)

    return visible_any, vis_count


# =============================================================================
# Synthetic DEM Generation
# =============================================================================

def generate_synthetic_dem(
    height: int = 512,
    width: int = 512,
    mode: str = 'hills',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic DEM for testing.

    Args:
        height: Number of rows.
        width: Number of columns.
        mode: Type of terrain - 'hills', 'ridge', 'flat', 'valley'.
        seed: Random seed for reproducibility.

    Returns:
        dem: float32 array (H, W) of terrain heights.
    """
    if seed is not None:
        np.random.seed(seed)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    if mode == 'flat':
        dem = np.zeros((height, width), dtype=np.float32)

    elif mode == 'hills':
        # Multiple Gaussian hills
        dem = np.zeros((height, width), dtype=np.float32)
        num_hills = 5
        for _ in range(num_hills):
            cx = np.random.uniform(0.2 * width, 0.8 * width)
            cy = np.random.uniform(0.2 * height, 0.8 * height)
            sigma = np.random.uniform(30, 80)
            amplitude = np.random.uniform(10, 50)
            dem += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    elif mode == 'ridge':
        # Central ridge running diagonally
        ridge_dir = np.array([1, 1]) / np.sqrt(2)
        center = np.array([width / 2, height / 2])
        dist = np.abs((x - center[0]) * ridge_dir[1] - (y - center[1]) * ridge_dir[0])
        dem = 40.0 * np.exp(-dist**2 / (2 * 40**2))
        # Add noise
        dem += 5 * np.random.randn(height, width).astype(np.float32)
        dem = np.maximum(dem, 0)

    elif mode == 'valley':
        # Valley in the center
        cx, cy = width / 2, height / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dem = 0.1 * dist
        dem = np.minimum(dem, 50)

    else:
        raise ValueError(f"Unknown DEM mode: {mode}")

    return dem.astype(np.float32)


def add_walls_to_dem(
    dem: np.ndarray,
    wall_cells: List[Tuple[int, int]],
    wall_height: float = 1e9
) -> np.ndarray:
    """
    Add walls to a DEM by setting specified cells to a very high value.

    Args:
        dem: Existing DEM array (will be modified in place).
        wall_cells: List of (row, col) tuples specifying wall locations.
        wall_height: Height value for walls.

    Returns:
        dem: Modified DEM with walls.
    """
    dem = dem.copy()
    for row, col in wall_cells:
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
            dem[row, col] = wall_height
    return dem


def add_wall_lines(
    dem: np.ndarray,
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    wall_height: float = 1e9,
    thickness: int = 1
) -> np.ndarray:
    """
    Add wall lines to a DEM (for indoor/floorplan scenarios).

    Args:
        dem: Existing DEM array.
        lines: List of ((r1, c1), (r2, c2)) line segments.
        wall_height: Height value for walls.
        thickness: Wall thickness in cells.

    Returns:
        dem: Modified DEM with wall lines.
    """
    dem = dem.copy()

    for (r1, c1), (r2, c2) in lines:
        # Bresenham-like line rasterization with thickness
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        steps = max(dr, dc, 1)

        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            r = int(round(r1 + t * (r2 - r1)))
            c = int(round(c1 + t * (c2 - c1)))

            # Add thickness
            for dr_off in range(-thickness // 2, thickness // 2 + 1):
                for dc_off in range(-thickness // 2, thickness // 2 + 1):
                    rr = r + dr_off
                    cc = c + dc_off
                    if 0 <= rr < dem.shape[0] and 0 <= cc < dem.shape[1]:
                        dem[rr, cc] = wall_height

    return dem


def create_floorplan_dem(
    height: int = 256,
    width: int = 256,
    wall_height: float = 1e9,
    base_height: float = 0.0
) -> np.ndarray:
    """
    Create a simple indoor floorplan DEM with rooms.

    Args:
        height: Number of rows.
        width: Number of columns.
        wall_height: Height value for walls.
        base_height: Floor height.

    Returns:
        dem: DEM with a simple room layout.
    """
    dem = np.full((height, width), base_height, dtype=np.float32)

    # Outer walls
    margin = 10
    lines = [
        # Outer boundary
        ((margin, margin), (margin, width - margin)),
        ((margin, width - margin), (height - margin, width - margin)),
        ((height - margin, width - margin), (height - margin, margin)),
        ((height - margin, margin), (margin, margin)),

        # Interior walls (creating rooms)
        ((margin, width // 2), (height // 2 - 20, width // 2)),  # Vertical wall with gap
        ((height // 2 + 20, width // 2), (height - margin, width // 2)),  # Continue vertical
        ((height // 2, margin), (height // 2, width // 3 - 10)),  # Horizontal wall with gap
        ((height // 2, width // 3 + 10), (height // 2, width // 2)),  # Continue horizontal
    ]

    dem = add_wall_lines(dem, lines, wall_height, thickness=2)
    return dem


# =============================================================================
# Camera Utilities
# =============================================================================

def create_random_cameras(
    num_cameras: int,
    dem: np.ndarray,
    height_above_ground: float = 5.0,
    hfov: float = np.pi / 2,  # 90 degrees
    vfov: float = np.pi / 3,  # 60 degrees
    max_range: float = 100.0,
    seed: Optional[int] = None,
    wall_threshold: float = 1e6
) -> List[Camera]:
    """
    Create randomly positioned cameras within the DEM bounds.

    Args:
        num_cameras: Number of cameras to create.
        dem: DEM array (for bounds and height reference).
        height_above_ground: Camera height above terrain.
        hfov: Horizontal FOV in radians.
        vfov: Vertical FOV in radians.
        max_range: Maximum viewing range.
        seed: Random seed.
        wall_threshold: Height above which cells are walls (cameras won't spawn there).

    Returns:
        List of Camera objects.
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = dem.shape
    cameras = []

    attempts = 0
    while len(cameras) < num_cameras and attempts < num_cameras * 100:
        attempts += 1

        # Random position within bounds (with margin)
        x = np.random.uniform(10, width - 10)
        y = np.random.uniform(10, height - 10)

        # Get terrain height at this position
        col = int(x)
        row = int(y)
        terrain_z = dem[row, col]

        # Skip wall positions
        if terrain_z >= wall_threshold:
            continue

        z = terrain_z + height_above_ground

        # Random yaw (full 360 degrees)
        yaw = np.random.uniform(-np.pi, np.pi)

        # Pitch: typically looking slightly down
        pitch = np.random.uniform(-np.pi / 6, 0)  # -30 to 0 degrees

        cameras.append(Camera(
            x=x, y=y, z=z,
            yaw=yaw, pitch=pitch,
            hfov=hfov, vfov=vfov,
            max_range=max_range
        ))

    return cameras


def create_preset_cameras(
    dem: np.ndarray,
    positions: List[Tuple[float, float]],
    height_above_ground: float = 5.0,
    hfov: float = np.pi / 2,
    vfov: float = np.pi / 3,
    max_range: float = 100.0,
    look_at_center: bool = True
) -> List[Camera]:
    """
    Create cameras at preset positions, optionally looking at DEM center.

    Args:
        dem: DEM array.
        positions: List of (x, y) positions.
        height_above_ground: Camera height above terrain.
        hfov, vfov: Field of view angles.
        max_range: Maximum range.
        look_at_center: If True, cameras point toward DEM center.

    Returns:
        List of Camera objects.
    """
    height, width = dem.shape
    center_x, center_y = width / 2, height / 2

    cameras = []
    for x, y in positions:
        col = int(np.clip(x, 0, width - 1))
        row = int(np.clip(y, 0, height - 1))
        terrain_z = dem[row, col]
        z = terrain_z + height_above_ground

        if look_at_center:
            dx = center_x - x
            dy = center_y - y
            yaw = np.arctan2(dy, dx)
            dist = np.sqrt(dx**2 + dy**2)
            dz = dem[int(center_y), int(center_x)] - z
            pitch = np.arctan2(dz, dist)
        else:
            yaw = 0.0
            pitch = 0.0

        cameras.append(Camera(
            x=x, y=y, z=z,
            yaw=yaw, pitch=pitch,
            hfov=hfov, vfov=vfov,
            max_range=max_range
        ))

    return cameras
