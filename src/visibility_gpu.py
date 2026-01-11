"""
GPU-based visibility computation for camera placement optimization.

This module provides a Python interface to the CUDA visibility kernel,
which computes per-cell visibility for multiple cameras viewing a DEM.

The CUDA kernel is JIT-compiled via PyTorch's cpp_extension system.
"""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# CUDA Extension Loading (JIT Compilation)
# =============================================================================

_cuda_module = None

def _get_cuda_module():
    """Load and cache the CUDA extension module."""
    global _cuda_module
    if _cuda_module is None:
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
        kernel_path = os.path.join(cuda_dir, 'visibility_kernel.cu')

        if not os.path.exists(kernel_path):
            raise RuntimeError(f"CUDA kernel not found: {kernel_path}")

        print("Compiling CUDA visibility kernel (first run only)...")
        _cuda_module = load(
            name='visibility_cuda',
            sources=[kernel_path],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print("CUDA kernel compiled successfully.")

    return _cuda_module


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CameraResolution:
    """
    Camera sensor resolution configuration.

    Attributes:
        horizontal_pixels: Horizontal resolution (e.g., 3840 for 4K)
        vertical_pixels: Vertical resolution (e.g., 2160 for 4K)
        pixels_per_meter: Required pixel density at max range (e.g., 50 PPM)
    """
    horizontal_pixels: int = 3840  # 4K default
    vertical_pixels: int = 2160   # 4K default
    pixels_per_meter: float = 50.0  # Required resolution at max range

    @property
    def aspect_ratio(self) -> float:
        """Width / Height aspect ratio."""
        return self.horizontal_pixels / self.vertical_pixels


def compute_max_range_from_fov(
    hfov: float,
    vfov: float,
    resolution: CameraResolution
) -> float:
    """
    Compute maximum viewing range based on FOV and resolution requirements.

    The max range is limited by the required pixels-per-meter (PPM) at the
    viewing distance. Narrower FOV = longer range, wider FOV = shorter range.

    Math:
        At distance d, the horizontal width covered is: 2 * d * tan(hfov/2)
        Pixels per meter at that distance: horizontal_pixels / width
        Required: pixels_per_meter = horizontal_pixels / (2 * d * tan(hfov/2))
        Solving for d: d = horizontal_pixels / (2 * ppm * tan(hfov/2))

    Args:
        hfov: Horizontal field of view in radians.
        vfov: Vertical field of view in radians.
        resolution: CameraResolution configuration.

    Returns:
        Maximum range in grid units (limited by both horizontal and vertical).
    """
    ppm = resolution.pixels_per_meter

    # Horizontal range limit
    h_range = resolution.horizontal_pixels / (2 * ppm * np.tan(hfov / 2))

    # Vertical range limit
    v_range = resolution.vertical_pixels / (2 * ppm * np.tan(vfov / 2))

    # Take minimum (we need sufficient resolution in both dimensions)
    return min(h_range, v_range)


@dataclass
class Camera:
    """
    Camera parameters for visibility computation.

    Attributes:
        x, y, z: Position in world coordinates (grid units)
        yaw: Horizontal viewing direction in radians (0 = +X, CCW positive)
        pitch: Vertical viewing angle in radians (0 = horizontal, negative = down)
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

    @classmethod
    def from_fov_and_resolution(
        cls,
        x: float, y: float, z: float,
        yaw: float, pitch: float,
        hfov: float, vfov: float,
        resolution: CameraResolution
    ) -> 'Camera':
        """
        Create a Camera with max_range computed from FOV and resolution.

        Args:
            x, y, z: Position.
            yaw, pitch: Orientation.
            hfov, vfov: Field of view in radians.
            resolution: CameraResolution for computing max_range.

        Returns:
            Camera instance with computed max_range.
        """
        max_range = compute_max_range_from_fov(hfov, vfov, resolution)
        return cls(x, y, z, yaw, pitch, hfov, vfov, max_range)


# =============================================================================
# Main Visibility Computation
# =============================================================================

def compute_visibility(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    occlusion_epsilon: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cell visibility from multiple cameras on GPU.

    Args:
        dem: 2D numpy array (H, W) of terrain heights (float32).
        cameras: List of Camera objects defining viewer positions and FOV.
        wall_threshold: Height value above which cells are treated as walls.
        occlusion_epsilon: Tolerance for occlusion test (prevents z-fighting).

    Returns:
        visible_any: uint8 array (H, W), 1 if any camera sees the cell.
        vis_count: int32 array (H, W), count of cameras seeing the cell.
    """
    # Validate inputs
    assert dem.ndim == 2, f"DEM must be 2D, got shape {dem.shape}"
    assert len(cameras) > 0, "Must provide at least one camera"
    assert len(cameras) <= 64, f"Maximum 64 cameras supported, got {len(cameras)}"

    # Get CUDA module
    cuda_module = _get_cuda_module()

    # Pack camera parameters into array [N, 8]
    camera_params = np.stack([cam.to_array() for cam in cameras], axis=0)

    # Transfer to GPU as PyTorch tensors
    dem_gpu = torch.from_numpy(dem.astype(np.float32)).cuda()
    cameras_gpu = torch.from_numpy(camera_params).cuda()

    # Call CUDA kernel
    vis_count_gpu, visible_any_gpu = cuda_module.compute_visibility(
        dem_gpu,
        cameras_gpu,
        wall_threshold,
        occlusion_epsilon
    )

    # Transfer back to CPU as numpy arrays
    visible_any = visible_any_gpu.cpu().numpy()
    vis_count = vis_count_gpu.cpu().numpy()

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
