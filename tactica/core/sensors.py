"""
Sensor (camera) data structures and utilities.

This module defines the core data structures for cameras and other sensors,
as well as utility functions for computing derived properties like max range.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class CameraResolution:
    """
    Camera sensor resolution configuration.

    This defines the pixel dimensions and required pixel density for a camera,
    which together determine the maximum effective viewing range.

    Attributes:
        horizontal_pixels: Horizontal resolution (e.g., 3840 for 4K)
        vertical_pixels: Vertical resolution (e.g., 2160 for 4K)
        pixels_per_meter: Required pixel density at max range (e.g., 30 PPM for face recognition)

    Example:
        >>> res = CameraResolution(3840, 2160, 30.0)  # 4K camera, 30 PPM requirement
        >>> print(f"Aspect ratio: {res.aspect_ratio:.2f}")
        Aspect ratio: 1.78
    """
    horizontal_pixels: int = 3840  # 4K default
    vertical_pixels: int = 2160   # 4K default
    pixels_per_meter: float = 30.0  # Required resolution at max range

    @property
    def aspect_ratio(self) -> float:
        """Width / Height aspect ratio."""
        return self.horizontal_pixels / self.vertical_pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "horizontal_pixels": self.horizontal_pixels,
            "vertical_pixels": self.vertical_pixels,
            "pixels_per_meter": self.pixels_per_meter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraResolution":
        """Create from dictionary."""
        return cls(
            horizontal_pixels=data["horizontal_pixels"],
            vertical_pixels=data["vertical_pixels"],
            pixels_per_meter=data["pixels_per_meter"],
        )

    # Common presets
    @classmethod
    def preset_4k(cls, ppm: float = 30.0) -> "CameraResolution":
        """4K (3840x2160) camera preset."""
        return cls(3840, 2160, ppm)

    @classmethod
    def preset_1080p(cls, ppm: float = 30.0) -> "CameraResolution":
        """1080p (1920x1080) camera preset."""
        return cls(1920, 1080, ppm)

    @classmethod
    def preset_720p(cls, ppm: float = 30.0) -> "CameraResolution":
        """720p (1280x720) camera preset."""
        return cls(1280, 720, ppm)


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

    Example:
        >>> res = CameraResolution(3840, 2160, 30.0)
        >>> max_range = compute_max_range_from_fov(np.deg2rad(90), np.deg2rad(60), res)
        >>> print(f"Max range: {max_range:.1f} units")
    """
    ppm = resolution.pixels_per_meter

    # Horizontal range limit
    h_tan = np.tan(hfov / 2)
    h_range = resolution.horizontal_pixels / (2 * ppm * h_tan) if h_tan > 0 else float('inf')

    # Vertical range limit
    v_tan = np.tan(vfov / 2)
    v_range = resolution.vertical_pixels / (2 * ppm * v_tan) if v_tan > 0 else float('inf')

    # Take minimum (we need sufficient resolution in both dimensions)
    return min(h_range, v_range)


@dataclass
class Camera:
    """
    Camera parameters for visibility computation.

    This represents a camera with full 6-DOF positioning plus optical parameters.
    The camera uses a pinhole model with symmetric horizontal and vertical FOV.

    Attributes:
        x: X position in world/grid coordinates
        y: Y position in world/grid coordinates
        z: Z position (height) in world coordinates
        yaw: Horizontal viewing direction in radians (0 = +X, CCW positive)
        pitch: Vertical viewing angle in radians (0 = horizontal, negative = down)
        hfov: Horizontal field of view in radians
        vfov: Vertical field of view in radians
        max_range: Maximum viewing distance (horizontal projection)

    Example:
        >>> cam = Camera(
        ...     x=100.0, y=100.0, z=15.0,
        ...     yaw=np.deg2rad(45), pitch=np.deg2rad(-10),
        ...     hfov=np.deg2rad(90), vfov=np.deg2rad(60),
        ...     max_range=200.0
        ... )
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

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "hfov": float(self.hfov),
            "vfov": float(self.vfov),
            "max_range": float(self.max_range),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            yaw=data["yaw"],
            pitch=data["pitch"],
            hfov=data["hfov"],
            vfov=data["vfov"],
            max_range=data["max_range"],
        )

    @classmethod
    def from_fov_and_resolution(
        cls,
        x: float, y: float, z: float,
        yaw: float, pitch: float,
        hfov: float,
        resolution: CameraResolution
    ) -> "Camera":
        """
        Create a Camera with max_range and vfov computed from FOV and resolution.

        The vertical FOV is derived from the horizontal FOV using the camera's
        aspect ratio: vfov = 2 * arctan(tan(hfov/2) / aspect_ratio)

        Args:
            x, y, z: Position.
            yaw, pitch: Orientation.
            hfov: Horizontal field of view in radians.
            resolution: CameraResolution for computing max_range and vfov.

        Returns:
            Camera instance with computed max_range and vfov.
        """
        # Compute VFOV from aspect ratio
        aspect = resolution.aspect_ratio
        vfov = 2 * np.arctan(np.tan(hfov / 2) / aspect)

        # Compute max range
        max_range = compute_max_range_from_fov(hfov, vfov, resolution)

        return cls(x, y, z, yaw, pitch, hfov, vfov, max_range)

    @property
    def position(self) -> Tuple[float, float, float]:
        """Return (x, y, z) position tuple."""
        return (self.x, self.y, self.z)

    @property
    def hfov_degrees(self) -> float:
        """Horizontal FOV in degrees."""
        return np.rad2deg(self.hfov)

    @property
    def vfov_degrees(self) -> float:
        """Vertical FOV in degrees."""
        return np.rad2deg(self.vfov)

    @property
    def yaw_degrees(self) -> float:
        """Yaw in degrees."""
        return np.rad2deg(self.yaw)

    @property
    def pitch_degrees(self) -> float:
        """Pitch in degrees."""
        return np.rad2deg(self.pitch)


def create_random_cameras(
    num_cameras: int,
    dem: np.ndarray,
    height_above_ground: float = 5.0,
    hfov: float = np.pi / 2,  # 90 degrees
    vfov: float = np.pi / 3,  # 60 degrees
    max_range: float = 100.0,
    seed: Optional[int] = None,
    wall_threshold: float = 1e6,
    margin: float = 10.0,
) -> List[Camera]:
    """
    Create randomly positioned cameras within the DEM bounds.

    Cameras are placed at random valid (non-wall) locations within the DEM,
    with random yaw orientations. This is useful for generating initial
    populations for optimization.

    Args:
        num_cameras: Number of cameras to create.
        dem: DEM array (for bounds and height reference).
        height_above_ground: Camera height above terrain.
        hfov: Horizontal FOV in radians.
        vfov: Vertical FOV in radians.
        max_range: Maximum viewing range.
        seed: Random seed for reproducibility.
        wall_threshold: Height above which cells are walls.
        margin: Minimum distance from DEM edges.

    Returns:
        List of Camera objects.
    """
    rng = np.random.default_rng(seed)

    height, width = dem.shape
    cameras = []

    attempts = 0
    max_attempts = num_cameras * 100

    while len(cameras) < num_cameras and attempts < max_attempts:
        attempts += 1

        # Random position within bounds (with margin)
        x = rng.uniform(margin, width - margin)
        y = rng.uniform(margin, height - margin)

        # Get terrain height at this position
        col = int(x)
        row = int(y)
        terrain_z = dem[row, col]

        # Skip wall positions
        if terrain_z >= wall_threshold:
            continue

        z = terrain_z + height_above_ground

        # Random yaw (full 360 degrees)
        yaw = rng.uniform(-np.pi, np.pi)

        # Pitch: typically looking slightly down
        pitch = rng.uniform(-np.pi / 6, 0)  # -30 to 0 degrees

        cameras.append(Camera(
            x=x, y=y, z=z,
            yaw=yaw, pitch=pitch,
            hfov=hfov, vfov=vfov,
            max_range=max_range
        ))

    if len(cameras) < num_cameras:
        raise ValueError(
            f"Could only place {len(cameras)} of {num_cameras} cameras. "
            f"DEM may have too many wall cells."
        )

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

    Useful for creating fixed camera configurations or for visualization.

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
            pitch = np.arctan2(dz, dist) if dist > 0 else 0.0
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
