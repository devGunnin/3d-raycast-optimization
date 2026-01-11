"""
Synthetic DEM generation for testing and development.

This module provides functions for generating various types of synthetic
terrain and indoor floorplans for testing camera placement algorithms.
"""

from typing import List, Tuple, Optional
import numpy as np


def generate_synthetic_dem(
    height: int = 512,
    width: int = 512,
    mode: str = 'hills',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic DEM for testing.

    Creates terrain with predictable features useful for testing visibility
    algorithms and optimization.

    Args:
        height: Number of rows.
        width: Number of columns.
        mode: Type of terrain:
            - 'hills': Multiple Gaussian hills (default)
            - 'ridge': Central diagonal ridge
            - 'flat': Flat terrain (all zeros)
            - 'valley': Valley in the center
        seed: Random seed for reproducibility.

    Returns:
        dem: float32 array (H, W) of terrain heights.

    Example:
        >>> dem = generate_synthetic_dem(256, 256, mode='hills', seed=42)
        >>> print(f"Height range: [{dem.min():.1f}, {dem.max():.1f}]")
    """
    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    if mode == 'flat':
        dem = np.zeros((height, width), dtype=np.float32)

    elif mode == 'hills':
        # Multiple Gaussian hills
        dem = np.zeros((height, width), dtype=np.float32)
        num_hills = 5
        for _ in range(num_hills):
            cx = rng.uniform(0.2 * width, 0.8 * width)
            cy = rng.uniform(0.2 * height, 0.8 * height)
            sigma = rng.uniform(30, 80)
            amplitude = rng.uniform(10, 50)
            dem += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    elif mode == 'ridge':
        # Central ridge running diagonally
        ridge_dir = np.array([1, 1]) / np.sqrt(2)
        center = np.array([width / 2, height / 2])
        dist = np.abs((x - center[0]) * ridge_dir[1] - (y - center[1]) * ridge_dir[0])
        dem = 40.0 * np.exp(-dist**2 / (2 * 40**2))
        # Add noise
        dem += 5 * rng.standard_normal((height, width)).astype(np.float32)
        dem = np.maximum(dem, 0)

    elif mode == 'valley':
        # Valley in the center
        cx, cy = width / 2, height / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dem = 0.1 * dist
        dem = np.minimum(dem, 50)

    else:
        raise ValueError(f"Unknown DEM mode: {mode}. Choose from: 'hills', 'ridge', 'flat', 'valley'")

    return dem.astype(np.float32)


def generate_random_dem(
    height: int = 256,
    width: int = 256,
    seed: Optional[int] = None,
    num_octaves: int = 4,
    persistence: float = 0.5,
    scale: float = 50.0,
    height_range: float = 60.0,
) -> np.ndarray:
    """
    Generate random terrain using multi-octave noise (Perlin-like).

    Creates more natural-looking terrain with features at multiple scales,
    useful for stress-testing optimization algorithms.

    Args:
        height: Number of rows.
        width: Number of columns.
        seed: Random seed for reproducibility.
        num_octaves: Number of noise layers (more = more detail).
        persistence: Amplitude decay per octave (0-1).
        scale: Base scale of features (larger = smoother).
        height_range: Maximum terrain height.

    Returns:
        dem: float32 array (H, W) of terrain heights in [0, height_range].

    Example:
        >>> dem = generate_random_dem(512, 512, seed=42)
        >>> print(f"Height range: [{dem.min():.1f}, {dem.max():.1f}]")
    """
    from scipy.ndimage import zoom

    rng = np.random.default_rng(seed)

    dem = np.zeros((height, width), dtype=np.float32)

    for octave in range(num_octaves):
        freq = 2 ** octave
        amp = persistence ** octave

        # Generate smooth noise at this frequency
        noise_h = max(2, int(height // (scale / freq)))
        noise_w = max(2, int(width // (scale / freq)))

        # Random values at grid points
        noise = rng.standard_normal((noise_h + 2, noise_w + 2)).astype(np.float32)

        # Interpolate to full size
        zoomed = zoom(noise, (height / noise.shape[0], width / noise.shape[1]), order=3)

        # Crop to exact size
        zoomed = zoomed[:height, :width]

        dem += amp * zoomed * 20

    # Normalize to desired range
    dem = dem - dem.min()
    dem = dem / (dem.max() + 1e-8) * height_range

    return dem.astype(np.float32)


def add_walls_to_dem(
    dem: np.ndarray,
    wall_cells: List[Tuple[int, int]],
    wall_height: float = 1e9
) -> np.ndarray:
    """
    Add walls to a DEM by setting specified cells to a very high value.

    Walls block line-of-sight and cannot be "seen" - they act as occluders.

    Args:
        dem: Existing DEM array (will be copied, not modified in place).
        wall_cells: List of (row, col) tuples specifying wall locations.
        wall_height: Height value for walls (default 1e9).

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

    Rasterizes line segments onto the DEM with the specified thickness.

    Args:
        dem: Existing DEM array.
        lines: List of ((r1, c1), (r2, c2)) line segments.
        wall_height: Height value for walls.
        thickness: Wall thickness in cells.

    Returns:
        dem: Modified DEM with wall lines.

    Example:
        >>> dem = np.zeros((100, 100), dtype=np.float32)
        >>> dem = add_wall_lines(dem, [((10, 10), (10, 90))], wall_height=1e9)
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
            half_thick = thickness // 2
            for dr_off in range(-half_thick, half_thick + 1):
                for dc_off in range(-half_thick, half_thick + 1):
                    rr = r + dr_off
                    cc = c + dc_off
                    if 0 <= rr < dem.shape[0] and 0 <= cc < dem.shape[1]:
                        dem[rr, cc] = wall_height

    return dem


def create_floorplan_dem(
    height: int = 256,
    width: int = 256,
    wall_height: float = 1e9,
    base_height: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Create an indoor floorplan DEM with rooms and corridors.

    Generates a realistic multi-room layout useful for testing indoor
    surveillance camera placement.

    Args:
        height: Number of rows.
        width: Number of columns.
        wall_height: Height value for walls.
        base_height: Floor height.
        seed: Random seed for reproducibility.

    Returns:
        dem: DEM with a room layout. Wall cells have height >= wall_height.

    Example:
        >>> dem = create_floorplan_dem(256, 256)
        >>> wall_cells = (dem >= 1e6).sum()
        >>> print(f"Wall cells: {wall_cells}")
    """
    rng = np.random.default_rng(seed)

    dem = np.full((height, width), base_height, dtype=np.float32)

    # Outer walls
    wall_thickness = 3
    dem[:wall_thickness, :] = wall_height
    dem[-wall_thickness:, :] = wall_height
    dem[:, :wall_thickness] = wall_height
    dem[:, -wall_thickness:] = wall_height

    # Create room divisions
    # Vertical walls with door gaps
    v_walls = [width // 4, width // 2, 3 * width // 4]
    for vw in v_walls:
        gap_pos = rng.integers(height // 4, 3 * height // 4)
        gap_size = 20
        dem[:gap_pos - gap_size // 2, vw - 1:vw + 2] = wall_height
        dem[gap_pos + gap_size // 2:, vw - 1:vw + 2] = wall_height

    # Horizontal walls with door gaps
    h_walls = [height // 3, 2 * height // 3]
    for hw in h_walls:
        for section in range(4):
            section_start = section * width // 4 + 5
            section_end = (section + 1) * width // 4 - 5
            if section_end - section_start > 20:
                gap_pos = rng.integers(section_start + 10, section_end - 10)
                gap_size = 15
                dem[hw - 1:hw + 2, section_start:gap_pos - gap_size // 2] = wall_height
                dem[hw - 1:hw + 2, gap_pos + gap_size // 2:section_end] = wall_height

    # Add some furniture-like obstacles (smaller blocks)
    num_obstacles = 8
    for _ in range(num_obstacles):
        ox = rng.integers(20, width - 40)
        oy = rng.integers(20, height - 40)
        ow = rng.integers(5, 15)
        oh = rng.integers(5, 15)

        # Only place if not on a wall
        if dem[oy:oy + oh, ox:ox + ow].max() < wall_height / 2:
            dem[oy:oy + oh, ox:ox + ow] = wall_height

    return dem


def generate_terrain_with_hills_and_valleys(
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
    num_hills: int = 5,
    num_valleys: int = 2,
    add_ridge: bool = True,
) -> np.ndarray:
    """
    Generate synthetic terrain with explicit hills, valleys, and optional ridge.

    This provides more control over terrain features than generate_synthetic_dem.

    Args:
        height: Number of rows.
        width: Number of columns.
        seed: Random seed.
        num_hills: Number of Gaussian hills to add.
        num_valleys: Number of valleys (negative Gaussians) to add.
        add_ridge: Whether to add a ridge feature.

    Returns:
        dem: float32 terrain array.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    dem = np.zeros((height, width), dtype=np.float32)

    # Add hills
    for _ in range(num_hills):
        cx = rng.uniform(0.1 * width, 0.9 * width)
        cy = rng.uniform(0.1 * height, 0.9 * height)
        sigma = rng.uniform(20, 60)
        amplitude = rng.uniform(15, 45)
        dem += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    # Add ridge
    if add_ridge:
        ridge_angle = rng.uniform(0, np.pi)
        ridge_center_x = width / 2 + rng.uniform(-50, 50)
        ridge_center_y = height / 2 + rng.uniform(-50, 50)
        cos_a, sin_a = np.cos(ridge_angle), np.sin(ridge_angle)
        dist_from_ridge = np.abs((x - ridge_center_x) * sin_a - (y - ridge_center_y) * cos_a)
        ridge_amplitude = rng.uniform(20, 40)
        ridge_width = rng.uniform(30, 60)
        dem += ridge_amplitude * np.exp(-dist_from_ridge**2 / (2 * ridge_width**2))

    # Add valleys
    for _ in range(num_valleys):
        vx = rng.uniform(0.2 * width, 0.8 * width)
        vy = rng.uniform(0.2 * height, 0.8 * height)
        valley_depth = rng.uniform(10, 25)
        valley_sigma = rng.uniform(40, 80)
        dem -= valley_depth * np.exp(-((x - vx)**2 + (y - vy)**2) / (2 * valley_sigma**2))

    # Ensure non-negative
    dem = np.maximum(dem, 0)

    return dem.astype(np.float32)
