"""
Camera placement optimization problem definition.

This module defines the CameraPlacementProblem class, which is the single
source of truth for the camera placement optimization problem. It consolidates
the previously duplicated problem definitions from multiple scripts.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np

from tactica.core.sensors import Camera, CameraResolution, compute_max_range_from_fov
from tactica.core.visibility import compute_visibility
from tactica.optimization.constraints import OptimizationConstraints


@dataclass
class CameraPlacementProblem:
    """
    Camera placement optimization problem.

    This class defines the complete optimization problem for placing cameras
    to maximize visibility coverage of a DEM. It handles:
    - Parameter encoding/decoding
    - Bounds generation
    - Objective function evaluation
    - Constraint handling
    - Solution tracking for visualization

    Attributes:
        dem: Digital Elevation Model array (H, W)
        num_cameras: Number of cameras to optimize
        camera_resolution: Camera sensor resolution configuration
        wall_threshold: Height above which cells are walls
        constraints: Optional optimization constraints
        z_bounds: (min, max) camera height above ground
        pitch_bounds: (min, max) pitch angle in radians
        fov_bounds: (min, max) horizontal FOV in radians
        margin: Minimum distance from DEM edges
        coverage_weight: Weight for coverage in objective
        redundancy_weight: Weight for redundancy bonus
        wall_penalty: Penalty for cameras on walls
        topology_name: Optional name for logging/visualization

    Example:
        >>> from tactica.dem import generate_synthetic_dem
        >>> dem = generate_synthetic_dem(256, 256)
        >>> problem = CameraPlacementProblem(
        ...     dem=dem,
        ...     num_cameras=4,
        ... )
        >>> x0 = problem.get_initial_solution(seed=42)
        >>> fitness = problem.objective(x0)
    """
    dem: np.ndarray
    num_cameras: int = 4
    camera_resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(3840, 2160, 30.0)
    )
    wall_threshold: float = 1e6
    constraints: Optional[OptimizationConstraints] = None

    # Parameter bounds
    margin: float = 15.0
    z_bounds: Tuple[float, float] = (2, 15)
    pitch_bounds: Tuple[float, float] = (-np.pi / 4, 0)
    fov_bounds: Tuple[float, float] = (np.deg2rad(30), np.deg2rad(140))

    # Objective weights
    coverage_weight: float = 1.0
    redundancy_weight: float = 0.1
    wall_penalty: float = 0.5

    # Metadata
    topology_name: str = "unknown"

    # Internal state (set in __post_init__)
    height: int = field(init=False)
    width: int = field(init=False)
    valid_mask: np.ndarray = field(init=False)
    num_valid_cells: int = field(init=False)
    params_per_camera: int = field(init=False, default=6)
    dim: int = field(init=False)
    bounds: np.ndarray = field(init=False)

    # Tracking state
    best_x: Optional[np.ndarray] = field(init=False, default=None)
    best_coverage: float = field(init=False, default=0.0)
    trace_x: List[np.ndarray] = field(init=False, default_factory=list)
    trace_coverage: List[float] = field(init=False, default_factory=list)
    eval_count: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize derived fields."""
        self.height, self.width = self.dem.shape

        # Compute valid mask (non-wall cells)
        self.valid_mask = self.dem < self.wall_threshold

        # Apply constraint masks
        if self.constraints is not None:
            if self.constraints.placement_mask is not None:
                self.valid_mask = self.valid_mask & self.constraints.placement_mask
            if self.constraints.exclusion_mask is not None:
                self.valid_mask = self.valid_mask & ~self.constraints.exclusion_mask

        self.num_valid_cells = int(self.valid_mask.sum())

        # Dimension: 6 params per camera (x, y, z, yaw, pitch, fov)
        self.dim = self.num_cameras * self.params_per_camera

        # Build bounds
        self._build_bounds()

        # Initialize tracking
        self.reset_tracking()

    def _build_bounds(self) -> None:
        """Build bounds array for all parameters."""
        lower = []
        upper = []

        for _ in range(self.num_cameras):
            # x, y, z_above, yaw, pitch, hfov
            lower.extend([
                self.margin,               # x
                self.margin,               # y
                self.z_bounds[0],          # z above ground
                -np.pi,                    # yaw
                self.pitch_bounds[0],      # pitch
                self.fov_bounds[0],        # hfov
            ])
            upper.extend([
                self.width - self.margin,  # x
                self.height - self.margin, # y
                self.z_bounds[1],          # z above ground
                np.pi,                     # yaw
                self.pitch_bounds[1],      # pitch
                self.fov_bounds[1],        # hfov
            ])

        self.bounds = np.array([lower, upper]).T  # Shape: (dim, 2)

    def decode_cameras(self, x: np.ndarray) -> List[Camera]:
        """
        Convert flat parameter vector to Camera objects.

        Args:
            x: Flat parameter vector of length dim

        Returns:
            List of Camera objects
        """
        cameras = []

        for i in range(self.num_cameras):
            idx = i * self.params_per_camera
            cx = x[idx + 0]
            cy = x[idx + 1]
            cz_above = x[idx + 2]
            cyaw = x[idx + 3]
            cpitch = x[idx + 4]
            cfov = x[idx + 5]

            # Compute VFOV from aspect ratio
            aspect = self.camera_resolution.aspect_ratio
            cvfov = 2 * np.arctan(np.tan(cfov / 2) / aspect)

            # Compute max range from FOV
            cmax_range = compute_max_range_from_fov(cfov, cvfov, self.camera_resolution)

            # Get terrain height at camera position
            col = int(np.clip(cx, 0, self.width - 1))
            row = int(np.clip(cy, 0, self.height - 1))
            terrain_z = self.dem[row, col]
            cz = terrain_z + cz_above

            cameras.append(Camera(
                x=cx, y=cy, z=cz,
                yaw=cyaw, pitch=cpitch,
                hfov=cfov, vfov=cvfov,
                max_range=cmax_range
            ))

        return cameras

    def encode_cameras(self, cameras: List[Camera]) -> np.ndarray:
        """
        Convert Camera objects to flat parameter vector.

        Args:
            cameras: List of Camera objects

        Returns:
            Flat parameter vector
        """
        x = np.zeros(self.dim, dtype=np.float32)

        for i, cam in enumerate(cameras):
            idx = i * self.params_per_camera

            # Get terrain height to compute z_above
            col = int(np.clip(cam.x, 0, self.width - 1))
            row = int(np.clip(cam.y, 0, self.height - 1))
            terrain_z = self.dem[row, col]
            z_above = cam.z - terrain_z

            x[idx + 0] = cam.x
            x[idx + 1] = cam.y
            x[idx + 2] = z_above
            x[idx + 3] = cam.yaw
            x[idx + 4] = cam.pitch
            x[idx + 5] = cam.hfov

        return x

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function to MINIMIZE.

        Returns negative coverage (for minimization) plus penalties.

        Args:
            x: Flat parameter vector

        Returns:
            Objective value (lower is better)
        """
        self.eval_count += 1

        # Decode cameras
        cameras = self.decode_cameras(x)

        # Add fixed cameras from constraints
        all_cameras = cameras.copy()
        if self.constraints and self.constraints.fixed_cameras:
            all_cameras.extend(self.constraints.fixed_cameras)

        # Compute penalty for wall violations
        penalty = 0.0
        for cam in cameras:  # Only penalize optimized cameras
            col = int(np.clip(cam.x, 0, self.width - 1))
            row = int(np.clip(cam.y, 0, self.height - 1))
            if self.dem[row, col] >= self.wall_threshold:
                penalty += self.wall_penalty

        # Compute visibility
        try:
            visible_any, vis_count = compute_visibility(
                self.dem, all_cameras, wall_threshold=self.wall_threshold
            )
        except Exception:
            return 1.0 + penalty  # Return high loss on error

        # Compute coverage
        visible_valid = (visible_any > 0) & self.valid_mask

        # Apply priority weights if available
        if self.constraints and self.constraints.priority_weights is not None:
            weights = self.constraints.priority_weights
            total_weight = (weights * self.valid_mask).sum()
            visible_weight = (weights * visible_valid).sum()
            coverage = visible_weight / total_weight if total_weight > 0 else 0.0
        else:
            coverage = visible_valid.sum() / self.num_valid_cells if self.num_valid_cells > 0 else 0.0

        # Compute redundancy bonus
        if self.num_valid_cells > 0:
            redundant_cells = ((vis_count > 1) & self.valid_mask).sum()
            redundancy_bonus = redundant_cells / self.num_valid_cells
        else:
            redundancy_bonus = 0.0

        # Track best
        if coverage > self.best_coverage:
            self.best_coverage = coverage
            self.best_x = x.copy()

        # Record trace periodically
        if self.eval_count % 10 == 0:
            self.trace_x.append(
                self.best_x.copy() if self.best_x is not None else x.copy()
            )
            self.trace_coverage.append(self.best_coverage)

        # Compute total fitness (to maximize)
        fitness = (
            self.coverage_weight * coverage +
            self.redundancy_weight * redundancy_bonus
        )

        # Return negative for minimization
        return -fitness + penalty

    def reset_tracking(self) -> None:
        """Reset tracking state for a new optimization run."""
        self.best_x = None
        self.best_coverage = 0.0
        self.trace_x = []
        self.trace_coverage = []
        self.eval_count = 0

    def get_initial_solution(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random initial solution within bounds.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Random parameter vector
        """
        rng = np.random.default_rng(seed)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + rng.random(self.dim) * (upper - lower)

    def get_bounds_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds as (lower, upper) tuple.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        return self.bounds[:, 0], self.bounds[:, 1]

    def evaluate_solution(self, x: np.ndarray) -> dict:
        """
        Evaluate a solution and return detailed metrics.

        Unlike objective(), this returns metrics for analysis rather than
        optimization, and doesn't affect tracking state.

        Args:
            x: Parameter vector

        Returns:
            Dict with coverage, redundancy, cameras, visibility maps, etc.
        """
        cameras = self.decode_cameras(x)

        # Add fixed cameras
        all_cameras = cameras.copy()
        if self.constraints and self.constraints.fixed_cameras:
            all_cameras.extend(self.constraints.fixed_cameras)

        visible_any, vis_count = compute_visibility(
            self.dem, all_cameras, wall_threshold=self.wall_threshold
        )

        visible_valid = (visible_any > 0) & self.valid_mask
        coverage = visible_valid.sum() / self.num_valid_cells if self.num_valid_cells > 0 else 0.0

        # Redundancy stats
        redundant_mask = (vis_count > 1) & self.valid_mask
        redundancy_fraction = redundant_mask.sum() / self.num_valid_cells if self.num_valid_cells > 0 else 0.0

        if visible_valid.sum() > 0:
            avg_cameras_per_cell = vis_count[visible_valid].mean()
        else:
            avg_cameras_per_cell = 0.0

        return {
            "coverage": float(coverage),
            "redundancy_fraction": float(redundancy_fraction),
            "avg_cameras_per_cell": float(avg_cameras_per_cell),
            "visible_cells": int(visible_valid.sum()),
            "valid_cells": int(self.num_valid_cells),
            "cameras": cameras,
            "all_cameras": all_cameras,
            "visible_any": visible_any,
            "vis_count": vis_count,
        }

    def to_dict(self) -> dict:
        """Convert problem configuration to dictionary."""
        return {
            "num_cameras": self.num_cameras,
            "dem_shape": list(self.dem.shape),
            "wall_threshold": self.wall_threshold,
            "margin": self.margin,
            "z_bounds": list(self.z_bounds),
            "pitch_bounds": list(self.pitch_bounds),
            "fov_bounds": list(self.fov_bounds),
            "coverage_weight": self.coverage_weight,
            "redundancy_weight": self.redundancy_weight,
            "camera_resolution": self.camera_resolution.to_dict(),
            "constraints": self.constraints.to_dict() if self.constraints else None,
            "topology_name": self.topology_name,
        }
