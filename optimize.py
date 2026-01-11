#!/usr/bin/env python3
"""
Camera Placement Optimization with Multiple Optimizers.

This script optimizes camera positions and orientations to maximize
visibility coverage of a DEM (terrain or floorplan).

Available optimizers:
  - cma: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
  - pso: Particle Swarm Optimization
  - de: Differential Evolution
  - ga: Genetic Algorithm with SBX crossover
  - abc: Artificial Bee Colony
  - dual: Dual Annealing

Parameters per camera (6):
  - x: horizontal position in grid coordinates
  - y: vertical position in grid coordinates
  - z: height above ground (0 < z < 25)
  - yaw (θ): horizontal viewing angle (-π, π)
  - pitch (φ): vertical viewing angle (bounded based on scenario)
  - fov: horizontal field of view (vfov derived from aspect ratio)
"""

import sys
import os
import time
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OptimizationFramework', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import cma

from visibility_gpu import (
    compute_visibility,
    generate_synthetic_dem,
    create_floorplan_dem,
    Camera,
    CameraResolution,
    compute_max_range_from_fov
)
from scipy.ndimage import zoom

# Import optimizers from OptimizationFramework
try:
    from OptimizationFramework.optimizers import (
        run_pso,
        run_de_scipy,
        run_ga_sbx,
        run_abc,
        run_dual_annealing,
    )
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False
    print("Warning: OptimizationFramework not found. Only CMA-ES available.")


# Optimizer registry
OPTIMIZER_NAMES = {
    'cma': 'CMA-ES',
    'pso': 'Particle Swarm Optimization',
    'de': 'Differential Evolution',
    'ga': 'Genetic Algorithm (SBX)',
    'abc': 'Artificial Bee Colony',
    'dual': 'Dual Annealing',
}


# =============================================================================
# Random DEM Generation
# =============================================================================

def generate_random_dem(
    height: int = 256,
    width: int = 256,
    seed: Optional[int] = None,
    num_octaves: int = 4,
    persistence: float = 0.5,
    scale: float = 50.0
) -> np.ndarray:
    """
    Generate random terrain using simplified Perlin-like noise.

    Args:
        height, width: DEM dimensions
        seed: Random seed
        num_octaves: Number of noise layers (more = more detail)
        persistence: Amplitude decay per octave
        scale: Base scale of features

    Returns:
        Random terrain DEM
    """
    if seed is not None:
        np.random.seed(seed)

    dem = np.zeros((height, width), dtype=np.float32)

    for octave in range(num_octaves):
        freq = 2 ** octave
        amp = persistence ** octave

        # Generate smooth noise at this frequency
        noise_h = max(2, height // (scale / freq))
        noise_w = max(2, width // (scale / freq))

        # Random values at grid points
        noise = np.random.randn(int(noise_h) + 2, int(noise_w) + 2).astype(np.float32)

        # Interpolate to full size
        zoomed = zoom(noise, (height / noise.shape[0], width / noise.shape[1]), order=3)

        # Crop to exact size
        zoomed = zoomed[:height, :width]

        dem += amp * zoomed * 20

    # Normalize to reasonable range
    dem = dem - dem.min()
    dem = dem / (dem.max() + 1e-8) * 60  # 0-60 height range

    return dem.astype(np.float32)


# =============================================================================
# Optimization Problem Definition
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for camera placement optimization."""

    # DEM settings
    dem: np.ndarray
    wall_threshold: float = 1e6

    # Number of cameras to optimize
    num_cameras: int = 4

    # Parameter bounds for position
    x_bounds: Tuple[float, float] = (10, 502)  # Grid x bounds
    y_bounds: Tuple[float, float] = (10, 502)  # Grid y bounds
    z_bounds: Tuple[float, float] = (1, 25)    # Height above ground
    yaw_bounds: Tuple[float, float] = (-np.pi, np.pi)  # Full rotation
    pitch_bounds: Tuple[float, float] = (-np.pi/3, 0)  # Looking down to horizontal

    # FOV bounds (optimizable) - in radians
    # 20 degrees to 150 degrees (capped to avoid tan(90°) = infinity at 180°)
    # Only HFOV is optimized; VFOV is computed from aspect ratio
    fov_bounds: Tuple[float, float] = (np.deg2rad(20), np.deg2rad(150))

    # Camera resolution settings (determines max_range from FOV)
    # 4K resolution by default
    camera_resolution: CameraResolution = None  # Will be initialized in __post_init__

    # Legacy fixed values (only used if optimize_fov=False)
    fixed_hfov: float = np.pi / 2     # 90 degrees
    fixed_vfov: float = np.pi / 3     # 60 degrees
    fixed_max_range: float = 150.0

    # Whether to optimize FOV (if False, uses fixed values)
    optimize_fov: bool = True

    # Objective weights
    coverage_weight: float = 1.0
    redundancy_weight: float = 0.1  # Bonus for overlapping coverage

    # Penalty weights
    wall_penalty: float = 100.0     # Penalty for cameras on walls
    bounds_penalty: float = 50.0    # Penalty for out-of-bounds

    def __post_init__(self):
        """Initialize default camera resolution if not provided."""
        if self.camera_resolution is None:
            # Default: 4K resolution, 30 pixels per meter requirement
            self.camera_resolution = CameraResolution(
                horizontal_pixels=3840,
                vertical_pixels=2160,
                pixels_per_meter=30.0
            )


class CameraOptimizer:
    """
    Optimizer for camera placement using CMA-ES.

    When optimize_fov=True (default):
        Decision vector: [x1, y1, z1, yaw1, pitch1, hfov1, vfov1, x2, ...]
        Total dimensions: num_cameras * 7

    When optimize_fov=False:
        Decision vector: [x1, y1, z1, yaw1, pitch1, x2, ...]
        Total dimensions: num_cameras * 5
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.dem = config.dem
        self.dem_height, self.dem_width = self.dem.shape

        # Precompute valid cell mask (non-wall cells)
        self.valid_mask = self.dem < config.wall_threshold
        self.num_valid_cells = self.valid_mask.sum()

        # Track optimization history
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_solution': [],
            'best_cameras': [],
            'best_visible_any': [],
            'best_vis_count': [],
            'coverage': []
        }

        # Dimensions depend on whether FOV is optimized
        # 6 params: x, y, z, yaw, pitch, fov (vfov derived from aspect ratio)
        # 5 params: x, y, z, yaw, pitch (fixed FOV)
        self.params_per_camera = 6 if config.optimize_fov else 5
        self.num_params = config.num_cameras * self.params_per_camera

    def _decode_solution(self, x: np.ndarray) -> List[Camera]:
        """Convert flat parameter vector to list of Camera objects."""
        cameras = []
        cfg = self.config
        ppc = self.params_per_camera  # params per camera

        for i in range(cfg.num_cameras):
            idx = i * ppc
            cx = x[idx + 0]
            cy = x[idx + 1]
            cz_above = x[idx + 2]  # Height above ground
            cyaw = x[idx + 3]
            cpitch = x[idx + 4]

            if cfg.optimize_fov:
                # FOV is optimized - get HFOV from solution vector
                chfov = x[idx + 5]
                # Compute VFOV from aspect ratio: vfov = 2 * atan(tan(hfov/2) / aspect_ratio)
                aspect = cfg.camera_resolution.aspect_ratio
                cvfov = 2 * np.arctan(np.tan(chfov / 2) / aspect)
                # Compute max_range from FOV and resolution
                cmax_range = compute_max_range_from_fov(
                    chfov, cvfov, cfg.camera_resolution
                )
            else:
                # Use fixed FOV values
                chfov = cfg.fixed_hfov
                cvfov = cfg.fixed_vfov
                cmax_range = cfg.fixed_max_range

            # Get terrain height at camera position
            col = int(np.clip(cx, 0, self.dem_width - 1))
            row = int(np.clip(cy, 0, self.dem_height - 1))
            terrain_z = self.dem[row, col]

            # Absolute z = terrain + height above ground
            cz = terrain_z + cz_above

            cameras.append(Camera(
                x=cx, y=cy, z=cz,
                yaw=cyaw, pitch=cpitch,
                hfov=chfov, vfov=cvfov,
                max_range=cmax_range
            ))

        return cameras

    def _compute_penalty(self, x: np.ndarray, cameras: List[Camera]) -> float:
        """Compute penalty for constraint violations."""
        penalty = 0.0
        cfg = self.config
        ppc = self.params_per_camera

        for i, cam in enumerate(cameras):
            idx = i * ppc

            # Check if camera is on a wall
            col = int(np.clip(cam.x, 0, self.dem_width - 1))
            row = int(np.clip(cam.y, 0, self.dem_height - 1))
            if self.dem[row, col] >= cfg.wall_threshold:
                penalty += cfg.wall_penalty

            # Bounds violations (soft penalty)
            # x bounds
            if x[idx + 0] < cfg.x_bounds[0]:
                penalty += cfg.bounds_penalty * (cfg.x_bounds[0] - x[idx + 0])
            if x[idx + 0] > cfg.x_bounds[1]:
                penalty += cfg.bounds_penalty * (x[idx + 0] - cfg.x_bounds[1])

            # y bounds
            if x[idx + 1] < cfg.y_bounds[0]:
                penalty += cfg.bounds_penalty * (cfg.y_bounds[0] - x[idx + 1])
            if x[idx + 1] > cfg.y_bounds[1]:
                penalty += cfg.bounds_penalty * (x[idx + 1] - cfg.y_bounds[1])

            # z bounds
            if x[idx + 2] < cfg.z_bounds[0]:
                penalty += cfg.bounds_penalty * (cfg.z_bounds[0] - x[idx + 2])
            if x[idx + 2] > cfg.z_bounds[1]:
                penalty += cfg.bounds_penalty * (x[idx + 2] - cfg.z_bounds[1])

            # FOV bounds (if optimizing FOV)
            if cfg.optimize_fov:
                if x[idx + 5] < cfg.fov_bounds[0]:
                    penalty += cfg.bounds_penalty * (cfg.fov_bounds[0] - x[idx + 5])
                if x[idx + 5] > cfg.fov_bounds[1]:
                    penalty += cfg.bounds_penalty * (x[idx + 5] - cfg.fov_bounds[1])

        return penalty

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function to MINIMIZE.

        Returns negative coverage (since CMA-ES minimizes).
        Includes penalties for constraint violations.
        """
        # Decode solution
        cameras = self._decode_solution(x)

        # Compute visibility
        try:
            visible_any, vis_count = compute_visibility(
                self.dem, cameras,
                wall_threshold=self.config.wall_threshold
            )
        except Exception as e:
            # Return large penalty on error
            return 1000.0

        # Compute coverage (fraction of valid cells visible)
        visible_valid = (visible_any > 0) & self.valid_mask
        coverage = visible_valid.sum() / self.num_valid_cells

        # Compute redundancy bonus (cells seen by multiple cameras)
        redundant_cells = ((vis_count > 1) & self.valid_mask).sum()
        redundancy_bonus = redundant_cells / self.num_valid_cells

        # Compute fitness (to maximize)
        fitness = (
            self.config.coverage_weight * coverage +
            self.config.redundancy_weight * redundancy_bonus
        )

        # Compute penalty
        penalty = self._compute_penalty(x, cameras)

        # Return negative (CMA-ES minimizes)
        return -fitness + penalty

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds for all parameters."""
        cfg = self.config

        lower = []
        upper = []

        for _ in range(cfg.num_cameras):
            # Base parameters: x, y, z, yaw, pitch
            lower.extend([cfg.x_bounds[0], cfg.y_bounds[0], cfg.z_bounds[0],
                         cfg.yaw_bounds[0], cfg.pitch_bounds[0]])
            upper.extend([cfg.x_bounds[1], cfg.y_bounds[1], cfg.z_bounds[1],
                         cfg.yaw_bounds[1], cfg.pitch_bounds[1]])

            # Add FOV parameter if optimizing (single HFOV, VFOV derived from aspect ratio)
            if cfg.optimize_fov:
                lower.append(cfg.fov_bounds[0])
                upper.append(cfg.fov_bounds[1])

        return np.array(lower), np.array(upper)

    def get_initial_solution(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random initial solution within bounds."""
        if seed is not None:
            np.random.seed(seed)

        lower, upper = self.get_bounds()
        return lower + np.random.rand(self.num_params) * (upper - lower)

    def callback(self, es: cma.CMAEvolutionStrategy):
        """Callback called after each generation."""
        gen = es.countiter
        best_fitness = -es.result.fbest  # Negate back to coverage
        mean_fitness = -np.mean(es.fit.fit)

        # Decode best solution
        best_x = es.result.xbest
        cameras = self._decode_solution(best_x)

        # Compute visibility for best solution
        visible_any, vis_count = compute_visibility(
            self.dem, cameras,
            wall_threshold=self.config.wall_threshold
        )

        coverage = ((visible_any > 0) & self.valid_mask).sum() / self.num_valid_cells

        # Store history
        self.history['generation'].append(gen)
        self.history['best_fitness'].append(best_fitness)
        self.history['mean_fitness'].append(mean_fitness)
        self.history['best_solution'].append(best_x.copy())
        self.history['best_cameras'].append(cameras)
        self.history['best_visible_any'].append(visible_any.copy())
        self.history['best_vis_count'].append(vis_count.copy())
        self.history['coverage'].append(coverage)

        print(f"  Gen {gen:3d}: coverage={100*coverage:.2f}%, "
              f"fitness={best_fitness:.4f}")

    def optimize(
        self,
        max_generations: int = 100,
        population_size: Optional[int] = None,
        sigma0: float = 0.3,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[List[Camera], dict]:
        """
        Run CMA-ES optimization.

        Args:
            max_generations: Maximum number of generations.
            population_size: Population size (None = automatic).
            sigma0: Initial step size (fraction of search space).
            seed: Random seed.
            verbose: Print progress.

        Returns:
            best_cameras: List of optimized Camera objects.
            history: Optimization history dictionary.
        """
        if verbose:
            print(f"\nStarting CMA-ES optimization")
            print(f"  Cameras: {self.config.num_cameras}")
            print(f"  Parameters: {self.num_params}")
            print(f"  Max generations: {max_generations}")
            print(f"  Valid cells: {self.num_valid_cells}")

        # Get bounds
        lower, upper = self.get_bounds()

        # Initial solution (center of bounds)
        x0 = self.get_initial_solution(seed)

        # CMA-ES options
        opts = {
            'bounds': [lower.tolist(), upper.tolist()],
            'maxiter': max_generations,
            'verb_disp': 0,  # Disable default output
            'verb_log': 0,
            'seed': seed if seed is not None else np.random.randint(1e6),
        }

        if population_size is not None:
            opts['popsize'] = population_size

        # Compute initial sigma based on search space
        sigma = sigma0 * np.mean(upper - lower)

        # Create CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(x0, sigma, opts)

        if verbose:
            print(f"  Population size: {es.popsize}")
            print(f"  Initial sigma: {sigma:.2f}")
            print("\nOptimization progress:")

        # Run optimization
        t0 = time.time()

        while not es.stop():
            # Get candidate solutions
            solutions = es.ask()

            # Evaluate fitness
            fitnesses = [self.objective(x) for x in solutions]

            # Update CMA-ES
            es.tell(solutions, fitnesses)

            # Callback
            self.callback(es)

        t1 = time.time()

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Time: {t1-t0:.1f}s")
            print(f"  Generations: {es.countiter}")
            print(f"  Final coverage: {100*self.history['coverage'][-1]:.2f}%")

        # Return best solution
        best_cameras = self._decode_solution(es.result.xbest)

        return best_cameras, self.history

    def optimize_with(
        self,
        optimizer: str = 'pso',
        budget: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[List[Camera], dict]:
        """
        Run optimization using OptimizationFramework optimizers.

        Args:
            optimizer: One of 'pso', 'de', 'ga', 'abc', 'dual'
            budget: Maximum number of function evaluations
            seed: Random seed
            verbose: Print progress

        Returns:
            best_cameras: List of optimized Camera objects
            history: Optimization history dictionary
        """
        if not OPTIMIZERS_AVAILABLE:
            raise RuntimeError("OptimizationFramework not available. Use optimizer='cma'")

        optimizer_name = OPTIMIZER_NAMES.get(optimizer, optimizer)

        if verbose:
            print(f"\nStarting {optimizer_name} optimization")
            print(f"  Cameras: {self.config.num_cameras}")
            print(f"  Parameters: {self.num_params}")
            print(f"  Budget: {budget} evaluations")
            print(f"  Valid cells: {self.num_valid_cells}")

        # Get bounds in format expected by OptimizationFramework
        lower, upper = self.get_bounds()
        bounds = np.array([lower, upper]).T  # Shape: (dim, 2)

        # Reset tracking
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_solution': [],
            'best_cameras': [],
            'best_visible_any': [],
            'best_vis_count': [],
            'coverage': []
        }

        # Track best during optimization
        self._best_x = None
        self._best_coverage = 0.0
        self._eval_count = 0

        def tracked_objective(x):
            """Wrapper to track progress."""
            self._eval_count += 1
            fitness = self.objective(x)
            coverage = -fitness  # Flip sign

            if coverage > self._best_coverage:
                self._best_coverage = coverage
                self._best_x = x.copy()

            # Record history periodically
            if self._eval_count % max(1, budget // 50) == 0:
                gen = len(self.history['generation']) + 1
                cameras = self._decode_solution(self._best_x if self._best_x is not None else x)

                try:
                    visible_any, vis_count = compute_visibility(
                        self.dem, cameras, wall_threshold=self.config.wall_threshold
                    )
                except:
                    visible_any = np.zeros_like(self.dem, dtype=np.uint8)
                    vis_count = np.zeros_like(self.dem, dtype=np.int32)

                self.history['generation'].append(gen)
                self.history['best_fitness'].append(self._best_coverage)
                self.history['mean_fitness'].append(coverage)
                self.history['best_solution'].append(self._best_x.copy() if self._best_x is not None else x.copy())
                self.history['best_cameras'].append(cameras)
                self.history['best_visible_any'].append(visible_any.copy())
                self.history['best_vis_count'].append(vis_count.copy())
                self.history['coverage'].append(self._best_coverage)

                if verbose:
                    print(f"  Eval {self._eval_count:5d}: coverage={100*self._best_coverage:.2f}%")

            return fitness

        # Run selected optimizer
        t0 = time.time()

        if optimizer == 'pso':
            result = run_pso(
                tracked_objective, bounds, budget, seed=seed,
                n_particles=40, w=0.7, c1=1.5, c2=1.5
            )
        elif optimizer == 'de':
            result = run_de_scipy(
                tracked_objective, bounds, budget, seed=seed
            )
        elif optimizer == 'ga':
            result = run_ga_sbx(
                tracked_objective, bounds, budget, seed=seed
            )
        elif optimizer == 'abc':
            result = run_abc(
                tracked_objective, bounds, budget, seed=seed
            )
        elif optimizer == 'dual':
            result = run_dual_annealing(
                tracked_objective, bounds, budget, seed=seed
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use one of: {list(OPTIMIZER_NAMES.keys())}")

        t1 = time.time()

        # Ensure we have the final result in history
        if len(self.history['generation']) == 0 or self._best_x is not None:
            cameras = self._decode_solution(result.x)
            try:
                visible_any, vis_count = compute_visibility(
                    self.dem, cameras, wall_threshold=self.config.wall_threshold
                )
            except:
                visible_any = np.zeros_like(self.dem, dtype=np.uint8)
                vis_count = np.zeros_like(self.dem, dtype=np.int32)

            final_coverage = -result.fx
            self.history['generation'].append(len(self.history['generation']) + 1)
            self.history['best_fitness'].append(final_coverage)
            self.history['mean_fitness'].append(final_coverage)
            self.history['best_solution'].append(result.x.copy())
            self.history['best_cameras'].append(cameras)
            self.history['best_visible_any'].append(visible_any.copy())
            self.history['best_vis_count'].append(vis_count.copy())
            self.history['coverage'].append(final_coverage)

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Time: {t1-t0:.1f}s")
            print(f"  Evaluations: {result.nfev}")
            print(f"  Final coverage: {100*self.history['coverage'][-1]:.2f}%")

        # Return best solution
        best_cameras = self._decode_solution(result.x)

        return best_cameras, self.history


# =============================================================================
# Visualization and GIF Generation
# =============================================================================

def draw_camera_markers(ax, cameras, scale=1.0, color='red'):
    """Draw camera positions with directional arrows."""
    for i, cam in enumerate(cameras):
        ax.plot(cam.x, cam.y, 'o', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)

        arrow_len = 15 * scale
        dx = arrow_len * np.cos(cam.yaw)
        dy = arrow_len * np.sin(cam.yaw)

        ax.annotate('', xy=(cam.x + dx, cam.y + dy), xytext=(cam.x, cam.y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    zorder=9)

        ax.annotate(f'{i+1}', (cam.x + 3, cam.y + 3), fontsize=8, color='white',
                    fontweight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))


def create_frame(
    dem: np.ndarray,
    cameras: List[Camera],
    visible_any: np.ndarray,
    vis_count: np.ndarray,
    generation: int,
    coverage: float,
    wall_threshold: float = 1e6,
    figsize: Tuple[int, int] = (12, 5)
) -> np.ndarray:
    """Create a single frame for the optimization GIF."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    # Left: DEM + cameras + visibility
    ax1.set_title(f'Generation {generation} - Coverage: {100*coverage:.1f}%',
                  fontsize=12, fontweight='bold')
    ax1.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal', alpha=0.5)

    if wall_mask.any():
        ax1.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)

    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax1.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal',
               alpha=0.6, vmin=0, vmax=1)

    draw_camera_markers(ax1, cameras, scale=1.0, color='red')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Right: Camera count heatmap
    ax2.set_title('Camera Coverage Count', fontsize=12)
    count_display = np.ma.masked_where(wall_mask | (vis_count == 0), vis_count)
    im = ax2.imshow(count_display, cmap='hot', origin='upper', aspect='equal',
                    vmin=0, vmax=max(len(cameras), 1))

    if wall_mask.any():
        ax2.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)

    draw_camera_markers(ax2, cameras, scale=1.0, color='cyan')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, label='# Cameras', shrink=0.7)

    plt.tight_layout()

    # Convert to image array using modern matplotlib API
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]  # Drop alpha channel

    plt.close(fig)

    return image


def create_optimization_gif(
    history: dict,
    dem: np.ndarray,
    output_path: str,
    wall_threshold: float = 1e6,
    fps: int = 5,
    skip_frames: int = 1
):
    """
    Create a GIF showing optimization progress.

    Args:
        history: Optimization history from CameraOptimizer.
        dem: DEM array.
        output_path: Path to save GIF.
        wall_threshold: Wall height threshold.
        fps: Frames per second.
        skip_frames: Only include every N-th frame.
    """
    import imageio.v2 as imageio

    print(f"\nGenerating optimization GIF...")

    frames = []
    generations = history['generation']

    # Select frames to include
    indices = list(range(0, len(generations), skip_frames))
    # Always include first and last
    if 0 not in indices:
        indices.insert(0, 0)
    if len(generations) - 1 not in indices:
        indices.append(len(generations) - 1)

    for i, idx in enumerate(indices):
        gen = history['generation'][idx]
        cameras = history['best_cameras'][idx]
        visible_any = history['best_visible_any'][idx]
        vis_count = history['best_vis_count'][idx]
        coverage = history['coverage'][idx]

        frame = create_frame(
            dem, cameras, visible_any, vis_count,
            gen, coverage, wall_threshold
        )
        frames.append(frame)

        print(f"  Frame {i+1}/{len(indices)} (gen {gen})")

    # Add extra copies of final frame for pause effect
    for _ in range(fps * 2):  # 2 second pause at end
        frames.append(frames[-1])

    # Save GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")


def plot_convergence(history: dict, output_path: str):
    """Plot convergence curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    generations = history['generation']
    coverage = [c * 100 for c in history['coverage']]

    # Coverage over generations
    ax1.plot(generations, coverage, 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Coverage vs Generation')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # Best and mean fitness
    ax2.plot(generations, history['best_fitness'], 'g-', label='Best', linewidth=2)
    ax2.plot(generations, history['mean_fitness'], 'r--', label='Mean', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Fitness Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to: {output_path}")


def save_final_configuration(
    dem: np.ndarray,
    cameras: List[Camera],
    history: dict,
    output_path: str,
    title: str = "Final Configuration",
    wall_threshold: float = 1e6
):
    """
    Save a high-quality PNG of the final optimized configuration.

    Args:
        dem: DEM array.
        cameras: List of optimized Camera objects.
        history: Optimization history.
        output_path: Path to save PNG.
        title: Plot title.
        wall_threshold: Wall height threshold.
    """
    # Get final visibility data
    visible_any = history['best_visible_any'][-1]
    vis_count = history['best_vis_count'][-1]
    coverage = history['coverage'][-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    # Plot 1: DEM with cameras
    ax1.set_title(f'{title}\nTerrain + Camera Positions', fontsize=11, fontweight='bold')
    im1 = ax1.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal')
    if wall_mask.any():
        ax1.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)
    draw_camera_markers(ax1, cameras, scale=1.0, color='red')
    ax1.set_xlabel('X (columns)')
    ax1.set_ylabel('Y (rows)')
    plt.colorbar(im1, ax=ax1, label='Height', shrink=0.7)

    # Plot 2: Visibility coverage
    ax2.set_title(f'Visibility Coverage: {100*coverage:.1f}%', fontsize=11, fontweight='bold')
    ax2.imshow(dem_display, cmap='gray', origin='upper', aspect='equal', alpha=0.3)
    if wall_mask.any():
        ax2.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)
    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax2.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal',
               alpha=0.7, vmin=0, vmax=1)
    draw_camera_markers(ax2, cameras, scale=1.0, color='red')
    ax2.set_xlabel('X (columns)')
    ax2.set_ylabel('Y (rows)')

    # Legend
    visible_patch = mpatches.Patch(color='green', alpha=0.7, label='Visible')
    not_visible_patch = mpatches.Patch(color='gray', alpha=0.3, label='Not visible')
    ax2.legend(handles=[visible_patch, not_visible_patch], loc='upper right', fontsize=8)

    # Plot 3: Camera count heatmap
    ax3.set_title('Camera Redundancy (count per cell)', fontsize=11, fontweight='bold')
    count_display = np.ma.masked_where(wall_mask | (vis_count == 0), vis_count)
    im3 = ax3.imshow(count_display, cmap='hot', origin='upper', aspect='equal',
                     vmin=0, vmax=max(len(cameras), 1))
    if wall_mask.any():
        ax3.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)
    draw_camera_markers(ax3, cameras, scale=1.0, color='cyan')
    ax3.set_xlabel('X (columns)')
    ax3.set_ylabel('Y (rows)')
    plt.colorbar(im3, ax=ax3, label='# Cameras', shrink=0.7)

    # Add camera info as text
    cam_info = "Camera Parameters:\n"
    for i, cam in enumerate(cameras):
        cam_info += (f"  {i+1}: pos=({cam.x:.0f},{cam.y:.0f},{cam.z:.1f}) "
                    f"yaw={np.rad2deg(cam.yaw):.0f}° pitch={np.rad2deg(cam.pitch):.0f}° "
                    f"FOV=({np.rad2deg(cam.hfov):.0f}°x{np.rad2deg(cam.vfov):.0f}°) "
                    f"range={cam.max_range:.0f}\n")

    fig.text(0.02, 0.02, cam_info, fontsize=7, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for camera info
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Final configuration saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def run_outdoor_optimization(
    seed: int = 42,
    max_generations: int = 100,
    num_cameras: int = 8,
    optimizer_type: str = 'cma'
):
    """
    Run optimization on outdoor terrain.

    Args:
        seed: Random seed
        max_generations: Max generations (CMA-ES) or budget multiplier (others)
        num_cameras: Number of cameras to optimize
        optimizer_type: One of 'cma', 'pso', 'de', 'ga', 'abc', 'dual'
    """
    optimizer_name = OPTIMIZER_NAMES.get(optimizer_type, optimizer_type)
    print("=" * 70)
    print(f"OUTDOOR TERRAIN OPTIMIZATION ({num_cameras} cameras, {optimizer_name})")
    print("=" * 70)

    # Generate terrain
    print("\nGenerating synthetic terrain (512x512)...")
    dem = generate_synthetic_dem(height=512, width=512, mode='hills', seed=seed)
    print(f"  Height range: [{dem.min():.1f}, {dem.max():.1f}]")

    # Camera resolution configuration
    # 4K camera, 25 pixels per meter for outdoor (allows longer range)
    camera_res = CameraResolution(
        horizontal_pixels=3840,
        vertical_pixels=2160,
        pixels_per_meter=25.0  # Lower PPM for outdoor allows longer range
    )

    print(f"\nCamera Resolution: {camera_res.horizontal_pixels}x{camera_res.vertical_pixels}")
    print(f"  Pixels per meter requirement: {camera_res.pixels_per_meter}")

    # Show example ranges for different FOVs
    example_fovs = [20, 45, 90, 120, 180]
    print("  Example max ranges for different FOVs:")
    for fov_deg in example_fovs:
        fov_rad = np.deg2rad(fov_deg)
        max_r = compute_max_range_from_fov(fov_rad, fov_rad * 0.5625, camera_res)
        print(f"    HFOV={fov_deg}°: range={max_r:.1f} units")

    # Configure optimization
    config = OptimizationConfig(
        dem=dem,
        num_cameras=num_cameras,
        x_bounds=(20, 492),
        y_bounds=(20, 492),
        z_bounds=(5, 25),
        yaw_bounds=(-np.pi, np.pi),
        pitch_bounds=(-np.pi/4, 0),  # -45 to 0 degrees
        # FOV bounds: 20° to 150° (wide range for outdoor)
        fov_bounds=(np.deg2rad(20), np.deg2rad(150)),
        camera_resolution=camera_res,
        optimize_fov=True,
        coverage_weight=1.0,
        redundancy_weight=0.05
    )

    # Create optimizer
    optimizer = CameraOptimizer(config)

    # Run optimization with selected method
    if optimizer_type == 'cma':
        best_cameras, history = optimizer.optimize(
            max_generations=max_generations,
            sigma0=0.3,
            seed=seed
        )
    else:
        # For other optimizers, budget = generations * population size estimate
        budget = max_generations * 20  # Approximate population equivalent
        best_cameras, history = optimizer.optimize_with(
            optimizer=optimizer_type,
            budget=budget,
            seed=seed
        )

    return dem, best_cameras, history, config


def run_indoor_optimization(
    seed: int = 123,
    max_generations: int = 100,
    num_cameras: int = 4,
    optimizer_type: str = 'cma'
):
    """
    Run optimization on indoor floorplan.

    Args:
        seed: Random seed
        max_generations: Max generations (CMA-ES) or budget multiplier (others)
        num_cameras: Number of cameras to optimize
        optimizer_type: One of 'cma', 'pso', 'de', 'ga', 'abc', 'dual'
    """
    optimizer_name = OPTIMIZER_NAMES.get(optimizer_type, optimizer_type)
    print("\n" + "=" * 70)
    print(f"INDOOR FLOORPLAN OPTIMIZATION ({num_cameras} cameras, {optimizer_name})")
    print("=" * 70)

    # Generate floorplan
    print("\nGenerating indoor floorplan (256x256)...")
    dem = create_floorplan_dem(height=256, width=256, wall_height=1e9)
    wall_cells = (dem >= 1e6).sum()
    print(f"  Wall cells: {wall_cells}")

    # Camera resolution configuration
    # 4K camera, 30 pixels per meter for indoor (allows reasonable range with wide FOV)
    camera_res = CameraResolution(
        horizontal_pixels=3840,
        vertical_pixels=2160,
        pixels_per_meter=30.0  # Lower PPM for indoor to allow wider FOV usage
    )

    print(f"\nCamera Resolution: {camera_res.horizontal_pixels}x{camera_res.vertical_pixels}")
    print(f"  Pixels per meter requirement: {camera_res.pixels_per_meter}")

    # Configure optimization
    config = OptimizationConfig(
        dem=dem,
        wall_threshold=1e6,
        num_cameras=num_cameras,
        x_bounds=(15, 241),
        y_bounds=(15, 241),
        z_bounds=(2, 10),  # Indoor: lower ceiling
        yaw_bounds=(-np.pi, np.pi),
        pitch_bounds=(-np.pi/6, 0),  # -30 to 0 degrees (more horizontal indoors)
        # FOV bounds for indoor: wider FOVs preferred (capped at 150° to avoid zero range)
        fov_bounds=(np.deg2rad(40), np.deg2rad(150)),
        camera_resolution=camera_res,
        optimize_fov=True,
        coverage_weight=1.0,
        redundancy_weight=0.1
    )

    # Create optimizer
    optimizer = CameraOptimizer(config)

    # Run optimization with selected method
    if optimizer_type == 'cma':
        best_cameras, history = optimizer.optimize(
            max_generations=max_generations,
            sigma0=0.25,
            seed=seed
        )
    else:
        # For other optimizers, budget = generations * population size estimate
        budget = max_generations * 20  # Approximate population equivalent
        best_cameras, history = optimizer.optimize_with(
            optimizer=optimizer_type,
            budget=budget,
            seed=seed
        )

    return dem, best_cameras, history, config


def run_random_optimization(
    seed: int = 456,
    max_generations: int = 100,
    num_cameras: int = 6,
    optimizer_type: str = 'cma'
):
    """
    Run optimization on random Perlin-noise terrain.

    Args:
        seed: Random seed
        max_generations: Max generations (CMA-ES) or budget multiplier (others)
        num_cameras: Number of cameras to optimize
        optimizer_type: One of 'cma', 'pso', 'de', 'ga', 'abc', 'dual'
    """
    optimizer_name = OPTIMIZER_NAMES.get(optimizer_type, optimizer_type)
    print("\n" + "=" * 70)
    print(f"RANDOM TERRAIN OPTIMIZATION ({num_cameras} cameras, {optimizer_name})")
    print("=" * 70)

    # Generate random terrain
    print("\nGenerating random Perlin-noise terrain (512x512)...")
    dem = generate_random_dem(height=512, width=512, seed=seed)
    print(f"  Height range: [{dem.min():.1f}, {dem.max():.1f}]")

    # Camera resolution configuration
    # 4K camera, 25 pixels per meter for outdoor-like terrain
    camera_res = CameraResolution(
        horizontal_pixels=3840,
        vertical_pixels=2160,
        pixels_per_meter=25.0
    )

    print(f"\nCamera Resolution: {camera_res.horizontal_pixels}x{camera_res.vertical_pixels}")
    print(f"  Pixels per meter requirement: {camera_res.pixels_per_meter}")

    # Configure optimization
    config = OptimizationConfig(
        dem=dem,
        wall_threshold=1e9,  # No walls in random terrain
        num_cameras=num_cameras,
        x_bounds=(20, 492),
        y_bounds=(20, 492),
        z_bounds=(5, 25),
        yaw_bounds=(-np.pi, np.pi),
        pitch_bounds=(-np.pi/4, 0),  # -45 to 0 degrees
        fov_bounds=(np.deg2rad(20), np.deg2rad(150)),
        camera_resolution=camera_res,
        optimize_fov=True,
        coverage_weight=1.0,
        redundancy_weight=0.05
    )

    # Create optimizer
    optimizer = CameraOptimizer(config)

    # Run optimization with selected method
    if optimizer_type == 'cma':
        best_cameras, history = optimizer.optimize(
            max_generations=max_generations,
            sigma0=0.3,
            seed=seed
        )
    else:
        # For other optimizers, budget = generations * population size estimate
        budget = max_generations * 20
        best_cameras, history = optimizer.optimize_with(
            optimizer=optimizer_type,
            budget=budget,
            seed=seed
        )

    return dem, best_cameras, history, config


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Camera Placement Optimization')
    parser.add_argument('--scenario', choices=['outdoor', 'indoor', 'random', 'all'],
                        default='all', help='Scenario to optimize')
    parser.add_argument('--optimizer', choices=['cma', 'pso', 'de', 'ga', 'abc', 'dual'],
                        default='cma', help='Optimization algorithm to use')
    parser.add_argument('--generations', type=int, default=100,
                        help='Maximum generations (or budget multiplier for non-CMA)')
    parser.add_argument('--cameras', type=int, default=None,
                        help='Number of cameras (default: 8 outdoor, 4 indoor, 6 random)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-gif', action='store_true',
                        help='Skip GIF generation')
    args = parser.parse_args()

    optimizer_name = OPTIMIZER_NAMES.get(args.optimizer, args.optimizer)
    print(f"Camera Placement Optimization using {optimizer_name}")
    print("=" * 70)

    # Check if non-CMA optimizer is requested but framework not available
    if args.optimizer != 'cma' and not OPTIMIZERS_AVAILABLE:
        print(f"Error: {optimizer_name} requires OptimizationFramework.")
        print("Please ensure OptimizationFramework is installed or use --optimizer cma")
        sys.exit(1)

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: {e}")

    # Create output directory
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Run outdoor optimization
    if args.scenario in ['outdoor', 'all']:
        num_cams = args.cameras if args.cameras is not None else 8
        dem, cameras, history, config = run_outdoor_optimization(
            seed=args.seed,
            max_generations=args.generations,
            num_cameras=num_cams,
            optimizer_type=args.optimizer
        )
        results['outdoor'] = {
            'dem': dem, 'cameras': cameras, 'history': history, 'config': config
        }

        # Generate outputs
        if not args.skip_gif:
            create_optimization_gif(
                history, dem,
                str(output_dir / 'optimization_outdoor.gif'),
                wall_threshold=config.wall_threshold,
                fps=5, skip_frames=max(1, args.generations // 25)
            )

        plot_convergence(history, str(output_dir / 'convergence_outdoor.png'))

        # Save final configuration PNG
        save_final_configuration(
            dem, cameras, history,
            str(output_dir / 'final_outdoor.png'),
            title="Outdoor Optimization Result",
            wall_threshold=config.wall_threshold
        )

    # Run indoor optimization
    if args.scenario in ['indoor', 'all']:
        num_cams = args.cameras if args.cameras is not None else 4
        dem, cameras, history, config = run_indoor_optimization(
            seed=args.seed,
            max_generations=args.generations,
            num_cameras=num_cams,
            optimizer_type=args.optimizer
        )
        results['indoor'] = {
            'dem': dem, 'cameras': cameras, 'history': history, 'config': config
        }

        # Generate outputs
        if not args.skip_gif:
            create_optimization_gif(
                history, dem,
                str(output_dir / 'optimization_indoor.gif'),
                wall_threshold=config.wall_threshold,
                fps=5, skip_frames=max(1, args.generations // 25)
            )

        plot_convergence(history, str(output_dir / 'convergence_indoor.png'))

        # Save final configuration PNG
        save_final_configuration(
            dem, cameras, history,
            str(output_dir / 'final_indoor.png'),
            title="Indoor Optimization Result",
            wall_threshold=config.wall_threshold
        )

    # Run random terrain optimization
    if args.scenario in ['random', 'all']:
        num_cams = args.cameras if args.cameras is not None else 6
        dem, cameras, history, config = run_random_optimization(
            seed=args.seed,
            max_generations=args.generations,
            num_cameras=num_cams,
            optimizer_type=args.optimizer
        )
        results['random'] = {
            'dem': dem, 'cameras': cameras, 'history': history, 'config': config
        }

        # Generate outputs
        if not args.skip_gif:
            create_optimization_gif(
                history, dem,
                str(output_dir / 'optimization_random.gif'),
                wall_threshold=config.wall_threshold,
                fps=5, skip_frames=max(1, args.generations // 25)
            )

        plot_convergence(history, str(output_dir / 'convergence_random.png'))

        # Save final configuration PNG
        save_final_configuration(
            dem, cameras, history,
            str(output_dir / 'final_random.png'),
            title="Random Terrain Optimization Result",
            wall_threshold=config.wall_threshold
        )

    # Print final results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)

    for scenario, data in results.items():
        print(f"\n{scenario.upper()}:")
        final_coverage = data['history']['coverage'][-1] * 100
        print(f"  Final coverage: {final_coverage:.2f}%")
        print(f"  Cameras: {len(data['cameras'])}")
        print(f"  {'='*65}")
        for i, cam in enumerate(data['cameras']):
            print(f"    Camera {i+1}:")
            print(f"      Position: ({cam.x:.1f}, {cam.y:.1f}, {cam.z:.1f})")
            print(f"      Orientation: yaw={np.rad2deg(cam.yaw):.0f}°, pitch={np.rad2deg(cam.pitch):.0f}°")
            print(f"      FOV: H={np.rad2deg(cam.hfov):.0f}°, V={np.rad2deg(cam.vfov):.0f}°")
            print(f"      Max Range: {cam.max_range:.1f} units")

    print(f"\nOutputs saved to: {output_dir}/")
    print("\nDone!")


if __name__ == '__main__':
    main()
