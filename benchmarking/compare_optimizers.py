#!/usr/bin/env python3
"""
Compare all optimization methods from OptimizationFramework on camera placement.

This script uses the tactica package and should produce identical results to
../compare_optimizers.py when using the same seed and parameters.

This script:
1. Tests 10 different optimization algorithms
2. Generates GIFs showing optimization progress for each method
3. Creates a final comparison plot of all methods
"""

import sys
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Add paths for tactica and OptimizationFramework
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'OptimizationFramework', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Use tactica package
from tactica.core.sensors import Camera, CameraResolution, compute_max_range_from_fov
from tactica.core.visibility import compute_visibility
from tactica.dem.synthetic import generate_synthetic_dem, create_floorplan_dem

# Use OptimizationFramework directly (same as original)
from OptimizationFramework.optimizers import (
    run_cma,
    run_de_scipy,
    run_ga_sbx,
    run_abc,
    run_dual_annealing,
    run_nelder_mead,
    run_powell,
    run_pso,
    run_bho,
    run_pgo,
)


# =============================================================================
# Camera Placement Problem (using tactica classes, same logic as original)
# =============================================================================

class CameraPlacementProblem:
    """
    Defines the camera placement optimization problem.

    This class maintains the same interface as the original to ensure
    identical optimization behavior and reproducibility.
    """

    def __init__(
        self,
        dem: np.ndarray,
        num_cameras: int,
        camera_resolution: CameraResolution,
        wall_threshold: float = 1e6,
        margin: float = 15.0,
        z_bounds: Tuple[float, float] = (2, 15),
        pitch_bounds: Tuple[float, float] = (-np.pi/4, 0),
        fov_bounds: Tuple[float, float] = (np.deg2rad(30), np.deg2rad(140)),
    ):
        self.dem = dem
        self.num_cameras = num_cameras
        self.camera_resolution = camera_resolution
        self.wall_threshold = wall_threshold
        self.margin = margin
        self.z_bounds = z_bounds
        self.pitch_bounds = pitch_bounds
        self.fov_bounds = fov_bounds

        self.height, self.width = dem.shape
        self.valid_mask = dem < wall_threshold
        self.num_valid_cells = self.valid_mask.sum()

        # 6 params per camera: x, y, z, yaw, pitch, fov
        self.params_per_camera = 6
        self.dim = num_cameras * self.params_per_camera

        self._build_bounds()
        self.reset_tracking()

    def _build_bounds(self):
        """Build bounds array for all parameters."""
        lower = []
        upper = []

        for _ in range(self.num_cameras):
            # x, y, z, yaw, pitch, fov
            lower.extend([
                self.margin,              # x
                self.margin,              # y
                self.z_bounds[0],         # z
                -np.pi,                   # yaw
                self.pitch_bounds[0],     # pitch
                self.fov_bounds[0],       # fov
            ])
            upper.extend([
                self.width - self.margin,  # x
                self.height - self.margin, # y
                self.z_bounds[1],          # z
                np.pi,                     # yaw
                self.pitch_bounds[1],      # pitch
                self.fov_bounds[1],        # fov
            ])

        self.bounds = np.array([lower, upper]).T  # Shape: (dim, 2)

    def decode_cameras(self, x: np.ndarray) -> List[Camera]:
        """Convert flat parameter vector to Camera objects."""
        cameras = []
        ppc = self.params_per_camera

        for i in range(self.num_cameras):
            idx = i * ppc
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

            # Get terrain height
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

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function to MINIMIZE (returns negative coverage).

        Args:
            x: Flat parameter vector

        Returns:
            Negative coverage (for minimization)
        """
        self.eval_count += 1

        # Decode cameras
        cameras = self.decode_cameras(x)

        # Check for wall violations
        penalty = 0.0
        for cam in cameras:
            col = int(np.clip(cam.x, 0, self.width - 1))
            row = int(np.clip(cam.y, 0, self.height - 1))
            if self.dem[row, col] >= self.wall_threshold:
                penalty += 0.5

        # Compute visibility using tactica
        try:
            visible_any, vis_count = compute_visibility(
                self.dem, cameras, wall_threshold=self.wall_threshold
            )
        except Exception:
            return 1.0 + penalty  # Return high loss on error

        # Compute coverage
        visible_valid = (visible_any > 0) & self.valid_mask
        coverage = visible_valid.sum() / self.num_valid_cells

        # Track best
        if coverage > self.best_coverage:
            self.best_coverage = coverage
            self.best_x = x.copy()

        # Record trace periodically
        if self.eval_count % 10 == 0:
            self.trace_x.append(self.best_x.copy() if self.best_x is not None else x.copy())
            self.trace_coverage.append(self.best_coverage)

        return -coverage + penalty  # Minimize negative coverage

    def reset_tracking(self):
        """Reset tracking for new optimization run."""
        self.best_x = None
        self.best_coverage = 0.0
        self.trace_x = []
        self.trace_coverage = []
        self.eval_count = 0


# =============================================================================
# Optimizer Registry (same as original)
# =============================================================================

METHODS: Dict[str, Callable] = {
    "CMA-ES": run_cma,
    "Diff. Evolution": run_de_scipy,
    "Genetic Alg.": run_ga_sbx,
    "Artificial Bee": run_abc,
    "Dual Annealing": run_dual_annealing,
    "Nelder-Mead": run_nelder_mead,
    "Powell": run_powell,
    "PSO": run_pso,
    "Beehive": run_bho,
    "PGO": run_pgo,
}


# =============================================================================
# Visualization (same as original)
# =============================================================================

def draw_camera_markers(ax, cameras, scale=1.0, color='red'):
    """Draw camera positions with directional arrows."""
    for cam in cameras:
        ax.plot(cam.x, cam.y, 'o', color=color, markersize=6,
                markeredgecolor='white', markeredgewidth=1, zorder=10)

        arrow_len = 12 * scale
        dx = arrow_len * np.cos(cam.yaw)
        dy = arrow_len * np.sin(cam.yaw)

        ax.annotate('', xy=(cam.x + dx, cam.y + dy), xytext=(cam.x, cam.y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=9)


def create_frame(
    problem: CameraPlacementProblem,
    x: np.ndarray,
    coverage: float,
    method_name: str,
    eval_num: int,
    figsize: Tuple[int, int] = (10, 5)
) -> np.ndarray:
    """Create a single frame for the optimization GIF."""
    dem = problem.dem
    cameras = problem.decode_cameras(x)

    try:
        visible_any, vis_count = compute_visibility(
            dem, cameras, wall_threshold=problem.wall_threshold
        )
    except Exception:
        visible_any = np.zeros_like(dem, dtype=np.uint8)
        vis_count = np.zeros_like(dem, dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    wall_mask = dem >= problem.wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    ax1.set_title(f'{method_name}\nEval {eval_num} - Coverage: {100*coverage:.1f}%',
                  fontsize=11, fontweight='bold')
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

    ax2.set_title('Camera Coverage Count', fontsize=11)
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
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return image


def create_optimization_gif(
    problem: CameraPlacementProblem,
    method_name: str,
    output_path: str,
    fps: int = 5,
    max_frames: int = 50
):
    """Create a GIF showing optimization progress."""
    if len(problem.trace_x) == 0:
        print(f"  No trace data for {method_name}")
        return

    print(f"  Generating GIF for {method_name}...")

    n_frames = len(problem.trace_x)
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
    else:
        indices = range(n_frames)

    frames = []
    for idx in indices:
        x = problem.trace_x[idx]
        coverage = problem.trace_coverage[idx]
        eval_num = (idx + 1) * 10
        frame = create_frame(problem, x, coverage, method_name, eval_num)
        frames.append(frame)

    for _ in range(fps * 2):
        frames.append(frames[-1])

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"    Saved: {output_path}")


def plot_comparison(
    results: List[Tuple[str, float, float, int, List[float]]],
    output_path: str
):
    """Create comparison plot of all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results_sorted = sorted(results, key=lambda t: t[1], reverse=True)

    ax1 = axes[0]
    names = [r[0] for r in results_sorted]
    coverages = [r[1] * 100 for r in results_sorted]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax1.barh(range(len(names)), coverages, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Coverage (%)')
    ax1.set_title('Final Coverage by Method', fontweight='bold')
    ax1.invert_yaxis()

    for bar, cov in zip(bars, coverages):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cov:.1f}%', va='center', fontsize=9)

    ax1.set_xlim(0, max(coverages) * 1.15)
    ax1.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, coverage, runtime, nfev, trace) in enumerate(results):
        if len(trace) > 0:
            x = np.linspace(0, 1, len(trace))
            y = [c * 100 for c in trace]
            ax2.plot(x, y, label=name, linewidth=2, color=colors_cycle[i % 10])

    ax2.set_xlabel('Progress (fraction of budget)')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Convergence Curves', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def save_final_result(
    problem: CameraPlacementProblem,
    x: np.ndarray,
    coverage: float,
    method_name: str,
    output_path: str
):
    """Save final result visualization."""
    dem = problem.dem
    cameras = problem.decode_cameras(x)

    try:
        visible_any, vis_count = compute_visibility(
            dem, cameras, wall_threshold=problem.wall_threshold
        )
    except Exception:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    wall_mask = dem >= problem.wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    ax1.set_title(f'{method_name} - Coverage: {100*coverage:.1f}%', fontweight='bold')
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

    ax2.axis('off')
    info_text = f"Method: {method_name}\nCoverage: {100*coverage:.2f}%\nCameras: {len(cameras)}\n\n"
    for i, cam in enumerate(cameras):
        info_text += f"Cam {i+1}: ({cam.x:.0f},{cam.y:.0f}) FOV={np.rad2deg(cam.hfov):.0f} deg range={cam.max_range:.0f}\n"

    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_comparison(
    scenario: str = 'indoor',
    num_cameras: int = 4,
    budget: int = 1000,
    seed: int = 42,
    skip_slow: bool = False
):
    """Run comparison of all optimization methods."""

    print("=" * 70)
    print(f"OPTIMIZER COMPARISON (tactica package): {scenario.upper()} scenario")
    print(f"Cameras: {num_cameras}, Budget: {budget} evaluations")
    print("=" * 70)

    # Setup problem using tactica
    if scenario == 'indoor':
        dem = create_floorplan_dem(height=256, width=256, wall_height=1e9)
        wall_threshold = 1e6
        z_bounds = (2, 8)
    else:
        dem = generate_synthetic_dem(height=256, width=256, mode='hills', seed=seed)
        wall_threshold = 1e6
        z_bounds = (5, 20)

    camera_res = CameraResolution(
        horizontal_pixels=3840,
        vertical_pixels=2160,
        pixels_per_meter=30.0
    )

    problem = CameraPlacementProblem(
        dem=dem,
        num_cameras=num_cameras,
        camera_resolution=camera_res,
        wall_threshold=wall_threshold,
        z_bounds=z_bounds,
    )

    print(f"\nProblem dimension: {problem.dim} parameters")
    print(f"Valid cells: {problem.num_valid_cells}")

    output_dir = Path(__file__).parent / 'outputs' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    methods_to_run = list(METHODS.keys())
    if skip_slow:
        methods_to_run = ["CMA-ES", "PSO", "Diff. Evolution", "Genetic Alg.", "PGO"]

    results = []

    for name in methods_to_run:
        runner = METHODS[name]
        problem.reset_tracking()

        print(f"\n{'─'*50}")
        print(f"Running: {name}")
        print(f"{'─'*50}")

        t0 = time.time()
        try:
            # Run with method-specific params (same as original)
            if name == "PGO":
                res = runner(
                    problem.objective,
                    problem.bounds,
                    budget,
                    seed=seed,
                    graph_max_size=25,
                    init_temperature=5,
                    final_temperature=0.5,
                    final_epsilon=0.001,
                )
            elif name == "PSO":
                res = runner(
                    problem.objective,
                    problem.bounds,
                    budget,
                    seed=seed,
                    n_particles=40,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                )
            elif name == "Beehive":
                n_particles = max(20, int(np.sqrt(budget)) // 2)
                res = runner(
                    problem.objective,
                    problem.bounds,
                    budget,
                    seed=seed,
                    n_particles=n_particles,
                    c=0.95,
                    q=0.05,
                    rho=0.92,
                    gamma=0.7,
                )
            elif name == "CMA-ES":
                res = runner(
                    problem.objective,
                    problem.bounds,
                    budget,
                    seed=seed,
                    sigma0=0.3,
                )
            else:
                res = runner(problem.objective, problem.bounds, budget, seed=seed)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        dt = time.time() - t0
        coverage = -float(res.fx)

        print(f"  Coverage: {100*coverage:.2f}%")
        print(f"  Evaluations: {res.nfev}")
        print(f"  Time: {dt:.2f}s")

        results.append((
            name,
            coverage,
            dt,
            res.nfev,
            problem.trace_coverage.copy()
        ))

        gif_path = output_dir / f'optimization_{name.replace(" ", "_").replace(".", "")}.gif'
        create_optimization_gif(problem, name, str(gif_path), fps=5)

        if problem.best_x is not None:
            img_path = output_dir / f'final_{name.replace(" ", "_").replace(".", "")}.png'
            save_final_result(problem, problem.best_x, coverage, name, str(img_path))

    print("\n" + "=" * 70)
    print("SCOREBOARD (by coverage)")
    print("=" * 70)

    results_sorted = sorted(results, key=lambda t: t[1], reverse=True)
    for i, (name, coverage, dt, nfev, _) in enumerate(results_sorted, 1):
        print(f"{i:2d}. {name:16s}  coverage={100*coverage:.2f}%  evals={nfev:5d}  time={dt:.2f}s")

    plot_path = output_dir / 'comparison_all_methods.png'
    plot_comparison(results, str(plot_path))

    winner = results_sorted[0]
    print(f"\n{'='*70}")
    print(f"WINNER: {winner[0]} with {100*winner[1]:.2f}% coverage")
    print(f"{'='*70}")

    print(f"\nAll outputs saved to: {output_dir}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare optimization methods for camera placement (using tactica package)'
    )
    parser.add_argument('--scenario', choices=['outdoor', 'indoor'], default='outdoor',
                        help='Scenario type')
    parser.add_argument('--cameras', type=int, default=25,
                        help='Number of cameras')
    parser.add_argument('--budget', type=int, default=10000,
                        help='Evaluation budget per method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip slower methods for quick testing')
    args = parser.parse_args()

    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: {e}")

    run_comparison(
        scenario=args.scenario,
        num_cameras=args.cameras,
        budget=args.budget,
        seed=args.seed,
        skip_slow=args.skip_slow
    )


if __name__ == '__main__':
    main()
