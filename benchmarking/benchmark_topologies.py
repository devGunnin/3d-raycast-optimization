#!/usr/bin/env python3
"""
Benchmark optimization methods across different DEM topologies.

This script uses the tactica package and should produce identical results to
../benchmark_topologies.py when using the same seed and parameters.

This script tests all optimization algorithms from OptimizationFramework
across three distinct terrain types:
1. Realistic: Indoor floorplan with walls and rooms
2. Synthetic: Hills terrain with structured elevation
3. Random: Perlin-noise based random terrain

Outputs:
- GIFs for best method per topology
- Comparison plots per topology
- Final summary comparing all methods across all topologies
"""

import sys
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

# Add paths for tactica and OptimizationFramework
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'OptimizationFramework', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import zoom

# Use tactica package
from tactica.core.sensors import Camera, CameraResolution, compute_max_range_from_fov
from tactica.core.visibility import compute_visibility

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
    run_pgo,
)


# =============================================================================
# DEM Generation Functions (same logic as original)
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
    """
    if seed is not None:
        np.random.seed(seed)

    dem = np.zeros((height, width), dtype=np.float32)

    for octave in range(num_octaves):
        freq = 2 ** octave
        amp = persistence ** octave

        noise_h = max(2, height // (scale / freq))
        noise_w = max(2, width // (scale / freq))

        noise = np.random.randn(int(noise_h) + 2, int(noise_w) + 2).astype(np.float32)
        zoomed = zoom(noise, (height / noise.shape[0], width / noise.shape[1]), order=3)
        zoomed = zoomed[:height, :width]
        dem += amp * zoomed * 20

    dem = dem - dem.min()
    dem = dem / (dem.max() + 1e-8) * 60

    return dem.astype(np.float32)


def generate_realistic_indoor(
    height: int = 256,
    width: int = 256,
    wall_height: float = 1e9,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a realistic indoor floorplan with multiple rooms and corridors.
    """
    if seed is not None:
        np.random.seed(seed)

    dem = np.zeros((height, width), dtype=np.float32)

    # Outer walls
    wall_thickness = 3
    dem[:wall_thickness, :] = wall_height
    dem[-wall_thickness:, :] = wall_height
    dem[:, :wall_thickness] = wall_height
    dem[:, -wall_thickness:] = wall_height

    # Vertical walls
    v_walls = [width // 4, width // 2, 3 * width // 4]
    for vw in v_walls:
        gap_pos = np.random.randint(height // 4, 3 * height // 4)
        gap_size = 20
        dem[:gap_pos - gap_size//2, vw-1:vw+2] = wall_height
        dem[gap_pos + gap_size//2:, vw-1:vw+2] = wall_height

    # Horizontal walls
    h_walls = [height // 3, 2 * height // 3]
    for hw in h_walls:
        for section in range(4):
            section_start = section * width // 4 + 5
            section_end = (section + 1) * width // 4 - 5
            gap_pos = np.random.randint(section_start + 10, section_end - 10)
            gap_size = 15
            dem[hw-1:hw+2, section_start:gap_pos - gap_size//2] = wall_height
            dem[hw-1:hw+2, gap_pos + gap_size//2:section_end] = wall_height

    # Add obstacles
    num_obstacles = 8
    for _ in range(num_obstacles):
        ox = np.random.randint(20, width - 40)
        oy = np.random.randint(20, height - 40)
        ow = np.random.randint(5, 15)
        oh = np.random.randint(5, 15)

        if dem[oy:oy+oh, ox:ox+ow].max() < wall_height / 2:
            dem[oy:oy+oh, ox:ox+ow] = wall_height

    return dem


def generate_synthetic_terrain(
    height: int = 256,
    width: int = 256,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic terrain with hills, valleys, and ridges.
    """
    if seed is not None:
        np.random.seed(seed)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    dem = np.zeros((height, width), dtype=np.float32)

    # Hills
    num_hills = np.random.randint(4, 8)
    for _ in range(num_hills):
        cx = np.random.uniform(0.1 * width, 0.9 * width)
        cy = np.random.uniform(0.1 * height, 0.9 * height)
        sigma = np.random.uniform(20, 60)
        amplitude = np.random.uniform(15, 45)
        dem += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    # Ridge
    ridge_angle = np.random.uniform(0, np.pi)
    ridge_center_x = width / 2 + np.random.uniform(-50, 50)
    ridge_center_y = height / 2 + np.random.uniform(-50, 50)

    cos_a, sin_a = np.cos(ridge_angle), np.sin(ridge_angle)
    dist_from_ridge = np.abs((x - ridge_center_x) * sin_a - (y - ridge_center_y) * cos_a)
    ridge_amplitude = np.random.uniform(20, 40)
    ridge_width = np.random.uniform(30, 60)
    dem += ridge_amplitude * np.exp(-dist_from_ridge**2 / (2 * ridge_width**2))

    # Valley
    vx = np.random.uniform(0.2 * width, 0.8 * width)
    vy = np.random.uniform(0.2 * height, 0.8 * height)
    valley_depth = np.random.uniform(10, 25)
    valley_sigma = np.random.uniform(40, 80)
    dem -= valley_depth * np.exp(-((x - vx)**2 + (y - vy)**2) / (2 * valley_sigma**2))

    dem = np.maximum(dem, 0)

    return dem.astype(np.float32)


# =============================================================================
# Camera Placement Problem (same logic as original, using tactica)
# =============================================================================

class CameraPlacementProblem:
    """Defines the camera placement optimization problem."""

    def __init__(
        self,
        dem: np.ndarray,
        num_cameras: int,
        camera_resolution: CameraResolution,
        wall_threshold: float = 1e6,
        topology_name: str = "unknown",
        margin: float = 15.0,
        z_bounds: Tuple[float, float] = (2, 15),
        pitch_bounds: Tuple[float, float] = (-np.pi/4, 0),
        fov_bounds: Tuple[float, float] = (np.deg2rad(30), np.deg2rad(140)),
    ):
        self.dem = dem
        self.num_cameras = num_cameras
        self.camera_resolution = camera_resolution
        self.wall_threshold = wall_threshold
        self.topology_name = topology_name
        self.margin = margin
        self.z_bounds = z_bounds
        self.pitch_bounds = pitch_bounds
        self.fov_bounds = fov_bounds

        self.height, self.width = dem.shape
        self.valid_mask = dem < wall_threshold
        self.num_valid_cells = self.valid_mask.sum()

        self.params_per_camera = 6
        self.dim = num_cameras * self.params_per_camera
        self._build_bounds()
        self.reset_tracking()

    def _build_bounds(self):
        lower, upper = [], []
        for _ in range(self.num_cameras):
            lower.extend([self.margin, self.margin, self.z_bounds[0], -np.pi,
                         self.pitch_bounds[0], self.fov_bounds[0]])
            upper.extend([self.width - self.margin, self.height - self.margin,
                         self.z_bounds[1], np.pi, self.pitch_bounds[1], self.fov_bounds[1]])
        self.bounds = np.array([lower, upper]).T

    def decode_cameras(self, x: np.ndarray) -> List[Camera]:
        cameras = []
        for i in range(self.num_cameras):
            idx = i * 6
            cx, cy, cz_above = x[idx], x[idx+1], x[idx+2]
            cyaw, cpitch, cfov = x[idx+3], x[idx+4], x[idx+5]

            aspect = self.camera_resolution.aspect_ratio
            cvfov = 2 * np.arctan(np.tan(cfov / 2) / aspect)
            cmax_range = compute_max_range_from_fov(cfov, cvfov, self.camera_resolution)

            col = int(np.clip(cx, 0, self.width - 1))
            row = int(np.clip(cy, 0, self.height - 1))
            terrain_z = self.dem[row, col]
            cz = terrain_z + cz_above

            cameras.append(Camera(x=cx, y=cy, z=cz, yaw=cyaw, pitch=cpitch,
                                  hfov=cfov, vfov=cvfov, max_range=cmax_range))
        return cameras

    def objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        cameras = self.decode_cameras(x)

        penalty = 0.0
        for cam in cameras:
            col = int(np.clip(cam.x, 0, self.width - 1))
            row = int(np.clip(cam.y, 0, self.height - 1))
            if self.dem[row, col] >= self.wall_threshold:
                penalty += 0.5

        try:
            visible_any, vis_count = compute_visibility(
                self.dem, cameras, wall_threshold=self.wall_threshold)
        except Exception:
            return 1.0 + penalty

        visible_valid = (visible_any > 0) & self.valid_mask
        coverage = visible_valid.sum() / self.num_valid_cells

        if coverage > self.best_coverage:
            self.best_coverage = coverage
            self.best_x = x.copy()

        if self.eval_count % 10 == 0:
            self.trace_x.append(self.best_x.copy() if self.best_x is not None else x.copy())
            self.trace_coverage.append(self.best_coverage)

        return -coverage + penalty

    def reset_tracking(self):
        self.best_x = None
        self.best_coverage = 0.0
        self.trace_x = []
        self.trace_coverage = []
        self.eval_count = 0


# =============================================================================
# Optimizer Registry
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
    "PGO": run_pgo,
}


# =============================================================================
# Visualization
# =============================================================================

def draw_camera_markers(ax, cameras, color='red'):
    for cam in cameras:
        ax.plot(cam.x, cam.y, 'o', color=color, markersize=6,
                markeredgecolor='white', markeredgewidth=1, zorder=10)
        dx = 12 * np.cos(cam.yaw)
        dy = 12 * np.sin(cam.yaw)
        ax.annotate('', xy=(cam.x + dx, cam.y + dy), xytext=(cam.x, cam.y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=9)


def create_frame(problem, x, coverage, method_name, eval_num, figsize=(10, 5)):
    dem = problem.dem
    cameras = problem.decode_cameras(x)

    try:
        visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold=problem.wall_threshold)
    except:
        visible_any = np.zeros_like(dem, dtype=np.uint8)
        vis_count = np.zeros_like(dem, dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    wall_mask = dem >= problem.wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    ax1.set_title(f'{method_name} on {problem.topology_name}\nEval {eval_num} - Coverage: {100*coverage:.1f}%',
                  fontsize=10, fontweight='bold')
    ax1.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal', alpha=0.5)
    if wall_mask.any():
        ax1.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)
    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax1.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal', alpha=0.6, vmin=0, vmax=1)
    draw_camera_markers(ax1, cameras)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.set_title('Camera Coverage Count', fontsize=10)
    count_display = np.ma.masked_where(wall_mask | (vis_count == 0), vis_count)
    im = ax2.imshow(count_display, cmap='hot', origin='upper', aspect='equal', vmin=0, vmax=max(len(cameras), 1))
    if wall_mask.any():
        ax2.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                   aspect='equal', alpha=0.9, vmin=0, vmax=1)
    draw_camera_markers(ax2, cameras, color='cyan')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, label='# Cameras', shrink=0.7)

    plt.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def create_optimization_gif(problem, method_name, output_path, fps=5, max_frames=50):
    if len(problem.trace_x) == 0:
        return

    n_frames = len(problem.trace_x)
    indices = np.linspace(0, n_frames - 1, min(n_frames, max_frames), dtype=int) if n_frames > max_frames else range(n_frames)

    frames = []
    for idx in indices:
        frame = create_frame(problem, problem.trace_x[idx], problem.trace_coverage[idx],
                            method_name, (idx + 1) * 10)
        frames.append(frame)

    for _ in range(fps * 2):
        frames.append(frames[-1])

    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def plot_topology_comparison(results: Dict[str, List[Tuple]], output_path: str):
    """Create comparison plot for all methods across all topologies."""
    topologies = list(results.keys())
    n_topo = len(topologies)

    fig, axes = plt.subplots(2, n_topo, figsize=(6 * n_topo, 10))
    if n_topo == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, topo in enumerate(topologies):
        topo_results = results[topo]
        topo_results_sorted = sorted(topo_results, key=lambda t: t[1], reverse=True)

        ax1 = axes[0, i]
        names = [r[0] for r in topo_results_sorted]
        coverages = [r[1] * 100 for r in topo_results_sorted]
        bars = ax1.barh(range(len(names)), coverages,
                        color=[colors[list(METHODS.keys()).index(n) % 10] for n in names])
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('Coverage (%)')
        ax1.set_title(f'{topo}\nFinal Coverage', fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_xlim(0, max(coverages) * 1.15)
        ax1.grid(axis='x', alpha=0.3)

        for bar, cov in zip(bars, coverages):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{cov:.1f}%', va='center', fontsize=8)

        ax2 = axes[1, i]
        for j, (name, coverage, runtime, nfev, trace) in enumerate(topo_results):
            if len(trace) > 0:
                x = np.linspace(0, 1, len(trace))
                y = [c * 100 for c in trace]
                ax2.plot(x, y, label=name, linewidth=1.5,
                        color=colors[list(METHODS.keys()).index(name) % 10])

        ax2.set_xlabel('Progress')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('Convergence', fontweight='bold')
        ax2.legend(loc='lower right', fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_summary_heatmap(all_results: Dict[str, Dict], output_path: str):
    """Create a summary heatmap of all methods vs all topologies."""
    topologies = list(all_results.keys())
    methods = list(METHODS.keys())

    coverage_matrix = np.zeros((len(methods), len(topologies)))

    for j, topo in enumerate(topologies):
        topo_results = {r[0]: r[1] for r in all_results[topo]}
        for i, method in enumerate(methods):
            coverage_matrix[i, j] = topo_results.get(method, 0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(coverage_matrix, cmap='YlGn', aspect='auto')

    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels(topologies, fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    for i in range(len(methods)):
        for j in range(len(topologies)):
            val = coverage_matrix[i, j]
            text_color = 'white' if val > coverage_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                   fontsize=9, color=text_color, fontweight='bold')

    for j in range(len(topologies)):
        best_idx = np.argmax(coverage_matrix[:, j])
        ax.add_patch(plt.Rectangle((j - 0.5, best_idx - 0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=3))

    ax.set_title('Coverage (%) by Method and Topology\n(Red border = best for topology)',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Coverage (%)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_overall_ranking(all_results: Dict[str, Dict], output_path: str):
    """Create overall ranking plot averaging across topologies."""
    methods = list(METHODS.keys())
    topologies = list(all_results.keys())

    avg_coverage = {}
    for method in methods:
        coverages = []
        for topo in topologies:
            topo_results = {r[0]: r[1] for r in all_results[topo]}
            if method in topo_results:
                coverages.append(topo_results[method])
        if coverages:
            avg_coverage[method] = np.mean(coverages)

    sorted_methods = sorted(avg_coverage.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [m[0] for m in sorted_methods]
    avgs = [m[1] * 100 for m in sorted_methods]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.barh(range(len(names)), avgs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Average Coverage (%)', fontsize=12)
    ax.set_title(f'Overall Ranking (Average Across {len(topologies)} Topologies)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (bar, avg) in enumerate(zip(bars, avgs)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{avg:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, max(avgs) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Benchmark
# =============================================================================

def run_single_topology(
    topology_name: str,
    dem: np.ndarray,
    num_cameras: int,
    budget: int,
    seed: int,
    wall_threshold: float,
    z_bounds: Tuple[float, float],
    output_dir: Path
) -> List[Tuple]:
    """Run all optimizers on a single topology."""

    print(f"\n{'='*70}")
    print(f"TOPOLOGY: {topology_name}")
    print(f"{'='*70}")

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
        topology_name=topology_name
    )

    print(f"DEM shape: {dem.shape}")
    print(f"Valid cells: {problem.num_valid_cells}")
    print(f"Cameras: {num_cameras}, Budget: {budget}")

    results = []
    best_coverage = 0
    best_method = None
    best_problem = None

    for name, runner in METHODS.items():
        problem.reset_tracking()

        print(f"\n  Running {name}...", end=" ", flush=True)

        t0 = time.time()
        try:
            if name == "PGO":
                res = runner(problem.objective, problem.bounds, budget, seed=seed,
                           graph_max_size=25, init_temperature=5, final_temperature=0.5)
            elif name == "PSO":
                res = runner(problem.objective, problem.bounds, budget, seed=seed,
                           n_particles=40, w=0.7, c1=1.5, c2=1.5)
            elif name == "CMA-ES":
                res = runner(problem.objective, problem.bounds, budget, seed=seed, sigma0=0.3)
            else:
                res = runner(problem.objective, problem.bounds, budget, seed=seed)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        dt = time.time() - t0
        coverage = -float(res.fx)

        print(f"Coverage: {100*coverage:.1f}% ({dt:.2f}s)")

        results.append((name, coverage, dt, res.nfev, problem.trace_coverage.copy()))

        if coverage > best_coverage:
            best_coverage = coverage
            best_method = name
            best_problem = CameraPlacementProblem(
                dem=dem.copy(),
                num_cameras=num_cameras,
                camera_resolution=camera_res,
                wall_threshold=wall_threshold,
                z_bounds=z_bounds,
                topology_name=topology_name
            )
            best_problem.best_x = problem.best_x.copy() if problem.best_x is not None else None
            best_problem.trace_x = [x.copy() for x in problem.trace_x]
            best_problem.trace_coverage = problem.trace_coverage.copy()

    if best_problem is not None and best_problem.best_x is not None:
        topo_safe = topology_name.replace(" ", "_").replace("(", "").replace(")", "")
        gif_path = output_dir / f'best_{topo_safe}_{best_method.replace(" ", "_")}.gif'
        print(f"\n  Generating GIF for best method ({best_method})...")
        create_optimization_gif(best_problem, f"{best_method}", str(gif_path))
        print(f"    Saved: {gif_path}")

    return results


def run_benchmark(
    num_cameras: int = 4,
    budget: int = 1000,
    seed: int = 42,
    dem_size: int = 256
):
    """Run complete benchmark across all topologies."""

    print("=" * 70)
    print("MULTI-TOPOLOGY OPTIMIZER BENCHMARK (tactica package)")
    print("=" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass

    output_dir = Path(__file__).parent / 'outputs' / 'benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = {
        "Realistic (Indoor)": {
            "dem": generate_realistic_indoor(dem_size, dem_size, seed=seed),
            "wall_threshold": 1e6,
            "z_bounds": (2, 8),
        },
        "Synthetic (Hills)": {
            "dem": generate_synthetic_terrain(dem_size, dem_size, seed=seed),
            "wall_threshold": 1e9,
            "z_bounds": (5, 20),
        },
        "Random (Noise)": {
            "dem": generate_random_dem(dem_size, dem_size, seed=seed),
            "wall_threshold": 1e9,
            "z_bounds": (5, 20),
        },
    }

    # Visualize DEMs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, config) in enumerate(topologies.items()):
        dem = config["dem"]
        wall_mask = dem >= config["wall_threshold"]
        dem_display = np.ma.masked_where(wall_mask, dem)

        axes[i].imshow(dem_display, cmap='terrain', origin='upper')
        if wall_mask.any():
            axes[i].imshow(np.where(wall_mask, 1, np.nan), cmap='binary',
                          origin='upper', alpha=0.9, vmin=0, vmax=1)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')

    plt.suptitle('DEM Topologies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'topologies_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTopology overview saved: {output_dir / 'topologies_overview.png'}")

    all_results = {}

    for topo_name, config in topologies.items():
        results = run_single_topology(
            topology_name=topo_name,
            dem=config["dem"],
            num_cameras=num_cameras,
            budget=budget,
            seed=seed,
            wall_threshold=config["wall_threshold"],
            z_bounds=config["z_bounds"],
            output_dir=output_dir
        )
        all_results[topo_name] = results

    print("\n" + "=" * 70)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 70)

    plot_topology_comparison(all_results, str(output_dir / 'comparison_per_topology.png'))
    print(f"Per-topology comparison saved: {output_dir / 'comparison_per_topology.png'}")

    plot_summary_heatmap(all_results, str(output_dir / 'summary_heatmap.png'))
    print(f"Summary heatmap saved: {output_dir / 'summary_heatmap.png'}")

    plot_overall_ranking(all_results, str(output_dir / 'overall_ranking.png'))
    print(f"Overall ranking saved: {output_dir / 'overall_ranking.png'}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for topo_name, results in all_results.items():
        print(f"\n{topo_name}:")
        results_sorted = sorted(results, key=lambda t: t[1], reverse=True)
        for i, (name, cov, dt, nfev, _) in enumerate(results_sorted[:5], 1):
            print(f"  {i}. {name:16s} {100*cov:5.1f}%  ({dt:.2f}s)")

    methods = list(METHODS.keys())
    avg_coverage = {}
    for method in methods:
        coverages = []
        for topo in all_results:
            topo_results = {r[0]: r[1] for r in all_results[topo]}
            if method in topo_results:
                coverages.append(topo_results[method])
        if coverages:
            avg_coverage[method] = np.mean(coverages)

    winner = max(avg_coverage.items(), key=lambda x: x[1])
    print(f"\n{'='*70}")
    print(f"OVERALL WINNER: {winner[0]} with {100*winner[1]:.1f}% average coverage")
    print(f"{'='*70}")

    print(f"\nAll outputs saved to: {output_dir}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark optimizers across DEM topologies (using tactica package)'
    )
    parser.add_argument('--cameras', type=int, default=4, help='Number of cameras')
    parser.add_argument('--budget', type=int, default=1000, help='Evaluation budget per method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--size', type=int, default=256, help='DEM size (square)')
    args = parser.parse_args()

    run_benchmark(
        num_cameras=args.cameras,
        budget=args.budget,
        seed=args.seed,
        dem_size=args.size
    )


if __name__ == '__main__':
    main()
