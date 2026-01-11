"""
Static plotting functions for camera placement visualization.

This module consolidates all plotting functions that were previously
duplicated across multiple scripts.
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tactica.core.sensors import Camera


def draw_camera_markers(
    ax: plt.Axes,
    cameras: List[Camera],
    scale: float = 1.0,
    color: str = 'red',
    show_labels: bool = False,
    arrow_length: float = 12.0,
) -> None:
    """
    Draw camera positions with directional arrows on a matplotlib axes.

    Args:
        ax: Matplotlib axes to draw on
        cameras: List of Camera objects
        scale: Scale factor for arrows
        color: Marker and arrow color
        show_labels: Whether to show camera number labels
        arrow_length: Base length for direction arrows
    """
    for i, cam in enumerate(cameras):
        # Draw camera position marker
        ax.plot(
            cam.x, cam.y, 'o',
            color=color,
            markersize=6,
            markeredgecolor='white',
            markeredgewidth=1,
            zorder=10
        )

        # Draw direction arrow
        arrow_len = arrow_length * scale
        dx = arrow_len * np.cos(cam.yaw)
        dy = arrow_len * np.sin(cam.yaw)

        ax.annotate(
            '', xy=(cam.x + dx, cam.y + dy), xytext=(cam.x, cam.y),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            zorder=9
        )

        # Optional label
        if show_labels:
            ax.annotate(
                f'{i+1}', (cam.x + 3, cam.y + 3),
                fontsize=8, color='white', fontweight='bold',
                zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
            )


def plot_dem(
    dem: np.ndarray,
    ax: Optional[plt.Axes] = None,
    wall_threshold: float = 1e6,
    cmap: str = 'terrain',
    title: Optional[str] = None,
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot a DEM with optional wall visualization.

    Args:
        dem: 2D array of elevation values
        ax: Matplotlib axes (creates new figure if None)
        wall_threshold: Height above which cells are walls
        cmap: Colormap for terrain
        title: Optional title
        show_colorbar: Whether to show colorbar

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    im = ax.imshow(dem_display, cmap=cmap, origin='upper', aspect='equal')

    # Show walls as black
    if wall_mask.any():
        ax.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )

    if title:
        ax.set_title(title, fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Height', shrink=0.7)

    return ax


def plot_visibility(
    dem: np.ndarray,
    visible_any: np.ndarray,
    vis_count: np.ndarray,
    cameras: List[Camera],
    ax: Optional[plt.Axes] = None,
    wall_threshold: float = 1e6,
    title: Optional[str] = None,
    coverage: Optional[float] = None,
) -> plt.Axes:
    """
    Plot visibility overlay on DEM.

    Args:
        dem: DEM array
        visible_any: Boolean visibility mask
        vis_count: Camera count per cell
        cameras: List of cameras to draw
        ax: Matplotlib axes
        wall_threshold: Wall height threshold
        title: Optional title
        coverage: Optional coverage value to show in title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    # Show terrain
    ax.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal', alpha=0.5)

    # Show walls
    if wall_mask.any():
        ax.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )

    # Show visibility
    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal', alpha=0.6, vmin=0, vmax=1)

    # Draw cameras
    draw_camera_markers(ax, cameras)

    # Title
    if title:
        if coverage is not None:
            title = f"{title} - Coverage: {100*coverage:.1f}%"
        ax.set_title(title, fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax


def plot_camera_configuration(
    dem: np.ndarray,
    cameras: List[Camera],
    visible_any: np.ndarray,
    vis_count: np.ndarray,
    wall_threshold: float = 1e6,
    title: str = "Camera Configuration",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Create a comprehensive visualization of a camera configuration.

    Shows three panels:
    1. Terrain with camera positions
    2. Visibility coverage
    3. Camera count heatmap

    Args:
        dem: DEM array
        cameras: List of cameras
        visible_any: Visibility mask
        vis_count: Camera count array
        wall_threshold: Wall height threshold
        title: Figure title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax1, ax2, ax3 = axes

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)
    valid_mask = ~wall_mask

    # Compute coverage
    visible_valid = (visible_any > 0) & valid_mask
    coverage = visible_valid.sum() / valid_mask.sum() if valid_mask.sum() > 0 else 0.0

    # Panel 1: Terrain + Cameras
    ax1.set_title(f'{title}\nTerrain + Camera Positions', fontsize=11, fontweight='bold')
    im1 = ax1.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal')
    if wall_mask.any():
        ax1.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )
    draw_camera_markers(ax1, cameras, show_labels=True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Height', shrink=0.7)

    # Panel 2: Visibility Coverage
    ax2.set_title(f'Visibility Coverage: {100*coverage:.1f}%', fontsize=11, fontweight='bold')
    ax2.imshow(dem_display, cmap='gray', origin='upper', aspect='equal', alpha=0.3)
    if wall_mask.any():
        ax2.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )
    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax2.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal', alpha=0.7, vmin=0, vmax=1)
    draw_camera_markers(ax2, cameras)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Legend
    visible_patch = mpatches.Patch(color='green', alpha=0.7, label='Visible')
    not_visible_patch = mpatches.Patch(color='gray', alpha=0.3, label='Not visible')
    ax2.legend(handles=[visible_patch, not_visible_patch], loc='upper right', fontsize=8)

    # Panel 3: Camera Count Heatmap
    ax3.set_title('Camera Redundancy', fontsize=11, fontweight='bold')
    count_display = np.ma.masked_where(wall_mask | (vis_count == 0), vis_count)
    im3 = ax3.imshow(
        count_display, cmap='hot', origin='upper', aspect='equal',
        vmin=0, vmax=max(len(cameras), 1)
    )
    if wall_mask.any():
        ax3.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )
    draw_camera_markers(ax3, cameras, color='cyan')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='# Cameras', shrink=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')

    return fig


def plot_comparison(
    results: List[Tuple[str, float, float, int, List[float]]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Create comparison plot of multiple optimization methods.

    Args:
        results: List of (name, coverage, runtime, nfev, trace) tuples
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by coverage (descending)
    results_sorted = sorted(results, key=lambda t: t[1], reverse=True)

    # Left: Bar chart of final coverage
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

    # Add value labels
    for bar, cov in zip(bars, coverages):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cov:.1f}%', va='center', fontsize=9)

    ax1.set_xlim(0, max(coverages) * 1.15)
    ax1.grid(axis='x', alpha=0.3)

    # Right: Convergence curves
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

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')

    return fig


def plot_convergence(
    trace_coverage: List[float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Optimization Convergence",
) -> plt.Figure:
    """
    Plot convergence curve for a single optimization run.

    Args:
        trace_coverage: List of coverage values during optimization
        output_path: Optional path to save figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    generations = list(range(1, len(trace_coverage) + 1))
    coverage_pct = [c * 100 for c in trace_coverage]

    ax.plot(generations, coverage_pct, 'b-', linewidth=2)
    ax.fill_between(generations, coverage_pct, alpha=0.3)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
