#!/usr/bin/env python3
"""
Demo script for GPU-based visibility computation.

This script demonstrates:
1. Outdoor scenario: Synthetic terrain with hills and randomly placed cameras
2. Indoor scenario: Simple floorplan with walls and strategic camera placement

Outputs are saved to outputs/demo.png
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow
import matplotlib.patches as mpatches

from visibility_gpu import (
    compute_visibility,
    generate_synthetic_dem,
    create_floorplan_dem,
    create_random_cameras,
    create_preset_cameras,
    Camera
)


def draw_camera_markers(ax, cameras, scale=1.0, color='red'):
    """
    Draw camera positions with directional arrows showing yaw.

    Args:
        ax: Matplotlib axes.
        cameras: List of Camera objects.
        scale: Arrow length scale.
        color: Marker/arrow color.
    """
    for i, cam in enumerate(cameras):
        # Camera position marker
        ax.plot(cam.x, cam.y, 'o', color=color, markersize=8, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)

        # Direction arrow (yaw)
        arrow_len = 15 * scale
        dx = arrow_len * np.cos(cam.yaw)
        dy = arrow_len * np.sin(cam.yaw)

        ax.annotate('', xy=(cam.x + dx, cam.y + dy), xytext=(cam.x, cam.y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    zorder=9)

        # Camera label
        ax.annotate(f'{i+1}', (cam.x + 3, cam.y + 3), fontsize=8, color='white',
                    fontweight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))


def plot_visibility_result(
    dem, cameras, visible_any, vis_count, title,
    ax_dem, ax_vis, ax_count,
    wall_threshold=1e6,
    cmap_terrain='terrain',
    cmap_vis='Greens'
):
    """
    Plot DEM, visibility mask, and visibility count.

    Args:
        dem: DEM array.
        cameras: List of Camera objects.
        visible_any: Binary visibility mask.
        vis_count: Camera count per cell.
        title: Plot title prefix.
        ax_dem, ax_vis, ax_count: Matplotlib axes for each subplot.
        wall_threshold: Height above which cells are walls.
        cmap_terrain: Colormap for terrain.
        cmap_vis: Colormap for visibility.
    """
    height, width = dem.shape

    # Create masked DEM for visualization (walls as special color)
    dem_display = np.ma.masked_where(dem >= wall_threshold, dem)

    # Plot 1: DEM with camera positions
    ax_dem.set_title(f'{title} - Terrain + Cameras')
    im1 = ax_dem.imshow(dem_display, cmap=cmap_terrain, origin='upper', aspect='equal')
    # Show walls in black
    wall_mask = dem >= wall_threshold
    if wall_mask.any():
        ax_dem.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                      aspect='equal', alpha=0.9, vmin=0, vmax=1)
    draw_camera_markers(ax_dem, cameras, scale=1.0, color='red')
    ax_dem.set_xlabel('X (columns)')
    ax_dem.set_ylabel('Y (rows)')
    plt.colorbar(im1, ax=ax_dem, label='Height', shrink=0.7)

    # Plot 2: Visible any overlay
    ax_vis.set_title(f'{title} - Visibility (Any Camera)')
    ax_vis.imshow(dem_display, cmap='gray', origin='upper', aspect='equal', alpha=0.3)
    # Show walls
    if wall_mask.any():
        ax_vis.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                      aspect='equal', alpha=0.9, vmin=0, vmax=1)
    # Visibility overlay
    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax_vis.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal',
                  alpha=0.7, vmin=0, vmax=1)
    draw_camera_markers(ax_vis, cameras, scale=1.0, color='red')
    ax_vis.set_xlabel('X (columns)')
    ax_vis.set_ylabel('Y (rows)')

    # Add legend
    visible_patch = mpatches.Patch(color='green', alpha=0.7, label='Visible')
    not_visible_patch = mpatches.Patch(color='gray', alpha=0.3, label='Not visible')
    ax_vis.legend(handles=[visible_patch, not_visible_patch], loc='upper right', fontsize=8)

    # Compute coverage statistics
    valid_cells = (dem < wall_threshold).sum()
    visible_cells = (visible_any > 0).sum()
    coverage_pct = 100 * visible_cells / valid_cells if valid_cells > 0 else 0

    # Plot 3: Visibility count heatmap
    ax_count.set_title(f'{title} - Camera Count (Coverage: {coverage_pct:.1f}%)')
    # Mask walls and non-visible
    count_display = np.ma.masked_where((dem >= wall_threshold) | (vis_count == 0), vis_count)
    im3 = ax_count.imshow(count_display, cmap='hot', origin='upper', aspect='equal',
                          vmin=0, vmax=max(len(cameras), 1))
    # Show walls
    if wall_mask.any():
        ax_count.imshow(np.where(wall_mask, 1, np.nan), cmap='binary', origin='upper',
                        aspect='equal', alpha=0.9, vmin=0, vmax=1)
    draw_camera_markers(ax_count, cameras, scale=1.0, color='cyan')
    ax_count.set_xlabel('X (columns)')
    ax_count.set_ylabel('Y (rows)')
    plt.colorbar(im3, ax=ax_count, label='# Cameras', shrink=0.7)


def run_outdoor_demo(seed=42):
    """
    Run outdoor terrain visibility demo with random cameras.

    Returns:
        dem, cameras, visible_any, vis_count
    """
    print("=" * 60)
    print("OUTDOOR DEMO: Hills terrain with 6 random cameras")
    print("=" * 60)

    # Generate synthetic terrain
    print("Generating synthetic DEM (512x512, hills mode)...")
    dem = generate_synthetic_dem(height=512, width=512, mode='hills', seed=seed)
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM height range: [{dem.min():.1f}, {dem.max():.1f}]")

    # Create random cameras
    print("\nCreating 6 random cameras...")
    cameras = create_random_cameras(
        num_cameras=6,
        dem=dem,
        height_above_ground=10.0,
        hfov=np.deg2rad(90),   # 90 degree horizontal FOV
        vfov=np.deg2rad(60),   # 60 degree vertical FOV
        max_range=150.0,
        seed=seed
    )

    for i, cam in enumerate(cameras):
        print(f"  Camera {i+1}: pos=({cam.x:.1f}, {cam.y:.1f}, {cam.z:.1f}), "
              f"yaw={np.rad2deg(cam.yaw):.0f}deg, pitch={np.rad2deg(cam.pitch):.0f}deg")

    # Compute visibility on GPU
    print("\nComputing visibility on GPU...")
    import time
    t0 = time.time()
    visible_any, vis_count = compute_visibility(dem, cameras)
    t1 = time.time()
    print(f"  Computation time: {(t1-t0)*1000:.1f} ms")

    # Statistics
    total_cells = dem.size
    visible_cells = (visible_any > 0).sum()
    print(f"\nResults:")
    print(f"  Total cells: {total_cells}")
    print(f"  Visible cells: {visible_cells} ({100*visible_cells/total_cells:.1f}%)")
    print(f"  Max camera count: {vis_count.max()}")

    return dem, cameras, visible_any, vis_count


def run_indoor_demo(seed=123):
    """
    Run indoor floorplan visibility demo with walls.

    Returns:
        dem, cameras, visible_any, vis_count
    """
    print("\n" + "=" * 60)
    print("INDOOR DEMO: Floorplan with walls and 4 strategic cameras")
    print("=" * 60)

    # Generate floorplan DEM
    print("Generating indoor floorplan DEM (256x256)...")
    dem = create_floorplan_dem(height=256, width=256, wall_height=1e9, base_height=0.0)

    wall_cells = (dem >= 1e6).sum()
    print(f"  DEM shape: {dem.shape}")
    print(f"  Wall cells: {wall_cells}")

    # Create strategic camera placements (corners of rooms)
    print("\nPlacing 4 cameras in room corners...")

    # Manual camera placement for good coverage
    cameras = [
        Camera(x=50.0, y=50.0, z=2.5, yaw=np.deg2rad(45), pitch=np.deg2rad(-10),
               hfov=np.deg2rad(110), vfov=np.deg2rad(80), max_range=100.0),
        Camera(x=200.0, y=50.0, z=2.5, yaw=np.deg2rad(135), pitch=np.deg2rad(-10),
               hfov=np.deg2rad(110), vfov=np.deg2rad(80), max_range=100.0),
        Camera(x=50.0, y=200.0, z=2.5, yaw=np.deg2rad(-45), pitch=np.deg2rad(-10),
               hfov=np.deg2rad(110), vfov=np.deg2rad(80), max_range=100.0),
        Camera(x=200.0, y=200.0, z=2.5, yaw=np.deg2rad(-135), pitch=np.deg2rad(-10),
               hfov=np.deg2rad(110), vfov=np.deg2rad(80), max_range=100.0),
    ]

    for i, cam in enumerate(cameras):
        print(f"  Camera {i+1}: pos=({cam.x:.1f}, {cam.y:.1f}, {cam.z:.1f}), "
              f"yaw={np.rad2deg(cam.yaw):.0f}deg")

    # Compute visibility on GPU
    print("\nComputing visibility on GPU...")
    import time
    t0 = time.time()
    visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold=1e6)
    t1 = time.time()
    print(f"  Computation time: {(t1-t0)*1000:.1f} ms")

    # Statistics (exclude walls)
    valid_cells = (dem < 1e6).sum()
    visible_cells = (visible_any > 0).sum()
    print(f"\nResults:")
    print(f"  Valid (non-wall) cells: {valid_cells}")
    print(f"  Visible cells: {visible_cells} ({100*visible_cells/valid_cells:.1f}%)")
    print(f"  Max camera count: {vis_count.max()}")

    return dem, cameras, visible_any, vis_count


def main():
    """Main demo entry point."""
    print("GPU Visibility Engine - Step 1 Demo")
    print("Using PyTorch CUDA extension with custom .cu kernel\n")

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA Device: {props.name}")
        print(f"Compute capability: {props.major}.{props.minor}")
        mem_total = props.total_memory / 1e9
        mem_free = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"GPU Memory: {mem_total:.1f} GB total, {mem_free:.1f} GB free\n")
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        print("This demo requires a CUDA-capable GPU with PyTorch CUDA support.")
        sys.exit(1)

    # Run both demos
    dem_outdoor, cams_outdoor, vis_any_outdoor, vis_count_outdoor = run_outdoor_demo(1)
    dem_indoor, cams_indoor, vis_any_indoor, vis_count_indoor = run_indoor_demo()
    dem_outdoor, cams_outdoor, vis_any_outdoor, vis_count_outdoor = run_outdoor_demo(2)
    dem_outdoor, cams_outdoor, vis_any_outdoor, vis_count_outdoor = run_outdoor_demo(3)
    dem_indoor, cams_indoor, vis_any_indoor, vis_count_indoor = run_indoor_demo(4)
    dem_indoor, cams_indoor, vis_any_indoor, vis_count_indoor = run_indoor_demo(5)

    # Create combined visualization
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 12))

    # Outdoor demo (top row)
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    plot_visibility_result(
        dem_outdoor, cams_outdoor, vis_any_outdoor, vis_count_outdoor,
        "Outdoor", ax1, ax2, ax3
    )

    # Indoor demo (bottom row)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    plot_visibility_result(
        dem_indoor, cams_indoor, vis_any_indoor, vis_count_indoor,
        "Indoor", ax4, ax5, ax6,
        cmap_terrain='gray'
    )

    plt.suptitle('GPU Visibility Engine Demo - Step 1\n'
                 'Per-cell visibility with FOV gating + LOS occlusion sampling',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save output
    output_path = os.path.join(os.path.dirname(__file__), 'outputs', 'demo.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nOutput saved to: {output_path}")

    plt.show()

    print("\nDemo complete!")


if __name__ == '__main__':
    main()
