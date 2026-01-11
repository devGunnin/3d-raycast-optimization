"""
Animation and GIF generation for camera placement optimization.

This module provides functions for creating animated visualizations
of the optimization process.
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tactica.core.sensors import Camera
from tactica.core.visibility import compute_visibility
from tactica.visualization.plotting import draw_camera_markers


def create_frame(
    dem: np.ndarray,
    cameras: List[Camera],
    coverage: float,
    wall_threshold: float = 1e6,
    title: str = "Optimization",
    eval_num: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> np.ndarray:
    """
    Create a single frame for an optimization GIF.

    Args:
        dem: DEM array
        cameras: List of cameras for this frame
        coverage: Coverage value to display
        wall_threshold: Wall height threshold
        title: Frame title
        eval_num: Optional evaluation number to display
        figsize: Figure size

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Compute visibility
    try:
        visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold)
    except Exception:
        visible_any = np.zeros_like(dem, dtype=np.uint8)
        vis_count = np.zeros_like(dem, dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    wall_mask = dem >= wall_threshold
    dem_display = np.ma.masked_where(wall_mask, dem)

    # Left: DEM + cameras + visibility
    if eval_num is not None:
        frame_title = f'{title}\nEval {eval_num} - Coverage: {100*coverage:.1f}%'
    else:
        frame_title = f'{title}\nCoverage: {100*coverage:.1f}%'

    ax1.set_title(frame_title, fontsize=10, fontweight='bold')
    ax1.imshow(dem_display, cmap='terrain', origin='upper', aspect='equal', alpha=0.5)

    if wall_mask.any():
        ax1.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )

    vis_overlay = np.ma.masked_where(visible_any == 0, visible_any)
    ax1.imshow(vis_overlay, cmap='Greens', origin='upper', aspect='equal', alpha=0.6, vmin=0, vmax=1)

    draw_camera_markers(ax1, cameras)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Right: Camera count heatmap
    ax2.set_title('Camera Coverage Count', fontsize=10)
    count_display = np.ma.masked_where(wall_mask | (vis_count == 0), vis_count)
    im = ax2.imshow(
        count_display, cmap='hot', origin='upper', aspect='equal',
        vmin=0, vmax=max(len(cameras), 1)
    )

    if wall_mask.any():
        ax2.imshow(
            np.where(wall_mask, 1, np.nan),
            cmap='binary', origin='upper',
            aspect='equal', alpha=0.9, vmin=0, vmax=1
        )

    draw_camera_markers(ax2, cameras, color='cyan')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, label='# Cameras', shrink=0.7)

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]  # Drop alpha channel

    plt.close(fig)

    return image


def create_optimization_gif(
    dem: np.ndarray,
    trace_x: List[np.ndarray],
    trace_coverage: List[float],
    decode_fn,
    output_path: Union[str, Path],
    wall_threshold: float = 1e6,
    title: str = "Optimization Progress",
    fps: int = 5,
    max_frames: int = 50,
    pause_at_end: int = 2,
) -> None:
    """
    Create a GIF showing optimization progress.

    Args:
        dem: DEM array
        trace_x: List of parameter vectors from optimization
        trace_coverage: List of coverage values from optimization
        decode_fn: Function to decode parameter vector to cameras
        output_path: Path to save the GIF
        wall_threshold: Wall height threshold
        title: GIF title
        fps: Frames per second
        max_frames: Maximum number of frames to include
        pause_at_end: Seconds to pause on final frame
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("imageio is required for GIF generation. Install with: pip install imageio")

    if len(trace_x) == 0:
        print(f"No trace data available for GIF generation")
        return

    n_frames = len(trace_x)

    # Select frames to include (evenly spaced)
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
    else:
        indices = list(range(n_frames))

    print(f"Generating optimization GIF with {len(indices)} frames...")

    frames = []
    for idx in indices:
        x = trace_x[idx]
        coverage = trace_coverage[idx]
        cameras = decode_fn(x)
        eval_num = (idx + 1) * 10  # Approximate evaluation number

        frame = create_frame(
            dem, cameras, coverage,
            wall_threshold=wall_threshold,
            title=title,
            eval_num=eval_num,
        )
        frames.append(frame)

    # Add pause at end
    for _ in range(fps * pause_at_end):
        frames.append(frames[-1])

    # Save GIF
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")


def create_comparison_gif(
    dem: np.ndarray,
    results: dict,
    decode_fn,
    output_path: Union[str, Path],
    wall_threshold: float = 1e6,
    fps: int = 3,
    max_frames_per_method: int = 20,
) -> None:
    """
    Create a GIF comparing multiple optimization methods.

    Args:
        dem: DEM array
        results: Dict mapping method name to (trace_x, trace_coverage)
        decode_fn: Function to decode parameter vector to cameras
        output_path: Path to save the GIF
        wall_threshold: Wall height threshold
        fps: Frames per second
        max_frames_per_method: Max frames per method
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("imageio is required for GIF generation")

    frames = []

    for method_name, (trace_x, trace_coverage) in results.items():
        if len(trace_x) == 0:
            continue

        n_frames = len(trace_x)
        if n_frames > max_frames_per_method:
            indices = np.linspace(0, n_frames - 1, max_frames_per_method, dtype=int)
        else:
            indices = list(range(n_frames))

        for idx in indices:
            x = trace_x[idx]
            coverage = trace_coverage[idx]
            cameras = decode_fn(x)

            frame = create_frame(
                dem, cameras, coverage,
                wall_threshold=wall_threshold,
                title=method_name,
                eval_num=(idx + 1) * 10,
            )
            frames.append(frame)

        # Pause between methods
        for _ in range(fps):
            frames.append(frames[-1])

    if frames:
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        print(f"Comparison GIF saved to: {output_path}")
