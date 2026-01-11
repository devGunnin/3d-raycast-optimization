"""
GPU-based visibility computation for camera placement optimization.

This module provides a Python interface to the CUDA visibility kernel,
which computes per-cell visibility for multiple cameras viewing a DEM.

The CUDA kernel is JIT-compiled via PyTorch's cpp_extension system.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.cpp_extension import load

from tactica.core.sensors import Camera

# =============================================================================
# CUDA Extension Loading (JIT Compilation)
# =============================================================================

_cuda_module = None


def _get_cuda_module():
    """Load and cache the CUDA extension module."""
    global _cuda_module
    if _cuda_module is None:
        # Look for CUDA kernel in multiple locations
        possible_paths = [
            Path(__file__).parent / 'cuda' / 'visibility_kernel.cu',
            Path(__file__).parent.parent.parent / 'src' / 'cuda' / 'visibility_kernel.cu',
        ]

        kernel_path = None
        for path in possible_paths:
            if path.exists():
                kernel_path = path
                break

        if kernel_path is None:
            raise RuntimeError(
                f"CUDA kernel not found. Searched paths: {possible_paths}"
            )

        print(f"Compiling CUDA visibility kernel from {kernel_path}...")
        _cuda_module = load(
            name='visibility_cuda',
            sources=[str(kernel_path)],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print("CUDA kernel compiled successfully.")

    return _cuda_module


def _check_cuda_available():
    """Check if CUDA is available and raise informative error if not."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This module requires a CUDA-capable GPU. "
            "Please check your PyTorch installation and GPU drivers."
        )


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

    This function runs a CUDA kernel that, for each DEM cell:
    1. Checks if the cell is within each camera's FOV and range
    2. Performs ray-marching occlusion tests against the terrain
    3. Aggregates visibility counts across all cameras

    Args:
        dem: 2D numpy array (H, W) of terrain heights (float32).
            Cells with height >= wall_threshold are treated as walls.
        cameras: List of Camera objects defining viewer positions and FOV.
            Maximum 64 cameras supported.
        wall_threshold: Height value above which cells are treated as walls.
            Wall cells block visibility and cannot themselves be "seen".
        occlusion_epsilon: Tolerance for occlusion test (prevents z-fighting).
            A small positive value prevents numerical issues at grazing angles.

    Returns:
        visible_any: uint8 array (H, W), 1 if any camera sees the cell, 0 otherwise.
        vis_count: int32 array (H, W), count of cameras seeing each cell.

    Raises:
        RuntimeError: If CUDA is not available.
        ValueError: If inputs are invalid (wrong dimensions, too many cameras).

    Example:
        >>> from tactica.dem import generate_synthetic_dem
        >>> from tactica.core import Camera
        >>> dem = generate_synthetic_dem(256, 256, mode='hills')
        >>> cameras = [Camera(128, 128, 20, 0, -0.1, 1.57, 1.0, 100)]
        >>> visible_any, vis_count = compute_visibility(dem, cameras)
        >>> coverage = visible_any.sum() / (dem < 1e6).sum()
        >>> print(f"Coverage: {coverage:.1%}")
    """
    # Check CUDA availability
    _check_cuda_available()

    # Validate inputs
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2D, got shape {dem.shape}")
    if len(cameras) == 0:
        raise ValueError("Must provide at least one camera")
    if len(cameras) > 64:
        raise ValueError(f"Maximum 64 cameras supported, got {len(cameras)}")

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


def compute_coverage(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    priority_weights: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute coverage statistics for a camera configuration.

    This is a convenience wrapper around compute_visibility that returns
    useful statistics about the coverage.

    Args:
        dem: 2D terrain array.
        cameras: List of cameras.
        wall_threshold: Height threshold for walls.
        priority_weights: Optional (H, W) array of importance weights per cell.
            Higher values indicate more important areas.
        valid_mask: Optional (H, W) boolean mask of valid cells to consider.
            If None, all non-wall cells are considered valid.

    Returns:
        Dictionary with coverage statistics:
        - coverage: Fraction of valid cells visible by at least one camera
        - weighted_coverage: Coverage weighted by priority_weights (if provided)
        - redundancy: Average number of cameras seeing each visible cell
        - visible_cells: Total number of visible cells
        - valid_cells: Total number of valid cells
        - visible_any: The visibility mask (H, W)
        - vis_count: The camera count array (H, W)
    """
    visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold)

    # Determine valid cells
    if valid_mask is None:
        valid_mask = dem < wall_threshold

    num_valid = valid_mask.sum()
    if num_valid == 0:
        return {
            "coverage": 0.0,
            "weighted_coverage": 0.0,
            "redundancy": 0.0,
            "visible_cells": 0,
            "valid_cells": 0,
            "visible_any": visible_any,
            "vis_count": vis_count,
        }

    # Basic coverage
    visible_valid = (visible_any > 0) & valid_mask
    num_visible = visible_valid.sum()
    coverage = num_visible / num_valid

    # Weighted coverage
    if priority_weights is not None:
        total_weight = (priority_weights * valid_mask).sum()
        visible_weight = (priority_weights * visible_valid).sum()
        weighted_coverage = visible_weight / total_weight if total_weight > 0 else 0.0
    else:
        weighted_coverage = coverage

    # Redundancy
    if num_visible > 0:
        redundancy = vis_count[visible_valid].mean()
    else:
        redundancy = 0.0

    return {
        "coverage": float(coverage),
        "weighted_coverage": float(weighted_coverage),
        "redundancy": float(redundancy),
        "visible_cells": int(num_visible),
        "valid_cells": int(num_valid),
        "visible_any": visible_any,
        "vis_count": vis_count,
    }
