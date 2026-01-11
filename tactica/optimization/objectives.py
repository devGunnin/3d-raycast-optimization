"""
Objective functions for camera placement optimization.

This module provides various objective functions that can be used
to evaluate camera placement solutions.
"""

from typing import Optional, List
import numpy as np

from tactica.core.sensors import Camera
from tactica.core.visibility import compute_visibility


def compute_coverage_objective(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute simple coverage fraction.

    Args:
        dem: DEM array
        cameras: List of cameras
        wall_threshold: Height threshold for walls
        valid_mask: Optional mask of cells to consider (default: all non-walls)

    Returns:
        Coverage fraction (0 to 1)
    """
    visible_any, _ = compute_visibility(dem, cameras, wall_threshold)

    if valid_mask is None:
        valid_mask = dem < wall_threshold

    num_valid = valid_mask.sum()
    if num_valid == 0:
        return 0.0

    visible_valid = (visible_any > 0) & valid_mask
    return float(visible_valid.sum() / num_valid)


def compute_weighted_coverage(
    dem: np.ndarray,
    cameras: List[Camera],
    weights: np.ndarray,
    wall_threshold: float = 1e6,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute weighted coverage using priority weights.

    Args:
        dem: DEM array
        cameras: List of cameras
        weights: Priority weights per cell (higher = more important)
        wall_threshold: Height threshold for walls
        valid_mask: Optional mask of cells to consider

    Returns:
        Weighted coverage fraction (0 to 1)
    """
    visible_any, _ = compute_visibility(dem, cameras, wall_threshold)

    if valid_mask is None:
        valid_mask = dem < wall_threshold

    total_weight = (weights * valid_mask).sum()
    if total_weight == 0:
        return 0.0

    visible_valid = (visible_any > 0) & valid_mask
    visible_weight = (weights * visible_valid).sum()

    return float(visible_weight / total_weight)


def compute_redundancy_score(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute redundancy metrics for a camera configuration.

    Args:
        dem: DEM array
        cameras: List of cameras
        wall_threshold: Height threshold for walls
        valid_mask: Optional mask of cells to consider

    Returns:
        Dict with redundancy metrics:
        - redundancy_fraction: Fraction of visible cells seen by 2+ cameras
        - avg_cameras_per_cell: Average number of cameras seeing each visible cell
        - max_cameras_per_cell: Maximum cameras seeing any single cell
    """
    _, vis_count = compute_visibility(dem, cameras, wall_threshold)

    if valid_mask is None:
        valid_mask = dem < wall_threshold

    visible_mask = (vis_count > 0) & valid_mask
    num_visible = visible_mask.sum()

    if num_visible == 0:
        return {
            "redundancy_fraction": 0.0,
            "avg_cameras_per_cell": 0.0,
            "max_cameras_per_cell": 0,
        }

    redundant_mask = (vis_count > 1) & valid_mask
    redundancy_fraction = redundant_mask.sum() / num_visible

    avg_cameras = vis_count[visible_mask].mean()
    max_cameras = int(vis_count.max())

    return {
        "redundancy_fraction": float(redundancy_fraction),
        "avg_cameras_per_cell": float(avg_cameras),
        "max_cameras_per_cell": max_cameras,
    }


def compute_zone_coverage(
    dem: np.ndarray,
    cameras: List[Camera],
    zones_mask: np.ndarray,
    wall_threshold: float = 1e6,
) -> dict:
    """
    Compute coverage per zone.

    Args:
        dem: DEM array
        cameras: List of cameras
        zones_mask: Integer array assigning cells to zones
        wall_threshold: Height threshold for walls

    Returns:
        Dict mapping zone_id to coverage fraction
    """
    visible_any, _ = compute_visibility(dem, cameras, wall_threshold)
    valid_mask = dem < wall_threshold

    zone_coverage = {}
    for zone_id in np.unique(zones_mask):
        zone_mask = (zones_mask == zone_id) & valid_mask
        zone_valid = zone_mask.sum()

        if zone_valid == 0:
            zone_coverage[int(zone_id)] = 0.0
        else:
            zone_visible = ((visible_any > 0) & zone_mask).sum()
            zone_coverage[int(zone_id)] = float(zone_visible / zone_valid)

    return zone_coverage


def compute_coverage_gap(
    dem: np.ndarray,
    cameras: List[Camera],
    wall_threshold: float = 1e6,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the "coverage gap" - cells that are valid but not visible.

    Useful for visualization and for guiding optimization.

    Args:
        dem: DEM array
        cameras: List of cameras
        wall_threshold: Height threshold for walls
        valid_mask: Optional mask of cells to consider

    Returns:
        Boolean mask where True = valid but not covered
    """
    visible_any, _ = compute_visibility(dem, cameras, wall_threshold)

    if valid_mask is None:
        valid_mask = dem < wall_threshold

    gap = valid_mask & (visible_any == 0)
    return gap


def compute_multi_objective_fitness(
    dem: np.ndarray,
    cameras: List[Camera],
    weights: Optional[np.ndarray] = None,
    wall_threshold: float = 1e6,
    coverage_weight: float = 1.0,
    redundancy_weight: float = 0.1,
    cost_weight: float = 0.0,
    camera_cost: float = 1.0,
) -> dict:
    """
    Compute multi-objective fitness with customizable weights.

    This function computes a weighted combination of:
    - Coverage (fraction of area visible)
    - Redundancy (bonus for overlapping coverage)
    - Cost (penalty for number of cameras)

    Args:
        dem: DEM array
        cameras: List of cameras
        weights: Optional priority weights
        wall_threshold: Height threshold for walls
        coverage_weight: Weight for coverage objective
        redundancy_weight: Weight for redundancy bonus
        cost_weight: Weight for camera cost penalty
        camera_cost: Cost per camera

    Returns:
        Dict with individual objectives and combined fitness
    """
    visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold)
    valid_mask = dem < wall_threshold
    num_valid = valid_mask.sum()

    if num_valid == 0:
        return {
            "coverage": 0.0,
            "weighted_coverage": 0.0,
            "redundancy": 0.0,
            "cost": len(cameras) * camera_cost,
            "fitness": 0.0,
        }

    # Coverage
    visible_valid = (visible_any > 0) & valid_mask
    coverage = visible_valid.sum() / num_valid

    # Weighted coverage
    if weights is not None:
        total_weight = (weights * valid_mask).sum()
        visible_weight = (weights * visible_valid).sum()
        weighted_coverage = visible_weight / total_weight if total_weight > 0 else 0.0
    else:
        weighted_coverage = coverage

    # Redundancy
    redundant = ((vis_count > 1) & valid_mask).sum()
    redundancy = redundant / num_valid

    # Cost
    cost = len(cameras) * camera_cost

    # Combined fitness (higher is better)
    fitness = (
        coverage_weight * weighted_coverage +
        redundancy_weight * redundancy -
        cost_weight * cost
    )

    return {
        "coverage": float(coverage),
        "weighted_coverage": float(weighted_coverage),
        "redundancy": float(redundancy),
        "cost": float(cost),
        "fitness": float(fitness),
    }
