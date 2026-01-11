"""
Constraint system for camera placement optimization.

This module defines constraints that control where cameras can be placed,
what areas need coverage, and other restrictions on the optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from tactica.core.sensors import Camera


@dataclass
class OptimizationConstraints:
    """
    Constraints for camera placement optimization.

    This dataclass defines all constraints that affect camera placement:
    - Where cameras can/cannot be placed
    - Priority weights for different areas
    - Fixed cameras that cannot be moved
    - Coverage requirements

    Attributes:
        placement_mask: Boolean array where True = valid camera placement.
            If None, cameras can be placed anywhere except walls.
        exclusion_mask: Boolean array where True = camera cannot be placed here.
            Takes precedence over placement_mask.
        priority_weights: Float array of importance weights per cell.
            Higher values indicate more important areas to cover.
            If None, all valid cells have equal priority.
        min_coverage: Minimum coverage fraction required (0.0 to 1.0).
        min_coverage_per_zone: Dict mapping zone_id to required coverage.
            Zone IDs are integer labels in a zones_mask array.
        zones_mask: Integer array assigning cells to zones for coverage requirements.
        fixed_cameras: List of Camera objects that cannot be moved.
            These are included in visibility but not optimized.
        max_cameras: Maximum number of cameras allowed.
            If None, uses num_cameras from problem config.
        height_constraints: Optional (H, W) array of maximum camera heights.
            Useful for indoor scenarios with varying ceiling heights.
        mounting_points: Optional list of (x, y) valid camera mounting positions.
            If provided, cameras must be placed at these exact locations.

    Example:
        >>> constraints = OptimizationConstraints(
        ...     min_coverage=0.8,  # Require 80% coverage
        ...     priority_weights=priority_map,  # Emphasize entrances
        ... )
    """
    # Placement constraints
    placement_mask: Optional[np.ndarray] = None
    exclusion_mask: Optional[np.ndarray] = None

    # Priority weights (importance per cell)
    priority_weights: Optional[np.ndarray] = None

    # Coverage requirements
    min_coverage: float = 0.0
    min_coverage_per_zone: Optional[Dict[int, float]] = None
    zones_mask: Optional[np.ndarray] = None

    # Fixed cameras
    fixed_cameras: Optional[List[Camera]] = None

    # Camera count constraints
    max_cameras: Optional[int] = None

    # Height constraints
    height_constraints: Optional[np.ndarray] = None

    # Mounting point constraints
    mounting_points: Optional[List[Tuple[float, float]]] = None

    def validate(self, dem_shape: Tuple[int, int]) -> None:
        """
        Validate constraints against a DEM shape.

        Args:
            dem_shape: (height, width) of the DEM

        Raises:
            ValueError: If constraints are invalid
        """
        height, width = dem_shape

        if self.placement_mask is not None:
            if self.placement_mask.shape != dem_shape:
                raise ValueError(
                    f"placement_mask shape {self.placement_mask.shape} "
                    f"doesn't match DEM shape {dem_shape}"
                )

        if self.exclusion_mask is not None:
            if self.exclusion_mask.shape != dem_shape:
                raise ValueError(
                    f"exclusion_mask shape {self.exclusion_mask.shape} "
                    f"doesn't match DEM shape {dem_shape}"
                )

        if self.priority_weights is not None:
            if self.priority_weights.shape != dem_shape:
                raise ValueError(
                    f"priority_weights shape {self.priority_weights.shape} "
                    f"doesn't match DEM shape {dem_shape}"
                )

        if self.zones_mask is not None:
            if self.zones_mask.shape != dem_shape:
                raise ValueError(
                    f"zones_mask shape {self.zones_mask.shape} "
                    f"doesn't match DEM shape {dem_shape}"
                )

        if self.height_constraints is not None:
            if self.height_constraints.shape != dem_shape:
                raise ValueError(
                    f"height_constraints shape {self.height_constraints.shape} "
                    f"doesn't match DEM shape {dem_shape}"
                )

        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {self.min_coverage}")

    def get_valid_placement_mask(
        self,
        dem: np.ndarray,
        wall_threshold: float = 1e6,
    ) -> np.ndarray:
        """
        Compute the combined valid placement mask.

        Combines placement_mask, exclusion_mask, and wall detection
        to produce a final mask of valid camera positions.

        Args:
            dem: DEM array
            wall_threshold: Height above which cells are walls

        Returns:
            Boolean array where True = valid camera placement
        """
        # Start with all non-wall cells as valid
        valid = dem < wall_threshold

        # Apply placement mask (intersection)
        if self.placement_mask is not None:
            valid = valid & self.placement_mask

        # Apply exclusion mask (subtraction)
        if self.exclusion_mask is not None:
            valid = valid & ~self.exclusion_mask

        return valid

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (without large arrays)."""
        return {
            "min_coverage": self.min_coverage,
            "min_coverage_per_zone": self.min_coverage_per_zone,
            "max_cameras": self.max_cameras,
            "has_placement_mask": self.placement_mask is not None,
            "has_exclusion_mask": self.exclusion_mask is not None,
            "has_priority_weights": self.priority_weights is not None,
            "has_zones_mask": self.zones_mask is not None,
            "has_height_constraints": self.height_constraints is not None,
            "num_fixed_cameras": len(self.fixed_cameras) if self.fixed_cameras else 0,
            "num_mounting_points": len(self.mounting_points) if self.mounting_points else 0,
        }


def create_priority_mask_from_zones(
    zones_mask: np.ndarray,
    zone_priorities: Dict[int, float],
    default_priority: float = 1.0,
) -> np.ndarray:
    """
    Create a priority weight array from a zones mask.

    Args:
        zones_mask: Integer array assigning cells to zones
        zone_priorities: Dict mapping zone_id to priority weight
        default_priority: Priority for zones not in the dict

    Returns:
        Float array of priority weights

    Example:
        >>> zones = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
        >>> priorities = create_priority_mask_from_zones(
        ...     zones, {0: 1.0, 1: 2.0, 2: 0.5}
        ... )
    """
    priority = np.full(zones_mask.shape, default_priority, dtype=np.float32)

    for zone_id, weight in zone_priorities.items():
        priority[zones_mask == zone_id] = weight

    return priority


def create_exclusion_from_buffer(
    dem: np.ndarray,
    wall_threshold: float = 1e6,
    buffer_cells: int = 5,
) -> np.ndarray:
    """
    Create an exclusion mask that buffers around walls.

    Cameras often shouldn't be placed right next to walls,
    so this creates an exclusion zone around all wall cells.

    Args:
        dem: DEM array
        wall_threshold: Height above which cells are walls
        buffer_cells: Number of cells to buffer around walls

    Returns:
        Boolean exclusion mask
    """
    from scipy.ndimage import binary_dilation

    walls = dem >= wall_threshold

    # Dilate walls to create buffer zone
    structure = np.ones((2 * buffer_cells + 1, 2 * buffer_cells + 1))
    exclusion = binary_dilation(walls, structure=structure)

    return exclusion


def create_placement_from_elevation(
    dem: np.ndarray,
    min_elevation: Optional[float] = None,
    max_elevation: Optional[float] = None,
    wall_threshold: float = 1e6,
) -> np.ndarray:
    """
    Create a placement mask based on elevation constraints.

    Useful for restricting cameras to certain terrain levels
    (e.g., only on ridges, or only in valleys).

    Args:
        dem: DEM array
        min_elevation: Minimum allowed elevation (None = no minimum)
        max_elevation: Maximum allowed elevation (None = no maximum)
        wall_threshold: Height above which cells are walls

    Returns:
        Boolean placement mask
    """
    valid = dem < wall_threshold

    if min_elevation is not None:
        valid = valid & (dem >= min_elevation)

    if max_elevation is not None:
        valid = valid & (dem <= max_elevation)

    return valid
