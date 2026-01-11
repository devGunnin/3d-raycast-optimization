"""
DEM (Digital Elevation Model) loading, generation, and coordinate handling.

This module provides tools for working with terrain data:
    - Loading real DEMs from GeoTIFF and other formats
    - Generating synthetic DEMs for testing
    - Coordinate system transformations
    - DEM manipulation (adding walls, resampling, etc.)
"""

from tactica.dem.synthetic import (
    generate_synthetic_dem,
    generate_random_dem,
    create_floorplan_dem,
    add_walls_to_dem,
    add_wall_lines,
)

from tactica.dem.loader import (
    load_dem,
    DEMMetadata,
)

from tactica.dem.coordinates import (
    grid_to_world,
    world_to_grid,
)

__all__ = [
    # Synthetic generation
    "generate_synthetic_dem",
    "generate_random_dem",
    "create_floorplan_dem",
    "add_walls_to_dem",
    "add_wall_lines",
    # Loading
    "load_dem",
    "DEMMetadata",
    # Coordinates
    "grid_to_world",
    "world_to_grid",
]
