"""
Tactica - GPU-accelerated camera/sensor placement optimization engine.

This package provides tools for optimizing the placement of cameras and sensors
to maximize visibility coverage over terrain (DEM) data.

Main modules:
    - tactica.core: GPU visibility computation and sensor models
    - tactica.dem: DEM loading, generation, and coordinate handling
    - tactica.optimization: Optimization problem definition and runners
    - tactica.visualization: Plotting and animation utilities
    - tactica.config: Configuration management
    - tactica.api: REST/WebSocket API schemas and routes

Quick start:
    >>> from tactica import optimize_camera_placement
    >>> from tactica.dem import load_dem, generate_synthetic
    >>> from tactica.core import Camera, CameraResolution
    >>>
    >>> dem = generate_synthetic(256, 256, mode='hills')
    >>> result = optimize_camera_placement(dem, num_cameras=4)
    >>> print(f"Coverage: {result.coverage:.1%}")
"""

__version__ = "0.2.0"
__author__ = "Tactica Team"

# Core exports
from tactica.core.sensors import Camera, CameraResolution, compute_max_range_from_fov
from tactica.core.visibility import compute_visibility

# DEM exports
from tactica.dem.synthetic import (
    generate_synthetic_dem,
    generate_random_dem,
    create_floorplan_dem,
)

# Optimization exports
from tactica.optimization.problem import CameraPlacementProblem
from tactica.optimization.constraints import OptimizationConstraints
from tactica.optimization.runner import optimize_camera_placement

# Config exports
from tactica.config.settings import TacticaConfig

__all__ = [
    # Version
    "__version__",
    # Core
    "Camera",
    "CameraResolution",
    "compute_max_range_from_fov",
    "compute_visibility",
    # DEM
    "generate_synthetic_dem",
    "generate_random_dem",
    "create_floorplan_dem",
    # Optimization
    "CameraPlacementProblem",
    "OptimizationConstraints",
    "optimize_camera_placement",
    # Config
    "TacticaConfig",
]
