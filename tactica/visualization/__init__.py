"""
Visualization tools for camera placement optimization.

This module provides:
    - Static plotting functions for DEMs, cameras, and visibility
    - Animation/GIF generation for optimization progress
    - Comparison and summary plots
"""

from tactica.visualization.plotting import (
    draw_camera_markers,
    plot_dem,
    plot_visibility,
    plot_camera_configuration,
    plot_comparison,
)
from tactica.visualization.animation import (
    create_frame,
    create_optimization_gif,
)

__all__ = [
    "draw_camera_markers",
    "plot_dem",
    "plot_visibility",
    "plot_camera_configuration",
    "plot_comparison",
    "create_frame",
    "create_optimization_gif",
]
