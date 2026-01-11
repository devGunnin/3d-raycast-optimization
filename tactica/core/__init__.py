"""
Core visibility computation and sensor models.

This module provides:
    - Camera and CameraResolution dataclasses for sensor configuration
    - GPU-accelerated visibility computation via CUDA kernels
    - Utility functions for FOV and range calculations
"""

from tactica.core.sensors import (
    Camera,
    CameraResolution,
    compute_max_range_from_fov,
    create_random_cameras,
    create_preset_cameras,
)
from tactica.core.visibility import compute_visibility

__all__ = [
    "Camera",
    "CameraResolution",
    "compute_max_range_from_fov",
    "compute_visibility",
    "create_random_cameras",
    "create_preset_cameras",
]
