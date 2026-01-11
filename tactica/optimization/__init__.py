"""
Camera placement optimization module.

This module provides:
    - CameraPlacementProblem: The core optimization problem definition
    - OptimizationConstraints: Constraint system for placement/exclusion zones
    - Objective functions for coverage, redundancy, and cost
    - Runner functions for various optimization algorithms
"""

from tactica.optimization.problem import CameraPlacementProblem
from tactica.optimization.constraints import OptimizationConstraints
from tactica.optimization.objectives import (
    compute_coverage_objective,
    compute_weighted_coverage,
)
from tactica.optimization.runner import (
    optimize_camera_placement,
    OptimizationResult,
)

__all__ = [
    "CameraPlacementProblem",
    "OptimizationConstraints",
    "compute_coverage_objective",
    "compute_weighted_coverage",
    "optimize_camera_placement",
    "OptimizationResult",
]
