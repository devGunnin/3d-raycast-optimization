"""
API schemas and utilities for frontend integration.

This module provides Pydantic models for the REST/WebSocket API,
defining the contracts between backend and frontend.
"""

from tactica.api.schemas import (
    # Camera schemas
    CameraSchema,
    CameraResolutionSchema,
    CameraListSchema,
    # DEM schemas
    DEMMetadataSchema,
    DEMUploadResponse,
    # Optimization schemas
    OptimizationRequest,
    OptimizationResponse,
    OptimizationProgress,
    # Constraint schemas
    ConstraintsSchema,
    # Result schemas
    CoverageResult,
    VisibilityResult,
)

__all__ = [
    # Camera
    "CameraSchema",
    "CameraResolutionSchema",
    "CameraListSchema",
    # DEM
    "DEMMetadataSchema",
    "DEMUploadResponse",
    # Optimization
    "OptimizationRequest",
    "OptimizationResponse",
    "OptimizationProgress",
    # Constraints
    "ConstraintsSchema",
    # Results
    "CoverageResult",
    "VisibilityResult",
]
