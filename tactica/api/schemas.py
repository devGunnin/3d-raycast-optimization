"""
Pydantic schemas for the Tactica API.

These schemas define the JSON contracts for communication between
the backend and frontend applications.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class CameraResolutionSchema:
    """Camera resolution configuration."""
    horizontal_pixels: int = 3840
    vertical_pixels: int = 2160
    pixels_per_meter: float = 30.0


@dataclass
class CameraSchema:
    """Single camera configuration."""
    id: Optional[int] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    hfov: float = 1.57  # 90 degrees
    vfov: float = 1.05  # 60 degrees
    max_range: float = 100.0

    # Optional metadata
    name: Optional[str] = None
    is_fixed: bool = False


@dataclass
class CameraListSchema:
    """List of cameras with metadata."""
    cameras: List[CameraSchema] = field(default_factory=list)
    total_count: int = 0


@dataclass
class DEMMetadataSchema:
    """DEM metadata for API responses."""
    width: int = 0
    height: int = 0
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y]
    resolution: List[float] = field(default_factory=lambda: [1.0, 1.0])
    units: str = "meters"
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    wall_cell_count: int = 0


@dataclass
class DEMUploadResponse:
    """Response after DEM upload."""
    success: bool = True
    dem_id: Optional[str] = None
    metadata: Optional[DEMMetadataSchema] = None
    message: str = ""
    preview_url: Optional[str] = None


@dataclass
class ConstraintsSchema:
    """Optimization constraints."""
    min_coverage: float = 0.0
    max_cameras: Optional[int] = None

    # Masks are sent as base64-encoded images or as references
    placement_mask_url: Optional[str] = None
    exclusion_mask_url: Optional[str] = None
    priority_weights_url: Optional[str] = None

    # Zone coverage requirements
    zone_coverage_requirements: Optional[Dict[int, float]] = None

    # Fixed cameras
    fixed_cameras: Optional[List[CameraSchema]] = None


@dataclass
class OptimizationRequest:
    """Request to start optimization."""
    dem_id: str = ""
    num_cameras: int = 4
    optimizer: str = "cma"
    budget: int = 1000
    seed: Optional[int] = None

    # Camera configuration
    camera_resolution: Optional[CameraResolutionSchema] = None
    z_bounds: List[float] = field(default_factory=lambda: [2.0, 15.0])
    fov_bounds: List[float] = field(default_factory=lambda: [0.52, 2.44])
    pitch_bounds: List[float] = field(default_factory=lambda: [-0.79, 0.0])

    # Objective weights
    coverage_weight: float = 1.0
    redundancy_weight: float = 0.1

    # Constraints
    constraints: Optional[ConstraintsSchema] = None


@dataclass
class OptimizationProgress:
    """Progress update during optimization (for WebSocket)."""
    job_id: str = ""
    status: str = "running"  # "pending", "running", "completed", "failed"
    progress: float = 0.0  # 0 to 1
    current_coverage: float = 0.0
    evaluations_completed: int = 0
    total_evaluations: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""

    # Optional: current best cameras for live preview
    current_cameras: Optional[List[CameraSchema]] = None


@dataclass
class OptimizationResponse:
    """Final optimization result."""
    job_id: str = ""
    success: bool = True
    message: str = ""

    # Results
    coverage: float = 0.0
    cameras: List[CameraSchema] = field(default_factory=list)

    # Statistics
    evaluations: int = 0
    runtime_seconds: float = 0.0
    optimizer_used: str = ""

    # Trace for visualization
    coverage_trace: List[float] = field(default_factory=list)

    # URLs for result visualization
    visibility_map_url: Optional[str] = None
    coverage_heatmap_url: Optional[str] = None


@dataclass
class CoverageResult:
    """Coverage analysis result."""
    coverage: float = 0.0
    weighted_coverage: float = 0.0
    visible_cells: int = 0
    valid_cells: int = 0
    redundancy_fraction: float = 0.0
    avg_cameras_per_cell: float = 0.0
    zone_coverage: Optional[Dict[int, float]] = None


@dataclass
class VisibilityResult:
    """Visibility computation result."""
    coverage: float = 0.0
    visible_cells: int = 0
    total_cells: int = 0
    wall_cells: int = 0

    # URLs for visualization
    visibility_map_url: Optional[str] = None
    count_heatmap_url: Optional[str] = None

    # Camera-level statistics
    camera_contributions: Optional[List[int]] = None


@dataclass
class JobStatus:
    """Status of an async job."""
    job_id: str = ""
    status: str = "pending"  # "pending", "running", "completed", "failed", "cancelled"
    progress: float = 0.0
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class APIError:
    """API error response."""
    error: str = ""
    code: str = "unknown_error"
    details: Optional[Dict[str, Any]] = None


# Helper functions for conversion

def camera_to_schema(camera) -> CameraSchema:
    """Convert internal Camera object to schema."""
    return CameraSchema(
        x=float(camera.x),
        y=float(camera.y),
        z=float(camera.z),
        yaw=float(camera.yaw),
        pitch=float(camera.pitch),
        hfov=float(camera.hfov),
        vfov=float(camera.vfov),
        max_range=float(camera.max_range),
    )


def cameras_to_schema(cameras: list) -> List[CameraSchema]:
    """Convert list of internal Camera objects to schemas."""
    return [camera_to_schema(cam) for cam in cameras]


def schema_to_camera(schema: CameraSchema):
    """Convert schema to internal Camera object."""
    from tactica.core.sensors import Camera
    return Camera(
        x=schema.x,
        y=schema.y,
        z=schema.z,
        yaw=schema.yaw,
        pitch=schema.pitch,
        hfov=schema.hfov,
        vfov=schema.vfov,
        max_range=schema.max_range,
    )


def schemas_to_cameras(schemas: List[CameraSchema]) -> list:
    """Convert list of schemas to internal Camera objects."""
    return [schema_to_camera(s) for s in schemas]
