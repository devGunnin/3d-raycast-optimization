"""
Configuration settings using Pydantic for validation.

This module provides typed configuration classes for all Tactica settings,
supporting loading from YAML/JSON files and environment variables.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np


@dataclass
class CameraConfig:
    """
    Camera/sensor configuration.

    Attributes:
        horizontal_pixels: Sensor horizontal resolution
        vertical_pixels: Sensor vertical resolution
        pixels_per_meter: Required pixel density at max range
        z_bounds: (min, max) height above ground
        pitch_bounds: (min, max) pitch angle in radians
        fov_bounds: (min, max) horizontal FOV in radians
    """
    horizontal_pixels: int = 3840
    vertical_pixels: int = 2160
    pixels_per_meter: float = 30.0
    z_bounds: Tuple[float, float] = (2.0, 15.0)
    pitch_bounds: Tuple[float, float] = (-0.785, 0.0)  # -45 to 0 degrees
    fov_bounds: Tuple[float, float] = (0.524, 2.443)   # 30 to 140 degrees

    @classmethod
    def for_indoor(cls) -> "CameraConfig":
        """Preset for indoor scenarios."""
        return cls(
            z_bounds=(2.0, 8.0),
            pitch_bounds=(-0.524, 0.0),  # -30 to 0 degrees
            fov_bounds=(0.698, 2.618),   # 40 to 150 degrees
        )

    @classmethod
    def for_outdoor(cls) -> "CameraConfig":
        """Preset for outdoor scenarios."""
        return cls(
            z_bounds=(5.0, 25.0),
            pitch_bounds=(-0.785, 0.0),  # -45 to 0 degrees
            fov_bounds=(0.349, 2.618),   # 20 to 150 degrees
            pixels_per_meter=25.0,
        )


@dataclass
class OptimizationConfig:
    """
    Optimization algorithm configuration.

    Attributes:
        num_cameras: Number of cameras to optimize
        optimizer: Optimization algorithm name
        budget: Maximum function evaluations
        seed: Random seed for reproducibility
        coverage_weight: Weight for coverage objective
        redundancy_weight: Weight for redundancy bonus
        wall_penalty: Penalty for cameras on walls
        margin: Minimum distance from DEM edges
    """
    num_cameras: int = 4
    optimizer: str = "cma"
    budget: int = 1000
    seed: Optional[int] = None
    coverage_weight: float = 1.0
    redundancy_weight: float = 0.1
    wall_penalty: float = 0.5
    margin: float = 15.0

    # Optimizer-specific settings
    cma_sigma0: float = 0.3
    pso_particles: int = 40
    pso_w: float = 0.7
    pso_c1: float = 1.5
    pso_c2: float = 1.5


@dataclass
class VisualizationConfig:
    """
    Visualization settings.

    Attributes:
        output_dir: Directory for output files
        generate_gifs: Whether to generate GIF animations
        gif_fps: Frames per second for GIFs
        gif_max_frames: Maximum frames in GIFs
        dpi: Resolution for saved images
        figsize: Default figure size (width, height)
    """
    output_dir: str = "outputs"
    generate_gifs: bool = True
    gif_fps: int = 5
    gif_max_frames: int = 50
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 5)


@dataclass
class TacticaConfig:
    """
    Main Tactica configuration.

    This is the top-level configuration class that contains all settings
    for the Tactica camera placement optimization engine.

    Attributes:
        camera: Camera/sensor configuration
        optimization: Optimization algorithm settings
        visualization: Visualization settings
        wall_threshold: Height above which DEM cells are walls
        dem_path: Optional path to DEM file
        constraints_path: Optional path to constraints file
    """
    camera: CameraConfig = field(default_factory=CameraConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    wall_threshold: float = 1e6
    dem_path: Optional[str] = None
    constraints_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        return convert(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TacticaConfig":
        """Create from dictionary."""
        camera = CameraConfig(**data.get('camera', {}))
        optimization = OptimizationConfig(**data.get('optimization', {}))
        visualization = VisualizationConfig(**data.get('visualization', {}))

        return cls(
            camera=camera,
            optimization=optimization,
            visualization=visualization,
            wall_threshold=data.get('wall_threshold', 1e6),
            dem_path=data.get('dem_path'),
            constraints_path=data.get('constraints_path'),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TacticaConfig":
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")

        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TacticaConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files")

        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def for_indoor(cls) -> "TacticaConfig":
        """Preset configuration for indoor scenarios."""
        return cls(
            camera=CameraConfig.for_indoor(),
            optimization=OptimizationConfig(
                num_cameras=4,
                budget=1000,
                redundancy_weight=0.1,
            ),
        )

    @classmethod
    def for_outdoor(cls) -> "TacticaConfig":
        """Preset configuration for outdoor scenarios."""
        return cls(
            camera=CameraConfig.for_outdoor(),
            optimization=OptimizationConfig(
                num_cameras=8,
                budget=2000,
                redundancy_weight=0.05,
            ),
        )


def load_config(path: Optional[Union[str, Path]] = None) -> TacticaConfig:
    """
    Load configuration from file or return defaults.

    Supports YAML and JSON files based on extension.

    Args:
        path: Path to configuration file (optional)

    Returns:
        TacticaConfig instance
    """
    if path is None:
        return TacticaConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return TacticaConfig.from_yaml(path)
    elif suffix == '.json':
        return TacticaConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")
