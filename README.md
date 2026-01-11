# Tactica - GPU Camera Placement Optimization Engine

**GPU-accelerated camera/sensor placement optimization for DEM (Digital Elevation Model) terrain.**

## Overview

Tactica is a Python package for optimizing camera and sensor placement on terrain data. It uses GPU-accelerated visibility computation via custom CUDA kernels and provides multiple metaheuristic optimization algorithms.

### Key Features

- **GPU Visibility**: Custom CUDA kernel for fast per-cell visibility computation
- **Multiple Optimizers**: CMA-ES (bundled), plus PSO, DE, GA, ABC via optional OptimizationFramework
- **Constraint System**: Placement zones, exclusion areas, priority weighting
- **GeoTIFF Support**: Load real DEMs with CRS and coordinate transformations
- **Configuration Management**: YAML/JSON configuration files
- **API Schemas**: Ready for frontend integration

## Installation

```bash
# Clone repository
git clone https://github.com/tactica/tactica-optimization.git
cd tactica-optimization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# With optional dependencies
pip install -e ".[all]"  # Everything
pip install -e ".[api]"  # FastAPI support
pip install -e ".[geo]"  # GeoTIFF support
pip install -e ".[dev]"  # Testing tools
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (Compute Capability 3.0+)
- CUDA Toolkit with `nvcc` compiler
- PyTorch with CUDA support

### Verify Setup

```bash
# Check CUDA
nvcc --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -v
```

## Quick Start

```python
from tactica import optimize_camera_placement
from tactica.dem import generate_synthetic_dem

# Generate terrain
dem = generate_synthetic_dem(256, 256, mode='hills', seed=42)

# Optimize camera placement
result = optimize_camera_placement(
    dem=dem,
    num_cameras=4,
    optimizer="cma",
    budget=500,
    verbose=True,
)

# Results
print(f"Coverage: {result.coverage:.1%}")
for cam in result.cameras:
    print(f"  Camera at ({cam.x:.0f}, {cam.y:.0f}, {cam.z:.0f})")
```

## Usage Examples

### Basic Visibility Computation

```python
from tactica.core import Camera, compute_visibility
from tactica.dem import generate_synthetic_dem
import numpy as np

# Generate terrain
dem = generate_synthetic_dem(512, 512, mode='hills', seed=42)

# Define cameras
cameras = [
    Camera(x=100, y=100, z=15, yaw=0, pitch=-0.3,
           hfov=np.deg2rad(90), vfov=np.deg2rad(60), max_range=150),
    Camera(x=400, y=400, z=20, yaw=np.pi, pitch=-0.2,
           hfov=np.deg2rad(90), vfov=np.deg2rad(60), max_range=150),
]

# Compute visibility
visible_any, vis_count = compute_visibility(dem, cameras)

print(f"Coverage: {visible_any.sum() / dem.size:.1%}")
print(f"Max redundancy: {vis_count.max()} cameras")
```

### With Constraints

```python
from tactica import optimize_camera_placement
from tactica.optimization import OptimizationConstraints
from tactica.dem import generate_synthetic_dem
import numpy as np

dem = generate_synthetic_dem(256, 256, mode='hills', seed=42)

# Priority weights (higher priority in center)
y, x = np.mgrid[0:256, 0:256]
priority = 1 + 2 * np.exp(-((x-128)**2 + (y-128)**2) / 5000)

# Exclusion zone (top-left corner)
exclusion = np.zeros((256, 256), dtype=bool)
exclusion[:50, :50] = True

constraints = OptimizationConstraints(
    priority_weights=priority.astype(np.float32),
    exclusion_mask=exclusion,
    min_coverage=0.5,
)

result = optimize_camera_placement(
    dem=dem,
    num_cameras=6,
    constraints=constraints,
    budget=500,
)
```

### Indoor / Floorplan

```python
from tactica import optimize_camera_placement
from tactica.dem import create_floorplan_dem
import numpy as np

# Generate indoor floorplan with walls
dem = create_floorplan_dem(256, 256, seed=42)

result = optimize_camera_placement(
    dem=dem,
    num_cameras=6,
    wall_threshold=1e6,
    z_bounds=(2, 6),  # Lower ceiling
    fov_bounds=(np.deg2rad(60), np.deg2rad(120)),  # Wide FOV
    budget=500,
)
```

### Load Real GeoTIFF

```python
from tactica.dem import load_dem, grid_to_world, world_to_grid

# Load GeoTIFF with metadata
dem, metadata = load_dem("terrain.tif")

print(f"CRS: {metadata.crs}")
print(f"Resolution: {metadata.resolution}")
print(f"Bounds: {metadata.bounds}")

# Coordinate transformations
world_x, world_y = grid_to_world(100, 100, metadata)
grid_row, grid_col = world_to_grid(world_x, world_y, metadata)
```

### Configuration File

```python
from tactica import TacticaConfig

# Load from YAML
config = TacticaConfig.from_yaml("config.yaml")

# Or use presets
config = TacticaConfig.for_indoor()
config = TacticaConfig.for_outdoor()

# Access settings
print(config.optimization.optimizer)
print(config.camera.resolution.horizontal_pixels)
```

Example `config.yaml`:

```yaml
camera:
  resolution:
    horizontal_pixels: 3840
    vertical_pixels: 2160
    pixels_per_meter: 30.0
  hfov_bounds: [0.52, 2.44]  # 30-140 degrees
  z_bounds: [2, 15]

optimization:
  optimizer: cma
  budget: 1000
  num_cameras: 4
  seed: 42

dem:
  wall_threshold: 1000000.0

visualization:
  colormap: viridis
  camera_marker_size: 8
  create_gif: true
  gif_fps: 10
```

## API

### Core Classes

**Camera** - Camera/sensor configuration

```python
Camera(x, y, z, yaw, pitch, hfov, vfov, max_range)

# Properties
cam.position          # (x, y, z) tuple
cam.yaw_degrees       # Yaw in degrees
cam.hfov_degrees      # HFOV in degrees

# Factory methods
Camera.from_fov_and_resolution(x, y, z, yaw, pitch, hfov, resolution)
```

**CameraResolution** - Sensor resolution settings

```python
CameraResolution(horizontal_pixels, vertical_pixels, pixels_per_meter)

# Presets
CameraResolution.preset_4k()
CameraResolution.preset_1080p()
```

**OptimizationConstraints** - Placement constraints

```python
OptimizationConstraints(
    placement_mask=None,     # Where cameras CAN be placed
    exclusion_mask=None,     # Where cameras CANNOT be placed
    priority_weights=None,   # Importance per cell (for objective)
    min_coverage=0.0,        # Minimum coverage requirement
    fixed_cameras=None,      # Pre-placed cameras to include
)
```

### Main Functions

**optimize_camera_placement()** - High-level optimization

```python
result = optimize_camera_placement(
    dem,                        # DEM array (H, W)
    num_cameras=4,              # Number of cameras
    optimizer="cma",            # Algorithm: cma, pso, de, ga, abc
    budget=1000,                # Max evaluations
    camera_resolution=None,     # CameraResolution (default: 4K @ 30 PPM)
    constraints=None,           # OptimizationConstraints
    wall_threshold=1e6,         # Wall height threshold
    z_bounds=(2, 15),           # Camera height bounds
    fov_bounds=(0.52, 2.44),    # HFOV bounds in radians
    seed=None,                  # Random seed
    verbose=True,               # Print progress
)
```

**compute_visibility()** - GPU visibility computation

```python
visible_any, vis_count = compute_visibility(
    dem,                        # DEM array
    cameras,                    # List of Camera objects
    wall_threshold=1e6,         # Wall height threshold
)
```

### DEM Generation

```python
from tactica.dem import (
    generate_synthetic_dem,     # Hills, ridge, flat, valley
    generate_random_dem,        # Perlin-like noise
    create_floorplan_dem,       # Indoor with walls
    add_walls_to_dem,           # Add wall cells
    add_wall_lines,             # Add wall lines
    load_dem,                   # Load GeoTIFF
    save_dem,                   # Save GeoTIFF
)
```

## Optimized Parameters (6 per camera)

| Parameter | Description | Typical Bounds |
|-----------|-------------|----------------|
| x | Grid x position | (margin, width-margin) |
| y | Grid y position | (margin, height-margin) |
| z | Height above ground | (1, 25) |
| yaw (θ) | Horizontal angle | (-π, π) |
| pitch (φ) | Vertical angle | (-π/3, 0) |
| hfov | Horizontal FOV | (20°, 150°) |

**Note:** Vertical FOV is derived from HFOV using aspect ratio.

## Package Structure

```
tactica-optimization/
├── tactica/
│   ├── __init__.py              # Package exports
│   ├── core/
│   │   ├── sensors.py           # Camera, CameraResolution
│   │   ├── visibility.py        # GPU visibility wrapper
│   │   └── cuda/
│   │       └── visibility_kernel.cu  # CUDA kernel
│   ├── dem/
│   │   ├── synthetic.py         # DEM generation
│   │   ├── loader.py            # GeoTIFF loading
│   │   └── coordinates.py       # CRS transformations
│   ├── optimization/
│   │   ├── problem.py           # CameraPlacementProblem
│   │   ├── constraints.py       # OptimizationConstraints
│   │   ├── objectives.py        # Objective functions
│   │   └── runner.py            # optimize_camera_placement()
│   ├── visualization/
│   │   ├── plotting.py          # Static plots
│   │   └── animation.py         # GIF generation
│   ├── config/
│   │   └── settings.py          # TacticaConfig
│   └── api/
│       └── schemas.py           # API schemas for frontend
├── tests/
│   ├── test_core.py
│   ├── test_dem.py
│   └── test_optimization.py
├── examples/
│   └── basic_usage.py
├── outputs/                     # Generated outputs
├── pyproject.toml               # Package configuration
├── requirements.txt
└── README.md
```

## Available Optimizers

| Optimizer | Bundled | Description |
|-----------|---------|-------------|
| CMA-ES | ✅ | Covariance Matrix Adaptation (recommended) |
| PSO | ❌ | Particle Swarm Optimization |
| DE | ❌ | Differential Evolution |
| GA | ❌ | Genetic Algorithm with SBX |
| ABC | ❌ | Artificial Bee Colony |
| Dual | ❌ | Dual Annealing |

Non-bundled optimizers require `OptimizationFramework` package.

## How Visibility Works

### Per-Cell Ray-Marching Algorithm

For each DEM cell and camera:

1. **FOV Gating (Quick Reject)**
   - Check if cell is within camera's horizontal FOV
   - Check if cell is within camera's vertical FOV
   - Check if cell is within max range

2. **Line-of-Sight Occlusion Test**
   - Sample K points along ray from camera to cell
   - At each point, compare LOS height to terrain height
   - If terrain exceeds LOS → ray is occluded

3. **Aggregation**
   - Each GPU thread (one per cell) loops over cameras
   - Accumulates visibility count and any-visible flag

### GPU Performance

- **Thread mapping**: One thread per DEM cell
- **Memory**: DEM in global memory, cameras in packed array
- **Block size**: 16×16 threads

| DEM Size | Cameras | Time (RTX 3080) |
|----------|---------|-----------------|
| 512×512  | 6       | ~5ms |
| 512×512  | 32      | ~20ms |
| 1024×1024| 6       | ~15ms |
| 2048×2048| 6       | ~55ms |

## Running Legacy Scripts

The original scripts are still available:

```bash
# Multi-scenario optimization
python optimize.py --scenario all --generations 100

# Compare optimizers
python compare_optimizers.py --scenario indoor --cameras 4 --budget 1000

# Benchmark across topologies
python benchmark_topologies.py --cameras 4 --budget 1000
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=tactica --cov-report=html

# Skip CUDA tests if no GPU
pytest tests/ -v -k "not cuda"
```

## License

MIT License

## Contributing

Contributions welcome for:
- Performance optimizations
- Additional terrain modes
- New optimization algorithms
- Documentation improvements
