# Tactica Optimization - GPU Visibility Engine

**Step 1: GPU-accelerated per-cell visibility computation for camera placement optimization.**

## Overview

This module computes visibility maps for multiple cameras viewing a DEM (Digital Elevation Model) terrain. Given N cameras with pose parameters (x, y, z, yaw, pitch) and field-of-view (HFOV, VFOV) plus max range R, it computes:

- `visible_any[y,x]` - 1 if **any** camera can see that DEM cell
- `vis_count[y,x]` - Number of cameras that can see each cell

The computation runs entirely on GPU using a CuPy RawKernel CUDA implementation.

## How It Works

### Visibility Algorithm (Per-Cell Ray-Marching)

For each DEM cell `p = (x, y, z_dem)` and each camera `i`:

1. **FOV Gating (Quick Reject)**
   - Compute vector `v` from camera to cell in world coordinates
   - **Horizontal check**: Is horizontal angle within `HFOV/2` of camera yaw?
   - **Vertical check**: Is elevation angle within `VFOV/2` of camera pitch?
   - **Range check**: Is `||v_xy|| <= max_range` and `||v|| > epsilon`?

2. **Line-of-Sight Occlusion Test**
   - Sample K points along the ray from camera to cell
   - K is based on distance: `K = clamp(ceil(distance / step), 8, 512)`
   - At each sample point `s`:
     - Compute LOS (line-of-sight) height via linear interpolation between camera z and cell z
     - Compute terrain height at `s.xy` via bilinear DEM interpolation
     - If `terrain_height > LOS_height + epsilon` → ray is **occluded**
   - If all samples pass → cell is **visible**

3. **Aggregation**
   - Each thread (one per DEM cell) loops over all cameras
   - Accumulates visibility count and any-visible flag

### Mathematical Details

**Horizontal FOV Gating:**
```
angle_to_cell = atan2(dy, dx)
h_diff = wrap_to_pi(angle_to_cell - cam_yaw)
visible_h = |h_diff| <= HFOV/2
```

**Vertical FOV Gating:**
```
v_angle_to_cell = atan2(dz, dist_xy)
v_diff = v_angle_to_cell - cam_pitch
visible_v = |v_diff| <= VFOV/2
```

**Bilinear Interpolation for DEM Heights:**
```
h = h00*(1-fx)*(1-fy) + h10*fx*(1-fy) + h01*(1-fx)*fy + h11*fx*fy
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (Compute Capability 3.0+)
- CUDA Toolkit (11.x, 12.x, or 13.x)
- NumPy, Matplotlib, CuPy

### Setup

```bash
# Navigate to project
cd tactica-optimization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (adjust cupy version to match your CUDA)
# - CUDA 11.x: pip install cupy-cuda11x
# - CUDA 12.x: pip install cupy-cuda12x
# - CUDA 13.x: pip install cupy-cuda13x
pip install -r requirements.txt

# If CUDA libraries are not in default path, set LD_LIBRARY_PATH
# Common locations: /usr/local/cuda/lib64, /opt/cuda/lib64
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Run demo
python demo.py
```

### Verify CUDA Setup

```bash
# Check CUDA version
nvcc --version

# Verify CuPy can access GPU
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device().compute_capability}')"
```

## Usage

### Basic Example

```python
from src.visibility_gpu import (
    compute_visibility,
    generate_synthetic_dem,
    create_random_cameras,
    Camera
)
import numpy as np

# Generate terrain
dem = generate_synthetic_dem(height=512, width=512, mode='hills', seed=42)

# Create cameras
cameras = create_random_cameras(
    num_cameras=6,
    dem=dem,
    height_above_ground=10.0,
    hfov=np.deg2rad(90),
    vfov=np.deg2rad(60),
    max_range=150.0
)

# Compute visibility on GPU
visible_any, vis_count = compute_visibility(dem, cameras)

# Results
print(f"Coverage: {100 * visible_any.sum() / dem.size:.1f}%")
print(f"Max cameras per cell: {vis_count.max()}")
```

### Manual Camera Definition

```python
from src.visibility_gpu import Camera

cam = Camera(
    x=100.0, y=100.0, z=15.0,    # Position in grid coordinates
    yaw=np.deg2rad(45),          # Looking toward NE
    pitch=np.deg2rad(-10),       # Looking slightly down
    hfov=np.deg2rad(90),         # 90° horizontal FOV
    vfov=np.deg2rad(60),         # 60° vertical FOV
    max_range=200.0              # 200 cell max range
)
```

### Indoor/Walls Mode

Walls are modeled by setting DEM cells to a very high value (e.g., `1e9`):

```python
from src.visibility_gpu import create_floorplan_dem, add_wall_lines

# Create floorplan with walls
dem = create_floorplan_dem(height=256, width=256, wall_height=1e9)

# Or add custom walls
dem = generate_synthetic_dem(256, 256, mode='flat')
dem = add_wall_lines(dem, [
    ((50, 50), (50, 200)),   # Vertical wall
    ((50, 200), (150, 200)), # Horizontal wall
], wall_height=1e9, thickness=2)

# Compute visibility (walls will block LOS)
visible_any, vis_count = compute_visibility(dem, cameras, wall_threshold=1e6)
```

## Performance

### GPU Parallelization

- **Thread mapping**: One thread per DEM cell
- **Camera loop**: Each thread iterates over all cameras (N ≤ 64)
- **Memory**: DEM in global memory, cameras in packed float array
- **Block size**: 16×16 threads per block (configurable)

### Benchmarks (RTX 3080)

| DEM Size | Cameras | Time |
|----------|---------|------|
| 512×512  | 6       | ~5ms |
| 512×512  | 32      | ~20ms |
| 1024×1024| 6       | ~15ms |
| 2048×2048| 6       | ~55ms |

## File Structure

```
tactica-optimization/
├── src/
│   └── visibility_gpu.py   # Core GPU engine + utilities
├── outputs/
│   └── demo.png            # Generated visualization
├── .venv/                  # Python virtual environment
├── demo.py                 # End-to-end demo script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Known Limitations

1. **Sampling-based occlusion**: Not a true z-buffer/depth-buffer approach. May miss thin occluders or have artifacts at grazing angles. K samples per ray trades accuracy vs. speed.

2. **No hierarchical acceleration**: Every cell checks all cameras. For very large DEMs (8K+), hierarchical culling would help.

3. **Single-bounce only**: No reflection, refraction, or secondary visibility effects.

4. **Discrete sampling**: Bilinear interpolation may miss sub-cell terrain features.

5. **Camera position constraints**: Cameras must be above terrain (no underground visibility).

## Next Steps (Future Work)

### Step 2: Camera Placement Optimization
- Gradient-free optimization (CMA-ES, genetic algorithms) over camera parameters
- Objective: maximize coverage, minimize camera count, ensure redundancy
- Constraints: camera must be on valid terrain, avoid walls

### Step 3: Modern Frontend Visualization
- WebGL/Three.js 3D terrain viewer
- Interactive camera placement and adjustment
- Real-time visibility preview
- Export/import camera configurations

### Step 4: Automatic DEM Acquisition
- Integration with elevation data APIs (Mapbox, Google Elevation, USGS)
- GeoTIFF/DEM file format support
- Coordinate system transformations (lat/lon → grid)

### Step 5: Advanced Features
- True z-buffer rendering approach (screen-space depth)
- Indoor mode with 3D wall meshes (z=inf wall plane)
- Multi-resolution hierarchical visibility
- Temporal visibility (moving cameras/objects)

## License

MIT License - See LICENSE file for details.

## Contributing

This is Step 1 of a larger project. Contributions welcome for:
- Performance optimizations
- Additional terrain generation modes
- Better visualization options
- Bug fixes and documentation improvements
