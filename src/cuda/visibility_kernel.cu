/*
 * GPU Visibility Kernel for Camera Placement Optimization
 *
 * This CUDA kernel computes per-cell visibility for multiple cameras
 * viewing a DEM (Digital Elevation Model) heightfield.
 *
 * Algorithm:
 *   For each DEM cell, for each camera:
 *     1. FOV gating (horizontal + vertical angle checks)
 *     2. Range gating
 *     3. Line-of-sight occlusion via ray sampling with bilinear DEM interpolation
 *     4. Aggregate visibility count
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;

// Wrap angle to [-pi, pi]
__device__ __forceinline__ float wrap_angle(float angle) {
    while (angle > PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

// Bilinear interpolation for DEM height lookup
__device__ __forceinline__ float bilinear_sample(
    const float* __restrict__ dem,
    int width, int height,
    float x, float y
) {
    // Clamp to valid range
    x = fmaxf(0.0f, fminf(x, (float)(width - 1) - 0.001f));
    y = fmaxf(0.0f, fminf(y, (float)(height - 1) - 0.001f));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = x - (float)x0;
    float fy = y - (float)y0;

    float h00 = dem[y0 * width + x0];
    float h10 = dem[y0 * width + x1];
    float h01 = dem[y1 * width + x0];
    float h11 = dem[y1 * width + x1];

    float h0 = h00 * (1.0f - fx) + h10 * fx;
    float h1 = h01 * (1.0f - fx) + h11 * fx;

    return h0 * (1.0f - fy) + h1 * fy;
}

/*
 * Main visibility kernel
 *
 * Each thread processes one DEM cell and loops over all cameras.
 *
 * Camera parameters layout (8 floats per camera):
 *   [0] x        - position x (grid coords)
 *   [1] y        - position y (grid coords)
 *   [2] z        - position z (height)
 *   [3] yaw      - horizontal angle (radians, 0 = +X, CCW positive)
 *   [4] pitch    - vertical angle (radians, 0 = horizontal, negative = down)
 *   [5] hfov     - horizontal field of view (radians)
 *   [6] vfov     - vertical field of view (radians)
 *   [7] range    - maximum viewing distance (horizontal)
 */
__global__ void visibility_kernel(
    const float* __restrict__ dem,           // [H, W] terrain heights
    const float* __restrict__ cameras,       // [N, 8] camera parameters
    int32_t* __restrict__ vis_count,         // [H, W] output: camera count per cell
    uint8_t* __restrict__ visible_any,       // [H, W] output: any camera sees cell
    int dem_height,
    int dem_width,
    int num_cameras,
    float wall_threshold,
    float occlusion_epsilon
) {
    // Thread -> DEM cell mapping
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= dem_height || col >= dem_width) return;

    int cell_idx = row * dem_width + col;
    float cell_z = dem[cell_idx];

    // Walls cannot be "seen" (they block, not receive)
    if (cell_z >= wall_threshold) {
        vis_count[cell_idx] = 0;
        visible_any[cell_idx] = 0;
        return;
    }

    // Cell center in world coordinates
    float cell_x = (float)col + 0.5f;
    float cell_y = (float)row + 0.5f;

    int count = 0;

    // Loop over all cameras
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        // Unpack camera parameters
        const float* cam = cameras + cam_idx * 8;
        float cam_x     = cam[0];
        float cam_y     = cam[1];
        float cam_z     = cam[2];
        float cam_yaw   = cam[3];
        float cam_pitch = cam[4];
        float cam_hfov  = cam[5];
        float cam_vfov  = cam[6];
        float cam_range = cam[7];

        // Vector from camera to cell
        float dx = cell_x - cam_x;
        float dy = cell_y - cam_y;
        float dz = cell_z - cam_z;

        float dist_xy = sqrtf(dx * dx + dy * dy);
        float dist_3d = sqrtf(dx * dx + dy * dy + dz * dz);

        // =====================================================
        // Gate 1: Range check
        // =====================================================
        if (dist_xy > cam_range || dist_3d < 0.01f) {
            continue;
        }

        // =====================================================
        // Gate 2: Horizontal FOV check
        // =====================================================
        // Angle from camera to cell in XY plane
        float angle_to_cell = atan2f(dy, dx);
        float h_diff = wrap_angle(angle_to_cell - cam_yaw);

        if (fabsf(h_diff) > cam_hfov * 0.5f) {
            continue;
        }

        // =====================================================
        // Gate 3: Vertical FOV check
        // =====================================================
        // Vertical angle: atan2(dz, dist_xy)
        // Positive = up, negative = down
        float v_angle_to_cell = atan2f(dz, dist_xy);
        float v_diff = v_angle_to_cell - cam_pitch;

        if (fabsf(v_diff) > cam_vfov * 0.5f) {
            continue;
        }

        // =====================================================
        // Gate 4: Line-of-sight occlusion test
        // =====================================================
        // Sample K points along ray from camera to cell
        // K based on distance, clamped to [8, 512]
        int K = (int)ceilf(dist_xy);
        K = max(8, min(512, K));

        bool occluded = false;

        for (int s = 1; s < K && !occluded; s++) {
            float t = (float)s / (float)K;

            // Sample point along ray
            float sx = cam_x + t * dx;
            float sy = cam_y + t * dy;
            float sz = cam_z + t * dz;  // LOS height

            // Convert to grid coordinates for DEM lookup
            float gx = sx - 0.5f;
            float gy = sy - 0.5f;

            // Get interpolated terrain height
            float terrain_z = bilinear_sample(dem, dem_width, dem_height, gx, gy);

            // Check occlusion: terrain above LOS + epsilon
            if (terrain_z > sz + occlusion_epsilon) {
                occluded = true;
            }
        }

        if (!occluded) {
            count++;
        }
    }

    // Write outputs
    vis_count[cell_idx] = count;
    visible_any[cell_idx] = (count > 0) ? 1 : 0;
}

} // anonymous namespace


// =============================================================================
// PyTorch C++ Interface
// =============================================================================

std::vector<torch::Tensor> compute_visibility_cuda(
    torch::Tensor dem,           // [H, W] float32
    torch::Tensor cameras,       // [N, 8] float32
    float wall_threshold,
    float occlusion_epsilon
) {
    // Input validation
    TORCH_CHECK(dem.is_cuda(), "DEM must be a CUDA tensor");
    TORCH_CHECK(cameras.is_cuda(), "Cameras must be a CUDA tensor");
    TORCH_CHECK(dem.dim() == 2, "DEM must be 2D");
    TORCH_CHECK(cameras.dim() == 2 && cameras.size(1) == 8, "Cameras must be [N, 8]");
    TORCH_CHECK(dem.dtype() == torch::kFloat32, "DEM must be float32");
    TORCH_CHECK(cameras.dtype() == torch::kFloat32, "Cameras must be float32");

    int dem_height = dem.size(0);
    int dem_width = dem.size(1);
    int num_cameras = cameras.size(0);

    // Allocate output tensors
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(dem.device());
    auto options_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(dem.device());

    torch::Tensor vis_count = torch::zeros({dem_height, dem_width}, options_int);
    torch::Tensor visible_any = torch::zeros({dem_height, dem_width}, options_uint8);

    // Kernel launch configuration
    dim3 block(16, 16);
    dim3 grid(
        (dem_width + block.x - 1) / block.x,
        (dem_height + block.y - 1) / block.y
    );

    // Launch kernel
    visibility_kernel<<<grid, block>>>(
        dem.data_ptr<float>(),
        cameras.data_ptr<float>(),
        vis_count.data_ptr<int32_t>(),
        visible_any.data_ptr<uint8_t>(),
        dem_height,
        dem_width,
        num_cameras,
        wall_threshold,
        occlusion_epsilon
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return {vis_count, visible_any};
}


// =============================================================================
// Python Binding
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_visibility", &compute_visibility_cuda,
          "Compute per-cell visibility from multiple cameras (CUDA)",
          py::arg("dem"),
          py::arg("cameras"),
          py::arg("wall_threshold") = 1e6f,
          py::arg("occlusion_epsilon") = 1e-3f);
}
