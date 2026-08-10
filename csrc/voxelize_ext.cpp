#include <torch/extension.h>
#include "voxelize.h"


void voxelize_cuda_forward(
    torch::Tensor points,
    torch::Tensor voxels,
    torch::Tensor coordinates,
    torch::Tensor num_points,
    torch::Tensor voxel_count,
    float vx, float vy, float vz,
    float x_min, float y_min, float z_min,
    float x_max, float y_max, float z_max,
    int grid_x, int grid_y, int grid_z,
    int max_points, int max_voxels) {

    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
    TORCH_CHECK(coordinates.is_cuda(), "coordinates must be a CUDA tensor");
    TORCH_CHECK(num_points.is_cuda(), "num_points must be a CUDA tensor");
    TORCH_CHECK(voxel_count.is_cuda(), "voxel_count must be a CUDA tensor");

    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(voxels.is_contiguous(), "voxels must be contiguous");
    TORCH_CHECK(coordinates.is_contiguous(), "coordinates must be contiguous");
    TORCH_CHECK(num_points.is_contiguous(), "num_points must be contiguous");

    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(voxels.dtype() == torch::kFloat32, "voxels must be float32");
    TORCH_CHECK(coordinates.dtype() == torch::kInt32, "coordinates must be int32");
    TORCH_CHECK(num_points.dtype() == torch::kInt32, "num_points must be int32");
    TORCH_CHECK(voxel_count.dtype() == torch::kInt32, "voxel_count must be int32");

    TORCH_CHECK(points.dim() == 2, "points must be 2D [N, C]");
    TORCH_CHECK(points.size(1) >= 4, "points must have at least 4 features");
    TORCH_CHECK(coordinates.size(1) == 3, "coordinates must have 3 columns");

    int N = points.size(0);
    int C = points.size(1);

    voxelize_launcher(
	points.data_ptr<float>(),
	voxels.data_ptr<float>(),
	coordinates.data_ptr<int>(),
	num_points.data_ptr<int>(),
	voxel_count.data_ptr<int>(),
	N, C,
	vx, vy, vz,
	x_min, y_min, z_min,
	x_max, y_max, z_max,
	grid_x, grid_y, grid_z,
	max_points, max_voxels
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize", &voxelize_cuda_forward,
          "LiDAR pillar voxelization CUDA kernel");
}
