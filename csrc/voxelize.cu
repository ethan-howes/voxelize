#include "voxelize.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define HASH_TABLE_SIZE(max_voxels) (2 * (max_voxels))

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        abort(); \
    } \
} while(0)


__device__ int compute_grid_idx(float x, float y, float x_min, float y_min, float vx, float vy, int grid_x, int grid_y) {
    if (isinf(x) || isnan(x) || isinf(y) || isnan(y)) return -1;

    int cx = (int)floorf((x - x_min) / vx);
    int cy = (int)floorf((y - y_min) / vy);

    if (cx < 0 || cx >= grid_x) return -1;
    if (cy < 0 || cy >= grid_y) return -1;

    return cy * grid_x + cx;
}


__global__ void voxelize_kernel(
	const float* points,
	float* voxels,
	int* coordinates,
	int* num_points_per_voxel,
	int* voxel_count,
	int* hash_keys,
	int* hash_values,
	int N, int C,
	float vx, float vy, float vz,
	float x_min, float y_min, float z_min,
	float x_max, float y_max, float z_max,
	int grid_x, int grid_y, int grid_z,
	int max_points, int max_voxels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // binning points
    float x = points[idx * C + 0];
    float y = points[idx * C + 1];
    float z = points[idx * C + 2];

    if (z <= z_min || z >= z_max) return;

    int voxel_id = compute_grid_idx(x, y, x_min, y_min, vx, vy, grid_x, grid_y);
    if (voxel_id == -1) return;

    int table_size = HASH_TABLE_SIZE(max_voxels);
    int h = ((voxel_id % table_size) + table_size) % table_size;

    int slot = -1;
    while (true) {
	int old = atomicCAS(&hash_keys[h], -1, voxel_id);

	if (old == -1) {
	    slot = atomicAdd(voxel_count, 1);
	    if (slot >= max_voxels) return;
	    hash_values[h] = slot;
	    int cy = voxel_id / grid_x;
	    int cx = voxel_id % grid_x;
	    coordinates[slot * 3 + 0] = 0;
	    coordinates[slot * 3 + 1] = cy;
	    coordinates[slot * 3 + 2] = cx;
	    break;
	}
	if (old == voxel_id) {
	    // if the pillar exists then wait for the slot to be written
	    while ((slot  = hash_values[h]) == -1) {}
	    break;
	}
	h = (h + 1) % table_size;
    }

    int pt_idx = atomicAdd(&num_points_per_voxel[slot], 1);
    if (pt_idx >= max_points) return;

    for (int c = 0; c < C; c++) {
	voxels[slot * max_points * C + pt_idx * C + c] = points[idx * C + c];
    }
}


void voxelize_launcher(
    const float* points,
    float * voxels,
    int* coordinates,
    int* num_points_per_voxel,
    int* voxel_count,
    int N, int C,
    float vx, float vy, float vz,
    float x_min, float y_min, float z_min, 
    float x_max, float y_max, float z_max,
    int grid_x, int grid_y, int grid_z, 
    int max_points, int max_voxels
) {
    int table_size = HASH_TABLE_SIZE(max_voxels);

    int* hash_keys;
    int* hash_values;

    CUDA_CHECK(cudaMalloc(&hash_keys, table_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hash_values, table_size * sizeof(int)));

    CUDA_CHECK(cudaMemset(hash_keys, 0xFF, table_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(hash_values, 0xFF, table_size * sizeof(int)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    voxelize_kernel<<<blocks, threads>>>(points, voxels, coordinates, num_points_per_voxel, voxel_count, hash_keys, hash_values, N, C, vx, vy, vz, x_min, y_min, z_min, x_max, y_max, z_max, grid_x, grid_y, grid_z, max_points, max_voxels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(hash_keys));
    CUDA_CHECK(cudaFree(hash_values));
}
