#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include "voxelize.h"

#define HASH_TABLE_SIZE(max_voxels) (8 * (max_voxels))

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
	int* hash_keys,
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

    while (true) {
	int old = atomicCAS(&hash_keys[h], -1, voxel_id);

	if (old == -1) {
	    int cy = voxel_id / grid_x;
	    int cx = voxel_id % grid_x;
	    coordinates[h * 3 + 0] = 0;
	    coordinates[h * 3 + 1] = cy;
	    coordinates[h * 3 + 2] = cx;
	    break;
	}
	if (old == voxel_id) break;
	h = (h + 1) % table_size;
    }
    int pt_idx = atomicAdd(&num_points_per_voxel[h], 1);
    if (pt_idx >= max_points) return;
    for (int c = 0; c < C; c++) {
        voxels[h * max_points * C + pt_idx * C + c] = points[idx * C + c];
    }
}


__global__ void compact_kernel(
    const float* voxels_large,
    const int* coords_large,
    const int* npts_large,
    float* voxels_out,
    int* coords_out,
    int* npts_out,
    const int* occupied_indices,
    int n_occupied,
    int max_points, int C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_occupied) return;

    int slot = occupied_indices[i];

    coords_out[i * 3 + 0] = coords_large[slot * 3 + 0];
    coords_out[i * 3 + 1] = coords_large[slot * 3 + 1];
    coords_out[i * 3 + 2] = coords_large[slot * 3 + 2];

    npts_out[i] = npts_large[slot];

    for (int j = 0; j < max_points * C; j++) {
	voxels_out[i * max_points * C + j] = voxels_large[slot * max_points * C + j];
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

    int*   hash_keys;
    float* voxels_large;
    int*   coords_large;
    int*   npts_large;

    CUDA_CHECK(cudaMalloc(&hash_keys, table_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&voxels_large, table_size * max_points * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coords_large, table_size * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&npts_large, table_size * sizeof(int)));

    CUDA_CHECK(cudaMemset(hash_keys, 0xFF, table_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(voxels_large, 0, table_size * max_points * C * sizeof(float)));
    CUDA_CHECK(cudaMemset(coords_large, 0, table_size * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(npts_large, 0, table_size * sizeof(int)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    voxelize_kernel<<<blocks, threads>>>(points, voxels_large, coords_large, npts_large, hash_keys, N, C, vx, vy, vz, x_min, y_min, z_min, x_max, y_max, z_max, grid_x, grid_y, grid_z, max_points, max_voxels);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // find occupied slot
    thrust::device_ptr<int> hash_keys_ptr(hash_keys);
    thrust::device_vector<int> occupied_indices(table_size);

    auto end_it = thrust::copy_if(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(table_size),
        hash_keys_ptr,
        occupied_indices.begin(),
        [] __device__ (int k) { return k != -1; }
    );

    int n_occupied = end_it - occupied_indices.begin();
    if (n_occupied > max_voxels) n_occupied = max_voxels;

    // copy data back
    int comp_blocks = (n_occupied + 255) / 256;
    if (comp_blocks > 0) {
        compact_kernel<<<comp_blocks, 256>>>(
            voxels_large, coords_large, npts_large,
            voxels, coordinates, num_points_per_voxel,
            thrust::raw_pointer_cast(occupied_indices.data()),
            n_occupied, max_points, C
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(voxel_count, &n_occupied, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFree(hash_keys));
    CUDA_CHECK(cudaFree(voxels_large));
    CUDA_CHECK(cudaFree(coords_large));
    CUDA_CHECK(cudaFree(npts_large));
}
