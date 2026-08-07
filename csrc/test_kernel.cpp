#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "voxelize.h"


int main() {
    float vx = 1.0f;
    float vy = 1.0f;
    float vz = 4.0f;
    float x_min = 0.0f;
    float y_min = 0.0f;
    float z_min = -3.0f;
    float x_max = 10.0f;
    float y_max = 10.0f;
    float z_max = 1.0f;
    int grid_x = 10;
    int grid_y = 10;
    int grid_z = 1;
    int max_points = 5;
    int max_voxels = 20000;
    int N = 12;
    int C = 4;

    float points_cpu[] = {
        // a
        0.5f, 0.5f, 0.0f, 1.0f,
        0.7f, 0.3f, 0.0f, 1.0f,
        0.1f, 0.2f, 0.0f, 1.0f,
        0.9f, 0.8f, 0.0f, 1.0f,
        // b
        1.1f, 1.2f, 0.0f, 1.0f,
        1.5f, 1.5f, 0.0f, 1.0f,
        1.9f, 1.8f, 0.0f, 1.0f,
        1.3f, 1.7f, 0.0f, 1.0f,
        1.6f, 1.4f, 0.0f, 1.0f,
        // c
        2.1f, 3.1f, 0.0f, 1.0f,
        2.8f, 3.9f, 0.0f, 1.0f,
        2.5f, 3.5f, 0.0f, 1.0f,
    };

    float* points_gpu;
    cudaMalloc(&points_gpu, N * C * sizeof(float));
    cudaMemcpy(points_gpu, points_cpu, N * C * sizeof(float), cudaMemcpyHostToDevice);

    float* voxels_gpu;
    int* coordinates_gpu;
    int* num_points_per_voxel_gpu;
    int* voxel_count_gpu;

    cudaMalloc(&voxels_gpu, max_voxels * max_points * C * sizeof(float));
    cudaMalloc(&coordinates_gpu, max_voxels * 3 * sizeof(int));
    cudaMalloc(&num_points_per_voxel_gpu, max_voxels * sizeof(int));
    cudaMalloc(&voxel_count_gpu, sizeof(int));

    cudaMemset(voxels_gpu, 0, max_voxels * max_points * C * sizeof(float));
    cudaMemset(coordinates_gpu, 0, max_voxels * 3 * sizeof(int));
    cudaMemset(num_points_per_voxel_gpu, 0, max_voxels * sizeof(int));
    cudaMemset(voxel_count_gpu, 0, sizeof(int));

    voxelize_launcher(points_gpu, voxels_gpu, coordinates_gpu, num_points_per_voxel_gpu, voxel_count_gpu, N, C, vx, vy, vz, x_min, y_min, z_min, x_max, y_max, z_max, grid_x, grid_y, grid_z, max_points, max_voxels);

    int voxel_count_cpu;
    cudaMemcpy(&voxel_count_cpu, voxel_count_gpu, sizeof(int), cudaMemcpyDeviceToHost);

    float* voxels_cpu = (float*)malloc(voxel_count_cpu * max_points * C * sizeof(float));
    int*   coordinates_cpu = (int*)malloc(voxel_count_cpu * 3 * sizeof(int));
    int*   num_points_per_voxel_cpu = (int*)malloc(voxel_count_cpu * sizeof(int));

    cudaMemcpy(voxels_cpu, voxels_gpu, voxel_count_cpu * max_points * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(coordinates_cpu, coordinates_gpu, voxel_count_cpu * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_points_per_voxel_cpu, num_points_per_voxel_gpu, voxel_count_cpu * sizeof(int), cudaMemcpyDeviceToHost);

    printf("pillars: %d\n", voxel_count_cpu);
    printf("\ncoordinates (z, y, x) and point counts:\n");
    for (int i = 0; i < voxel_count_cpu; i++) {
	printf("pillar %d: (%d, %d, %d) %d points\n", i, coordinates_cpu[i * 3 + 0], coordinates_cpu[i * 3 + 1], coordinates_cpu[i * 3 + 2], num_points_per_voxel_cpu[i]);
    }

    cudaFree(points_gpu);
    cudaFree(voxels_gpu);
    cudaFree(coordinates_gpu);
    cudaFree(num_points_per_voxel_gpu);
    cudaFree(voxel_count_gpu);

    free(voxels_cpu);
    free(coordinates_cpu);
    free(num_points_per_voxel_cpu);

    return 0;
}
