#pragma once

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
);
