#include "voxelize.h"
#include <cuda_runtime.h>


__device__ int compute_grid_idx(float x, float y, float x_min, float y_min, float vx, float vy, int grid_x, int grid_y) {
    if (isinf(x) || isnan(x) || isinf(y) || isnan(y)) return -1;

    int cx = (int)floorf((x - x_min) / vx);
    int cy = (int)floorf((y - y_min) / vy);

    if (cx < 0 || cx >= grid_x) return -1;
    if (cy < 0 || cy >= grid_y) return -1;

    return cy * grid_x + cx;
}
