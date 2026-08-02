// voxelize.cu
//
// Voxelize takes an input of N points shaped as (N, 4). They are then turned
// into pillars in an x-y grid since the z dimension is not needed for cars.
// The pillars are indexed into a grid and placed into bins using
// floor((x - x_min) / voxel_size). Outputs are a dense tensor of shape
// (D, P, N) representing features, max pillars, and max points per pillar.
// Overflow points are randomly sampled; underflow pillars are zero padded.
