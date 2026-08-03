import numpy as np

def voxelize_cpu(points, voxel_size, coors_range, max_points=32, max_voxels=20000):
    """
    A simple CPU implementation of voxelize to verify correctness.
    Uses NumPy so that potential bugs within PyTorch may be found.
    """
    voxel_size = np.array(voxel_size, dtype=np.float32)
    coors_range = np.array(coors_range, dtype=np.float32)

    grid_size = (np.round((coors_range[3:] - coors_range[:3]) / voxel_size)).astype(np.int32)


