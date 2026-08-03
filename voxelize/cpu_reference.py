import numpy as np

def voxelize_cpu(points, voxel_size, coors_range, max_points=32, max_voxels=20000):
    """
    A simple CPU implementation of voxelize to verify correctness.
    Uses NumPy so that potential bugs within PyTorch may be found.
    """
    voxel_size = np.array(voxel_size, dtype=np.float32)
    coors_range = np.array(coors_range, dtype=np.float32)

    grid_size = (np.round((coors_range[3:] - coors_range[:3]) / voxel_size)).astype(np.int32)

    num_points = points.shape[0]
    num_features = points.shape[1]

    voxels = np.zeros((max_voxels, max_points, num_features), dtype=np.float32)
    coordinates = np.zeros((max_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(max_voxels, dtype=np.int32)

    voxel_map = {}
    num_occupied = 0

    for i in range(num_points):
        point = points[i]

        c_x = int((np.floor((point[0] - coors_range[0]) / voxel_size[0])))
        c_y = int((np.floor((point[1] - coors_range[1]) / voxel_size[1])))
        c_z = int((np.floor((point[2] - coors_range[2]) / voxel_size[2])))

        if c_x < 0 or c_x >= grid_size[0]:
            continue

        if c_y < 0 or c_y >= grid_size[1]:
            continue

        if c_z < 0 or c_z >= grid_size[2]:
            continue

        key = (c_z, c_y, c_x)

        if key not in voxel_map:
            if num_occupied >= max_voxels:
                continue
            voxel_map[key] = num_occupied
            coordinates[num_occupied] = key
            num_occupied += 1

        slot = voxel_map[key]
        if num_points_per_voxel[slot] >= max_points:
                continue

        pt_idx = num_points_per_voxel[slot]
        voxels[slot, pt_idx] = point
        num_points_per_voxel[slot] += 1

    voxels = voxels[:num_occupied]
    coordinates = coordinates[:num_occupied]
    num_points_per_voxel = num_points_per_voxel[:num_occupied]

    return voxels, coordinates, num_points_per_voxel
