import torch
import voxelize_cuda


def voxelize(points, voxel_size, coors_range, max_points=32, max_voxels=20000):
    vx, vy, vz = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = coors_range

    grid_x = round((x_max - x_min) / vx)
    grid_y = round((y_max - y_min) / vy)
    grid_z = round((z_max - z_min) / vz)

    C = points.shape[1]

    voxels = torch.zeros((max_voxels, max_points, C), dtype=torch.float32, device=points.device)
    coordinates = torch.zeros((max_voxels, 3), dtype=torch.int32, device=points.device)
    num_points_per_voxel = torch.zeros(max_voxels, dtype=torch.int32, device=points.device)
    voxel_count = torch.zeros(1, dtype=torch.int32, device=points.device)

    with torch.no_grad():
        voxelize_cuda.voxelize(
            points, voxels, coordinates,
            num_points_per_voxel, voxel_count,
            vx, vy, vz,
            x_min, y_min, z_min,
            x_max, y_max, z_max,
            grid_x, grid_y, grid_z,
            max_points, max_voxels
        )

    n = voxel_count.item()
    return voxels[:n], coordinates[:n], num_points_per_voxel[:n]
