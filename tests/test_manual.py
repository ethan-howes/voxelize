from voxelize.cpu_reference import voxelize_cpu
import numpy as np

def test_voxelize_cpu():

    voxel_size = [1.0, 1.0, 4.0]
    coors_range = [0.0, 0.0, -3.0, 10.0, 10.0, 1.0]
    max_points = 5
    max_voxels = 20000
    
    points = np.array([
        [0.5, 0.5, 0.0, 1.0],
        [0.7, 0.3, 0.0, 1.0],
        [1.1, 1.2, 0.0, 1.0],
        [1.5, 1.5, 0.0, 1.0],
        [1.9, 1.8, 0.0, 1.0],
        [2.1, 3.1, 0.0, 1.0],
        [2.8, 3.9, 0.0, 1.0],
    ], dtype=np.float32)
    
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, voxel_size, coors_range, max_points, max_voxels)
    
    assert len(coordinates) == 3
    assert np.sum(num_points_per_voxel) == 7
    assert np.array_equal(np.sort(num_points_per_voxel), [2, 2, 3])
    
    expected_coords = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 3, 2],
    ], dtype=np.int32)
    
    assert np.array_equal(np.sort(coordinates, axis=0), np.sort(expected_coords, axis=0))
    
    print("all assertions passed")

if __name__ == "__main__":
    test_voxelize_cpu()
