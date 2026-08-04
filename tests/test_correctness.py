import pytest
import numpy as np
import os
from voxelize.cpu_reference import voxelize_cpu


# in case kitti data lives in another dir
KITTI_DIR = os.environ.get('KITTI_DIR', 'data/kitti/training/velodyne')


@pytest.fixture(params=list(range(5)))
def kitti_frame(request):
    frame_idx = request.param
    path = os.path.join(KITTI_DIR, f'{frame_idx:06d}.bin')
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


VOXEL_SIZE  = [0.16, 0.16, 4.0]
COORS_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
MAX_POINTS  = 32
MAX_VOXELS  = 20000


def test_voxelize_cpu_on_kitti(kitti_frame):
    points = kitti_frame
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, VOXEL_SIZE, COORS_RANGE, MAX_POINTS, MAX_VOXELS)

    sort_keys = (coordinates[:, 0] * 10000 * 10000 + coordinates[:, 1] * 10000 + coordinates[:, 2])
    sort_order = np.argsort(sort_keys)

    sorted_coords  = coordinates[sort_order]
    sorted_npoints = num_points_per_voxel[sort_order]

    # sanity checks
    assert len(coordinates) > 0, "No occupied pillars found"
    assert len(coordinates) <= MAX_VOXELS, "Exceeded max voxels"
    assert len(coordinates) == len(num_points_per_voxel), "Output length mismatch"
    assert np.all(num_points_per_voxel > 0), "Empty pillar in output"
    assert np.all(num_points_per_voxel <= MAX_POINTS), "Exceeded max points"

    # bounds checks
    assert np.all(sorted_coords[:, 1] >= 0), "Negative y index"
    assert np.all(sorted_coords[:, 2] >= 0), "Negative x index"

    # need cuda section here


# point exactly on x_max should be excluded
def test_point_on_x_max_excluded():
    points = np.array([[69.12, 0.0, 0.0, 1.0]], dtype=np.float32)
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, VOXEL_SIZE, COORS_RANGE, MAX_POINTS, MAX_VOXELS)
    assert len(coordinates) == 0, "point on x_max should be excluded"


# point exactly on x_min should be included
def test_point_on_x_min_included():
    points = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, VOXEL_SIZE, COORS_RANGE, MAX_POINTS, MAX_VOXELS)
    assert len(coordinates) == 1, "point on x_min should be included"


# pillar receiving more than max_points
def test_max_points_truncation():
    n = MAX_POINTS + 5
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = 1.0
    points[:, 1] = 1.0
    points[:, 2] = 0.0
    points[:, 3] = 1.0
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, VOXEL_SIZE, COORS_RANGE, MAX_POINTS, MAX_VOXELS)
    assert len(coordinates) == 1, "all points should land in one pillar"
    assert num_points_per_voxel[0] == MAX_POINTS, "pillar should be truncated to max_points"


# frame with more unique pillars than max_voxels
def test_max_voxels_truncation():
    n = MAX_VOXELS + 1000
    points = np.zeros((n, 4), dtype=np.float32)
    grid_x = int(round((69.12 - 0) / 0.16))
    for i in range(n):
        cx = i % grid_x
        cy = i // grid_x
        points[i, 0] = cx * 0.16 + 0.08
        points[i, 1] = -39.68 + cy * 0.16 + 0.08
        points[i, 2] = 0.0
        points[i, 3] = 1.0
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, VOXEL_SIZE, COORS_RANGE, MAX_POINTS, MAX_VOXELS)
    assert len(coordinates) == MAX_VOXELS, "Should be capped at max_voxels"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
