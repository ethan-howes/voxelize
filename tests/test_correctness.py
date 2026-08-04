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


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
