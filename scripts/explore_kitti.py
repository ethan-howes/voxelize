import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_kitti_frame(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def print_frame_stats(frame_idx, points):
    print(f"Total points : {points.shape[0]}")
    channels = ['x', 'y', 'z', 'intensity']
    for i, name in enumerate(channels):
        print(f"  {name} : min={points[:, i].min():.3f}  max={points[:, i].max():.3f}")


def plot_bev(frame_idx, points, output_dir):
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 3],
        s=0.2,
        cmap='plasma',
        vmin=0.0,
        vmax=1.0
    )


all_point_counts = []
all_point_counts.append(points.shape[0])

print(f"Min point count : {min(all_point_counts)}")
print(f"Max point count : {max(all_point_counts)}")
print(f"Avg point count : {sum(all_point_counts) / len(all_point_counts):.0f}")

if len(sys.argv) > 1:
    velodyne_dir = sys.argv[1]
else:
    velodyne_dir = "data/kitti/velodyne"
