import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from voxelize.cpu_reference import voxelize_cpu


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
    plt.colorbar(sc, ax=ax, label='Intensity')
    ax.set_xlabel('x (m) — forward')
    ax.set_ylabel('y (m) — lateral')
    ax.set_title(f'KITTI Frame {frame_idx:06d} — {points.shape[0]} points')
    ax.set_aspect('equal')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved BEV plot: {out_path}")


def plot_voxel_heatmap(frame_idx, points, voxel_size, coors_range, output_dir):
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, voxel_size, coors_range, max_points=32, max_voxels=20000)

    grid_x = int(round((coors_range[3] - coors_range[0]) / voxel_size[0]))
    grid_y = int(round((coors_range[4] - coors_range[1]) / voxel_size[1]))

    # fill with point counts
    heatmap = np.zeros((grid_y, grid_x), dtype=np.float32)
    for i in range(len(coordinates)):
        c_z, c_y, c_x = coordinates[i]
        heatmap[c_y, c_x] = num_points_per_voxel[i]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(heatmap, origin='lower', cmap='hot')
    ax.set_title(f'Frame {frame_idx:06d} — BEV pillar density')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {out_path}")


def plot_point_histogram(frame_idx, points, voxel_size, coors_range, output_dir):
    voxels, coordinates, num_points_per_voxel = voxelize_cpu(points, voxel_size, coors_range, max_points=32, max_voxels=20000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(num_points_per_voxel, bins=32, color='steelblue', edgecolor='black')
    ax.set_xlabel('Points per pillar')
    ax.set_ylabel('Number of pillars')
    ax.set_title(f'Frame {frame_idx:06d} — points per pillar distribution')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_histogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {out_path}")


def main():
    if len(sys.argv) > 1:
        velodyne_dir = sys.argv[1]
    else:
        velodyne_dir = "data/kitti/velodyne"

    output_dir = "scripts/kitti_analysis"

    bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])[:10]

    if not bin_files:
        print(f"No .bin files found in {velodyne_dir}")
        sys.exit(1)

    print(f"Found {len(bin_files)} frames in {velodyne_dir}")

    all_point_counts = []

    for i, fname in enumerate(bin_files):
        path = os.path.join(velodyne_dir, fname)
        points = load_kitti_frame(path)
        print_frame_stats(i, points)
        voxel_size  = [0.16, 0.16, 4.0]
        coors_range = [0, -39.68, -3, 69.12, 39.68, 1]
        plot_voxel_heatmap(i, points, voxel_size, coors_range, output_dir)
        plot_point_histogram(i, points, voxel_size, coors_range, output_dir)
        plot_bev(i, points, output_dir)
        all_point_counts.append(points.shape[0])

    print(f"\nSummary for {len(bin_files)} frames")
    print(f"  Min point count : {min(all_point_counts)}")
    print(f"  Max point count : {max(all_point_counts)}")
    print(f"  Avg point count : {sum(all_point_counts) / len(all_point_counts):.0f}")


if __name__ == "__main__":
    main()
