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
        plot_bev(i, points, output_dir)
        all_point_counts.append(points.shape[0])

    print(f"\nSummary for {len(bin_files)} frames")
    print(f"  Min point count : {min(all_point_counts)}")
    print(f"  Max point count : {max(all_point_counts)}")
    print(f"  Avg point count : {sum(all_point_counts) / len(all_point_counts):.0f}")


if __name__ == "__main__":
    main()
