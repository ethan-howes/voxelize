import numpy as np
import time
import json
from voxelize.cpu_reference import voxelize_cpu


voxel_size  = [0.16, 0.16, 4.0]
coors_range = [0, -39.68, -3, 69.12, 39.68, 1]
max_points  = 32
max_voxels  = 20000


def generate_points(n):
    np.random.seed(42)
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = np.random.uniform(0, 69.12, n)
    points[:, 1] = np.random.uniform(-39.68, 39.68, n)
    points[:, 2] = np.random.uniform(-3, 1, n)
    points[:, 3] = np.random.uniform(0, 1, n)
    return points


def time_cpu(points, n_iter=100):
    times = []

    for i in range(n_iter):
        start = time.perf_counter()
        voxelize_cpu(points, voxel_size, coors_range, max_points, max_voxels)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return times.mean(), times.std()


def time_cuda(fn, args, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn(*args)

    times = []

    time_start = torch.cuda.Event(enable_timing = True)
    time_end = torch.cuda.Event(enable_timing = True)

    for _ in range(n_iter):
        time_start.record()
        fn(*args)
        time_end.record()

        torch.cuda.synchronize()
        times.append(time_start.elapsed_time(time_end))

    nptimes = np.array(times)
    times_mean = np.mean(nptimes)
    times_std = np.std(nptimes)

    return times_mean, times_std


def main():
    point_counts = [25000, 50000, 100000, 130000]
    results = {}

    for n in point_counts:
        points = generate_points(n)
        mean, std = time_cpu(points)
        results[n] = {'mean_ms': round(mean, 3), 'std_ms': round(std, 3)}
        print(f"  {n:>7} points: {mean:.2f} ± {std:.2f} ms")

    return results


if __name__ == "__main__":
    results = main()
    with open('evals/baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nBaseline saved to evals/baseline.json")
