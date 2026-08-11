import numpy as np
import time
import torch
import json
from voxelize.cpu_reference import voxelize_cpu
from voxelize import voxelize


voxel_size = [0.16, 0.16, 4.0]
coors_range = [0, -39.68, -3, 69.12, 39.68, 1]
max_points = 32
max_voxels = 20000


def generate_points(n):
    np.random.seed(42)
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = np.random.uniform(0, 69.12, n)
    points[:, 1] = np.random.uniform(-39.68, 39.68, n)
    points[:, 2] = np.random.uniform(-3, 1, n)
    points[:, 3] = np.random.uniform(0, 1, n)
    return points


def time_cpu(points, n_iter=10):
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


def generate_points_gpu(n):
    torch.manual_seed(42)
    points = torch.zeros((n, 4), dtype=torch.float32)
    points[:, 0] = torch.rand(n) * 69.12
    points[:, 1] = torch.rand(n) * 79.36 - 39.68
    points[:, 2] = torch.rand(n) * 4.0 - 3.0
    points[:, 3] = torch.rand(n)
    return points.cuda()


def main():
    point_counts = [25000, 50000, 100000, 130000]
    results = {}

    for n in point_counts:
        points_cpu = generate_points(n)
        points_gpu = generate_points_gpu(n)

        cpu_mean, cpu_std = time_cpu(points_cpu)
        gpu_mean, gpu_std = time_cuda(voxelize, [points_gpu, voxel_size, coors_range])

        speedup = cpu_mean / gpu_mean

        results[n] = {
            'cpu_mean_ms': round(cpu_mean, 3),
            'cpu_std_ms': round(cpu_std, 3),
            'cuda_mean_ms': round(gpu_mean, 3),
            'cuda_std_ms': round(gpu_std, 3),
            'speedup': round(speedup, 2),
        } 
        print(f"{n:>7} points: {cpu_mean:>8.2f} ± {cpu_std:.2f} | {gpu_mean:>8.2f} ± {gpu_std:.2f} | {speedup:>6.1f}x")

    return results


if __name__ == "__main__":
    results = main()
    with open('evals/baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nBaseline saved to evals/baseline.json")
