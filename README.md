# voxelize

A CUDA C++ kernel that converts raw LiDAR point clouds into the pillar-voxel representation consumed by 3D object detection networks.

---

## Background

A self-driving vehicle's LiDAR sensor fires millions of laser pulses per second and receives back a point cloud: a set of (x, y, z, intensity) measurements with no inherent order or structure. A 64-beam sensor like the Velodyne HDL-64E produces roughly 130,000 points per frame at 10Hz. That data has to become something a neural network can process before any object detection can happen.

The dominant approach, introduced by PointPillars (Lang et al., CVPR 2019), is to project the 3D point cloud down onto a 2D bird's-eye view (BEV) grid. Each grid cell is a vertical column called a **pillar**. Every point gets assigned to the pillar it falls into based on its (x, y) coordinates. The result (a fixed-size tensor of pillar feature vectors) can be processed by a standard convolutional neural network as if it were a 2D image.

This assignment problem looks simple. At scale, it is not. 130,000 unordered points must each independently determine which pillar they belong to and write their features there, in parallel, without threads overwriting each other. That is a GPU problem.

---

## The Problem This Solves

The voxelization step is the computational bottleneck in every pillar-based LiDAR detection pipeline. At inference time in a production AV system running at 10Hz, you have 100ms per frame total. A slow CPU voxelization implementation burns a significant fraction of that budget before the neural network has even started.

---

## Algorithm

Each GPU thread handles exactly one point. The core challenge is that thousands of threads may belong to the same pillar and must coordinate to avoid overwriting each other's data.

The solution is a **hash table with linear probing** allocated in global memory, initialized to -1 (empty). Each thread:

1. Computes its 2D grid index `(cx, cy)` from the point's `(x, y)` coordinates and the voxel size parameters.
2. Computes a hash of the flattened voxel ID `cy * grid_x + cx` and probes the hash table with linear probing to find or claim a slot for that pillar.
3. Uses `atomicCAS` to atomically claim empty hash table slots — only the first thread to arrive at a new pillar wins the slot; all subsequent threads for that pillar find it already occupied.
4. Uses `atomicAdd` on a per-pillar point counter to claim a write position within the pillar's feature array without overwriting other threads' data.
5. Writes the point's C feature channels into the output voxel tensor at the claimed position.

The thread that wins the `atomicCAS` race for a new pillar also claims a sequential output slot index via `atomicAdd` on a global counter and writes the pillar's `(z=0, cy, cx)` coordinate to the output coordinates tensor.

The hash table means pillar assignment is O(1) per point (amortized) rather than requiring any serialized search. The atomic operations eliminate data races without requiring a reduction or synchronization barrier.

---

## Scope

**What this repo is:**

- A single CUDA C++ kernel (`csrc/voxelize.cu`) implementing parallel pillar voxelization
- A PyTorch C++ extension wrapper (`csrc/voxelize_ext.cpp`) exposing the kernel to Python via pybind11
- A clean Python API (`voxelize/`) that pre-allocates output tensors and trims them to the occupied voxel count
- A CPU reference implementation in NumPy (`voxelize/cpu_reference.py`) that serves as a correctness oracle
- A correctness test suite (`tests/`) comparing CUDA output against the CPU reference on real KITTI frames and edge cases
- A benchmarking harness (`benchmarks/`) using `torch.cuda.Event` timing against OpenPCDet's reference op
- An NSight Compute profile (`profiles/`) documenting achieved occupancy, memory throughput, and warp stall reasons
- A `Makefile` reducing build, test, and benchmark to four-letter commands
- A GitHub Actions CI/CD workflow with a self-hosted GPU runner that enforces correctness and a performance regression gate

---

## Deliverables

**1. Benchmark table** comparing this kernel against OpenPCDet's `voxel_generator` at four point cloud densities on an RTX 1060 (Pascal sm_61):

| Points | This kernel (ms ± σ) | OpenPCDet (ms ± σ) | Speedup |
|--------|----------------------|---------------------|---------|
| 25k    | —                    | —                   | —       |
| 50k    | —                    | —                   | —       |
| 100k   | —                    | —                   | —       |
| 130k   | —                    | —                   | —       |

*Benchmarked on NVIDIA RTX 1060 3GB, CUDA 12.x, PyTorch 2.x. Results measured with `torch.cuda.Event` timing over 200 iterations after 20 warmup runs.*

**2. NSight Compute profile** (`profiles/ncu_analysis.md`) documenting achieved occupancy, memory throughput relative to theoretical peak, L2 cache hit rate, and the top three warp stall reasons. The `.ncu-rep` file is attached as a release artifact.

**3. Correctness test suite** (`tests/`) with parametrized tests on 5 real KITTI frames and 4 explicit edge cases: boundary point inclusion/exclusion, `max_points` truncation, and `max_voxels` truncation. All tests run in CI on every push to `dev`.

**4. Optimization analysis** documenting which hardware bottleneck was targeted (`__ldg()` read-only cache, vectorized `float4` loads, or block size tuning) and the before/after benchmark comparison.

---

## Repository Structure

```
voxelize/
├── csrc/
│   ├── voxelize.cu           # CUDA kernel implementation
│   ├── voxelize.h            # Launcher declaration
│   └── voxelize_ext.cpp      # PyTorch C++ extension wrapper
├── voxelize/
│   ├── __init__.py           # Public Python API
│   └── cpu_reference.py      # NumPy CPU reference (correctness oracle)
├── tests/
│   ├── test_correctness.py   # CUDA vs CPU reference on KITTI frames
│   └── test_benchmark.py     # Regression guard
├── benchmarks/
│   ├── benchmark.py          # Full benchmark harness
│   ├── profile_kernel.py     # Single-call profiling entry point
│   └── baseline.json         # Stored baseline for CI regression gate
├── profiles/                 # NSight Compute output
│   └── ncu_analysis.md
├── scripts/
│   └── explore_kitti.py      # Point cloud visualization and statistics
├── .github/
│   └── workflows/
│       └── ci.yml            # Build → test → benchmark regression on GPU runner
├── setup.py
├── Makefile
├── IMPLEMENTATION.md         # Technical deep-dive into kernel design
└── README.md
```

---

## Hardware Requirements

This repo runs entirely on an RTX 1060 3GB. No model training happens here — only kernel execution and benchmarking.

| Resource | Requirement |
|----------|-------------|
| GPU | Any NVIDIA GPU with CUDA compute capability ≥ 6.0 (tested on RTX 1060, sm_61) |
| VRAM | 2GB minimum |
| CUDA Toolkit | 12.x |
| Python | 3.10+ |
| PyTorch | 2.x (CUDA build) |
