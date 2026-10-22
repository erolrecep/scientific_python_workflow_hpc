# Managing Scientific Python Environments and Workflows on HPC Systems

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Build-Make-orange.svg?logo=gnu&logoColor=white" alt="Make">
  <img src="https://img.shields.io/badge/Env-uv%20%7C%20mamba-success.svg" alt="Environment">
  <img src="https://img.shields.io/badge/HPC-SLURM_Ready-blueviolet.svg" alt="HPC">
  <a href="https://colab.research.google.com/github/erolrecep/scientific_python_workflow_hpc/blob/main/notebooks/scientific_python_workflow_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://erolrecep.github.io"><img src="https://img.shields.io/badge/Blog-erolrecep.github.io-red.svg" alt="Blog"></a>
</p>

Welcome to the workshop repository! This project demonstrates how to transition Python workflows from local development environments (laptops or Google Colab) to reproducible, large-scale High-Performance Computing (HPC) systems.

The core philosophy of this workshop is: **Experiment and debug locally, automate and scale globally.**

## 📌 Workshop Objectives

1. **Environment Management:** Learn to use `uv` and `mamba` to build strictly pinned Python environments.

2. **Workflow Automation:** Use `make` to build an idempotent, automated Directed Acyclic Graph (DAG) for data processing.

3. **Multi-Core Scaling:** Scale Python workloads across all allocated CPU cores (`ProcessPoolExecutor`) while avoiding thread oversubscription and respecting SLURM cgroups (`SLURM_CPUS_PER_TASK`).

4. **Cross-Verification:** Process images using multiple libraries (OpenCV, Pillow, PyTorch) and calculate pixel differences to understand the importance of reproducibility.

5. **Code Profiling:** Measure CPU, RAM, and Disk I/O bottlenecks using `scalene` and `memray` before deploying to an HPC cluster.

## 🚀 Prerequisites & Execution Environments

### 1. Google Colab (One-Click Cloud Environment — No Local Setup Required!)

For workshop attendees who do not have a Linux machine, macOS, or dedicated GPU:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erolrecep/scientific_python_workflow_hpc/blob/main/notebooks/scientific_python_workflow_colab.ipynb)

* Open the interactive notebook: [`notebooks/scientific_python_workflow_colab.ipynb`](file:///Users/erolrecep/Developer/codebase/apps/scientific_python_workflow_hpc/notebooks/scientific_python_workflow_colab.ipynb).
* Switch to a free cloud GPU: **Runtime** ➔ **Change runtime type** ➔ **T4 GPU** ➔ **Save**.
* Follow the guided notebook cells to run `uv` bootstrapping, dataset verification, single-core, multi-core, and CUDA GPU batch acceleration with interactive charts!

### 2. Local Environment (Linux / macOS)

* `git` and `make`
* Python 3.9+
* `curl` (for installing `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
* Mamba / Micromamba (optional, if using the Mamba workflow)

## ⚡ The Quickstart: One-Command Execution

Our master `Makefile` orchestrates the entire workflow. You can choose your preferred environment manager to run the complete pipeline:

* **To run using `uv` (Recommended for pure Python speed):**

  ```bash
  make all-uv
  ```

* **To run using `mamba` (Recommended for complex C/C++ system binaries):**

  ```bash
  make all-mamba
  ```

## 🛠️ Environment Managers: `make` vs. Manual Commands

Behind the scenes, our `Makefile` automates the environment creation. If you were to run these commands manually without `make`, here is what it looks like:

### The `uv` Workflow

`uv` is an extremely fast Python package manager written in Rust (10-100x faster than `pip`).

* **Make Command:** `make env-uv`

* **Manual Equivalent:**

  ```bash
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  ```

### The `mamba` Workflow

`mamba` is a fast drop-in replacement for `conda`, perfect when you need non-Python binaries (like complex GPU drivers or system-level C libraries).

* **Make Command:** `make env-mamba`

* **Manual Equivalent:**
  *Note: We use `./.mamba-env` to force mamba to install locally rather than globally.*

  ```bash
  mamba create -y -p ./.mamba-env python=3.10
  ./.mamba-env/bin/python -m pip install -r requirements.txt
  ```

## 🖼️ Image Operations & Cross-Verification

Our Python script (`src/process_images.py`) runs a specific pipeline on a sample image from the VGG 17flowers dataset:
**Original Image ➔ Grayscale ➔ Sharpen (Custom Kernel) ➔ Resize (229x229)**

### Why Strict Environments Matter (The Diff Test)

We run the exact same resizing operations across OpenCV, Pillow, and PyTorch. Because different libraries use different underlying interpolation algorithms, **the resulting pixel values are not identical!**

When you run the pipeline, you will see output proving this difference:

```text
--- Cross-Library Difference Analysis ---
OpenCV vs Pillow  | Mean Diff: 11.31 pixels | Max Diff: 135.0 pixels
OpenCV vs PyTorch | Mean Diff: 12.09 pixels | Max Diff: 140.9 pixels
Pillow vs PyTorch | Mean Diff:  1.01 pixels | Max Diff:  12.8 pixels

Conclusion: The exact same operations yield different pixel values.
This proves why strict environment pinning is mandatory for scientific reproducibility!
```

## ⚡ Multi-Core Scaling & Benchmarking

Scientific computing on HPC is fundamentally parallel. Running Python sequentially wastes costly multi-core allocations. Our pipeline implements **embarrassingly parallel batch processing** with critical HPC optimizations:

### 1. Accurate Core Allocation Detection
Instead of blindly calling `os.cpu_count()` (which returns all cores on a shared node and causes severe cgroup quota thrashing), our pipeline inspects:
1. `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE`
2. `os.sched_getaffinity(0)` (Linux CPU cgroups)
3. Host CPU count (fallback)

### 2. Anti-Oversubscription Worker Initialization
A classic HPC trap is *thread oversubscription*: if 4 worker processes each spawn 8 threads via OpenCV, OpenMP, or PyTorch, 32 threads fight for 4 cores. Our worker initializer restricts child libraries to single-threaded mode:
```python
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(1)
torch.set_num_threads(1)
```

### 3. Chunked Task Distribution
Images are batched into contiguous slices per worker process, eliminating 99% of inter-process serialization (IPC) overhead.

### 4. Running Benchmarks Across Backends & GPU
```bash
# Run 1-core vs. N-core benchmark for a specific backend:
make benchmark WORKERS=16 BACKEND=opencv MAX_IMAGES=1000
make benchmark WORKERS=16 BACKEND=pillow MAX_IMAGES=1000
make benchmark WORKERS=16 BACKEND=pytorch MAX_IMAGES=1000

# Run unified comparison across ALL backends (OpenCV, Pillow, PyTorch CPU & GPU):
make benchmark-all WORKERS=16 MAX_IMAGES=1000

# Run standalone batched GPU processing on CUDA / MPS:
make process-gpu DEVICE=cuda BATCH_SIZE=64 MAX_IMAGES=1000
```

## 🚀 Third Running Approach: GPU Acceleration (`dlbox`)

Beyond sequential and multi-core CPU execution, our third paradigm is **hardware-accelerated GPU batch processing** utilizing PyTorch with CUDA (or Apple Silicon MPS). 

### GPU Architecture Highlights
- **Pipelined Host-to-Device Streaming:** Multi-threaded SIMD host decoding (`cv2.imread` / `libjpeg-turbo`) streams zero-copy tensors directly to the GPU stream.
- **Batched 2D Tensor Convolutions:** The sharpening kernel (`3x3`) is computed in parallel across the entire tensor batch on CUDA cores via `F.conv2d(batch, weight, padding=1)`.
- **GPU Bilinear Interpolation:** Resizing to `(229, 229)` is executed directly on GPU memory via `F.interpolate()`.
- **Variable-Dimension Resilience:** Mini-batches with identical shapes execute in batched matrix operations; variable-resolution inputs stream dynamically without shape errors.

### Empirical Benchmark Results on `dlbox` (1,000 Images)
* **Hardware:** AMD Ryzen Threadripper 2950X (16 Cores / 32 Threads) + NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
* **Dataset:** 1,000 images from Oxford 17flowers

| Backend | Execution Mode | Hardware Target | Wall Time | Throughput | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenCV** | CPU (1-Core) | AMD Threadripper (1 Core) | 1.944 s | 514.4 img/s | 1.00x |
| **OpenCV** | CPU (16-Core) | AMD Threadripper (16 Cores) | **0.710 s** | **1,408.9 img/s** | **2.74x** |
| **Pillow** | CPU (1-Core) | AMD Threadripper (1 Core) | 3.568 s | 280.3 img/s | 0.54x |
| **Pillow** | CPU (16-Core) | AMD Threadripper (16 Cores) | **0.849 s** | **1,177.7 img/s** | **2.29x** |
| **PyTorch** | CPU (1-Core) | AMD Threadripper (1 Core) | 4.760 s | 210.1 img/s | 0.41x |
| **PyTorch** | CPU (16-Core) | AMD Threadripper (16 Cores) | 7.126 s | 140.3 img/s | 0.27x |
| **PyTorch (GPU)** | **GPU (CUDA)** | **NVIDIA RTX 5060 Ti** | **0.895 s** | **1,117.2 img/s** | **2.17x** *(5.32x vs PyTorch CPU)* |

> **Key Takeaway:** For CPU multi-core scaling, OpenCV and Pillow scale with 2.3x–2.7x speedup across 16 cores. For PyTorch, running on GPU via CUDA delivers an immediate **5.32x speedup** over PyTorch CPU execution while freeing CPU cores for upstream ETL.

## 📊 Profiling: What to Expect

During the pipeline execution, we run two profilers. **Note: the script runs twice** (once for memory, once for CPU) to prevent the profilers from interfering with each other's metrics.

### 1. Memory Profiling (`memray`)

Memray tracks memory allocations down to the underlying native C/C++ extensions.

* **Output:** It generates an interactive HTML flamegraph in the `results/` folder (`memray-flamegraph-memray_profile.html`). You can open this in your web browser to visually hunt down memory leaks.

### 2. CPU/GPU Profiling (`scalene`)

Scalene tracks exactly which lines of code consume the most compute time.

* **Output:** At the end of the `make` execution, you will see a success box with a custom command:

  ```text
  ======================================================================
  ✅ PROFILING COMPLETE!
  To view the CPU profile in your browser, copy and paste this command:
    .venv/bin/scalene view results/scalene_profile.json
  ======================================================================
  ```

  *(If using mamba, the path will be `./.mamba-env/bin/scalene view...`)*

## 🌉 The HPC Bridge: Moving from Laptop to Cluster

A compute cluster is a shared resource. You do not run heavy processing interactively on the login node. Once your `Makefile` works locally, moving to HPC simply involves submitting your automated DAG to a job scheduler (like SLURM).

### Example SLURM Submission Script (`job.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=image_pipeline
#SBATCH --output=results/hpc_job_%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute

# 1. Load system modules (cluster-specific)
# module load python/3.10 curl

# 2. Execute the entire automated DAG using uv
make all-uv
```

Submit using: `sbatch job.sbatch`. Once complete, you can review `results/hpc_job_<ID>.out` and compare the cluster's execution metrics against your laptop's baseline!


### Supercomputer SLURM Workflow (Rocky Linux & Modules)

For deployment on shared resource supercomputers running **Rocky Linux**, **SLURM**, and **Lmod / Environment Modules**, we provide 3 dedicated production `.sbatch` scripts managed via `Makefile.sbatch`:

| Script | Purpose | Allocation |
| :--- | :--- | :--- |
| **`sbatch/01_env_check.sbatch`** | Validates Rocky Linux, environment modules, bootstraps standalone `uv` directly from source, verifies dataset, and runs pytest. | 2 CPUs, 4GB RAM (`--partition=compute`) |
| **`sbatch/02_single_core.sbatch`** | Runs strict single-core sequential baseline with thread pinning, generates Memray flamegraphs and Scalene CPU profiles. | 1 CPU, 4GB RAM (`--partition=compute`) |
| **`sbatch/03_multicore_gpu.sbatch`** | Runs back-to-back 16-core parallel CPU scaling against hardware-accelerated NVIDIA CUDA GPU batch processing. | 16 CPUs, 1 GPU, 16GB RAM (`--partition=gpu`) |

#### Managing & Submitting SLURM Jobs via `Makefile.sbatch`

```bash
# 1. Run Job 1: Environment readiness and module check
make -f Makefile.sbatch check

# 2. Run Job 2: Strict single-core baseline and profiling
make -f Makefile.sbatch single

# 3. Run Job 3: Back-to-back multi-core and GPU scaling run
make -f Makefile.sbatch multi-gpu

# 4. Run the FULL 3-stage automated pipeline with SLURM job dependencies:
# (Job 2 runs only after Job 1 succeeds; Job 3 runs only after Job 2 succeeds)
make -f Makefile.sbatch pipeline

# 5. Monitor and inspect jobs:
make -f Makefile.sbatch status   # Check queue: squeue -u $USER
make -f Makefile.sbatch logs     # Tail latest SLURM output log
make -f Makefile.sbatch cancel   # Cancel all jobs: scancel -u $USER
```

1. **Validation (`check`):**
   * Submits `sbatch/01_env_check.sbatch` to verify Rocky Linux compatibility, load cluster modules, bootstrap `uv` if missing, verify datasets, and run pytest tests.

2. **Single-Core Baseline (`single`):**
   * Submits `sbatch/02_single_core.sbatch` with strict thread pinning (`OMP_NUM_THREADS=1`) to establish a ground-truth sequential execution speed and generate Memray flamegraphs & Scalene CPU profiles.

3. **Multi-Core & GPU Scaling (`multi-gpu`):**
   * Submits `sbatch/03_multicore_gpu.sbatch` on a GPU node (`--partition=gpu --gpus=1 --cpus-per-task=16`), comparing multi-process CPU workers against CUDA-accelerated batch processing.

4. **Automated Pipeline (`pipeline`):**
   * Submits all 3 jobs sequentially using SLURM job dependencies (`--dependency=afterok:<jobid>`). Job 2 only runs after Job 1 completes successfully, and Job 3 runs after Job 2.

5. **Monitoring & Management:**
   * `make -f Makefile.sbatch status`: Inspect active jobs in the queue via `squeue`.
   * `make -f Makefile.sbatch logs`: Automatically `tail -f` the most recent `.out` log in `results/`.
   * `make -f Makefile.sbatch cancel`: Safely cancel queued/running jobs via `scancel`.
   * `make -f Makefile.sbatch clean-logs`: Clean up old `.out` and `.err` logs.


## 📝 License

This project is licensed under the MIT License.