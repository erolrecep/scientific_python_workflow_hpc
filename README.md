# Managing Scientific Python Environments and Workflows on HPC Systems

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Build-Make-orange.svg?logo=gnu&logoColor=white" alt="Make">
  <img src="https://img.shields.io/badge/Env-uv%20%7C%20mamba-success.svg" alt="Environment">
  <img src="https://img.shields.io/badge/HPC-SLURM_Ready-blueviolet.svg" alt="HPC">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://erolrecep.github.io"><img src="https://img.shields.io/badge/Blog-erolrecep.github.io-red.svg" alt="Blog"></a>
</p>

Welcome to the workshop repository! This project demonstrates how to transition Python workflows from local development environments (laptops or Google Colab) to reproducible, large-scale High-Performance Computing (HPC) systems.

The core philosophy of this workshop is: **Experiment and debug locally, automate and scale globally.**

## 📌 Workshop Objectives

1. **Environment Management:** Learn to use `uv` and `mamba` to build strictly pinned Python environments.

2. **Workflow Automation:** Use `make` to build an idempotent, automated Directed Acyclic Graph (DAG) for data processing.

3. **Cross-Verification:** Process images using multiple libraries (OpenCV, Pillow, PyTorch) and calculate pixel differences to understand the importance of reproducibility.

4. **Code Profiling:** Measure CPU, RAM, and Disk I/O bottlenecks using `scalene` and `memray` before deploying to an HPC cluster.

## 🚀 Prerequisites

### Local Environment (Linux / macOS)

* `git` and `make`

* Python 3.9+

* `curl` (for installing `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

* Mamba / Micromamba (optional, if using the Mamba workflow)

### Google Colab (Alternative)

If you are on Windows without WSL, open a Google Colab notebook, mount your Google Drive, and open a terminal instance. (Free-tier users can run `!pip install colab-xterm` and `%xterm` to get an interactive terminal).

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

## 📝 License

This project is licensed under the MIT License.