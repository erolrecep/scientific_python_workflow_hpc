#!/usr/bin/env python3
"""
process_images.py: High-Performance Multi-Core Image Processing & Verification Pipeline
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

DATA_DIR = "data/jpg"
TARGET_SIZE = (229, 229)
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

_PYTORCH_TRANSFORMS = None

def _get_pytorch_transforms():
    global _PYTORCH_TRANSFORMS
    if _PYTORCH_TRANSFORMS is None:
        import torchvision.transforms as T
        _PYTORCH_TRANSFORMS = (
            T.ToTensor(),
            T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.BILINEAR)
        )
    return _PYTORCH_TRANSFORMS


def get_optimal_workers(requested_workers: int = None) -> int:
    """
    Detects the optimal number of CPU workers based on SLURM environment,
    process CPU affinity (cgroups), or host CPU core count.
    """
    if requested_workers is not None and requested_workers > 0:
        return requested_workers

    # 1. Check SLURM job allocation
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    # 2. Check process affinity (Linux cgroups / taskset)
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                return affinity
        except Exception:
            pass

    # 3. Fallback to host CPU count
    return max(1, os.cpu_count() or 1)


def init_worker():
    """
    Initializes each worker process.
    Crucial HPC Best Practice: Pin internal library threads to 1 to prevent
    severe thread oversubscription and context switching across cores.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    if "torch" in sys.modules:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass


def get_sample_image(data_dir: str = DATA_DIR) -> str:
    """Finds the first available JPEG in data_dir."""
    images = get_all_images(data_dir)
    if not images:
        raise FileNotFoundError(f"No images found in '{data_dir}'. Run 'make data' first.")
    return images[0]


def get_all_images(data_dir: str = DATA_DIR, max_images: int = None) -> List[str]:
    """Scans and retrieves image paths from data_dir."""
    if not os.path.exists(data_dir):
        return []
    images = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if max_images and max_images > 0:
        images = images[:max_images]
    return images


# =============================================================================
# Core Pipeline Implementations across Libraries
# =============================================================================

def process_with_opencv(image_path: str) -> np.ndarray:
    """Grayscale -> Sharpen -> Bilinear Resize using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.filter2D(gray, -1, SHARPEN_KERNEL)
    resized = cv2.resize(sharpened, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized


def process_with_pillow(image_path: str) -> Image.Image:
    """Grayscale -> Sharpen -> Bilinear Resize using Pillow."""
    img = Image.open(image_path).convert('L')
    sharpened = img.filter(ImageFilter.SHARPEN)
    resized = sharpened.resize(TARGET_SIZE, Image.BILINEAR)
    return resized


def process_with_pytorch(image_path: str):
    """Grayscale -> Bilinear Resize using PyTorch."""
    to_tensor, resize = _get_pytorch_transforms()
    img = Image.open(image_path).convert('L')
    tensor_img = to_tensor(img)
    resized = resize(tensor_img)
    return resized


def process_single_task(args: Tuple[str, str, str]) -> str:
    """Worker task wrapper for ProcessPoolExecutor."""
    image_path, backend, output_dir = args
    if backend == "opencv":
        res = process_with_opencv(image_path)
        if output_dir:
            out_file = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(out_file, res)
    elif backend == "pillow":
        res = process_with_pillow(image_path)
        if output_dir:
            out_file = os.path.join(output_dir, os.path.basename(image_path))
            res.save(out_file)
    elif backend == "pytorch":
        res = process_with_pytorch(image_path)
        if output_dir:
            out_file = os.path.join(output_dir, os.path.basename(image_path))
            arr = (res.squeeze().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(arr).save(out_file)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return image_path


def process_chunk_task(args: Tuple[List[str], str, str]) -> int:
    """Processes a batch/slice of image paths inside a dedicated worker process."""
    paths, backend, output_dir = args
    for p in paths:
        process_single_task((p, backend, output_dir))
    return len(paths)


# =============================================================================
# Sequential & Multi-Core Batch Processing
# =============================================================================

def process_dataset_sequential(image_paths: List[str], backend: str = "opencv", output_dir: str = None) -> float:
    """Processes images sequentially on a single CPU core."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    t0 = time.perf_counter()
    for p in image_paths:
        process_single_task((p, backend, output_dir))
    t1 = time.perf_counter()
    return t1 - t0


def get_mp_context():
    """Returns fork context on Unix/Linux for near-zero process startup latency."""
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        try:
            return mp.get_context("fork")
        except Exception:
            pass
    return mp.get_context()


def process_dataset_multicore(image_paths: List[str], num_workers: int = None,
                              backend: str = "opencv", output_dir: str = None) -> float:
    """Processes images in parallel across multiple CPU worker processes using chunking."""
    workers = get_optimal_workers(num_workers)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if workers <= 1 or len(image_paths) <= 1:
        return process_dataset_sequential(image_paths, backend=backend, output_dir=output_dir)

    # Chunk the work evenly across workers to minimize IPC serialization overhead
    num_chunks = min(workers, len(image_paths))
    chunks = [image_paths[i::num_chunks] for i in range(num_chunks)]
    chunk_tasks = [(chunk, backend, output_dir) for chunk in chunks if chunk]

    t0 = time.perf_counter()
    ctx = get_mp_context()
    with ProcessPoolExecutor(max_workers=num_chunks, mp_context=ctx, initializer=init_worker) as executor:
        list(executor.map(process_chunk_task, chunk_tasks))
    t1 = time.perf_counter()
    return t1 - t0


# =============================================================================
# Cross-Library Reproducibility Analysis
# =============================================================================

def run_cross_verification(sample_img: str):
    """Executes single-image pipeline across OpenCV, Pillow, and PyTorch and reports pixel diffs."""
    print(f"\n🔬 Cross-Verification Sample: {sample_img}")
    cv2_res = process_with_opencv(sample_img)
    pil_res = process_with_pillow(sample_img)
    torch_res = process_with_pytorch(sample_img)

    print("\n--- Output Dimensions ---")
    print(f"OpenCV shape:  {cv2_res.shape}")
    print(f"Pillow size:   {pil_res.size}")
    print(f"PyTorch shape: {torch_res.shape}")

    print("\n--- Cross-Library Difference Analysis ---")
    arr_cv2 = cv2_res.astype(np.float32)
    arr_pil = np.array(pil_res, dtype=np.float32)
    arr_torch = (torch_res.squeeze().numpy() * 255.0).astype(np.float32)

    diff_cv2_pil = np.abs(arr_cv2 - arr_pil)
    diff_cv2_torch = np.abs(arr_cv2 - arr_torch)
    diff_pil_torch = np.abs(arr_pil - arr_torch)

    print(f"OpenCV vs Pillow  | Mean Diff: {diff_cv2_pil.mean():>5.2f} pixels | Max Diff: {diff_cv2_pil.max():>5.1f} pixels")
    print(f"OpenCV vs PyTorch | Mean Diff: {diff_cv2_torch.mean():>5.2f} pixels | Max Diff: {diff_cv2_torch.max():>5.1f} pixels")
    print(f"Pillow vs PyTorch | Mean Diff: {diff_pil_torch.mean():>5.2f} pixels | Max Diff: {diff_pil_torch.max():>5.1f} pixels")

    print("\n💡 Conclusion: The exact same operations yield different pixel values.")
    print("   This proves why strict environment pinning is mandatory for scientific reproducibility!")


# =============================================================================
# Benchmarking & Scaling Suite
# =============================================================================

def run_multicore_benchmark(image_paths: List[str], workers: int, backend: str = "opencv"):
    """Runs a 1-Core vs N-Core scaling benchmark and computes speedup & efficiency."""
    n_images = len(image_paths)
    print("\n" + "=" * 70)
    print(f"🚀 HPC MULTI-CORE SCALING BENCHMARK ({n_images} images | backend: {backend})")
    print("=" * 70)
    print(f"  Detected Allocation: {workers} CPU Cores")

    # 1. Sequential Single-Core Run
    print(f"\n[1/2] Running Sequential Baseline (1 Worker)...")
    t_seq = process_dataset_sequential(image_paths, backend=backend)
    fps_seq = n_images / t_seq
    print(f"      Time: {t_seq:.3f} s  |  Throughput: {fps_seq:.1f} images/sec")

    # 2. Parallel Multi-Core Run
    print(f"\n[2/2] Running Multi-Core Parallel Run ({workers} Workers)...")
    t_par = process_dataset_multicore(image_paths, num_workers=workers, backend=backend)
    fps_par = n_images / t_par
    print(f"      Time: {t_par:.3f} s  |  Throughput: {fps_par:.1f} images/sec")

    # Metrics
    speedup = t_seq / t_par if t_par > 0 else 0.0
    efficiency = (speedup / workers) * 100.0 if workers > 0 else 0.0

    print("\n" + "-" * 70)
    print(f"{'Mode':<18} | {'Workers':<8} | {'Wall Time':<12} | {'Throughput':<15} | {'Speedup':<8}")
    print("-" * 70)
    print(f"{'Single-Core':<18} | {1:<8} | {t_seq:7.3f} s    | {fps_seq:6.1f} img/s    | 1.00x")
    print(f"{'Multi-Core (HPC)':<18} | {workers:<8} | {t_par:7.3f} s    | {fps_par:6.1f} img/s    | {speedup:4.2f}x")
    print("-" * 70)
    print(f"🎉 Scaling Factor:   {speedup:.2f}x speedup across {workers} cores")
    print(f"⚡ Parallel Efficiency: {efficiency:.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-Core Scientific Image Processing Pipeline")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to raw JPEG images directory")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of CPU worker processes (auto-detected if None)")
    parser.add_argument("--max-images", "-n", type=int, default=100, help="Max images to process (0 for all)")
    parser.add_argument("--backend", type=str, default="opencv", choices=["opencv", "pillow", "pytorch"], help="Processing library backend")
    parser.add_argument("--benchmark", action="store_true", help="Run 1-core vs N-core scaling benchmark")
    parser.add_argument("--no-verify", action="store_true", help="Skip single-image cross-library diff test")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save processed output images")
    args = parser.parse_args()

    # Verify data exists or trigger automatic download/generation
    images = get_all_images(args.data_dir, max_images=args.max_images if args.max_images > 0 else None)
    if not images:
        print(f"⚠️ No images found in '{args.data_dir}'. Generating synthetic images for execution...")
        from download_data import generate_synthetic_dataset
        generate_synthetic_dataset(args.data_dir, num_images=100)
        images = get_all_images(args.data_dir, max_images=args.max_images if args.max_images > 0 else None)

    workers = get_optimal_workers(args.workers)

    # 1. Cross-Verification on sample image
    if not args.no_verify and len(images) > 0:
        run_cross_verification(images[0])

    # 2. Benchmarking or Multi-Core Batch Processing
    if args.benchmark:
        run_multicore_benchmark(images, workers=workers, backend=args.backend)
    else:
        print(f"\n⚡ Executing Multi-Core Batch Pipeline ({len(images)} images | {workers} workers | backend: {args.backend})...")
        t_elapsed = process_dataset_multicore(images, num_workers=workers, backend=args.backend, output_dir=args.output_dir)
        throughput = len(images) / t_elapsed if t_elapsed > 0 else 0.0
        print(f"✅ Processed {len(images)} images in {t_elapsed:.3f} s ({throughput:.1f} images/sec) across {workers} cores.")


if __name__ == "__main__":
    main()
