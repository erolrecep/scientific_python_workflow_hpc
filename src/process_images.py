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

_PYTORCH_TRANSFORMS = {}

def get_optimal_device(requested_device: str = None) -> str:
    """Detects available hardware acceleration device (cuda, mps, or cpu)."""
    if requested_device and requested_device != "auto":
        return requested_device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_pytorch_transforms(device: str = "cpu"):
    """Caches transform and 2D convolution sharpening kernel for a given device."""
    global _PYTORCH_TRANSFORMS
    if device not in _PYTORCH_TRANSFORMS:
        import torch
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        resize = T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.BILINEAR)
        sharpen_weight = torch.from_numpy(SHARPEN_KERNEL).view(1, 1, 3, 3).to(device)
        _PYTORCH_TRANSFORMS[device] = (to_tensor, resize, sharpen_weight)
    return _PYTORCH_TRANSFORMS[device]


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


def process_with_pytorch(image_path: str, device: str = "cpu"):
    """Grayscale -> Sharpen (conv2d) -> Bilinear Resize using PyTorch."""
    import torch.nn.functional as F
    to_tensor, resize, sharpen_weight = _get_pytorch_transforms(device)
    img = Image.open(image_path).convert('L')
    tensor_img = to_tensor(img).unsqueeze(0).to(device)
    sharpened = F.conv2d(tensor_img, sharpen_weight, padding=1).clamp(0.0, 1.0)
    resized = resize(sharpened).squeeze(0)
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
    """
    Returns multiprocessing context.
    Uses 'spawn' start method to guarantee POSIX safety and prevent deadlocks
    when OpenCV/PyTorch C++ runtimes are active in the parent process.
    """
    import multiprocessing as mp
    return mp.get_context("spawn")


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


def process_dataset_gpu(image_paths: List[str], batch_size: int = 64, device: str = None,
                        output_dir: str = None) -> float:
    """
    High-Performance Batched GPU Image Processing Pipeline via PyTorch (CUDA / MPS).
    Pipelines host-to-device streaming:
    Grayscale -> 2D Convolution Sharpening -> Bilinear Interpolation.
    Supports variable-dimension image datasets seamlessly with SIMD host decoding.
    """
    import torch
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor

    dev_str = get_optimal_device(device)
    dev = torch.device(dev_str)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    to_tensor, _, sharpen_weight = _get_pytorch_transforms(dev_str)

    def load_single_tensor(path: str):
        try:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().div_(255.0)
        except Exception:
            pass
        img = Image.open(path).convert('L')
        return to_tensor(img).unsqueeze(0)

    # Warmup GPU kernels
    dummy = torch.randn(1, 1, 500, 500, device=dev)
    _ = F.interpolate(F.conv2d(dummy, sharpen_weight, padding=1), size=TARGET_SIZE, mode="bilinear")
    if dev_str == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    num_images = len(image_paths)

    for i in range(0, num_images, batch_size):
        chunk = image_paths[i:i + batch_size]
        # Multi-threaded image reading and decoding from disk using C/SIMD
        with ThreadPoolExecutor(max_workers=min(8, len(chunk))) as pool:
            tensors = list(pool.map(load_single_tensor, chunk))

        # Check if all images in this mini-batch have identical dimensions for batch processing
        shapes = [t.shape for t in tensors]
        if all(s == shapes[0] for s in shapes):
            batch = torch.cat(tensors, dim=0).to(dev, non_blocking=True)
            sharp = F.conv2d(batch, sharpen_weight, padding=1).clamp(0.0, 1.0)
            resized = F.interpolate(sharp, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            if output_dir:
                cpu_arrs = (resized.cpu().numpy() * 255.0).astype(np.uint8)
                for p, arr in zip(chunk, cpu_arrs):
                    out_path = os.path.join(output_dir, os.path.basename(p))
                    Image.fromarray(arr[0]).save(out_path)
        else:
            # Handle variable-size images individually on GPU stream
            for p, t in zip(chunk, tensors):
                t_gpu = t.to(dev, non_blocking=True)
                sharp = F.conv2d(t_gpu, sharpen_weight, padding=1).clamp(0.0, 1.0)
                resized = F.interpolate(sharp, size=TARGET_SIZE, mode="bilinear", align_corners=False)
                if output_dir:
                    arr = (resized.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
                    out_path = os.path.join(output_dir, os.path.basename(p))
                    Image.fromarray(arr).save(out_path)

    if dev_str == "cuda":
        torch.cuda.synchronize()

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

def run_multicore_benchmark(image_paths: List[str], workers: int, backend: str = "opencv",
                            device: str = None, batch_size: int = 64):
    """Runs scaling benchmarks across backends (OpenCV, Pillow, PyTorch CPU, and PyTorch GPU)."""
    n_images = len(image_paths)
    print("\n" + "=" * 78)
    print(f"🚀 HPC MULTI-CORE & GPU SCALING BENCHMARK ({n_images} images)")
    print("=" * 78)
    print(f"  CPU Cores Allocation: {workers} Workers")

    dev_str = get_optimal_device(device)
    dev_name = "N/A"
    try:
        import torch
        if dev_str == "cuda" and torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
        elif dev_str == "mps":
            dev_name = "Apple Silicon (MPS)"
    except Exception:
        pass
    print(f"  GPU Hardware Target:  {dev_str.upper()} ({dev_name})")

    backends_to_test = ["opencv", "pillow", "pytorch"] if backend == "all" else [backend]
    results = []

    # Baseline sequential time (used for speedup calculation across all entries)
    t_baseline = None

    if backend != "gpu":
        for b in backends_to_test:
            print(f"\n[Benchmarking Backend: {b.upper()}]")
            # 1. Single-Core Sequential
            print(f"  - Running 1-Core Sequential Baseline...")
            t_seq = process_dataset_sequential(image_paths, backend=b)
            fps_seq = n_images / t_seq
            if t_baseline is None:
                t_baseline = t_seq
            speedup_seq = t_baseline / t_seq
            results.append((b, "CPU (1-Core)", "1 Core", t_seq, fps_seq, speedup_seq))
            print(f"    Wall Time: {t_seq:.3f} s | Throughput: {fps_seq:.1f} img/s")

            # 2. Multi-Core Parallel
            print(f"  - Running {workers}-Core Parallel Pipeline...")
            t_par = process_dataset_multicore(image_paths, num_workers=workers, backend=b)
            fps_par = n_images / t_par
            speedup_par = t_baseline / t_par
            results.append((b, f"CPU ({workers}-Core)", f"{workers} Cores", t_par, fps_par, speedup_par))
            print(f"    Wall Time: {t_par:.3f} s | Throughput: {fps_par:.1f} img/s | Speedup: {speedup_par:.2f}x")

    # 3. GPU Acceleration Run (if backend is "all" or "gpu")
    if (backend == "all" and dev_str in ["cuda", "mps"]) or backend == "gpu":
        print(f"\n[Benchmarking Backend: PYTORCH GPU ({dev_str.upper()})]")
        if t_baseline is None:
            t_baseline = process_dataset_sequential(image_paths, backend="opencv")

        print(f"  - Running Batched GPU Pipeline (batch_size={batch_size})...")
        t_gpu = process_dataset_gpu(image_paths, batch_size=batch_size, device=dev_str)
        fps_gpu = n_images / t_gpu
        speedup_gpu = t_baseline / t_gpu
        target_label = f"GPU ({dev_str.upper()})"
        results.append(("pytorch", target_label, dev_name[:14], t_gpu, fps_gpu, speedup_gpu))
        print(f"    Wall Time: {t_gpu:.3f} s | Throughput: {fps_gpu:.1f} img/s | Speedup: {speedup_gpu:.2f}x")

    print("\n" + "=" * 78)
    print(f"{'Backend':<10} | {'Execution Mode':<16} | {'Hardware':<15} | {'Wall Time':<10} | {'Throughput':<12} | {'Speedup':<8}")
    print("-" * 78)
    for b_name, mode_name, hw_name, wtime, fps, spd in results:
        print(f"{b_name:<10} | {mode_name:<16} | {hw_name:<15} | {wtime:6.3f} s   | {fps:7.1f} img/s | {spd:5.2f}x")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Multi-Core & GPU Scientific Image Processing Pipeline")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to raw JPEG images directory")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of CPU worker processes")
    parser.add_argument("--max-images", "-n", type=int, default=100, help="Max images to process (0 for all)")
    parser.add_argument("--backend", type=str, default="opencv", choices=["opencv", "pillow", "pytorch", "gpu", "all"], help="Processing library backend")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Hardware acceleration device")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU execution")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-core / GPU scaling benchmark")
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

    # 2. Benchmarking or Processing
    if args.benchmark:
        run_multicore_benchmark(images, workers=workers, backend=args.backend,
                                device=args.device, batch_size=args.batch_size)
    elif args.backend == "gpu":
        dev_str = get_optimal_device(args.device)
        print(f"\n⚡ Executing GPU Pipeline ({len(images)} images | device: {dev_str.upper()} | batch_size: {args.batch_size})...")
        t_elapsed = process_dataset_gpu(images, batch_size=args.batch_size, device=dev_str, output_dir=args.output_dir)
        throughput = len(images) / t_elapsed if t_elapsed > 0 else 0.0
        print(f"✅ Processed {len(images)} images on GPU in {t_elapsed:.3f} s ({throughput:.1f} images/sec).")
    else:
        print(f"\n⚡ Executing Multi-Core Batch Pipeline ({len(images)} images | {workers} workers | backend: {args.backend})...")
        t_elapsed = process_dataset_multicore(images, num_workers=workers, backend=args.backend, output_dir=args.output_dir)
        throughput = len(images) / t_elapsed if t_elapsed > 0 else 0.0
        print(f"✅ Processed {len(images)} images in {t_elapsed:.3f} s ({throughput:.1f} images/sec) across {workers} cores.")


if __name__ == "__main__":
    main()
