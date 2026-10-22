#!/usr/bin/env python3
"""
download_data.py: Robust Dataset Fetcher & Synthetic Generator for 17flowers
"""

import os
import sys
import tarfile
import urllib.request
import argparse
import numpy as np
from PIL import Image

DATASET_URLS = [
    "https://thor.robots.ox.ac.uk/flowers/17/17flowers.tgz",
    "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
]

def generate_synthetic_dataset(output_dir: str, num_images: int = 100):
    """Generates synthetic multi-color JPEG test images for offline or CI runs."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"🌸 Generating {num_images} synthetic test images in '{output_dir}'...")

    np.random.seed(42)
    for i in range(1, num_images + 1):
        filename = f"image_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            # Create a synthetic flower-like pattern (radial color gradient)
            h, w = 500, 500
            y, x = np.ogrid[:h, :w]
            cx, cy = w // 2, h // 2
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            
            # Oscillating color rings
            red = (128 + 127 * np.sin(r / 20.0 + i)).astype(np.uint8)
            green = (128 + 127 * np.cos(r / 30.0 + i * 2)).astype(np.uint8)
            blue = (128 + 127 * np.sin(r / 15.0 - i)).astype(np.uint8)
            img_arr = np.stack([red, green, blue], axis=-1)
            
            img = Image.fromarray(img_arr)
            img.save(filepath, quality=90)

    print(f"✅ Generated {num_images} synthetic images successfully.")

def download_and_extract(data_dir: str = "data"):
    """Downloads and extracts the 17flowers dataset with URL redirection fallbacks."""
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "17flowers.tgz")
    jpg_dir = os.path.join(data_dir, "jpg")

    if os.path.exists(jpg_dir) and len([f for f in os.listdir(jpg_dir) if f.endswith(".jpg")]) > 0:
        print(f"✅ Dataset already extracted in '{jpg_dir}'. Skipping download.")
        return

    if not os.path.exists(archive_path):
        success = False
        for url in DATASET_URLS:
            try:
                print(f"📥 Downloading dataset from {url} ...")
                urllib.request.urlretrieve(url, archive_path)
                print(f"✅ Downloaded: {archive_path}")
                success = True
                break
            except Exception as e:
                print(f"⚠️ Failed to download from {url}: {e}")

        if not success:
            print("⚠️ Network download unavailable. Falling back to synthetic test dataset...")
            generate_synthetic_dataset(jpg_dir, num_images=100)
            return

    print(f"📦 Extracting {archive_path} to {data_dir} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"✅ Extracted dataset to '{jpg_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Dataset Fetcher for 17flowers")
    parser.add_argument("--data-dir", type=str, default="data", help="Target data directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic images instead of downloading")
    parser.add_argument("--num-synthetic", type=int, default=100, help="Number of synthetic images to generate")
    args = parser.parse_args()

    jpg_dir = os.path.join(args.data_dir, "jpg")
    if args.synthetic:
        generate_synthetic_dataset(jpg_dir, args.num_synthetic)
    else:
        download_and_extract(args.data_dir)

if __name__ == "__main__":
    main()
