import os
import shutil
import numpy as np
import pytest
from src.process_images import (
    process_with_opencv,
    process_with_pillow,
    process_with_pytorch,
    process_dataset_multicore,
    process_dataset_sequential,
    get_optimal_workers,
    TARGET_SIZE
)

DUMMY_DIR = "tests/test_data"
DUMMY_IMG_PATH = "tests/test_data/dummy_0.jpg"

def setup_module(module):
    from PIL import Image
    os.makedirs(DUMMY_DIR, exist_ok=True)
    for i in range(4):
        p = os.path.join(DUMMY_DIR, f"dummy_{i}.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        img.save(p)

def teardown_module(module):
    if os.path.exists(DUMMY_DIR):
        shutil.rmtree(DUMMY_DIR)

def test_opencv_processing():
    res = process_with_opencv(DUMMY_IMG_PATH)
    assert res.shape == TARGET_SIZE

def test_pillow_processing():
    res = process_with_pillow(DUMMY_IMG_PATH)
    assert res.size == TARGET_SIZE

def test_pytorch_processing():
    res = process_with_pytorch(DUMMY_IMG_PATH)
    assert res.shape[1:] == TARGET_SIZE

def test_get_optimal_workers(monkeypatch):
    # Test manual override
    assert get_optimal_workers(8) == 8
    # Test SLURM environment variable detection
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert get_optimal_workers() == 4
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    # Default fallback >= 1
    assert get_optimal_workers() >= 1

@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_multicore_processing():
    image_paths = [os.path.join(DUMMY_DIR, f"dummy_{i}.jpg") for i in range(4)]
    out_dir = "tests/test_output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        duration = process_dataset_multicore(image_paths, num_workers=2, backend="opencv", output_dir=out_dir)
        assert duration > 0.0
        assert len(os.listdir(out_dir)) == 4
    finally:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
