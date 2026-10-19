import os
import numpy as np
from src.process_images import process_with_opencv, process_with_pillow, process_with_pytorch, TARGET_SIZE

DUMMY_IMG_PATH = "tests/dummy.jpg"

def setup_module(module):
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    img.save(DUMMY_IMG_PATH)

def teardown_module(module):
    if os.path.exists(DUMMY_IMG_PATH):
        os.remove(DUMMY_IMG_PATH)

def test_opencv_processing():
    res = process_with_opencv(DUMMY_IMG_PATH)
    assert res.shape == TARGET_SIZE

def test_pillow_processing():
    res = process_with_pillow(DUMMY_IMG_PATH)
    assert res.size == TARGET_SIZE

def test_pytorch_processing():
    res = process_with_pytorch(DUMMY_IMG_PATH)
    assert res.shape[1:] == TARGET_SIZE
