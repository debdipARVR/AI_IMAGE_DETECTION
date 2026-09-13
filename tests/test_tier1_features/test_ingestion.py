"""
Tier 1: Feature Coverage - Image Ingestion, Resizing & Presets
Test ID: T1.2.1 to T1.2.5
Authoritative Source: PROJECT.md § Feature 2 & TEST_INFRA.md
"""

import os
import pytest
import numpy as np
from PIL import Image
import torch


def test_t1_2_1_pil_image_input(test_engine):
    """T1.2.1: PIL Image input produces normalized tensor [1, 3, 512, 512] in [-1.0, 1.0]."""
    engine = test_engine
    img = Image.new("RGB", (512, 512), (128, 64, 192))
    tensor, arr = engine.preprocess_image(img, target_size=(512, 512))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)
    assert float(arr.min()) >= -1.0
    assert float(arr.max()) <= 1.0


def test_t1_2_2_file_path_input(test_engine, tmp_path):
    """T1.2.2: File path string input opens, converts to RGB, and normalizes tensor."""
    engine = test_engine
    test_file = tmp_path / "sample_ingest.png"
    img = Image.new("RGB", (512, 512), (50, 150, 250))
    img.save(test_file)

    tensor, arr = engine.preprocess_image(str(test_file))
    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)
    assert arr.dtype == np.float32


def test_t1_2_3_numpy_ndarray_input(test_engine):
    """T1.2.3: Numpy ndarray (H, W, C) input converts without channel transposition error."""
    engine = test_engine
    np_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    tensor, arr = engine.preprocess_image(np_img)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)
    assert -1.0 <= arr.min() <= arr.max() <= 1.0


def test_t1_2_4_bilinear_lanczos_resizing(test_engine):
    """T1.2.4: Non-square input (1024, 768) resizes strictly to target [1, 3, 512, 512]."""
    engine = test_engine
    rect_img = Image.new("RGB", (1024, 768), (100, 100, 100))
    tensor, arr = engine.preprocess_image(rect_img, target_size=(512, 512))

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t1_2_5_unsupported_input_type(test_engine):
    """T1.2.5: Unsupported input type raises descriptive ValueError."""
    engine = test_engine
    with pytest.raises(ValueError, match="Unsupported image input type"):
        engine.preprocess_image([1, 2, 3])

    with pytest.raises(ValueError, match="Unsupported image input type"):
        engine.preprocess_image(42)
