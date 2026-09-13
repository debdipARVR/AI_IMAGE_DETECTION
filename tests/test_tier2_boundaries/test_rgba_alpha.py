"""
Tier 2: Boundary & Corner Cases - RGBA with Alpha Channel & Color Spaces
Test ID: T2.4.1 to T2.4.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_4_1_full_transparent_png(test_engine):
    """T2.4.1: Full transparent PNG (Alpha=0) converts to RGB without crashing."""
    engine = test_engine
    img_trans = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
    tensor, arr = engine.preprocess_image(img_trans)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_4_2_semi_transparent_overlay(test_engine):
    """T2.4.2: Semi-transparent overlay (Alpha=128) blends and evaluates cleanly."""
    engine = test_engine
    img_semi = Image.new("RGBA", (512, 512), (200, 100, 50, 128))
    tensor, arr = engine.preprocess_image(img_semi)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_4_3_transparent_border_letterbox(test_engine):
    """T2.4.3: Image with transparent border / letterbox strips alpha cleanly."""
    engine = test_engine
    arr = np.zeros((512, 512, 4), dtype=np.uint8)
    arr[:, :] = [0, 0, 0, 0]  # Transparent background
    arr[100:412, 100:412] = [180, 140, 90, 255]  # Central opaque patch
    img = Image.fromarray(arr, mode="RGBA")

    tensor, pre_arr = engine.preprocess_image(img)
    assert tensor.shape == (1, 3, 512, 512)


def test_t2_4_4_cmyk_print_format(test_engine):
    """T2.4.4: CMYK print format converts to RGB color profile without channel error."""
    engine = test_engine
    img_cmyk = Image.new("CMYK", (512, 512), (50, 100, 150, 20))
    tensor, arr = engine.preprocess_image(img_cmyk)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_4_5_premultiplied_rgba(test_engine):
    """T2.4.5: Premultiplied RGBA converts to RGB without overflow or nan values."""
    engine = test_engine
    # Alpha = 64 (25%), RGB pre-scaled
    arr_pre = np.full((512, 512, 4), [32, 64, 16, 64], dtype=np.uint8)
    img_pre = Image.fromarray(arr_pre, mode="RGBA")
    res = engine.analyze(img_pre)

    assert not np.isnan(res["spatial"]["mse"])
    assert res["spatial"]["psnr"] > 0.0
