"""
Tier 2: Boundary & Corner Cases - Grayscale & Non-RGB Color Modes
Test ID: T2.3.1 to T2.3.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_3_1_grayscale_mode_l(test_engine):
    """T2.3.1: Grayscale (PIL mode 'L') converts 1-channel to 3-channel RGB without broadcast error."""
    engine = test_engine
    img_gray = Image.new("L", (512, 512), 128)
    tensor, arr = engine.preprocess_image(img_gray)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_3_2_grayscale_with_alpha_la(test_engine):
    """T2.3.2: Grayscale + Alpha ('LA') converts cleanly to RGB without dropping channel dimension."""
    engine = test_engine
    img_la = Image.new("LA", (512, 512), (100, 200))
    tensor, arr = engine.preprocess_image(img_la)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_3_3_one_bit_bilevel_mode_1(test_engine):
    """T2.3.3: 1-bit bilevel ('1') converts boolean 0/1 to RGB values cleanly."""
    engine = test_engine
    img_bilevel = Image.new("1", (512, 512), 1)
    tensor, arr = engine.preprocess_image(img_bilevel)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_3_4_palette_based_color_mode_p(test_engine):
    """T2.3.4: Palette-based color ('P') resolves palette indices to 3-channel RGB."""
    engine = test_engine
    img_p = Image.new("P", (512, 512))
    img_p.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] * 85)
    tensor, arr = engine.preprocess_image(img_p)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_3_5_float_mode_f(test_engine):
    """T2.3.5: 32-bit Float ('F') converts cleanly and executes analysis."""
    engine = test_engine
    arr_f = np.random.uniform(0.0, 255.0, (512, 512)).astype(np.float32)
    img_f = Image.fromarray(arr_f, mode="F")
    tensor, arr = engine.preprocess_image(img_f)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)
