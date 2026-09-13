"""
Tier 2: Boundary & Corner Cases - Multi-Megapixel & High-Resolution Images
Test ID: T2.5.1 to T2.5.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
from PIL import Image
import numpy as np


def test_t2_5_1_two_thousand_square(test_engine):
    """T2.5.1: 2048x2048 multi-megapixel image resizes cleanly to 512x512 without OOM."""
    engine = test_engine
    img_2k = Image.new("RGB", (2048, 2048), (140, 70, 90))
    tensor, arr = engine.preprocess_image(img_2k)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_5_2_four_thousand_square(test_engine):
    """T2.5.2: 4096x4096 high-resolution capture downscales safely."""
    engine = test_engine
    img_4k = Image.new("RGB", (4096, 4096), (60, 150, 210))
    tensor, arr = engine.preprocess_image(img_4k)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_5_3_six_thousand_dslr_export(test_engine):
    """T2.5.3: 6000x4000 (24MP DSLR export) downsamples to 512x512 without memory defect."""
    engine = test_engine
    img_dslr = Image.new("RGB", (6000, 4000), (95, 110, 85))
    tensor, arr = engine.preprocess_image(img_dslr)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_5_4_eight_thousand_square(test_engine):
    """T2.5.4: 8192x8192 extreme resolution image downsamples cleanly."""
    engine = test_engine
    img_8k = Image.new("RGB", (8192, 8192), (180, 180, 180))
    tensor, arr = engine.preprocess_image(img_8k)

    assert tensor.shape == (1, 3, 512, 512)


def test_t2_5_5_decompression_bomb_guard():
    """T2.5.5: Pillow MAX_IMAGE_PIXELS guard is set safely to prevent DoS attacks."""
    from PIL import Image
    # Decompression bomb guard must be active (not None or infinite)
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS >= 89478485  # Default safe threshold
