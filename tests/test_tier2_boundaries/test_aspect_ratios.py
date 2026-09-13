"""
Tier 2: Boundary & Corner Cases - Extreme Aspect Ratios
Test ID: T2.2.1 to T2.2.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_2_1_ultra_tall_banner(test_engine):
    """T2.2.1: 10x1000 ultra-tall banner resamples cleanly to 512x512 without distortion crash."""
    engine = test_engine
    img_tall = Image.new("RGB", (10, 1000), (120, 40, 80))
    tensor, arr = engine.preprocess_image(img_tall)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)


def test_t2_2_2_ultra_wide_strip(test_engine):
    """T2.2.2: 1920x100 ultra-wide strip resamples cleanly and computes full 2D-FFT."""
    engine = test_engine
    img_wide = Image.new("RGB", (1920, 100), (30, 150, 75))
    res = engine.analyze(img_wide)

    assert res["spectral"]["log_magnitude"].shape == (512, 512)
    assert len(res["spectral"]["radial_profile"]) == 256


def test_t2_2_3_cinematic_crop_21_9(test_engine):
    """T2.2.3: 21:9 cinematic crop (840x360) resizes gracefully without array mismatch."""
    engine = test_engine
    img_cinema = Image.new("RGB", (840, 360), (210, 180, 140))
    res = engine.analyze(img_cinema)

    assert res["spatial"]["psnr"] is not None
    assert res["arr_recon"].shape == (512, 512, 3)


def test_t2_2_4_vertical_smartphone_photo_9_16(test_engine):
    """T2.2.4: 9:16 vertical smartphone photo (360x640) normalizes and evaluates correctly."""
    engine = test_engine
    img_phone = Image.new("RGB", (360, 640), (60, 120, 180))
    res = engine.analyze(img_phone)

    assert 0.0 <= res["ai_probability"] <= 1.0


def test_t2_2_5_fractional_aspect_ratio(test_engine):
    """T2.2.5: Non-integer aspect ratio (731x419) resizes without rounding defect."""
    engine = test_engine
    img_frac = Image.new("RGB", (731, 419), (145, 95, 205))
    tensor, arr = engine.preprocess_image(img_frac)

    assert tensor.shape == (1, 3, 512, 512)
    assert arr.shape == (512, 512, 3)
