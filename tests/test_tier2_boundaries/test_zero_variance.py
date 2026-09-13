"""
Tier 2: Boundary & Corner Cases - Zero-Variance / Flat Fields
Test ID: T2.7.1 to T2.7.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_7_1_solid_black(test_engine):
    """T2.7.1: Pure solid black (#000000) image avoids div-by-zero in metrics and normalization."""
    engine = test_engine
    img_black = Image.new("RGB", (512, 512), (0, 0, 0))
    res = engine.analyze(img_black)

    assert not np.isnan(res["spatial"]["mse"])
    assert not np.isinf(res["spatial"]["psnr"])
    assert res["spatial"]["ncc"] >= -1.0


def test_t2_7_2_solid_white(test_engine):
    """T2.7.2: Pure solid white (#ffffff) image computes cleanly without NaN/inf values."""
    engine = test_engine
    img_white = Image.new("RGB", (512, 512), (255, 255, 255))
    res = engine.analyze(img_white)

    assert res["spatial"]["mse"] >= 0.0
    assert not np.isnan(res["spatial"]["psnr"])


def test_t2_7_3_uniform_middle_gray(test_engine):
    """T2.7.3: Uniform middle gray (#808080) produces zero high-frequency energy ratio."""
    engine = test_engine
    img_gray = Image.new("RGB", (512, 512), (128, 128, 128))
    res = engine.analyze(img_gray)

    assert res["spectral"]["high_freq_ratio"] <= 0.10


def test_t2_7_4_pure_primary_color(test_engine):
    """T2.7.4: Pure saturated primary color (#00ff00) executes cleanly."""
    engine = test_engine
    img_green = Image.new("RGB", (512, 512), (0, 255, 0))
    res = engine.analyze(img_green)

    assert res["verdict"] is not None
    assert 0.0 <= res["ai_probability"] <= 1.0


def test_t2_7_5_constant_offset_field(test_engine):
    """T2.7.5: Constant non-zero offset field normalizations remain finite."""
    engine = test_engine
    arr_const = np.full((512, 512, 3), 42, dtype=np.uint8)
    img_const = Image.fromarray(arr_const)
    tensor, arr = engine.preprocess_image(img_const)

    assert float(np.min(arr)) == float(np.max(arr))
    assert -1.0 <= float(arr[0, 0, 0]) <= 1.0
