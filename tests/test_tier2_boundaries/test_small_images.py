"""
Tier 2: Boundary & Corner Cases - Miniature Images (1x1 to 128x128)
Test ID: T2.1.1 to T2.1.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_1_1_one_pixel_image(test_engine):
    """T2.1.1: 1x1 pixel image resizes smoothly to 512x512 without crash or div-by-zero."""
    engine = test_engine
    img_1x1 = Image.new("RGB", (1, 1), (255, 0, 128))
    res = engine.analyze(img_1x1)

    assert res["spatial"]["psnr"] is not None
    assert not np.isnan(res["spatial"]["mse"])
    assert res["spectral"]["radial_profile"].shape == (256,)


def test_t2_1_2_eight_by_eight_thumbnail(test_engine):
    """T2.1.2: 8x8 thumbnail resamples to 512x512 and executes full analysis pipeline."""
    engine = test_engine
    img_8x8 = Image.new("RGB", (8, 8), (45, 90, 180))
    res = engine.analyze(img_8x8)

    assert res["verdict"] in ["AI-Generated (Congruent)", "Authentic Photographic (Divergent)"]
    assert 0.0 <= res["ai_probability"] <= 1.0


def test_t2_1_3_sixteen_by_sixteen_icon(test_engine):
    """T2.1.3: 16x16 icon produces valid spatial and spectral metrics without indexing error."""
    engine = test_engine
    img_16x16 = Image.new("RGB", (16, 16), (200, 200, 50))
    tensor, arr = engine.preprocess_image(img_16x16)

    assert tensor.shape == (1, 3, 512, 512)
    spatial = engine.compute_spatial_metrics(arr, arr)
    assert spatial["mse"] == 0.0


def test_t2_1_4_sixty_four_avatar(test_engine):
    """T2.1.4: 64x64 avatar image normalizes and reconstructs cleanly."""
    engine = test_engine
    arr_64 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img_64 = Image.fromarray(arr_64)
    res = engine.analyze(img_64)

    assert res["arr_recon"].shape == (512, 512, 3)
    assert len(res["spectral"]["radial_profile"]) == 256


def test_t2_1_5_one_twenty_eight_patch(test_engine):
    """T2.1.5: 128x128 small patch executes 4-pass diagnostics without warning."""
    engine = test_engine
    img_128 = Image.new("RGB", (128, 128), (80, 160, 240))
    res = engine.analyze(img_128)

    assert res["spatial"]["mae"] >= 0.0
    assert res["spectral"]["total_spectral_energy"] >= 0.0
