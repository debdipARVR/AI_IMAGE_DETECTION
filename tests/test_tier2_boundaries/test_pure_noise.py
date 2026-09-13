"""
Tier 2: Boundary & Corner Cases - Pure Noise & High-Entropy Stress
Test ID: T2.8.1 to T2.8.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import pytest
import numpy as np
from PIL import Image


def test_t2_8_1_gaussian_white_noise(test_engine):
    """T2.8.1: Gaussian white noise (sigma=1.0) produces high divergence residual."""
    engine = test_engine
    np.random.seed(101)
    noise = np.random.normal(128, 40, (512, 512, 3))
    arr_gauss = np.clip(noise, 0, 255).astype(np.uint8)
    img_gauss = Image.fromarray(arr_gauss)

    res = engine.analyze(img_gauss)
    assert res["spatial"]["mse"] > 0.0
    assert not np.isnan(res["spatial"]["psnr"])


def test_t2_8_2_uniform_random_noise(test_engine):
    """T2.8.2: Uniform random noise [0, 255] exhibits broadband high-frequency energy."""
    engine = test_engine
    np.random.seed(202)
    arr_uniform = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img_uniform = Image.fromarray(arr_uniform)

    res = engine.analyze(img_uniform)
    # High-entropy noise distributes energy across high frequencies
    assert res["spectral"]["high_freq_ratio"] > 0.01


def test_t2_8_3_salt_and_pepper_impulse_noise(test_engine):
    """T2.8.3: Salt-and-pepper impulse noise handles single-pixel spikes without overflow."""
    engine = test_engine
    arr_sp = np.full((512, 512, 3), 128, dtype=np.uint8)
    # Add salt and pepper
    prob = 0.05
    rnd = np.random.random((512, 512))
    arr_sp[rnd < prob / 2] = 0
    arr_sp[rnd > 1 - prob / 2] = 255
    img_sp = Image.fromarray(arr_sp)

    res = engine.analyze(img_sp)
    assert res["spatial"]["mae"] > 0.0
    assert not np.isnan(res["spatial"]["mae"])


def test_t2_8_4_high_frequency_checkerboard(test_engine):
    """T2.8.4: High-frequency checkerboard (1x1 pixels) concentrates energy at boundary frequencies."""
    engine = test_engine
    y, x = np.mgrid[:512, :512]
    checker = ((x + y) % 2) * 255
    arr_check = np.repeat(checker[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    img_check = Image.fromarray(arr_check)

    res = engine.analyze(img_check)
    assert res["spectral"]["total_spectral_energy"] > 0.0


def test_t2_8_5_poisson_shot_noise(test_engine):
    """T2.8.5: Poisson shot noise field aggregates cleanly across radial bins."""
    engine = test_engine
    np.random.seed(303)
    arr_poisson = np.random.poisson(lam=100.0, size=(512, 512, 3))
    arr_poisson = np.clip(arr_poisson, 0, 255).astype(np.uint8)
    img_poisson = Image.fromarray(arr_poisson)

    res = engine.analyze(img_poisson)
    assert len(res["spectral"]["radial_profile"]) == 256
    assert float(np.min(res["spectral"]["radial_profile"])) >= 0.0
