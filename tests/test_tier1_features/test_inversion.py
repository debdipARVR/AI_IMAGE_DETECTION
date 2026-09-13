"""
Tier 1: Feature Coverage - Deterministic VAE Latent Inversion
Test ID: T1.3.1 to T1.3.5
Authoritative Source: PROJECT.md § Feature 3 & TEST_INFRA.md
"""

import pytest
import numpy as np
import torch
from PIL import Image


def test_t1_3_1_zero_noise_mode(test_engine, synthetic_clean_rgb):
    """T1.3.1: Zero-noise mode reconstruction executes deterministically."""
    engine = test_engine
    tensor_x, _ = engine.preprocess_image(synthetic_clean_rgb)
    recon = engine.reconstruct(tensor_x)
    assert recon is not None
    assert isinstance(recon, np.ndarray)


def test_t1_3_2_reconstruction_output_clamping(test_engine, synthetic_clean_rgb):
    """T1.3.2: Reconstruction output values are bounded strictly within [-1.0, 1.0]."""
    engine = test_engine
    tensor_x, _ = engine.preprocess_image(synthetic_clean_rgb)
    recon = engine.reconstruct(tensor_x)
    assert float(recon.min()) >= -1.0
    assert float(recon.max()) <= 1.0


def test_t1_3_3_inversion_deterministic_repeatability(test_engine, synthetic_clean_rgb):
    """T1.3.3: Reconstructing identical input twice yields identical arrays (MSE == 0.0)."""
    engine = test_engine
    tensor_x, _ = engine.preprocess_image(synthetic_clean_rgb)
    recon_1 = engine.reconstruct(tensor_x)
    recon_2 = engine.reconstruct(tensor_x)
    mse_diff = np.mean((recon_1 - recon_2) ** 2)
    assert mse_diff == 0.0


def test_t1_3_4_reconstruction_shape_integrity(test_engine, synthetic_clean_rgb):
    """T1.3.4: Reconstruction output shape matches expected (512, 512, 3)."""
    engine = test_engine
    tensor_x, _ = engine.preprocess_image(synthetic_clean_rgb)
    recon = engine.reconstruct(tensor_x)
    assert recon.shape == (512, 512, 3)


def test_t1_3_5_latent_representation_properties(test_engine, synthetic_clean_rgb):
    """T1.3.5: VAE encoder bottleneck downsamples 8x to 4 latent channels [1, 4, 64, 64]."""
    engine = test_engine
    tensor_x, _ = engine.preprocess_image(synthetic_clean_rgb)
    posterior = engine.vae.encode(tensor_x).latent_dist
    assert posterior.mean.shape == (1, 4, 64, 64)
