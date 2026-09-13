"""
Tier 1: Feature Coverage - 2D-FFT Frequency Analysis & Periodic Lattice
Test ID: T1.5.1 to T1.5.5
Authoritative Source: PROJECT.md § Feature 5 & TEST_INFRA.md
"""

import pytest
import numpy as np


def test_t1_5_1_fft_transform_shape(test_engine):
    """T1.5.1: 2D-FFT transform produces log magnitude of shape (512, 512)."""
    engine = test_engine
    delta = np.random.normal(0, 0.1, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    assert spectral["log_magnitude"].shape == (512, 512)
    assert spectral["radial_profile"] is not None


def test_t1_5_2_dc_component_centering(test_engine):
    """T1.5.2: DC component is centered at (256, 256) after fftshift."""
    engine = test_engine
    # Constant non-zero delta will concentrate all energy at DC (0 frequency)
    delta_const = np.ones((512, 512, 3), dtype=np.float32) * 0.5
    spectral = engine.compute_spectral_metrics(delta_const)

    log_mag = spectral["log_magnitude"]
    cy, cx = 256, 256
    # Maximum value in spectrum must be exactly at the center (256, 256)
    max_pos = np.unravel_index(np.argmax(log_mag), log_mag.shape)
    assert max_pos == (cy, cx)


def test_t1_5_3_power_spectrum_non_negativity(test_engine):
    """T1.5.3: Spectral power spectrum |X(u, v)|^2 and log magnitude are strictly non-negative."""
    engine = test_engine
    delta = np.random.uniform(-1.0, 1.0, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    assert float(np.min(spectral["log_magnitude"])) >= 0.0
    assert float(np.min(spectral["radial_profile"])) >= 0.0


def test_t1_5_4_log_magnitude_scaling(test_engine):
    """T1.5.4: Log magnitude uses log(1.0 + |X|), mapping zero input to exactly 0.0."""
    engine = test_engine
    delta_zero = np.zeros((512, 512, 3), dtype=np.float32)
    spectral = engine.compute_spectral_metrics(delta_zero)

    assert float(np.max(spectral["log_magnitude"])) == 0.0
    assert float(np.min(spectral["log_magnitude"])) == 0.0


def test_t1_5_5_total_spectral_energy_positive(test_engine):
    """T1.5.5: Total spectral energy is strictly positive for non-zero inputs."""
    engine = test_engine
    delta = np.random.normal(0, 0.05, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    assert spectral["total_spectral_energy"] > 0.0
    assert 0.0 <= spectral["high_freq_ratio"] <= 1.0
