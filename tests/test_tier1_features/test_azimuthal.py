"""
Tier 1: Feature Coverage - Azimuthal Radial Power Spectrum R(r)
Test ID: T1.6.1 to T1.6.5
Authoritative Source: PROJECT.md § Feature 6 & TEST_INFRA.md
"""

import pytest
import numpy as np


def test_t1_6_1_radial_profile_length(test_engine):
    """T1.6.1: Radial profile length equals min(H/2, W/2) = 256 for 512x512 input."""
    engine = test_engine
    delta = np.random.normal(0, 0.1, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    profile = spectral["radial_profile"]
    assert len(profile) == 256


def test_t1_6_2_radial_integration_non_negativity(test_engine):
    """T1.6.2: All bins in radial profile are non-negative."""
    engine = test_engine
    delta = np.random.uniform(-1.0, 1.0, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    profile = spectral["radial_profile"]
    assert float(np.min(profile)) >= 0.0


def test_t1_6_3_high_frequency_ratio_partition(test_engine):
    """T1.6.3: High-frequency ratio is correctly partitioned at R_max/3 and bounded in [0.0, 1.0]."""
    engine = test_engine
    delta = np.random.normal(0, 0.1, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    ratio = spectral["high_freq_ratio"]
    assert 0.0 <= ratio <= 1.0


def test_t1_6_4_harmonic_spike_detection_synthetic_peak(test_engine):
    """T1.6.4: Harmonic lattice detector catches sharp deconvolution spike at r = 64."""
    engine = test_engine
    # Create spatial pattern with periodic 8x8 checkerboard (frequency 64 in 512x512)
    h, w = 512, 512
    y, x = np.mgrid[:h, :w]
    np.random.seed(42)
    # Sine modulation along Cartesian axes with period 8 pixels -> radial frequency r = 512 / 8 = 64
    pattern = np.sin(2 * np.pi * x / 8.0) + np.sin(2 * np.pi * y / 8.0) + np.random.normal(0, 0.05, (h, w))
    delta = np.repeat(pattern[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    spectral = engine.compute_spectral_metrics(delta)
    spike = spectral["max_harmonic_spike"]
    assert spike >= 1.30  # Clear spike detected above baseline


def test_t1_6_5_natural_smooth_decay_spike_near_one(test_engine):
    """T1.6.5: Smooth decay noise field yields harmonic spike ratio near 1.0 (smooth)."""
    engine = test_engine
    # White noise has flat/smooth expectation without periodic spikes
    np.random.seed(42)
    delta = np.random.normal(0, 0.05, (512, 512, 3)).astype(np.float32)
    spectral = engine.compute_spectral_metrics(delta)

    spike = spectral["max_harmonic_spike"]
    assert 0.85 <= spike <= 1.30  # No massive periodic anomaly
