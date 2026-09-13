"""
Tier 1: Feature Coverage - Spatial Residual Extraction (PSNR/MSE/MAE/NCC)
Test ID: T1.4.1 to T1.4.5
Authoritative Source: PROJECT.md § Feature 4 & TEST_INFRA.md
"""

import pytest
import numpy as np


def test_t1_4_1_delta_residual_range(test_engine):
    """T1.4.1: Delta residual delta = orig - recon is bounded within [-2.0, 2.0]."""
    engine = test_engine
    arr_orig = np.ones((512, 512, 3), dtype=np.float32) * 1.0
    arr_recon = np.ones((512, 512, 3), dtype=np.float32) * (-1.0)
    spatial = engine.compute_spatial_metrics(arr_orig, arr_recon)

    assert float(spatial["delta"].min()) >= -2.0
    assert float(spatial["delta"].max()) <= 2.0
    assert spatial["delta"].shape == (512, 512, 3)


def test_t1_4_2_exact_psnr_formula(test_engine):
    """T1.4.2: Exact PSNR matches 10 * log10(4.0 / (MSE + 1e-12))."""
    engine = test_engine
    arr_orig = np.zeros((512, 512, 3), dtype=np.float32)
    # Known MSE: constant error of 0.02 -> MSE = 0.0004
    arr_recon = arr_orig + 0.02
    spatial = engine.compute_spatial_metrics(arr_orig, arr_recon)

    expected_mse = 0.0004
    expected_psnr = 10.0 * np.log10(4.0 / (expected_mse + 1e-12))
    assert pytest.approx(spatial["mse"], rel=1e-3) == expected_mse
    assert pytest.approx(spatial["psnr"], rel=1e-3) == expected_psnr


def test_t1_4_3_perfect_reconstruction_zero_mse(test_engine):
    """T1.4.3: Perfect reconstruction yields zero MSE and gracefully capped high PSNR."""
    engine = test_engine
    arr = np.random.uniform(-1.0, 1.0, (512, 512, 3)).astype(np.float32)
    spatial = engine.compute_spatial_metrics(arr, arr)

    assert spatial["mse"] == 0.0
    assert spatial["mae"] == 0.0
    assert spatial["psnr"] >= 90.0  # Capped safely without div-by-zero
    assert pytest.approx(spatial["ncc"], abs=1e-4) == 1.0


def test_t1_4_4_mae_metric_validity(test_engine):
    """T1.4.4: MAE metric validity: MAE > 0 for non-identical arrays and MAE <= sqrt(MSE)."""
    engine = test_engine
    arr_orig = np.random.uniform(-1.0, 1.0, (512, 512, 3)).astype(np.float32)
    arr_recon = arr_orig + np.random.normal(0, 0.05, (512, 512, 3)).astype(np.float32)
    spatial = engine.compute_spatial_metrics(arr_orig, arr_recon)

    assert spatial["mae"] > 0.0
    # By Jensen's inequality / Cauchy-Schwarz: E[|X|] <= sqrt(E[X^2])
    assert spatial["mae"] <= np.sqrt(spatial["mse"]) + 1e-6


def test_t1_4_5_normalized_cross_correlation(test_engine):
    """T1.4.5: Normalized cross correlation bounds in [-1.0, 1.0] and handles zero norm."""
    engine = test_engine
    arr_1 = np.ones((512, 512, 3), dtype=np.float32)
    arr_2 = arr_1 * 0.5
    spatial = engine.compute_spatial_metrics(arr_1, arr_2)
    assert pytest.approx(spatial["ncc"], abs=1e-4) == 1.0

    # Zero array handling
    arr_zero = np.zeros((512, 512, 3), dtype=np.float32)
    spatial_zero = engine.compute_spatial_metrics(arr_zero, arr_1)
    assert spatial_zero["ncc"] == 0.0
