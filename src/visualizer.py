"""
Forensic Spectral and Spatial Visualization Module
Author: Debdip Bandyopadhyay

Generates publication-ready figures:
1. Spatial Residual Magnification (Original vs Reconstruction vs 10x Residual Heatmap)
2. 2D-FFT Power Spectrum & Periodic Lattice Harmonic Spikes
3. Azimuthal Radial Decay Curves (Real 1/f decay vs AI Transposed Convolutions)
4. AUROC / ROC Performance Curve
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_forensic_comparison(real_res, ai_res, output_path: str):
    """
    Creates a 2x4 comprehensive diagnostic panel comparing Real vs AI.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # ------------------ Row 1: Authentic Human / Camera ------------------
    # Original
    axes[0, 0].imshow(((real_res["arr_orig"] + 1.0) * 0.5).clip(0, 1))
    axes[0, 0].set_title("Authentic Photo (Original)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # VAE Reconstruction
    axes[0, 1].imshow(((real_res["arr_recon"] + 1.0) * 0.5).clip(0, 1))
    axes[0, 1].set_title(f"VAE Reconstruction\nPSNR: {real_res['spatial']['psnr']:.2f} dB", fontsize=12)
    axes[0, 1].axis("off")

    # Residual Heatmap (10x magnified)
    delta_real = np.mean(np.abs(real_res["delta"]), axis=2) * 5.0
    im1 = axes[0, 2].imshow(delta_real, cmap="inferno", vmin=0, vmax=1.0)
    axes[0, 2].set_title(f"Residual |Δx| (High Sensor Noise)\nMSE: {real_res['spatial']['mse']:.5f}", fontsize=12)
    axes[0, 2].axis("off")
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 2D-FFT Power Spectrum
    axes[0, 3].imshow(real_res["log_magnitude"], cmap="viridis")
    axes[0, 3].set_title("2D-FFT Residual Spectrum\n(Smooth Power-Law Decay)", fontsize=12)
    axes[0, 3].axis("off")

    # ------------------ Row 2: AI Diffusion (Native SD VAE) ------------------
    # Original
    axes[1, 0].imshow(((ai_res["arr_orig"] + 1.0) * 0.5).clip(0, 1))
    axes[1, 0].set_title("AI Generated (Native Diffusion)", fontsize=12, fontweight="bold", color="darkred")
    axes[1, 0].axis("off")

    # VAE Reconstruction
    axes[1, 1].imshow(((ai_res["arr_recon"] + 1.0) * 0.5).clip(0, 1))
    axes[1, 1].set_title(f"VAE Reconstruction (Resonant)\nPSNR: {ai_res['spatial']['psnr']:.2f} dB", fontsize=12, color="darkred")
    axes[1, 1].axis("off")

    # Residual Heatmap (10x magnified)
    delta_ai = np.mean(np.abs(ai_res["delta"]), axis=2) * 5.0
    im2 = axes[1, 2].imshow(delta_ai, cmap="inferno", vmin=0, vmax=1.0)
    axes[1, 2].set_title(f"Residual |Δx| (Near-Zero Loss)\nMSE: {ai_res['spatial']['mse']:.5f}", fontsize=12, color="darkred")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 2D-FFT Power Spectrum
    axes[1, 3].imshow(ai_res["log_magnitude"], cmap="viridis")
    axes[1, 3].set_title(f"2D-FFT Residual Spectrum\n(Periodic Lattice Spike: {ai_res['spectral']['max_harmonic_spike']:.2f}x)", fontsize=12, color="darkred")
    axes[1, 3].axis("off")

    plt.suptitle("Zero-Shot VAE Reconstruction Resonance & 2D-FFT Spectral Fingerprint Diagnostic", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved diagnostic comparison to {output_path}")


def plot_radial_power_spectrum(real_res, ai_res, output_path: str):
    """
    Plots the Azimuthal Radial Power Spectrum R(r) comparing Real vs AI decay.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    r_real = real_res["radial_profile"]
    r_ai = ai_res["radial_profile"]
    freqs = np.arange(len(r_real))

    ax.plot(freqs, np.log10(r_real + 1e-6), label="Real Camera Photo (Smooth 1/f Decay)", color="#2a4a6b", linewidth=2.5)
    ax.plot(freqs, np.log10(r_ai + 1e-6), label=f"AI Diffusion (Transposed Conv Periodic Spikes)", color="#8b2000", linewidth=2.5, linestyle="--")

    # Mark 8x8 deconvolution stride harmonics
    stride_f = len(freqs) // 8
    for mult in [1, 2, 3]:
        fx = mult * stride_f
        if fx < len(freqs):
            ax.axvline(x=fx, color="gray", linestyle=":", alpha=0.7, label=f"8x8 Lattice Harmonic ({mult}x)" if mult == 1 else "")

    ax.set_title("Azimuthally Averaged Radial Power Spectrum of VAE Residual $\Delta x$", fontsize=14, fontweight="bold")
    ax.set_xlabel("Spatial Frequency Radius $r$ (cycles)", fontsize=12)
    ax.set_ylabel("Log Radial Power $\log_{10} R(r)$", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved radial spectrum plot to {output_path}")
