"""
Forensic Spectral and Spatial Visualization Module
Author: Debdip Bandyopadhyay

Generates publication-ready figures & interactive UI assets:
1. Spatial Residual Magnification (Original vs Reconstruction vs Residual Heatmap)
2. Interactive Colormapped Residual Heatmap Generator
3. 2D-FFT Power Spectrum with 8x8 Periodic Lattice Harmonic Annotations
4. Parchment-Themed Plotly Azimuthal Radial Decay Curves
5. Comprehensive Diagnostic Comparison Panels
"""

import os
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go


def generate_residual_heatmap(
    delta: np.ndarray,
    colormap: str = "inferno",
    gain: float = 5.0
) -> Image.Image:
    """
    Generates a color-mapped spatial residual heatmap |?x| with amplification gain.

    Parameters
    ----------
    delta : np.ndarray
        Spatial residual array ?x = x - x_hat, shape (H, W, 3) or (H, W).
    colormap : str
        Matplotlib colormap identifier ('inferno', 'viridis', 'magma', 'bone', 'turbo').
    gain : float
        Linear amplification factor (1.0x to 20.0x).

    Returns
    -------
    PIL.Image.Image
        RGB color-mapped residual heatmap image.
    """
    if delta.ndim == 3:
        # Mean absolute error across RGB channels
        spatial_error = np.mean(np.abs(delta), axis=2)
    else:
        spatial_error = np.abs(delta)

    # Apply amplification gain and clip to [0, 1]
    amplified = np.clip(spatial_error * gain, 0.0, 1.0)

    # Retrieve colormap safely with fallback to inferno
    try:
        cmap = matplotlib.colormaps.get_cmap(colormap)
    except (ValueError, KeyError):
        cmap = matplotlib.colormaps.get_cmap("inferno")

    # Colormap produces RGBA float array in [0, 1]
    rgba = cmap(amplified)
    rgb_uint8 = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8)


def annotate_fft_spectrum(
    log_magnitude: np.ndarray,
    stride: int = 64,
    colormap: str = "viridis"
) -> Image.Image:
    """
    Generates a normalized 2D-FFT residual power spectrum with overlaid
    8x8 transposed convolution harmonic lattice crosshairs.

    Parameters
    ----------
    log_magnitude : np.ndarray
        Centered 2D-FFT log magnitude array, shape (H, W).
    stride : int
        Deconvolution stride in frequency domain (default: 64 for 512x512 with 8x8 stride).
    colormap : str
        Matplotlib colormap identifier.

    Returns
    -------
    PIL.Image.Image
        Annotated RGB power spectrum image with lattice markers.
    """
    h, w = log_magnitude.shape
    v_min = float(np.min(log_magnitude))
    v_max = float(np.max(log_magnitude))
    
    if v_max > v_min:
        norm_mag = (log_magnitude - v_min) / (v_max - v_min)
    else:
        norm_mag = np.zeros_like(log_magnitude)

    try:
        cmap = matplotlib.colormaps.get_cmap(colormap)
    except (ValueError, KeyError):
        cmap = matplotlib.colormaps.get_cmap("viridis")

    rgba = cmap(norm_mag)
    rgb_uint8 = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    base_img = Image.fromarray(rgb_uint8)

    # Draw 8x8 lattice crosshairs on spectrum
    draw = ImageDraw.Draw(base_img)
    cx, cy = w // 2, h // 2

    # Draw harmonic frequency markers at +/- 64, +/- 128, +/- 192 cycles
    for mult in [1, 2, 3]:
        offset = mult * stride
        # Horizontal lines (u = +/- offset)
        for sign in [-1, 1]:
            y = cy + sign * offset
            if 0 <= y < h:
                draw.line([(0, y), (w, y)], fill=(200, 180, 120), width=1)
            x = cx + sign * offset
            if 0 <= x < w:
                draw.line([(x, 0), (x, h)], fill=(200, 180, 120), width=1)

    # Center DC component cross
    draw.line([(cx - 8, cy), (cx + 8, cy)], fill=(255, 60, 60), width=2)
    draw.line([(cx, cy - 8), (cx, cy + 8)], fill=(255, 60, 60), width=2)

    return base_img


def create_parchment_radial_plot(
    radial_profile: np.ndarray,
    is_ai: bool = False,
    alpha: float = 1.85
) -> go.Figure:
    """
    Creates a publication-grade Plotly line chart of the Azimuthal Radial Power
    Spectrum R(r) styled in the ScribeMark warm editorial parchment theme.

    Parameters
    ----------
    radial_profile : np.ndarray
        Azimuthally integrated radial power array (length 256).
    is_ai : bool
        Whether the sample is classified as AI diffusion or authentic camera.
    alpha : float
        Power-law exponent for the natural optical 1/f^alpha baseline.

    Returns
    -------
    plotly.graph_objects.Figure
        Responsive Plotly figure styled in warm editorial parchment.
    """
    freqs = np.arange(len(radial_profile))
    log_power = np.log10(np.maximum(radial_profile, 1e-6))

    # Natural optical baseline curve: R(r) ~ r^(-alpha) => log R(r) = C - alpha * log(r)
    baseline = log_power[1] - alpha * np.log10(np.maximum(freqs, 1))

    fig = go.Figure()

    # 1. Natural Optical Baseline Curve
    fig.add_trace(go.Scatter(
        x=freqs,
        y=baseline,
        mode="lines",
        name="Natural Optical Baseline (1/f^a)",
        line=dict(color="#1c3652", width=2, dash="dash"),
        hovertemplate="Freq: %{x} cyc<br>Baseline: %{y:.2f} dB<extra></extra>"
    ))

    # 2. Analyzed Sample Radial Profile Curve
    curve_color = "#8b2000" if is_ai else "#4a6b3a"
    fig.add_trace(go.Scatter(
        x=freqs,
        y=log_power,
        mode="lines",
        name="Sample Radial Profile R(r)",
        line=dict(color=curve_color, width=2.5),
        hovertemplate="Freq: %{x} cyc<br>Power: %{y:.2f} dB<extra></extra>"
    ))

    # 3. 8x8 Deconvolution Stride Harmonics Vertical Reference Lines
    stride_f = len(freqs) // 8
    for mult in [1, 2, 3]:
        fx = mult * stride_f
        if fx < len(freqs):
            fig.add_vline(
                x=fx,
                line_width=1.5,
                line_dash="dot",
                line_color="#7a6040",
                annotation_text=f"8x8 Harmonic ({mult}x)" if mult == 1 else "",
                annotation_position="top right",
                annotation_font=dict(family="JetBrains Mono", size=9, color="#7a6040")
            )

    # Layout matching ScribeMark Editorial Parchment
    fig.update_layout(
        paper_bgcolor="#eae0c5",
        plot_bgcolor="#eae0c5",
        font=dict(family="JetBrains Mono", size=10, color="#2c1f0e"),
        xaxis=dict(
            title="Spatial Frequency Radius r (cycles)",
            gridcolor="#c9b88a",
            linecolor="#b5a47e",
            zeroline=False
        ),
        yaxis=dict(
            title="Log Radial Power log10 R(r)",
            gridcolor="#c9b88a",
            linecolor="#b5a47e",
            zeroline=False
        ),
        margin=dict(l=50, r=30, t=35, b=45),
        height=360,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(234, 224, 197, 0.8)",
            bordercolor="#c9b88a",
            borderwidth=1
        )
    )
    return fig


def plot_forensic_comparison(real_res, ai_res, output_path: str):
    """
    Creates a 2x4 comprehensive diagnostic panel comparing Real vs AI.
    Preserved for offline publication script and benchmark reproduction.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # ------------------ Row 1: Authentic Human / Camera ------------------
    axes[0, 0].imshow(((real_res["arr_orig"] + 1.0) * 0.5).clip(0, 1))
    axes[0, 0].set_title("Authentic Photo (Original)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(((real_res["arr_recon"] + 1.0) * 0.5).clip(0, 1))
    axes[0, 1].set_title(f"VAE Reconstruction\nPSNR: {real_res['spatial']['psnr']:.2f} dB", fontsize=12)
    axes[0, 1].axis("off")

    delta_real = np.mean(np.abs(real_res["delta"]), axis=2) * 5.0
    im1 = axes[0, 2].imshow(delta_real, cmap="inferno", vmin=0, vmax=1.0)
    axes[0, 2].set_title(f"Residual |?x| (High Sensor Noise)\nMSE: {real_res['spatial']['mse']:.5f}", fontsize=12)
    axes[0, 2].axis("off")
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].imshow(real_res["log_magnitude"], cmap="viridis")
    axes[0, 3].set_title("2D-FFT Residual Spectrum\n(Smooth Power-Law Decay)", fontsize=12)
    axes[0, 3].axis("off")

    # ------------------ Row 2: AI Diffusion (Native SD VAE) ------------------
    axes[1, 0].imshow(((ai_res["arr_orig"] + 1.0) * 0.5).clip(0, 1))
    axes[1, 0].set_title("AI Generated (Native Diffusion)", fontsize=12, fontweight="bold", color="darkred")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(((ai_res["arr_recon"] + 1.0) * 0.5).clip(0, 1))
    axes[1, 1].set_title(f"VAE Reconstruction (Resonant)\nPSNR: {ai_res['spatial']['psnr']:.2f} dB", fontsize=12, color="darkred")
    axes[1, 1].axis("off")

    delta_ai = np.mean(np.abs(ai_res["delta"]), axis=2) * 5.0
    im2 = axes[1, 2].imshow(delta_ai, cmap="inferno", vmin=0, vmax=1.0)
    axes[1, 2].set_title(f"Residual |?x| (Near-Zero Loss)\nMSE: {ai_res['spatial']['mse']:.5f}", fontsize=12, color="darkred")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

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
    Preserved for offline publication script and benchmark reproduction.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    r_real = real_res["radial_profile"]
    r_ai = ai_res["radial_profile"]
    freqs = np.arange(len(r_real))

    ax.plot(freqs, np.log10(r_real + 1e-6), label="Real Camera Photo (Smooth 1/f Decay)", color="#2a4a6b", linewidth=2.5)
    ax.plot(freqs, np.log10(r_ai + 1e-6), label="AI Diffusion (Transposed Conv Periodic Spikes)", color="#8b2000", linewidth=2.5, linestyle="--")

    stride_f = len(freqs) // 8
    for mult in [1, 2, 3]:
        fx = mult * stride_f
        if fx < len(freqs):
            ax.axvline(x=fx, color="gray", linestyle=":", alpha=0.7, label=f"8x8 Lattice Harmonic ({mult}x)" if mult == 1 else "")

    ax.set_title(r"Azimuthally Averaged Radial Power Spectrum of VAE Residual $\Delta x$", fontsize=14, fontweight="bold")
    ax.set_xlabel("Spatial Frequency Radius $r$ (cycles)", fontsize=12)
    ax.set_ylabel(r"$\log_{10} R(r)$", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved radial spectrum plot to {output_path}")
