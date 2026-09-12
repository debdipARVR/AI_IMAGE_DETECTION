"""
Streamlit Web Forensic Dashboard: Zero-Shot VAE Reconstruction Resonance & 2D-FFT Spectral Analyzer
Author: Debdip Bandyopadhyay
"""

import os
import sys
import io
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vae_resonance import VAEResonanceEngine

st.set_page_config(
    page_title="VAE Resonance Forensic Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Zero-Shot VAE Reconstruction Resonance & Spectral Forensics")
st.caption("Deterministic Latent Manifold Resonance & 2D-FFT Periodic Lattice Detection for AI Image Forensics")

@st.cache_resource
def load_engine():
    return VAEResonanceEngine(model_name="stabilityai/sd-vae-ft-mse")

with st.spinner("Loading Stable Diffusion VAE Forensic Backbone..."):
    engine = load_engine()

# Sidebar options
st.sidebar.header("Forensic Parameters")
model_info = st.sidebar.info("Model: stabilityai/sd-vae-ft-mse\nMode: Deterministic (Zero-Noise Mode)")
threshold = st.sidebar.slider("AI Classification Threshold", 0.0, 1.0, 0.50, 0.05)

# Preset loaders or File Uploader
col_up, col_pre = st.columns([2, 1])

uploaded_file = col_up.file_uploader("Upload an Image (.png, .jpg, .webp)", type=["png", "jpg", "jpeg", "webp"])

data_dir = os.path.join(os.path.dirname(__file__), "data")
real_presets = [os.path.join(data_dir, "real_photos", f) for f in os.listdir(os.path.join(data_dir, "real_photos")) if f.endswith(".png")] if os.path.exists(os.path.join(data_dir, "real_photos")) else []
ai_presets = [os.path.join(data_dir, "ai_synthetic", f) for f in os.listdir(os.path.join(data_dir, "ai_synthetic")) if f.endswith(".png")] if os.path.exists(os.path.join(data_dir, "ai_synthetic")) else []

selected_preset = col_pre.selectbox(
    "Or Select a Benchmark Sample:",
    ["None"] + [f"Authentic Camera: {os.path.basename(p)}" for p in real_presets[:5]] + [f"AI Diffusion: {os.path.basename(p)}" for p in ai_presets[:5]]
)

target_image = None
if uploaded_file is not None:
    target_image = Image.open(uploaded_file).convert("RGB")
elif selected_preset != "None":
    if "Authentic Camera:" in selected_preset:
        fn = selected_preset.split(": ")[1]
        target_image = Image.open(os.path.join(data_dir, "real_photos", fn)).convert("RGB")
    else:
        fn = selected_preset.split(": ")[1]
        target_image = Image.open(os.path.join(data_dir, "ai_synthetic", fn)).convert("RGB")

if target_image is not None:
    with st.spinner("Running Zero-Shot VAE Latent Encoding & 2D-FFT Spectral Decomposition..."):
        res = engine.analyze(target_image)

    # Verdict Banner
    prob = res["ai_probability"]
    is_ai = prob >= threshold

    st.markdown("---")
    if is_ai:
        st.error(f"### 🚨 Verdict: AI-Generated Diffusion Image (Probability: {prob*100:.1f}%)")
        st.write("**Resonance Diagnostics**: High VAE Reconstruction Congruence + Periodic Transposed Convolution Lattice Spikes detected.")
    else:
        st.success(f"### 🛡️ Verdict: Authentic Photographic Camera Image (Probability: {(1.0-prob)*100:.1f}% Authentic)")
        st.write("**Divergence Diagnostics**: Natural Sensor Noise (PRNU) & Stochastic High-Frequency Details cannot be reconstructed by lossy VAE.")

    # Top Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reconstruction PSNR", f"{res['spatial']['psnr']:.2f} dB", delta="High = AI" if res['spatial']['psnr'] > 28 else "Normal")
    m2.metric("Mean Squared Error (MSE)", f"{res['spatial']['mse']:.6f}")
    m3.metric("Harmonic Lattice Spike", f"{res['spectral']['max_harmonic_spike']:.2f}x", delta="Spike = Transposed Conv" if res['spectral']['max_harmonic_spike'] > 1.3 else "Smooth")
    m4.metric("High-Freq Energy Ratio", f"{res['spectral']['high_freq_ratio']:.3f}")

    # Visual Inspection Panels
    st.markdown("### Forensic Multi-Domain Inspection")
    v1, v2, v3, v4 = st.columns(4)

    v1.image(((res["arr_orig"] + 1.0) * 0.5).clip(0, 1), caption="Original Input Image", use_container_width=True)
    v2.image(((res["arr_recon"] + 1.0) * 0.5).clip(0, 1), caption=f"Deterministic VAE Reconstruction\n(PSNR: {res['spatial']['psnr']:.1f} dB)", use_container_width=True)
    
    delta_vis = np.mean(np.abs(res["delta"]), axis=2) * 5.0
    v3.image(delta_vis.clip(0, 1), caption=f"Reconstruction Residual |Δx| (5x Mag)\nMSE: {res['spatial']['mse']:.5f}", use_container_width=True)
    
    v4.image(res["log_magnitude"] / res["log_magnitude"].max(), caption=f"2D-FFT Residual Spectrum\nSpike: {res['spectral']['max_harmonic_spike']:.2f}x", use_container_width=True)

    # Radial Spectrum Chart
    st.markdown("### Azimuthally Averaged Radial Power Spectrum $R(r)$")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    freqs = np.arange(len(res["radial_profile"]))
    ax.plot(freqs, np.log10(res["radial_profile"] + 1e-6), color="#8b2000" if is_ai else "#2a4a6b", linewidth=2.0)
    
    # 8x8 stride mark
    stride_f = len(freqs) // 8
    for mult in [1, 2, 3]:
        fx = mult * stride_f
        if fx < len(freqs):
            ax.axvline(x=fx, color="gray", linestyle=":", alpha=0.7, label="8x8 Deconv Harmonic" if mult == 1 else "")

    ax.set_xlabel("Spatial Frequency Radius $r$")
    ax.set_ylabel("$\log_{10} R(r)$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
