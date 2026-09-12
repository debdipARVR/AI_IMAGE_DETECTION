"""
Zero-Shot VAE Reconstruction Resonance and Spectral Forensic Engine
Author: Debdip Bandyopadhyay

Evaluates deterministic VAE latent reconstruction delta (spatial and frequency domain)
to distinguish between native diffusion synthetic images and natural photographic cameras.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import scipy.fftpack as fft


class VAEResonanceEngine:
    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[VAEResonanceEngine] Initializing with {model_name} on {self.device}...")
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        print("[VAEResonanceEngine] VAE loaded successfully.")

    def preprocess_image(self, img_input, target_size=(512, 512)):
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1.0, 1.0]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, arr

    def reconstruct(self, tensor_x):
        with torch.no_grad():
            posterior = self.vae.encode(tensor_x).latent_dist
            # Deterministic: use posterior mean (mode) directly, zero sigma sampling
            z = posterior.mean
            x_recon = self.vae.decode(z).sample
            
        x_recon = x_recon.clamp(-1.0, 1.0)
        arr_recon = x_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return arr_recon

    def compute_spatial_metrics(self, arr_orig, arr_recon):
        delta = arr_orig - arr_recon  # Range [-2.0, 2.0]
        mse = float(np.mean(delta ** 2))
        mae = float(np.mean(np.abs(delta)))
        psnr = float(10.0 * np.log10(4.0 / (mse + 1e-12)))  # peak signal is 2.0 (range -1 to 1)
        
        # Normalized Cross Correlation (NCC)
        norm_orig = np.linalg.norm(arr_orig)
        norm_recon = np.linalg.norm(arr_recon)
        if norm_orig > 1e-8 and norm_recon > 1e-8:
            ncc = float(np.sum(arr_orig * arr_recon) / (norm_orig * norm_recon))
        else:
            ncc = 0.0

        return {
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "ncc": ncc,
            "delta": delta
        }

    def compute_spectral_metrics(self, delta_spatial):
        gray_delta = np.mean(delta_spatial, axis=2)
        h, w = gray_delta.shape

        # 2D Discrete Fourier Transform
        f_transform = np.fft.fft2(gray_delta)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift) ** 2
        log_magnitude = np.log(1.0 + np.abs(f_shift))

        # Radial Profile (Azimuthal Integration)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
        max_r = min(cy, cx)

        radial_bins = np.bincount(r.ravel()[:], weights=power_spectrum.ravel()[:], minlength=max_r + 1)[:max_r]
        radial_counts = np.bincount(r.ravel()[:], minlength=max_r + 1)[:max_r]
        radial_profile = radial_bins / np.maximum(radial_counts, 1)

        # High-frequency vs Low-frequency energy ratio
        cutoff = max_r // 3
        low_freq_energy = float(np.sum(radial_bins[:cutoff]))
        high_freq_energy = float(np.sum(radial_bins[cutoff:max_r]))
        total_energy = low_freq_energy + high_freq_energy + 1e-12
        high_freq_ratio = float(high_freq_energy / total_energy)

        # Harmonic lattice peak detection:
        # Check for sharp periodic peaks corresponding to the 8x8 deconvolution stride
        stride_freq = h // 8
        harmonic_peaks = []
        for mult in [1, 2, 3]:
            freq_idx = mult * stride_freq
            if freq_idx < max_r - 2 and freq_idx > 2:
                local_window = radial_profile[freq_idx - 2 : freq_idx + 3]
                peak_val = radial_profile[freq_idx]
                background = np.mean([local_window[0], local_window[1], local_window[3], local_window[4]])
                ratio = float(peak_val / (background + 1e-12))
                harmonic_peaks.append(ratio)

        max_harmonic_spike = float(np.max(harmonic_peaks)) if harmonic_peaks else 1.0

        return {
            "log_magnitude": log_magnitude,
            "radial_profile": radial_profile,
            "high_freq_ratio": high_freq_ratio,
            "max_harmonic_spike": max_harmonic_spike,
            "total_spectral_energy": float(total_energy)
        }

    def analyze(self, img_input, target_size=(512, 512)):
        tensor_x, arr_orig = self.preprocess_image(img_input, target_size=target_size)
        arr_recon = self.reconstruct(tensor_x)
        
        spatial = self.compute_spatial_metrics(arr_orig, arr_recon)
        spectral = self.compute_spectral_metrics(spatial["delta"])

        psnr_norm = np.clip((spatial["psnr"] - 22.0) / 12.0, 0.0, 1.0)
        spike_norm = np.clip((spectral["max_harmonic_spike"] - 1.0) / 2.0, 0.0, 1.0)
        ai_probability = float(0.70 * psnr_norm + 0.30 * spike_norm)

        return {
            "ai_probability": ai_probability,
            "verdict": "AI-Generated (Congruent)" if ai_probability >= 0.50 else "Authentic Photographic (Divergent)",
            "spatial": {
                "mse": spatial["mse"],
                "mae": spatial["mae"],
                "psnr": spatial["psnr"],
                "ncc": spatial["ncc"]
            },
            "spectral": {
                "high_freq_ratio": spectral["high_freq_ratio"],
                "max_harmonic_spike": spectral["max_harmonic_spike"],
                "total_spectral_energy": spectral["total_spectral_energy"]
            },
            "arr_orig": arr_orig,
            "arr_recon": arr_recon,
            "delta": spatial["delta"],
            "log_magnitude": spectral["log_magnitude"],
            "radial_profile": spectral["radial_profile"]
        }
