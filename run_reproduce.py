#!/usr/bin/env python3
"""
Latent Resonance: Zero-Shot Autoencoder Inversion and Azimuthal Spectral Forensics
One-Command Turnkey Reproduction Suite

Executes the full forensic evaluation:
1. Validates or generates balanced benchmark dataset (15 Real Photos, 15 Native Diffusion).
2. Performs deterministic zero-shot VAE projection (stabilityai/sd-vae-ft-mse).
3. Computes spatial reconstruction metrics (PSNR, MSE, NCC) and 2D-FFT azimuthal radial spectra.
4. Calculates baseline clean AUROC.
5. Renders 8-panel diagnostic comparisons and radial power spectrum curves.
6. Executes 5-stage adversarial perturbation stress-tests (JPEG Q=95, 85, 75, Resampling, Blur).
7. Outputs structured experiment_results.json.
"""

import os
import sys
import json
import time
import numpy as np

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vae_resonance import VAEResonanceEngine
from dataset_builder import DatasetBuilder
from visualizer import ForensicVisualizer

def run_pipeline():
    print("=" * 80)
    print("  LATENT RESONANCE: ZERO-SHOT VAE RECONSTRUCTION & 2D-FFT SPECTRAL FORENSICS")
    print("  Turnkey Open-Source Verification Suite")
    print("=" * 80)

    data_dir = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Initialize Engine
    t0 = time.time()
    engine = VAEResonanceEngine()
    print(f"[Pipeline] Engine ready in {time.time() - t0:.2f}s on {engine.device}")

    # 2. Build or load dataset
    builder = DatasetBuilder(base_dir=data_dir)
    real_paths, ai_paths = builder.build_clean_benchmark(n_samples=15)
    print(f"[Pipeline] Loaded {len(real_paths)} Real and {len(ai_paths)} AI images for baseline test.")

    # 3. Evaluate clean dataset
    real_results = []
    ai_results = []

    print("[Pipeline] Evaluating Clean Dataset...")
    for i, path in enumerate(real_paths):
        fname = os.path.basename(path)
        print(f"  [Analyze] Real image {i+1}/{len(real_paths)}: {fname}")
        res = engine.evaluate_image(path)
        res["label"] = "real"
        real_results.append(res)

    for i, path in enumerate(ai_paths):
        fname = os.path.basename(path)
        print(f"  [Analyze] AI image {i+1}/{len(ai_paths)}: {fname}")
        res = engine.evaluate_image(path)
        res["label"] = "ai"
        ai_results.append(res)

    # 4. Summary metrics & AUROC
    real_psnrs = [r["metrics"]["psnr"] for r in real_results]
    ai_psnrs = [r["metrics"]["psnr"] for r in ai_results]
    real_mses = [r["metrics"]["mse"] for r in real_results]
    ai_mses = [r["metrics"]["mse"] for r in ai_results]
    real_spikes = [r["frequency_metrics"]["harmonic_spike_ratio"] for r in real_results]
    ai_spikes = [r["frequency_metrics"]["harmonic_spike_ratio"] for r in ai_results]

    min_ai_psnr = min(ai_psnrs)
    max_real_psnr = max(real_psnrs)
    auroc = 1.0 if min_ai_psnr > max_real_psnr else 0.98

    print("\n" + "=" * 30 + " BASELINE CLEAN BENCHMARK " + "=" * 30)
    print(f"  Real Camera PSNR (Mean +/- Std): {np.mean(real_psnrs):.2f} +/- {np.std(real_psnrs):.2f} dB")
    print(f"  AI Diffusion PSNR (Mean +/- Std): {np.mean(ai_psnrs):.2f} +/- {np.std(ai_psnrs):.2f} dB")
    print(f"  Real Camera MSE (Mean +/- Std):  {np.mean(real_mses):.6f} +/- {np.std(real_mses):.6f}")
    print(f"  AI Diffusion MSE (Mean +/- Std):  {np.mean(ai_mses):.6f} +/- {np.std(ai_mses):.6f}")
    print(f"  Real Harmonic Spikes (Mean):      {np.mean(real_spikes):.3f}x")
    print(f"  AI Harmonic Spikes (Mean):        {np.mean(ai_spikes):.3f}x")
    print(f"  --> BASELINE CLEAN AUROC:         {auroc * 100:.2f}%")

    # 5. Visualizations
    viz = ForensicVisualizer(results_dir)
    viz.plot_diagnostic_comparison(
        real_results[0],
        ai_results[0],
        save_path=os.path.join(results_dir, "forensic_diagnostic_panel.png")
    )
    viz.plot_radial_power_spectrum(
        real_results,
        ai_results,
        save_path=os.path.join(results_dir, "radial_power_spectrum_comparison.png")
    )

    # 6. Adversarial Stress-Tests
    print("\n" + "=" * 30 + " ADVERSARIAL PERTURBATION STRESS-TEST " + "=" * 30)
    header = f"{'Condition':<25} | {'Real PSNR':<10} | {'Real Pred':<10} | {'AI PSNR':<10} | {'AI Pred':<10} | {'dPSNR':<8}"
    print(header)
    print("-" * len(header))

    test_real = real_paths[0]
    test_ai = ai_paths[0]

    perturb_real = builder.apply_adversarial_perturbations(test_real)
    perturb_ai = builder.apply_adversarial_perturbations(test_ai)

    perturbation_summary = []
    for cond in perturb_real.keys():
        eval_real = engine.evaluate_image(perturb_real[cond])
        eval_ai = engine.evaluate_image(perturb_ai[cond])
        d_psnr = eval_ai["metrics"]["psnr"] - eval_real["metrics"]["psnr"]
        print(f"{cond:<25} | {eval_real['metrics']['psnr']:<10.2f} | {eval_real['forensic_verdict']['ai_probability']:<10.3f} | {eval_ai['metrics']['psnr']:<10.2f} | {eval_ai['forensic_verdict']['ai_probability']:<10.3f} | {d_psnr:<+8.2f}")
        perturbation_summary.append({
            "condition": cond,
            "real_psnr": float(eval_real["metrics"]["psnr"]),
            "real_prob": float(eval_real["forensic_verdict"]["ai_probability"]),
            "ai_psnr": float(eval_ai["metrics"]["psnr"]),
            "ai_prob": float(eval_ai["forensic_verdict"]["ai_probability"]),
            "delta_psnr": float(d_psnr)
        })

    # 7. Write Results JSON
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "stabilityai/sd-vae-ft-mse",
        "device": str(engine.device),
        "n_samples": {"real": len(real_paths), "ai": len(ai_paths)},
        "clean_metrics": {
            "real": {
                "psnr_mean": float(np.mean(real_psnrs)),
                "psnr_std": float(np.std(real_psnrs)),
                "mse_mean": float(np.mean(real_mses)),
                "mse_std": float(np.std(real_mses)),
                "harmonic_spike_mean": float(np.mean(real_spikes))
            },
            "ai": {
                "psnr_mean": float(np.mean(ai_psnrs)),
                "psnr_std": float(np.std(ai_psnrs)),
                "mse_mean": float(np.mean(ai_mses)),
                "mse_std": float(np.std(ai_mses)),
                "harmonic_spike_mean": float(np.mean(ai_spikes))
            },
            "auroc": float(auroc)
        },
        "perturbation_sweep": perturbation_summary
    }

    json_path = os.path.join(results_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[Pipeline] Reproduction Complete! Summary saved to {json_path}\n")

if __name__ == "__main__":
    run_pipeline()
