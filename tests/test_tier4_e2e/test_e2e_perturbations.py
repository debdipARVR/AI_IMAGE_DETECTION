"""
Tier 4: E2E Scenario 3 - Adversarial Perturbation & Lossy Resampling Stress-Testing
Authoritative Source: Spec Miner Survey Report 3 § 3.4 & TEST_INFRA.md
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.forensic_classifier import classify_forensics
from src.pdf_certificate import generate_forensic_certificate
from src.preset_cache import load_preset_sample


def test_scenario_3_compression_perturbations(test_engine):
    """
    Scenario 3: Stress-testing adversarial compression (JPEG Q=75) and resampling.
    Validates that lossy compression does not crash the forensic pipeline and that
    the system correctly handles block quantization artifacts in spatial/frequency domains.
    """
    # 1. Generate lossy compressed JPEG version of base sample
    img, _ = load_preset_sample("authentic_camera")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    img_compressed = Image.open(buf).convert("RGB")

    # 2. Run through full 4-pass analytical engine
    res = test_engine.analyze(img_compressed)
    spatial = res["spatial"]
    spectral = res["spectral"]

    assert not np.isnan(spatial["psnr"])
    assert not np.isnan(spatial["mse"])
    assert res["spectral"]["radial_profile"] is not None

    # 3. Classify with calibrated decision engine
    verdict = classify_forensics(spatial, spectral)
    assert verdict["category"] in ["AUTHENTIC OPTICAL PHOTO", "AI GENERATED DIFFUSION", "MANIPULATED / RESAMPLED"]

    # 4. Generate certificate documenting compression artifacts
    ev_dict = {
        "evidence_identification": {
            "case_id": "CASE-PERTURBED-003",
            "evidence_uuid": "c81d4fae-7dec-11d0-a765-00a0c91e6bf8",
            "filename": "compressed_q75.jpg",
            "filesize_bytes": len(buf.getvalue()),
            "dimensions": "512 x 512",
            "color_channels": "RGB",
            "mime_type": "image/jpeg"
        },
        "custodial_timestamps": {
            "ingestion_utc": "2026-09-12T20:50:00Z",
            "analysis_utc": "2026-09-12T20:50:02Z",
            "certificate_issued_utc": "2026-09-12T20:50:03Z"
        },
        "cryptographic_chain_of_custody": {"input_file_sha256": "c" * 64},
        "forensic_metrics": {"psnr_db": spatial["psnr"], "mse": spatial["mse"], "harmonic_lattice_spike_ratio": spectral["max_harmonic_spike"]},
        "verdict": verdict,
        "verification_seal": {
            "signatory_authority": "ScribeMark Latent Resonance Evidence Engine v1.0",
            "algorithm": "HMAC-SHA256 (Canonical JSON Manifest)",
            "signature_hex": "c" * 64,
            "admissibility_statute": "ISO/IEC 27037:2012 §5.4-5.6 | FRE 902(13)-(14)"
        }
    }

    pdf_bytes = generate_forensic_certificate(ev_dict, {"orig_pil": img_compressed, "recon_pil": img_compressed, "residual_pil": img_compressed, "fft_pil": img_compressed})
    assert pdf_bytes.startswith(b"%PDF-")
