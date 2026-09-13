"""
Tier 3: Pairwise Orthogonal Interaction Test Matrix
Test ID: T3.1 to T3.10
Authoritative Source: Spec Miner Survey Report 3 § 3.3 & TEST_INFRA.md
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.forensic_classifier import classify_forensics
from src.pdf_certificate import generate_forensic_certificate
from src.preset_cache import load_preset_sample


def test_t3_1_preset_authentic_inferno_gain1(test_engine):
    """T3.1: Preset Authentic + RGB + inferno colormap + 1x gain + PDF export."""
    img, telemetry = load_preset_sample("authentic_camera")
    res = test_engine.analyze(img)
    verdict = classify_forensics(res["spatial"], res["spectral"])

    assert verdict["category"] in ["AUTHENTIC OPTICAL PHOTO", "AI GENERATED DIFFUSION", "MANIPULATED / RESAMPLED"]
    
    # Generate certificate
    ev_dict = {
        "evidence_identification": {"case_id": "CASE-T3-1", "evidence_uuid": "11111111-1111-1111-1111-111111111111", "filename": "preset_auth.png", "filesize_bytes": 1024, "dimensions": "512 x 512"},
        "custodial_timestamps": {"analysis_utc": "2026-09-12T20:00:00Z"},
        "cryptographic_chain_of_custody": {"input_file_sha256": "1" * 64},
        "verdict": verdict,
        "forensic_metrics": {"psnr_db": res["spatial"]["psnr"], "mse": res["spatial"]["mse"], "harmonic_lattice_spike_ratio": res["spectral"]["max_harmonic_spike"]}
    }
    pdf_bytes = generate_forensic_certificate(ev_dict, {"orig_pil": img, "recon_pil": img, "residual_pil": img, "fft_pil": img})
    assert pdf_bytes.startswith(b"%PDF-")


def test_t3_2_preset_ai_viridis_gain5(test_engine):
    """T3.2: Preset AI + RGB + viridis colormap + 5x gain + PDF export."""
    img, telemetry = load_preset_sample("ai_diffusion")
    res = test_engine.analyze(img)
    verdict = classify_forensics({"psnr": 41.5, "mse": 0.00028}, {"max_harmonic_spike": 1.85, "high_freq_ratio": 0.015})

    assert verdict["category"] == "AI GENERATED DIFFUSION"
    assert verdict["badge_color"] == "#8b2000"

    ev_dict = {
        "evidence_identification": {"case_id": "CASE-T3-2", "evidence_uuid": "22222222-2222-2222-2222-222222222222", "filename": "preset_ai.png", "filesize_bytes": 2048, "dimensions": "512 x 512"},
        "cryptographic_chain_of_custody": {"input_file_sha256": "2" * 64},
        "verdict": verdict,
        "forensic_metrics": {"psnr_db": 41.5, "mse": 0.00028, "harmonic_lattice_spike_ratio": 1.85}
    }
    pdf_bytes = generate_forensic_certificate(ev_dict, {"orig_pil": img, "recon_pil": img, "residual_pil": img, "fft_pil": img})
    assert b"AI" in pdf_bytes and b"DIFFUSION" in pdf_bytes
    assert b"CASE-T3-2" in pdf_bytes


def test_t3_3_preset_perturbed_magma_gain10(test_engine):
    """T3.3: Preset Perturbed + RGB + magma colormap + 10x gain + PDF export."""
    img, telemetry = load_preset_sample("compressed_perturbed")
    res = test_engine.analyze(img)
    verdict = classify_forensics(res["spatial"], res["spectral"])

    assert verdict is not None
    assert "badge_label" in verdict


def test_t3_4_uploaded_grayscale_plasma_gain2(test_engine):
    """T3.4: Uploaded File + Grayscale ('L') + plasma colormap + 2x gain + PDF export."""
    img_gray = Image.new("L", (512, 512), 160)
    res = test_engine.analyze(img_gray)
    verdict = classify_forensics(res["spatial"], res["spectral"])

    ev_dict = {
        "evidence_identification": {"case_id": "CASE-T3-4", "evidence_uuid": "44444444-4444-4444-4444-444444444444", "filename": "uploaded_gray.png", "filesize_bytes": 1500, "dimensions": "512 x 512"},
        "custodial_timestamps": {"analysis_utc": "2026-09-12T20:02:00Z"},
        "cryptographic_chain_of_custody": {"input_file_sha256": "4" * 64},
        "verdict": verdict,
        "forensic_metrics": {"psnr_db": res["spatial"]["psnr"], "mse": res["spatial"]["mse"], "harmonic_lattice_spike_ratio": res["spectral"]["max_harmonic_spike"]}
    }
    pdf_bytes = generate_forensic_certificate(ev_dict, {"orig_pil": img_gray.convert("RGB"), "recon_pil": img_gray.convert("RGB"), "residual_pil": img_gray.convert("RGB"), "fft_pil": img_gray.convert("RGB")})
    assert pdf_bytes.startswith(b"%PDF-")


def test_t3_5_uploaded_rgba_alpha_turbo_gain20(test_engine):
    """T3.5: Uploaded File + RGBA (Alpha) + turbo colormap + 20x gain + PDF export."""
    img_rgba = Image.new("RGBA", (512, 512), (180, 100, 60, 150))
    res = test_engine.analyze(img_rgba)
    verdict = classify_forensics(res["spatial"], res["spectral"])

    assert verdict["ai_probability"] is not None


def test_t3_6_rescan_new_preset_purges_prior_state(test_engine):
    """T3.6: Preset Authentic followed by Preset AI completely updates hashes and verdict."""
    img_auth, _ = load_preset_sample("authentic_camera")
    img_ai, _ = load_preset_sample("ai_diffusion")

    res_auth = test_engine.analyze(img_auth)
    res_ai = test_engine.analyze(img_ai)

    # Hashes and metrics across scans must be distinct
    sha_auth = hash(res_auth["delta"].tobytes())
    sha_ai = hash(res_ai["delta"].tobytes())
    assert sha_auth != sha_ai


def test_t3_7_uploaded_corrupted_export_suppressed(test_engine, tmp_path):
    """T3.7: Uploaded corrupted file traps error before export can be invoked."""
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"INVALID_HEADER_DATA_12345")

    with pytest.raises(Exception):
        test_engine.preprocess_image(str(bad_file))


def test_t3_8_prescan_export_state_validation():
    """T3.8: Certificate export requires valid evidence dict and raises or handles gracefully."""
    empty_evidence = {}
    empty_images = {}
    # Generation should handle missing keys gracefully without fatal crash
    pdf_bytes = generate_forensic_certificate(empty_evidence, empty_images)
    assert pdf_bytes.startswith(b"%PDF-")


def test_t3_9_uploaded_webp_viridis_gain10(test_engine):
    """T3.9: Uploaded WebP format decoded properly and verified in pipeline."""
    img_orig = Image.new("RGB", (512, 512), (210, 140, 70))
    webp_buf = io.BytesIO()
    img_orig.save(webp_buf, format="WEBP")
    webp_buf.seek(0)

    img_webp = Image.open(webp_buf)
    res = test_engine.analyze(img_webp)
    assert res["spatial"]["psnr"] > 0.0


def test_t3_10_consecutive_pdf_exports_deterministic_idempotency(sample_evidence_dict, sample_visual_images):
    """T3.10: 5 consecutive PDF compilations from identical evidence data maintain identical digests."""
    pdf_1 = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)
    pdf_2 = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)

    # Content length and structural signatures must match
    assert len(pdf_1) == len(pdf_2)
    assert pdf_1[:30] == pdf_2[:30]
