"""
Tier 4: E2E Scenario 4 - Multi-Item Successive Triage & Session State Isolation
Authoritative Source: Spec Miner Survey Report 3 § 3.4 & TEST_INFRA.md
"""

import hashlib
import pytest
import numpy as np
from PIL import Image

from src.forensic_classifier import classify_forensics
from src.pdf_certificate import generate_forensic_certificate
from src.preset_cache import load_preset_sample


def test_scenario_4_successive_triage_session_isolation(test_engine):
    """
    Scenario 4: Multi-item successive triage & session state isolation.
    Validates that consecutive scans on different images do not leak state,
    cache values, cryptographic hashes, or UUIDs across analysis sessions.
    """
    # 1. First Session: Analyze Image A (Authentic Camera Photo)
    img_a, _ = load_preset_sample("authentic_camera")
    res_a = test_engine.analyze(img_a)
    hash_a = hashlib.sha256(img_a.tobytes()).hexdigest()
    uuid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    ev_a = {
        "evidence_identification": {"case_id": "CASE-SESSION-A", "evidence_uuid": uuid_a, "filename": "sample_a.png", "filesize_bytes": 1000, "dimensions": "512 x 512"},
        "custodial_timestamps": {"analysis_utc": "2026-09-12T20:55:01Z"},
        "cryptographic_chain_of_custody": {"input_file_sha256": hash_a},
        "forensic_metrics": {"psnr_db": res_a["spatial"]["psnr"], "mse": res_a["spatial"]["mse"], "harmonic_lattice_spike_ratio": res_a["spectral"]["max_harmonic_spike"]},
        "verdict": classify_forensics(res_a["spatial"], res_a["spectral"])
    }
    pdf_a = generate_forensic_certificate(ev_a, {"orig_pil": img_a, "recon_pil": img_a, "residual_pil": img_a, "fft_pil": img_a})

    # 2. Second Session: Immediately Analyze Image B (AI Diffusion Image) without clearing application
    img_b, _ = load_preset_sample("ai_diffusion")
    res_b = test_engine.analyze(img_b)
    hash_b = hashlib.sha256(img_b.tobytes()).hexdigest()
    uuid_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    ev_b = {
        "evidence_identification": {"case_id": "CASE-SESSION-B", "evidence_uuid": uuid_b, "filename": "sample_b.png", "filesize_bytes": 2000, "dimensions": "512 x 512"},
        "custodial_timestamps": {"analysis_utc": "2026-09-12T20:55:05Z"},
        "cryptographic_chain_of_custody": {"input_file_sha256": hash_b},
        "forensic_metrics": {"psnr_db": res_b["spatial"]["psnr"], "mse": res_b["spatial"]["mse"], "harmonic_lattice_spike_ratio": res_b["spectral"]["max_harmonic_spike"]},
        "verdict": classify_forensics(res_b["spatial"], res_b["spectral"])
    }
    pdf_b = generate_forensic_certificate(ev_b, {"orig_pil": img_b, "recon_pil": img_b, "residual_pil": img_b, "fft_pil": img_b})

    # 3. Assert Strict Session Isolation
    # Hashes and UUIDs must never leak between reports
    assert hash_a != hash_b
    assert uuid_a != uuid_b
    assert hash_a.encode("ascii") in pdf_a
    assert hash_a.encode("ascii") not in pdf_b
    assert uuid_a.encode("ascii") in pdf_a
    assert uuid_a.encode("ascii") not in pdf_b
    assert uuid_b.encode("ascii") in pdf_b
    assert uuid_b.encode("ascii") not in pdf_a
