"""
Tier 4: E2E Scenario 5 - Air-Gapped Standalone Offline Execution Verification
Authoritative Source: Spec Miner Survey Report 3 § 3.4 & TEST_INFRA.md
"""

import socket
import pytest
from PIL import Image

from src.forensic_classifier import classify_forensics
from src.pdf_certificate import generate_forensic_certificate
from src.preset_cache import load_preset_sample


def test_scenario_5_airgapped_offline_execution(test_engine, monkeypatch):
    """
    Scenario 5: Air-gapped standalone offline execution verification.
    Disables all network socket connections and verifies that the complete
    forensic analysis and ISO/IEC 27037 PDF certificate generation execute
    100% locally with zero network or cloud dependencies.
    """
    # 1. Sever all socket connections (Air-Gapped Simulation)
    def airgap_guard(*args, **kwargs):
        raise ConnectionRefusedError("Air-gapped security boundary: outbound network calls prohibited.")

    monkeypatch.setattr(socket.socket, "connect", airgap_guard)

    # 2. Ingestion & Analysis in Air-Gapped Environment
    img, telemetry = load_preset_sample("authentic_camera")
    res = test_engine.analyze(img)
    verdict = classify_forensics(res["spatial"], res["spectral"])

    # 3. Assemble Evidence & Export PDF in Air-Gapped Mode
    evidence_data = {
        "evidence_identification": {
            "case_id": "CASE-AIRGAP-005",
            "evidence_uuid": "99999999-9999-9999-9999-999999999999",
            "filename": "offline_airgap_sample.png",
            "filesize_bytes": 1024,
            "dimensions": "512 x 512"
        },
        "custodial_timestamps": {
            "analysis_utc": "2026-09-12T20:59:00Z"
        },
        "cryptographic_chain_of_custody": {
            "input_file_sha256": "9" * 64
        },
        "forensic_metrics": {
            "psnr_db": res["spatial"]["psnr"],
            "mse": res["spatial"]["mse"],
            "harmonic_lattice_spike_ratio": res["spectral"]["max_harmonic_spike"]
        },
        "verdict": verdict,
        "verification_seal": {
            "signatory_authority": "ScribeMark Latent Resonance Evidence Engine v1.0",
            "algorithm": "HMAC-SHA256 (Canonical JSON Manifest)",
            "signature_hex": "9" * 64,
            "admissibility_statute": "ISO/IEC 27037:2012 §5.4-5.6 | FRE 902(13)-(14)"
        }
    }

    visual_images = {
        "orig_pil": img,
        "recon_pil": img,
        "residual_pil": img,
        "fft_pil": img
    }

    # PDF generation must succeed locally with standard Helvetica/Courier fonts
    pdf_bytes = generate_forensic_certificate(evidence_data, visual_images)
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"CASE-AIRGAP-005" in pdf_bytes
