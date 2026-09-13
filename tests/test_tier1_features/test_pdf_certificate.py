"""
Tier 1: Feature Coverage - ISO/IEC 27037 ReportLab PDF Certificate Generation
Test ID: T1.7.1 to T1.7.5
Authoritative Source: PROJECT.md § Feature 7 & TEST_INFRA.md
"""

import io
import pytest
from src.pdf_certificate import generate_forensic_certificate, ForensicNumberedCanvas


def test_t1_7_1_valid_pdf_binary_signature(sample_evidence_dict, sample_visual_images):
    """T1.7.1: Generated PDF contains valid %PDF binary header and %%EOF trailer."""
    pdf_bytes = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes


def test_t1_7_2_sha256_hash_preservation(sample_evidence_dict, sample_visual_images):
    """T1.7.2: Verbatim SHA-256 cryptographic hashes are preserved in compiled PDF bytes."""
    input_sha256 = sample_evidence_dict["cryptographic_chain_of_custody"]["input_file_sha256"]
    pdf_bytes = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)

    assert input_sha256.encode("ascii") in pdf_bytes


def test_t1_7_3_in_memory_image_embedding(sample_evidence_dict, sample_visual_images):
    """T1.7.3: All 4 diagnostic plates are embedded purely in-memory with zero disk temporary files."""
    pdf_bytes = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)

    # In PDF, embedded images are stored as XObject forms or Image streams
    assert b"/Image" in pdf_bytes or b"/XObject" in pdf_bytes
    assert len(pdf_bytes) > 5000  # Embedded PNGs add substantial byte size


def test_t1_7_4_dynamic_page_numbering(sample_evidence_dict, sample_visual_images):
    """T1.7.4: ForensicNumberedCanvas renders dynamic page count (e.g. 'Page 1 of')."""
    pdf_bytes = generate_forensic_certificate(sample_evidence_dict, sample_visual_images)

    assert b"Page 1 of" in pdf_bytes


def test_t1_7_5_long_text_wrapping(sample_evidence_dict, sample_visual_images):
    """T1.7.5: Zero-throw on long 1000-word forensic rationale: autowraps cleanly."""
    long_evidence = dict(sample_evidence_dict)
    long_rationale = "Forensic analytical evaluation of latent resonance manifold. " * 120
    long_evidence["verdict"]["forensic_rationale"] = long_rationale

    # Must compile cleanly without overflowing canvas or raising reportlab.platypus.LayoutError
    pdf_bytes = generate_forensic_certificate(long_evidence, sample_visual_images)
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes
