"""
Tier 2: Boundary & Corner Cases - Corrupted & Malformed Files
Test ID: T2.6.1 to T2.6.5
Authoritative Source: Spec Miner Survey Report 3 § 3.2 & TEST_INFRA.md
"""

import io
import pytest
from PIL import Image, UnidentifiedImageError


def test_t2_6_1_zero_byte_empty_file(test_engine, tmp_path):
    """T2.6.1: 0-byte empty file is trapped via UnidentifiedImageError or ValueError."""
    engine = test_engine
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises((UnidentifiedImageError, ValueError)):
        engine.preprocess_image(str(empty_file))


def test_t2_6_2_partial_truncated_jpeg_stream(test_engine, tmp_path):
    """T2.6.2: Partial byte stream (truncated JPEG header) is trapped without segmentation fault."""
    engine = test_engine
    # First 16 bytes of JPEG header followed by truncation
    partial_jpeg = tmp_path / "broken.jpg"
    partial_jpeg.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01")

    with pytest.raises((UnidentifiedImageError, ValueError, OSError)):
        engine.preprocess_image(str(partial_jpeg))


def test_t2_6_3_text_file_renamed_to_png(test_engine, tmp_path):
    """T2.6.3: Plain ASCII text file renamed to .png is rejected before tensor conversion."""
    engine = test_engine
    txt_png = tmp_path / "fake_image.png"
    txt_png.write_text("This is not a PNG file, just plaintext data.", encoding="utf-8")

    with pytest.raises((UnidentifiedImageError, ValueError)):
        engine.preprocess_image(str(txt_png))


def test_t2_6_4_corrupted_png_crc_chunk(test_engine, tmp_path):
    """T2.6.4: PNG with corrupted CRC chunk header is trapped gracefully."""
    engine = test_engine
    # PNG signature followed by corrupt IHDR chunk
    corrupt_png = tmp_path / "corrupt_crc.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\xFF\xFF\xFF\xFF")

    with pytest.raises((UnidentifiedImageError, ValueError, OSError)):
        engine.preprocess_image(str(corrupt_png))


def test_t2_6_5_random_binary_garbage_stream(test_engine, tmp_path):
    """T2.6.5: Random high-entropy binary payload is safely trapped."""
    engine = test_engine
    import os
    garbage_file = tmp_path / "garbage.webp"
    garbage_file.write_bytes(os.urandom(1024))

    with pytest.raises((UnidentifiedImageError, ValueError, OSError)):
        engine.preprocess_image(str(garbage_file))
