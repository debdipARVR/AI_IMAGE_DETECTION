"""
Tier 1: Feature Coverage - Streamlit Headless UI State Transitions
Test ID: T1.8.1 to T1.8.5
Authoritative Source: PROJECT.md § Feature 8 & TEST_INFRA.md
"""

import os
import io
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


@pytest.fixture
def headless_app():
    """Initializes headless Streamlit AppTest runner from app.py."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def test_t1_8_1_initial_state_landing_view(headless_app):
    """T1.8.1: Initial application state renders title, dropzone, and preset loader."""
    at = headless_app
    assert len(at.exception) == 0
    assert len(at.file_uploader) >= 1
    assert len(at.selectbox) >= 1


def test_t1_8_2_preset_selection_populates_image(headless_app):
    """T1.8.2: Preset selection populates image preview and metrics."""
    at = headless_app
    # Select authentic camera preset (index 1 if presets available)
    if len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        assert len(at.image) >= 1


def test_t1_8_3_file_upload_triggers_processing(headless_app):
    """T1.8.3: File upload to dropzone triggers VAE analysis and metric display."""
    at = headless_app
    img = Image.new("RGB", (64, 64), (120, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    at.file_uploader[0].upload("test_upload.png", buf.getvalue()).run()
    assert len(at.exception) == 0
    
    # Verify metric cards rendered
    labels = [m.label for m in at.metric]
    assert "Reconstruction PSNR" in labels
    assert "Mean Squared Error (MSE)" in labels


def test_t1_8_4_traffic_light_badge_rendering(headless_app):
    """T1.8.4: Analysis renders verdict banner without unhandled exception."""
    at = headless_app
    if len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        # Verdict displays in error or success block
        has_verdict = len(at.error) > 0 or len(at.success) > 0 or len(at.warning) > 0
        assert has_verdict


def test_t1_8_5_visual_diagnostic_panels_render(headless_app):
    """T1.8.5: Multi-domain visual inspection panels display images without error."""
    at = headless_app
    if len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        # Check that multiple images (Orig, Recon, Residual, FFT) are rendered
        assert len(at.image) >= 4
