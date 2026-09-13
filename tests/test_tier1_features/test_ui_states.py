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


@pytest.fixture
def unlocked_app(headless_app):
    """Headless AppTest runner with terms accepted to unlock ingestion pipeline."""
    at = headless_app
    if len(at.checkbox) > 0 and not at.session_state["terms_accepted"]:
        at.checkbox[0].check().run()
    return at


def test_t1_8_1_initial_state_landing_view(headless_app):
    """T1.8.1: Initial application state renders title, gated dropzone, and terms checkbox."""
    at = headless_app
    assert len(at.exception) == 0
    assert len(at.file_uploader) >= 1
    # Image upload must be locked until terms are accepted
    assert at.file_uploader[0].disabled is True
    assert len(at.checkbox) >= 1
    assert at.session_state["terms_accepted"] is False


def test_t1_8_1_terms_acceptance_unlocks_ingestion(unlocked_app):
    """T1.8.1b: Accepting terms unlocks file upload dropzone."""
    at = unlocked_app
    assert len(at.exception) == 0
    assert at.session_state["terms_accepted"] is True
    assert len(at.file_uploader) >= 1
    assert at.file_uploader[0].disabled is False


def test_t1_8_2_preset_selection_populates_image(unlocked_app):
    """T1.8.2: Preset selection populates image preview and metrics."""
    at = unlocked_app
    # Select authentic camera preset (index 1 if presets available)
    if len(at.selectbox) > 0 and len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        assert len(at.image) >= 1


def test_t1_8_3_file_upload_triggers_processing(unlocked_app):
    """T1.8.3: File upload to dropzone triggers VAE analysis and metric display."""
    at = unlocked_app
    img = Image.new("RGB", (64, 64), (120, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    at.file_uploader[0].upload("test_upload.png", buf.getvalue()).run()
    assert len(at.exception) == 0
    
    # Verify metric cards rendered
    labels = [m.label for m in at.metric]
    assert "Reconstruction PSNR" in labels
    assert "Mean Squared Error (MSE)" in labels


def test_t1_8_4_traffic_light_badge_rendering(unlocked_app):
    """T1.8.4: Analysis renders verdict banner without unhandled exception."""
    at = unlocked_app
    if len(at.selectbox) > 0 and len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        # Verdict displays in error or success block
        has_verdict = len(at.error) > 0 or len(at.success) > 0 or len(at.warning) > 0
        assert has_verdict


def test_t1_8_5_visual_diagnostic_panels_render(unlocked_app):
    """T1.8.5: Multi-domain visual inspection panels display images without error."""
    at = unlocked_app
    if len(at.selectbox) > 0 and len(at.selectbox[0].options) > 1:
        at.selectbox[0].select_index(1).run()
        assert len(at.exception) == 0
        # Check that multiple images (Orig, Recon, Residual, FFT) are rendered
        assert len(at.image) >= 4
