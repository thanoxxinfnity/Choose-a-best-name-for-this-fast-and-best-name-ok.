"""Live checks against the real Puter API.

Skipped unless PUTER_API_KEY is set, so the normal suite stays offline. These
exist because the failures they catch are invisible to every other kind of
test: a voice id that Polly does not have, or a driver method that was renamed
away, is a perfectly valid string in a config file and a 400 at runtime.

    PUTER_API_KEY=... python -m pytest tests/test_puter_live.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from voices import VOICE_PROFILES  # noqa: E402

KEY = os.environ.get("PUTER_API_KEY", "")
pytestmark = pytest.mark.skipif(not KEY, reason="PUTER_API_KEY is not set")


def _call(interface, driver, method, args, timeout=120):
    return requests.post(
        f"{settings.puter_base_url}/drivers/call",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"interface": interface, "driver": driver, "method": method, "args": args},
        timeout=timeout,
    )


def test_the_key_authenticates():
    response = requests.get(
        f"{settings.puter_base_url}/whoami",
        headers={"Authorization": f"Bearer {KEY}"}, timeout=30,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("key", sorted(VOICE_PROFILES))
def test_every_voice_we_ship_actually_exists(key):
    """A voice id Polly does not have is a valid string and a 400 at runtime."""
    profile = VOICE_PROFILES[key]
    response = _call(
        settings.puter_tts_interface, settings.puter_tts_driver, settings.puter_tts_method,
        {"text": "test", "voice": profile.voice_id,
         "engine": profile.engine, "language": profile.language},
    )
    assert response.status_code == 200, (
        f"{key} -> voice '{profile.voice_id}' ({profile.engine}, {profile.language}): "
        f"{response.text[:200]}"
    )
    assert response.headers.get("content-type", "").startswith("audio")
    assert len(response.content) > 1000


def test_text_to_speech_returns_real_audio():
    response = _call(
        settings.puter_tts_interface, settings.puter_tts_driver, settings.puter_tts_method,
        {"text": "Namaste, Moja AI test.", "voice": "Kajal",
         "engine": "neural", "language": "en-IN"},
    )
    assert response.status_code == 200
    assert response.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3"), "not an MP3"


def test_image_generation_returns_a_png():
    response = _call(
        settings.puter_txt2img_interface, settings.puter_txt2img_driver,
        settings.puter_txt2img_method,
        {"prompt": "a glowing fire emoji sticker on a plain background"},
        timeout=240,
    )
    assert response.status_code == 200, response.text[:300]
    assert "image/png" in response.text[:200] or response.content[:4] == b"\x89PNG"


def test_the_inpaint_method_we_configured_is_one_the_driver_has():
    """Puter's image driver exposes generation only - there is no edit method.

    This is why inpainting composites locally instead of asking the API to do
    it: a call to a method that does not exist fails, and a call to 'generate'
    with an image attached silently ignores the image and returns an unrelated
    picture, which is worse.
    """
    response = _call(
        settings.puter_inpaint_interface, settings.puter_inpaint_driver,
        settings.puter_inpaint_method, {"prompt": "test"},
    )
    assert response.status_code != 404, (
        f"'{settings.puter_inpaint_method}' is not a method on "
        f"{settings.puter_inpaint_driver}: {response.text[:200]}"
    )


def test_puter_still_has_no_video_generation():
    """The blueprint assumed wan2.2 through Puter; Puter serves no video driver.

    If this ever starts passing, Puter gained video generation and
    PuterVideoProvider can be switched back on ahead of the local fallback.
    """
    response = _call(
        settings.puter_video_interface, settings.puter_video_driver,
        settings.puter_video_method, {"prompt": "test"},
    )
    assert response.status_code == 404
    assert "Driver not found" in response.text
