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


def test_the_video_driver_we_configured_is_the_one_that_exists():
    """The driver name was wrong for most of this project's life.

    Probing found "Driver not found: puter-video-generation:wan-ai" and that
    was read as "there is no video interface" - but the error names the
    *pair*, and the interface was there all along under a different driver.
    A valid pair answers "Method not found" for a nonsense method; an invalid
    one says "Driver not found". That is the distinction this asserts.
    """
    response = _call(
        settings.puter_video_interface, settings.puter_video_driver,
        "__no_such_method__", {},
    )
    assert response.status_code == 404
    assert "Driver not found" not in response.text, (
        f"{settings.puter_video_interface}:{settings.puter_video_driver} "
        f"is not a real driver pair: {response.text[:200]}"
    )
    assert "Method" in response.text


def test_video_generation_answers_in_test_mode():
    """test_mode proves the endpoint is reachable without spending credits.

    It proves only that. The same canned sample comes back for a model name
    that does not exist, so it cannot be used to check which models or which
    arguments actually work - only a paid call can.
    """
    response = _call(
        settings.puter_video_interface, settings.puter_video_driver,
        settings.puter_video_method,
        {"prompt": "a slow drift through dark clouds",
         "model": settings.puter_t2v_model, "seconds": 4, "test_mode": True},
        timeout=240,
    )
    assert response.status_code == 200, response.text[:300]
    assert response.json().get("success") is True


def test_generation_still_needs_a_prompt():
    """A guard that the driver validates anything at all in test mode."""
    response = _call(
        settings.puter_video_interface, settings.puter_video_driver,
        settings.puter_video_method, {"test_mode": True},
    )
    assert response.status_code == 400
    assert "prompt" in response.text
