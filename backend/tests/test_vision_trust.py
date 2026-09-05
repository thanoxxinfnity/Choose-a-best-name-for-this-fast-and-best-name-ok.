"""What happens when the vision model is confidently wrong."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_analyzer as va  # noqa: E402
from config import settings  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402
from themes import choose_theme  # noqa: E402


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vt") / "clip.mp4"
    subprocess.run(
        [ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=360x640:rate=15:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _payload(model: str) -> dict:
    """The shape of a confident hallucination, taken from a real one."""
    return {
        "content_type": "gaming",
        "subjects": ["gamer", "controller", "screen"],
        "art_style": "screen recording",
        "mood": "exciting",
        "energy": "high",
        "recognisable": "Fortnite",
        "suggested_theme": "gaming",
        "sticker_ideas": ["controller in hand"],
        "text_ideas": ["GAME ON", "LEVEL UP"],
        "summary": "A gamer playing a game on a screen.",
        "_model": model,
    }


def _analyse(clip: Path, model: str):
    with patch.object(va, "describe_with_vision", return_value=_payload(model)):
        return va.analyse_video(clip, nim_api_key="test-key")


def test_the_primary_models_description_is_used_as_written(clip):
    analysis = _analyse(clip, settings.nim_vision_model)
    assert analysis.vision_trusted is True
    assert analysis.content_type == "gaming"
    assert analysis.subjects == ["gamer", "controller", "screen"]
    assert analysis.text_ideas == ["GAME ON", "LEVEL UP"]


def test_a_fallback_description_never_names_subjects_or_writes_copy(clip):
    """This is how an anime edit ended up captioned LEVEL UP."""
    analysis = _analyse(clip, settings.nim_vision_fallback_model)

    assert analysis.vision_trusted is False
    assert analysis.subjects == []
    assert analysis.sticker_ideas == []
    assert analysis.text_ideas == []
    assert analysis.recognisable == ""
    assert analysis.summary == ""


def test_the_measured_signal_survives_the_distrust(clip):
    """Energy and mood are cheap to read; they are not what gets confabulated."""
    analysis = _analyse(clip, settings.nim_vision_fallback_model)
    assert analysis.energy == "high"
    assert analysis.mood == "exciting"
    # And everything that never came from the model at all.
    assert analysis.duration > 0
    assert analysis.width and analysis.height
    assert analysis.motion_curve


def test_the_user_is_told_the_description_was_not_reliable(clip):
    analysis = _analyse(clip, settings.nim_vision_fallback_model)
    assert settings.nim_vision_fallback_model in analysis.vision_error
    assert "not reliable" in analysis.vision_error


def test_a_distrusted_content_type_cannot_steer_the_theme(clip):
    """'gaming' routes straight to the hype theme - on an invented reading."""
    trusted = _analyse(clip, settings.nim_vision_model)
    distrusted = _analyse(clip, settings.nim_vision_fallback_model)

    assert choose_theme(trusted).key == "ae_hype"
    # With the reading discarded, the theme falls back to the measured signal
    # rather than to a content type nobody verified.
    assert distrusted.content_type == "unknown"


def test_the_flag_is_reported_to_the_app(clip):
    analysis = _analyse(clip, settings.nim_vision_fallback_model)
    assert analysis.to_dict()["vision_trusted"] is False
