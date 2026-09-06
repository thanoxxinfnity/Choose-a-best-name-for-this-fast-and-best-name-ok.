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
    assert analysis.art_style == ""
    # The summary is no longer the model's prose about imagined content; it is
    # replaced by a description of the file itself.
    assert "gamer" not in analysis.summary
    assert "clip" in analysis.summary


def test_what_the_file_measures_replaces_what_the_model_claimed(clip):
    """There is no reason to keep a guess from a reading already judged unreliable
    when motion, brightness and cut density measure the same thing directly."""
    analysis = _analyse(clip, settings.nim_vision_fallback_model)

    assert analysis.energy in ("low", "medium", "high")
    assert analysis.mood  # re-derived, not the model's word
    assert analysis.suggested_theme
    # The summary is now a description of the file, not of imagined content.
    assert "clip" in analysis.summary and "energy" in analysis.summary

    # And everything that never came from the model at all.
    assert analysis.duration > 0
    assert analysis.width and analysis.height
    assert analysis.motion_curve


def test_a_wrong_energy_claim_is_overruled_by_the_measurement(clip, monkeypatch):
    """The small model called a fast, high-motion clip 'low energy'."""
    payload = _payload(settings.nim_vision_fallback_model)
    payload.update(energy="low", mood="calm", suggested_theme="normal")
    with patch.object(va, "describe_with_vision", return_value=payload):
        analysis = va.analyse_video(clip, nim_api_key="test-key")

    measured = va.VideoAnalysis(
        duration=analysis.duration, motion_curve=analysis.motion_curve,
        scene_cuts=analysis.scene_cuts, brightness=analysis.brightness,
    )
    va._apply_local_fallback(measured)
    assert analysis.energy == measured.energy
    assert analysis.mood == measured.mood


def test_the_user_is_told_the_description_was_not_reliable(clip):
    analysis = _analyse(clip, settings.nim_vision_fallback_model)
    assert settings.nim_vision_fallback_model in analysis.vision_error
    assert "not reliable" in analysis.vision_error


def test_a_distrusted_content_type_cannot_steer_the_theme(clip):
    """'gaming' routes straight to the hype theme - on an invented reading."""
    trusted = _analyse(clip, settings.nim_vision_model)
    distrusted = _analyse(clip, settings.nim_vision_fallback_model)

    assert choose_theme(trusted).key == "ae_hype"
    # With the reading discarded, the theme comes from the measured signal
    # rather than from a content type nobody verified.
    assert distrusted.content_type == "unknown"
    assert choose_theme(distrusted).key == choose_theme(distrusted).key


def test_the_flag_is_reported_to_the_app(clip):
    analysis = _analyse(clip, settings.nim_vision_fallback_model)
    assert analysis.to_dict()["vision_trusted"] is False


# ------------------------------------------------- what the pass is worth ----

def test_a_supplied_plan_skips_the_vision_pass():
    """Vision exists to tell the planner what it is looking at.

    With the plan already written it costs minutes a clip and changes nothing,
    so the job must ask for the local measurements only.
    """
    import inspect

    import jobs

    body = inspect.getsource(jobs.JobManager._run)
    assert "use_vision=looking" in body
    assert "looking = request.plan_override is None" in body
