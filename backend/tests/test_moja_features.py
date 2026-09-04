"""Tests for the Moja AI feature modules: analysis, stickers, VFX, themes, i2v."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puter_integration import ffmpeg_binary  # noqa: E402
from puter_video import (  # noqa: E402
    IMAGE_TO_VIDEO_MODEL,
    TEXT_TO_VIDEO_MODEL,
    PuterVideoClient,
    build_animation_prompt,
    extract_keyframe,
)
from sticker_art import SHAPES, draw_sticker, pick_palette, pick_shape  # noqa: E402
from themes import THEMES, resolve_theme  # noqa: E402
from vfx import (  # noqa: E402
    ShakeSpec,
    build_effect_chain,
    colour_grade,
    directional_blur,
    rgb_split,
    shake_at,
    shift_frame,
    vignette,
)
from video_analyzer import (  # noqa: E402
    VideoAnalysis,
    analyse_audio,
    build_contact_sheet,
    scan_visual_timeline,
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """A short clip with two hard scene changes and a beeping soundtrack."""
    directory = tmp_path_factory.mktemp("analysis")
    path = directory / "clip.mp4"
    subprocess.run(
        [
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=360x640:rate=24:duration=2",
            "-f", "lavfi", "-i", "smptebars=size=360x640:rate=24:duration=2",
            "-f", "lavfi", "-i", "color=c=black:size=360x640:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=500:duration=6",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


# ------------------------------------------------------------------ analysis --

def test_scan_visual_timeline_finds_the_scene_changes(clip):
    result = scan_visual_timeline(clip)
    cuts = result["scene_cuts"]
    assert cuts, "the two hard cuts should be detected"
    # Cuts sit near 2s and 4s.
    assert any(1.5 < cut < 2.6 for cut in cuts)
    assert any(3.5 < cut < 4.6 for cut in cuts)
    assert 0.0 <= result["brightness"] <= 1.0
    assert all(colour.startswith("#") and len(colour) == 7 for colour in result["palette"])


def test_analyse_audio_reports_a_tone_as_continuous(clip):
    result = analyse_audio(clip)
    # A constant sine is loud throughout, so almost nothing is silent.
    assert result["speech_ratio"] > 0.8
    assert isinstance(result["beats"], list)
    assert result["bpm"] >= 0.0


def test_contact_sheet_respects_the_pixel_budget():
    frames = [(float(i), np.full((640, 360, 3), i * 40, dtype=np.uint8)) for i in range(4)]
    payload = build_contact_sheet(frames, cell=224, max_pixels=190_000)
    assert payload and payload[:2] == b"\xff\xd8"  # JPEG magic
    with Image.open(__import__("io").BytesIO(payload)) as sheet:
        assert sheet.width * sheet.height <= 190_000 * 1.02


def test_contact_sheet_of_nothing_is_none():
    assert build_contact_sheet([]) is None


def test_analysis_prompt_block_carries_the_findings():
    analysis = VideoAnalysis(
        duration=16.0, content_type="anime_edit", subjects=["girl", "monster"],
        art_style="2D anime", mood="dark fantasy", energy="high",
        scene_cuts=[4.7, 6.3], high_motion_moments=[4.7], beats=[0.5, 1.0], bpm=172.0,
        sticker_ideas=["glowing sword slash"], text_ideas=["FIGHT"],
        suggested_theme="anime_edits", summary="an anime fight",
    )
    block = analysis.to_prompt_block()
    for expected in ("anime_edit", "2D anime", "girl", "FIGHT", "172", "anime_edits", "4.7s"):
        assert expected in block


def test_local_fallback_classifies_without_a_vision_model():
    from video_analyzer import _apply_local_fallback

    dark = VideoAnalysis(duration=10.0, brightness=0.12,
                         motion_curve=[(t / 2, 0.02) for t in range(20)])
    _apply_local_fallback(dark)
    assert dark.suggested_theme == "haunted"

    busy = VideoAnalysis(duration=10.0, brightness=0.5,
                         motion_curve=[(t / 2, 0.20) for t in range(20)])
    _apply_local_fallback(busy)
    assert busy.energy == "high" and busy.suggested_theme == "anime_edits"


# ------------------------------------------------------------------ stickers --

@pytest.mark.parametrize(
    "prompt,shape",
    [
        ("glowing energy sword slash", "sword"),
        ("anime explosion impact", "explosion"),
        ("cartoon fire flame", "flame"),
        ("lightning bolt speed lines", "bolt"),
        ("cursed demon skull", "skull"),
        ("3D glowing subscribe button", "play"),
        ("something entirely unrelated", "burst"),
    ],
)
def test_pick_shape(prompt, shape):
    assert pick_shape(prompt) == shape


def test_every_shape_renders_a_transparent_png(tmp_path):
    for name in SHAPES:
        path = draw_sticker("test", tmp_path / f"{name}.png", size=128, shape=name)
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            alpha = np.array(image)[:, :, 3]
            assert alpha.max() == 255, f"{name} drew nothing opaque"
            assert alpha.min() == 0, f"{name} has no transparent margin"


def test_sticker_rendering_is_fast_enough_for_a_render(tmp_path):
    """A sticker used to take ~75s because of PIL's MaxFilter; it must not regress."""
    started = time.time()
    draw_sticker("glowing sword slash", tmp_path / "s.png", size=512)
    assert time.time() - started < 15.0


def test_palette_follows_the_prompt():
    assert pick_palette("golden crown")[1] != pick_palette("green leaf")[1]
    assert pick_palette("blue lightning")[1] == pick_palette("electric energy")[1]


# ----------------------------------------------------------------------- vfx --

def _frame(height: int = 64, width: int = 36) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


def test_shift_preserves_shape_and_dtype():
    frame = _frame()
    out = shift_frame(frame, 5.0, -3.0, zoom=1.08)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_rgb_split_moves_only_the_outer_channels():
    frame = _frame()
    out = rgb_split(frame, 4.0)
    assert not np.array_equal(out[:, :, 0], frame[:, :, 0])
    assert np.array_equal(out[:, :, 1], frame[:, :, 1])
    assert out.shape == frame.shape


def test_rgb_split_below_threshold_is_a_no_op():
    frame = _frame()
    assert np.array_equal(rgb_split(frame, 0.2), frame)


def test_directional_blur_reduces_local_variance():
    frame = _frame(96, 96)
    blurred = directional_blur(frame, 9.0, 0.0)
    assert blurred.std() < frame.std()


def test_colour_grade_saturation_extremes():
    frame = _frame()
    grey = colour_grade(frame, saturation=0.0)
    assert np.allclose(grey[:, :, 0], grey[:, :, 1], atol=2)
    assert not np.array_equal(colour_grade(frame, contrast=1.5), frame)


def test_vignette_darkens_corners_more_than_the_centre():
    frame = np.full((128, 128, 3), 200, dtype=np.uint8)
    out = vignette(frame, 0.6)
    assert out[0, 0].mean() < out[64, 64].mean()


def test_directional_shake_decays_after_a_hit():
    spec = ShakeSpec(kind="directional", intensity=20.0, hits=(1.0,), decay=0.2)
    assert shake_at(spec, 0.5) == (0.0, 0.0, 1.0, 0.0)      # before the hit
    at_hit = shake_at(spec, 1.005)
    assert abs(at_hit[0]) + abs(at_hit[1]) > 1.0            # kicks
    assert shake_at(spec, 1.5)[:2] == (0.0, 0.0)            # settled


def test_pulse_shake_only_zooms():
    spec = ShakeSpec(kind="pulse", intensity=1.0, hits=(0.5,), decay=0.25, zoom=0.08)
    dx, dy, zoom, _split = shake_at(spec, 0.51)
    assert (dx, dy) == (0.0, 0.0) and zoom > 1.0


def test_effect_chain_is_none_when_nothing_is_configured():
    assert build_effect_chain() is None


def test_effect_chain_runs_every_stage():
    spec = ShakeSpec(kind="directional", intensity=8.0, hits=(0.2,), rgb_split=4.0,
                     motion_blur=True, zoom=0.05)
    chain = build_effect_chain(spec, {"saturation": 1.3}, 0.4, 0.3)
    frame = _frame(120, 68)
    out = chain(frame, 0.21)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not np.array_equal(out, frame)


# -------------------------------------------------------------------- themes --

def test_every_theme_is_coherent():
    for key, theme in THEMES.items():
        assert theme.key == key
        assert theme.segment_seconds[0] < theme.segment_seconds[1]
        assert theme.prompt_hint()
        spec = theme.shake_spec([1.0, 2.0])
        assert (spec is None) == (theme.shake_kind == "none")


def test_theme_aliases_and_default():
    assert resolve_theme("anime").key == "anime_edits"
    assert resolve_theme("HORROR").key == "haunted"
    assert resolve_theme("Playful Mode").key == "playful"
    assert resolve_theme(None).key == "normal"
    assert resolve_theme("who knows").key == "normal"


def test_shake_intensity_scales_with_width():
    theme = THEMES["anime_edits"]
    assert theme.shake_spec([1.0], width=2160).intensity == pytest.approx(
        theme.shake_spec([1.0], width=1080).intensity * 2
    )


# ---------------------------------------------------------------- video / i2v --

def test_video_models_match_the_spec():
    assert TEXT_TO_VIDEO_MODEL == "wan-ai/wan2.2-t2v-a14b"
    assert IMAGE_TO_VIDEO_MODEL == "wan-ai/wan2.2-i2v-a14b"


def test_job_handle_detection():
    assert PuterVideoClient._job_handle({"result": {"job_id": "j1"}}) == "j1"
    # A payload that already carries media is finished, not a job.
    assert PuterVideoClient._job_handle({"result": {"url": "https://x/y.mp4", "id": "j1"}}) is None
    assert PuterVideoClient._job_handle({"result": "https://x/y.mp4"}) is None


def test_status_parsing():
    assert PuterVideoClient._status_of({"result": {"status": "SUCCEEDED"}}) == "succeeded"
    assert PuterVideoClient._status_of({"state": "failed"}) == "failed"
    assert PuterVideoClient._status_of("nonsense") == ""


def test_animation_prompt_folds_in_the_story_context():
    prompt = build_animation_prompt(
        "slow push in", "hero powers up", ["Gojo"], "2D anime", "dark fantasy",
    )
    for expected in ("slow push in", "Gojo", "2D anime", "dark fantasy", "hero powers up"):
        assert expected in prompt


def test_extract_keyframe_writes_a_still(clip, tmp_path):
    path = extract_keyframe(clip, 3.0, tmp_path / "kf.png")
    with Image.open(path) as image:
        assert image.size[0] > 0 and image.size[1] > 0


def test_extract_keyframe_past_the_end_falls_back(clip, tmp_path):
    path = extract_keyframe(clip, 9999.0, tmp_path / "kf2.png")
    assert path.exists()


def test_video_client_without_a_key_refuses_clearly(tmp_path):
    from puter_integration import PuterError

    client = PuterVideoClient(api_key="")
    with pytest.raises(PuterError, match="Puter.js API key"):
        client.text_to_video("a cat", tmp_path / "out.mp4")
