"""Offline tests for the parts of the pipeline that do not need an API key.

    cd backend && python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from orchestrator import (  # noqa: E402
    KimiOrchestrator,
    build_fallback_plan,
    extract_json,
    extract_youtube_id,
    sanitise_plan,
)
from puter_integration import PuterClient, _chunk_text, build_mask  # noqa: E402
from schemas import ClipInfo, EditPlan, format_timecode, parse_timecode  # noqa: E402
from video_renderer import anchor_for, render_text_rgba  # noqa: E402


# ---------------------------------------------------------------- timecodes --

@pytest.mark.parametrize(
    "value,expected",
    [
        ("00:00:05", 5.0),
        ("00:01:30", 90.0),
        ("01:02:03", 3723.0),
        ("00:00:02.500", 2.5),
        ("1:05", 65.0),
        (12, 12.0),
        ("7.25", 7.25),
        ("", 0.0),
        (None, 0.0),
        ("garbage", 0.0),
    ],
)
def test_parse_timecode(value, expected):
    assert parse_timecode(value) == pytest.approx(expected)


def test_format_timecode_roundtrip():
    assert format_timecode(3723) == "01:02:03"
    assert parse_timecode(format_timecode(95.4)) == 95.0


# --------------------------------------------------------------- JSON parsing --

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_extract_json_with_prose_and_braces_in_strings():
    raw = 'Sure! Here you go:\n{"tts_script": "use { and } freely"}\nHope that helps.'
    assert extract_json(raw) == {"tts_script": "use { and } freely"}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json at all")


# -------------------------------------------------------------------- plans --

def _clips() -> list[ClipInfo]:
    return [
        ClipInfo(filename="a.mp4", path="a.mp4", duration=12.0, width=1920, height=1080, fps=30.0),
        ClipInfo(filename="b.mp4", path="b.mp4", duration=6.0, width=1080, height=1920, fps=60.0),
    ]


def test_blueprint_example_validates():
    """The exact JSON shape from the blueprint must load without changes."""
    payload = {
        "project_meta": {"resolution": "1080x1920", "fps": 60},
        "audio": {
            "use_puter_tts": True,
            "voice_accent": "indian_accent",
            "tts_script": "Doston, aaj hum dekhenge...",
        },
        "edit_timeline": [
            {
                "start_time": "00:00:00",
                "end_time": "00:00:05",
                "cut_type": "jump_cut",
                "puter_sticker": {
                    "generate_prompt": "3D glowing subscribe button",
                    "position": "bottom_center",
                    "animation": "pop_up",
                },
                "puter_inpaint": {
                    "active": True,
                    "target_object": "sky",
                    "replace_prompt": "dark stormy sky with lightning",
                },
            }
        ],
    }
    plan = EditPlan.model_validate(payload)
    assert plan.project_meta.size == (1080, 1920)
    assert plan.audio.has_voiceover
    assert plan.edit_timeline[0].duration == 5.0
    assert plan.edit_timeline[0].puter_sticker.active
    assert plan.edit_timeline[0].puter_inpaint.is_enabled


def test_sanitise_clamps_out_of_range_segments():
    payload = {
        "edit_timeline": [
            # Starts past the end of a 12 s clip.
            {"start_time": "00:00:40", "end_time": "00:00:50", "source_index": 0},
            # 30 s long -> clamped to 8 s.
            {"start_time": "00:00:00", "end_time": "00:00:30", "source_index": 1},
            # Bogus index -> repaired.
            {"start_time": "00:00:01", "end_time": "00:00:03", "source_index": 99},
        ]
    }
    plan, _warnings = sanitise_plan(EditPlan.model_validate(payload), _clips())
    for segment, clip in ((s, _clips()[s.source_index]) for s in plan.edit_timeline):
        assert 0 <= segment.start_seconds < clip.duration
        assert segment.end_seconds <= clip.duration
        assert 0.4 <= segment.duration <= 8.0


def test_sanitise_enforces_asset_budget():
    segments = [
        {
            "start_time": "00:00:00",
            "end_time": "00:00:02",
            "source_index": 0,
            "puter_sticker": {"generate_prompt": f"sticker {index}"},
            "puter_inpaint": {"active": True, "replace_prompt": "stormy sky"},
        }
        for index in range(8)
    ]
    plan, _ = sanitise_plan(
        EditPlan.model_validate({"edit_timeline": segments}), _clips(),
        max_stickers=2, max_inpaints=1,
    )
    assert sum(1 for s in plan.edit_timeline if s.puter_sticker) == 2
    assert sum(1 for s in plan.edit_timeline if s.puter_inpaint) == 1


def test_sanitise_disables_tts_without_a_script():
    plan = EditPlan.model_validate({"audio": {"use_puter_tts": True, "tts_script": "   "}})
    plan, warnings = sanitise_plan(plan, _clips())
    assert plan.audio.use_puter_tts is False
    assert any("TTS" in warning for warning in warnings)


def test_fallback_plan_is_renderable():
    plan = build_fallback_plan("hindi voiceover reel about coding", _clips(), None, 20.0)
    assert plan.edit_timeline
    assert plan.audio.use_puter_tts and plan.audio.tts_script
    assert plan.project_meta.resolution == "1080x1920"
    assert plan.project_meta.fps == 60
    for segment in plan.edit_timeline:
        clip = _clips()[segment.source_index]
        assert 0 <= segment.start_seconds < segment.end_seconds <= clip.duration


def test_fallback_plan_without_voiceover_request():
    plan = build_fallback_plan("just fast cuts, no talking", _clips(), None, 15.0)
    assert plan.audio.use_puter_tts is False


def test_orchestrator_without_key_falls_back():
    orchestrator = KimiOrchestrator(api_key="")
    plan, warnings = orchestrator.build_plan("make a reel", _clips())
    assert plan.edit_timeline
    assert any("NVIDIA NIM" in warning for warning in warnings)


# ----------------------------------------------------------------- youtube --

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/nope", None),
    ],
)
def test_extract_youtube_id(url, expected):
    assert extract_youtube_id(url) == expected


# ------------------------------------------------------------------- puter --

def test_voice_catalogue_is_indian_by_default():
    for accent in ("indian_accent", "INDIAN-ACCENT", "india", "", "unknown"):
        voice = PuterClient.resolve_voice(accent)
        assert voice.language == "en-IN"


def test_chunk_text_respects_the_limit():
    script = ("Doston, aaj hum dekhenge kuch naya. " * 300).strip()
    chunks = _chunk_text(script, 2800)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2800 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == script.replace(" ", "")


def test_chunk_text_splits_a_single_giant_sentence():
    chunks = _chunk_text("x" * 7000, 2800)
    assert all(len(chunk) <= 2800 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == 7000


def test_find_media_value_handles_every_envelope():
    finder = PuterClient._find_media_value
    assert finder({"result": {"url": "https://cdn.puter.com/a.png"}}) == "https://cdn.puter.com/a.png"
    assert finder({"result": [{"b64_json": "A" * 300}]}) == "A" * 300
    assert finder({"success": True, "result": "data:image/png;base64,AAAA"}) == "data:image/png;base64,AAAA"
    assert finder({"nothing": "useful"}) is None


def test_build_mask_targets_the_sky(tmp_path):
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    frame[:150, :] = (220, 140, 40)   # BGR blue sky
    frame[150:, :] = (40, 90, 40)     # green ground
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), frame)

    mask = cv2.imread(
        str(build_mask(frame_path, tmp_path / "mask.png", target_object="sky")),
        cv2.IMREAD_GRAYSCALE,
    )
    assert mask.shape == (400, 300)
    assert mask[:120, :].mean() > 200   # sky selected
    assert mask[300:, :].mean() < 40    # ground untouched


def test_build_mask_region_hints(tmp_path):
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    frame_path = tmp_path / "flat.png"
    cv2.imwrite(str(frame_path), frame)

    left = cv2.imread(
        str(build_mask(frame_path, tmp_path / "left.png", region="left")), cv2.IMREAD_GRAYSCALE
    )
    assert left[:, :80].mean() > 200 and left[:, 130:].mean() < 40


# ---------------------------------------------------------------- rendering --

def test_render_text_rgba_produces_visible_pixels():
    array = render_text_rgba("JUMP CUT", max_width=900, font_size=72, three_d=True)
    assert array.ndim == 3 and array.shape[2] == 4
    assert array[:, :, 3].max() == 255       # opaque glyphs
    assert array.shape[1] <= 900 + 64


def test_render_text_rgba_wraps_long_text():
    single = render_text_rgba("SHORT", max_width=600, font_size=64)
    wrapped = render_text_rgba("THIS HEADLINE IS FAR TOO LONG FOR ONE LINE", max_width=600, font_size=64)
    assert wrapped.shape[0] > single.shape[0]
    assert wrapped.shape[1] <= 600 + 64


def test_anchor_lookup_is_forgiving():
    assert anchor_for("bottom_center") == anchor_for("BOTTOM-CENTER") == anchor_for("Bottom Center")
    assert anchor_for("nonsense") == anchor_for("bottom_center")
    assert anchor_for("center") == (0.5, 0.5)


def test_plan_serialises_to_the_blueprint_shape():
    plan = build_fallback_plan("voiceover reel", _clips(), None, 12.0)
    payload = json.loads(plan.model_dump_json())
    assert set(payload) >= {"project_meta", "audio", "edit_timeline"}
    assert set(payload["project_meta"]) >= {"resolution", "fps"}
    assert set(payload["audio"]) >= {"use_puter_tts", "voice_accent", "tts_script"}
    assert set(payload["edit_timeline"][0]) >= {"start_time", "end_time", "cut_type"}
