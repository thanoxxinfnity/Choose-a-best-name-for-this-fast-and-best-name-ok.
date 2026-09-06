"""The hold at the end of an edit, and what it is allowed to cost.

A voice line placed past the last cut used to buy itself unlimited frozen
frame: the renderer held the final image until the line finished. On a real
edit that was three seconds of a motion-blurred still, which reads as a crash
rather than as an ending. The hold is now rationed, and the mix follows the
picture instead of the other way round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_renderer  # noqa: E402
from schemas import EditPlan, ProjectMeta, TimelineSegment  # noqa: E402
from video_renderer import MAX_FREEZE_TAIL, VideoRenderer  # noqa: E402


def _renderer(tmp_path) -> VideoRenderer:
    plan = EditPlan(
        project_meta=ProjectMeta(resolution="720x1280", fps=30),
        edit_timeline=[TimelineSegment(start_time="00:00:00", end_time="00:00:02",
                                       source_index=0)],
    )
    return VideoRenderer(
        plan=plan, clip_paths=[tmp_path / "nothing.mp4"],
        workspace=tmp_path / "ws", puter=None, export="720p60",
    )


def _colour(seconds: float):
    from moviepy import ColorClip  # noqa: PLC0415

    clip = ColorClip((64, 64), color=(20, 30, 40))
    clip = video_renderer._with_duration(clip, seconds)
    return video_renderer._with_fps(clip, 30)


def test_a_short_tail_is_held_in_full(tmp_path):
    """Half a second keeps the last word; that is what the hold is for."""
    renderer = _renderer(tmp_path)
    held = renderer._extend_to(_colour(2.0), 2.4)
    assert held.duration == pytest.approx(2.4, abs=0.05)
    assert not renderer.warnings


def test_a_long_tail_is_capped_and_said_out_loud(tmp_path):
    renderer = _renderer(tmp_path)
    held = renderer._extend_to(_colour(2.0), 5.0)

    assert held.duration == pytest.approx(2.0 + MAX_FREEZE_TAIL, abs=0.05)
    assert any("past the last cut" in warning for warning in renderer.warnings)


def test_a_tail_worth_nothing_leaves_the_clip_alone(tmp_path):
    renderer = _renderer(tmp_path)
    clip = _colour(2.0)
    assert renderer._extend_to(clip, 2.01) is clip
