"""Auto silence-cut, auto beat-sync and auto-reframe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from micro_features import (  # noqa: E402
    ReframeTrack,
    _nearest,
    _smooth,
    _subtract_spans,
    apply_micro_features,
    auto_beat_sync,
    auto_silence_cut,
    plan_reframe,
)
from puter_integration import ffmpeg_binary  # noqa: E402
from schemas import EditPlan, TimelineSegment  # noqa: E402
from video_analyzer import VideoAnalysis, analyse_audio  # noqa: E402

FF = ffmpeg_binary()


def _plan(*windows, source: int = 0) -> EditPlan:
    return EditPlan(edit_timeline=[
        TimelineSegment(start_time=str(a), end_time=str(b), source_index=source)
        for a, b in windows
    ])


def _windows(plan: EditPlan):
    return [(round(s.start_seconds, 2), round(s.end_seconds, 2)) for s in plan.edit_timeline]


# ------------------------------------------------------------ span algebra --

def test_subtract_spans_splits_around_a_hole():
    assert _subtract_spans(0.0, 10.0, [(4.0, 6.0)], 0.0) == [(0.0, 4.0), (6.0, 10.0)]


def test_subtract_spans_trims_an_edge():
    assert _subtract_spans(0.0, 10.0, [(0.0, 3.0)], 0.0) == [(3.0, 10.0)]


def test_subtract_spans_can_remove_everything():
    assert _subtract_spans(2.0, 5.0, [(0.0, 9.0)], 0.0) == []


def test_subtract_spans_padding_keeps_a_breath():
    # A 4s hole with 0.5s padding only removes the middle 3s.
    assert _subtract_spans(0.0, 10.0, [(3.0, 7.0)], 0.5) == [(0.0, 3.5), (6.5, 10.0)]


def test_subtract_spans_ignores_spans_outside_the_window():
    assert _subtract_spans(5.0, 8.0, [(0.0, 2.0), (12.0, 14.0)], 0.0) == [(5.0, 8.0)]


# ----------------------------------------------------------- silence cut ----

def test_auto_silence_cut_splits_a_segment_around_dead_air():
    plan = _plan((0, 10))
    analysis = VideoAnalysis(silence_spans=[(4.0, 6.0)])
    plan, notes = auto_silence_cut(plan, [analysis], keep_padding=0.0)
    assert _windows(plan) == [(0.0, 4.0), (6.0, 10.0)]
    assert notes and "2.0s of dead air" in notes[0]


def test_auto_silence_cut_drops_a_fully_silent_segment():
    plan = _plan((0, 3), (5, 8))
    analysis = VideoAnalysis(silence_spans=[(4.5, 9.0)])
    plan, notes = auto_silence_cut(plan, [analysis], keep_padding=0.0)
    assert _windows(plan) == [(0.0, 3.0)]
    assert notes


def test_auto_silence_cut_ignores_short_gaps():
    plan = _plan((0, 10))
    analysis = VideoAnalysis(silence_spans=[(4.0, 4.2)])
    plan, notes = auto_silence_cut(plan, [analysis], min_silence=0.6)
    assert _windows(plan) == [(0.0, 10.0)]
    assert notes == []


def test_auto_silence_cut_discards_slivers_below_min_segment():
    plan = _plan((0, 10))
    analysis = VideoAnalysis(silence_spans=[(0.3, 6.0)])
    plan, _notes = auto_silence_cut(plan, [analysis], keep_padding=0.0, min_segment=0.5)
    # The 0.0-0.3s sliver is dropped; only the tail survives.
    assert _windows(plan) == [(6.0, 10.0)]


def test_auto_silence_cut_is_per_clip():
    plan = EditPlan(edit_timeline=[
        TimelineSegment(start_time="0", end_time="10", source_index=0),
        TimelineSegment(start_time="0", end_time="10", source_index=1),
    ])
    analyses = [VideoAnalysis(silence_spans=[(4.0, 6.0)]), VideoAnalysis()]
    plan, _notes = auto_silence_cut(plan, analyses, keep_padding=0.0)
    assert _windows(plan) == [(0.0, 4.0), (6.0, 10.0), (0.0, 10.0)]


def test_auto_silence_cut_without_analysis_is_a_no_op():
    plan = _plan((0, 5))
    plan, notes = auto_silence_cut(plan, [])
    assert _windows(plan) == [(0.0, 5.0)] and notes == []


# -------------------------------------------------------------- beat sync ---

def test_nearest_snaps_within_tolerance_only():
    beats = [0.0, 0.5, 1.0, 1.5]
    assert _nearest(beats, 0.55, 0.1) == 0.5
    assert _nearest(beats, 0.72, 0.1) == 0.72   # too far from any beat
    assert _nearest(beats, 1.48, 0.1) == 1.5


def test_auto_beat_sync_pulls_cuts_onto_the_beat():
    plan = _plan((0.06, 2.04))
    analysis = VideoAnalysis(beats=[0.0, 0.5, 1.0, 1.5, 2.0], bpm=120.0)
    plan, notes = auto_beat_sync(plan, [analysis], tolerance=0.15)
    assert _windows(plan) == [(0.0, 2.0)]
    assert notes and "120 BPM" in notes[0]


def test_auto_beat_sync_leaves_deliberate_cuts_alone():
    plan = _plan((0.0, 3.7))
    analysis = VideoAnalysis(beats=[0.0, 0.5, 1.0, 1.5, 2.0])
    plan, notes = auto_beat_sync(plan, [analysis], tolerance=0.15)
    assert _windows(plan) == [(0.0, 3.7)], "no beat is near 3.7s"
    assert notes == []


def test_auto_beat_sync_never_collapses_a_segment():
    plan = _plan((1.02, 1.18))
    analysis = VideoAnalysis(beats=[1.0, 1.2])
    plan, _notes = auto_beat_sync(plan, [analysis], tolerance=0.3, min_segment=0.5)
    assert _windows(plan) == [(1.02, 1.18)], "snapping would have made it too short"


def test_apply_micro_features_runs_silence_then_beats():
    plan = _plan((0, 10))
    analysis = VideoAnalysis(silence_spans=[(4.0, 6.0)], beats=[0.0, 3.9, 6.1, 10.0])
    plan, notes = apply_micro_features(plan, [analysis], silence_cut=True, beat_sync=True)
    # Split at 4/6 by the silence pass, then pulled onto 3.9/6.1 by the beats.
    assert _windows(plan) == [(0.0, 3.9), (6.1, 10.0)]
    assert len(notes) == 2


def test_apply_micro_features_off_changes_nothing():
    plan = _plan((0, 10))
    analysis = VideoAnalysis(silence_spans=[(4.0, 6.0)], beats=[3.9])
    plan, notes = apply_micro_features(plan, [analysis])
    assert _windows(plan) == [(0.0, 10.0)] and notes == []


# --------------------------------------------------------------- smoothing --

def test_smooth_reduces_jitter_without_shifting_the_mean():
    noisy = [100.0, 140.0, 95.0, 145.0, 98.0, 142.0, 101.0]
    smoothed = _smooth(noisy, 0.25)
    assert len(smoothed) == len(noisy)
    assert np.std(smoothed) < np.std(noisy) / 2
    assert abs(np.mean(smoothed) - np.mean(noisy)) < 12


def test_smooth_handles_the_degenerate_cases():
    assert _smooth([], 0.3) == []
    assert _smooth([5.0], 0.3) == [5.0]


# ------------------------------------------------------------- reframe -----

def test_reframe_track_interpolates_and_clamps():
    track = ReframeTrack(times=[0.0, 2.0], centres_x=[100.0, 300.0],
                         centres_y=[50.0, 50.0], width=640, height=360)
    assert track.centre_at(1.0)[0] == pytest.approx(200.0)
    assert track.centre_at(-5.0)[0] == 100.0     # before the first sample
    assert track.centre_at(99.0)[0] == 300.0     # after the last


@pytest.mark.slow
def test_plan_reframe_follows_a_moving_subject(tmp_path):
    """A white box slides left to right across a wide frame; the crop must follow."""
    path = tmp_path / "pan.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:size=960x540:rate=24:duration=4",
         "-f", "lavfi", "-i", "color=c=white:size=120x120:rate=24:duration=4",
         "-filter_complex", "[0][1]overlay=x='(W-w)*t/4':y=(H-h)/2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    track = plan_reframe(path, 9 / 16, sample_fps=6.0)
    assert track is not None, "a 16:9 source reframed to 9:16 must produce a track"
    assert track.width == 960 and track.height == 540

    early = track.centre_at(0.4)[0]
    late = track.centre_at(3.4)[0]
    assert late > early + 100, f"the crop should pan right: {early:.0f} -> {late:.0f}"

    half = min(track.width, int(track.height * 9 / 16)) / 2
    assert all(half - 1 <= x <= track.width - half + 1 for x in track.centres_x), \
        "the window must stay inside the frame"


@pytest.mark.slow
def test_plan_reframe_skips_already_vertical_footage(tmp_path):
    path = tmp_path / "tall.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=360x640:rate=24:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    assert plan_reframe(path, 9 / 16) is None, "nothing to pan across"


def test_plan_reframe_on_a_missing_file_returns_none(tmp_path):
    assert plan_reframe(tmp_path / "nope.mp4", 9 / 16) is None


# ------------------------------------------------------ audio integration ---

@pytest.mark.slow
def test_silence_analysis_feeds_the_cut(tmp_path):
    """End to end: a real clip with a gap in the middle loses the gap."""
    path = tmp_path / "gap.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-filter_complex", "[1][2][3]concat=n=3:v=0:a=1[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    audio = analyse_audio(path)
    spans = audio["silence_spans"]
    assert spans, "the 2s gap should be detected"
    assert any(1.7 < a < 2.6 and 3.4 < b < 4.4 for a, b in spans), spans

    plan = _plan((0, 6))
    plan, notes = auto_silence_cut(plan, [VideoAnalysis(silence_spans=spans)], keep_padding=0.1)
    assert len(plan.edit_timeline) == 2, _windows(plan)
    assert notes
