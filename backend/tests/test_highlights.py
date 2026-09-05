"""Finding the short inside a long source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import highlights  # noqa: E402
from highlights import (  # noqa: E402
    Highlight,
    _rise,
    find_highlights,
    loudness_per_second,
    motion_per_second,
    pick_windows,
    score_seconds,
    snap_to_cuts,
)
from puter_integration import ffmpeg_binary  # noqa: E402

FF = ffmpeg_binary()


# ------------------------------------------------------------- scoring -----

def test_a_sudden_loud_moment_outscores_a_sustained_one():
    """A loud stretch is a music bed; a loud jump is an event.

    A plain loudness score cannot tell them apart, which is exactly how a
    highlight finder ends up recommending the credits music.
    """
    sustained = np.full(40, 0.8, dtype=np.float32)
    sudden = np.concatenate([np.full(20, 0.1), np.full(20, 0.8)]).astype(np.float32)
    assert float(_rise(sudden).max()) > float(_rise(sustained).max()) + 0.3


def test_the_score_needs_both_signals_but_survives_losing_one():
    quiet = np.zeros(60, dtype=np.float32)
    busy = np.concatenate([np.zeros(30), np.ones(30)]).astype(np.float32)
    score = score_seconds(quiet, busy)
    assert score.size == 60
    assert float(score[45]) > float(score[10])


def test_no_signal_means_no_score():
    assert score_seconds(np.zeros(0), np.zeros(0)).size == 0


def test_shot_changes_lift_the_score_where_they_cluster():
    flat = np.full(60, 0.3, dtype=np.float32)
    dense = [float(t) for t in range(40, 50)]
    with_cuts = score_seconds(flat, flat, dense)
    without = score_seconds(flat, flat, [])
    assert float(with_cuts[45]) > float(without[45])


# ------------------------------------------------------------- windows -----

def test_the_best_window_is_the_densest_stretch():
    score = np.zeros(120, dtype=np.float32)
    score[60:90] = 1.0
    windows = pick_windows(score, window=30, count=1)
    assert len(windows) == 1
    start, end, _mean = windows[0]
    assert 55 <= start <= 62
    assert end - start == 30


def test_picked_windows_never_overlap():
    """Two overlapping highlights are one highlight reported twice."""
    score = np.zeros(300, dtype=np.float32)
    score[50:80] = 1.0
    score[200:230] = 0.9
    windows = pick_windows(score, window=30, count=3, min_gap=15)
    spans = [(s, e) for s, e, _ in windows]
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        assert a_end <= b_start or b_end <= a_start


def test_a_source_shorter_than_the_window_yields_nothing():
    assert pick_windows(np.ones(10, dtype=np.float32), window=30, count=2) == []


def test_asking_for_no_windows_returns_none():
    assert pick_windows(np.ones(100, dtype=np.float32), window=10, count=0) == []


# --------------------------------------------------------------- snapping --

def test_the_clip_opens_on_a_shot_change_when_one_is_near():
    start, end = snap_to_cuts(10.4, 45.6, [3.0, 10.0, 46.0, 90.0])
    assert (start, end) == (10.0, 46.0)


def test_a_distant_cut_does_not_drag_the_window():
    assert snap_to_cuts(10.0, 45.0, [0.0, 80.0]) == (10.0, 45.0)


def test_snapping_never_collapses_the_window():
    """Two cuts on top of each other must not produce a zero-length clip."""
    assert snap_to_cuts(10.0, 10.9, [10.0, 10.05]) == (10.0, 10.9)


def test_no_cuts_leaves_the_window_alone():
    assert snap_to_cuts(5.0, 20.0, []) == (5.0, 20.0)


# ------------------------------------------------------------ end to end ---

@pytest.fixture(scope="module")
def long_source(tmp_path_factory) -> Path:
    """Three minutes of quiet still footage with one loud, busy 30s in it."""
    path = tmp_path_factory.mktemp("long") / "movie.mp4"
    subprocess.run(
        [
            FF, "-y", "-hide_banner", "-loglevel", "error",
            # A near-static picture for the whole runtime...
            "-f", "lavfi", "-i", "color=c=gray:size=320x180:rate=10:duration=180",
            # ...an obviously busy one to splice into the middle...
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=180",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=180",
            "-filter_complex",
            # Swap to the busy source for 100s-130s, and open the volume with it.
            "[0:v][1:v]overlay=enable='between(t,100,130)'[v];"
            "[2:a]volume='if(between(t,100,130),1.0,0.02)':eval=frame[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-t", "180", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def test_the_loud_busy_stretch_is_what_gets_found(long_source):
    report = find_highlights(long_source, duration=180.0, target_seconds=25.0, count=2)
    assert report.searched and not report.error
    assert report.highlights

    best = report.highlights[0]
    # The planted moment runs 100s-130s; the top pick must land inside it.
    assert 90.0 <= best.start <= 125.0, report.to_dict()
    assert best.end <= 140.0
    assert best.reasons


def test_the_audio_pass_hears_the_loud_stretch(long_source):
    loudness = loudness_per_second(long_source, FF)
    assert loudness.size > 150
    assert float(loudness[100:130].mean()) > float(loudness[0:90].mean()) * 3


def test_the_video_pass_sees_the_busy_stretch(long_source):
    motion, cuts = motion_per_second(long_source, 180.0, FF)
    assert motion.size > 150
    assert float(motion[100:130].mean()) > float(motion[0:90].mean())


def test_a_source_that_is_already_short_is_returned_whole(tmp_path):
    clip = tmp_path / "short.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=20",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    report = find_highlights(clip, duration=20.0, target_seconds=30.0)
    assert not report.searched
    assert len(report.highlights) == 1
    assert report.highlights[0].start == 0.0
    assert report.highlights[0].end == pytest.approx(20.0)


def test_an_unreadable_source_reports_instead_of_raising(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    report = find_highlights(broken, duration=600.0, target_seconds=30.0)
    assert report.error
    assert report.highlights == []


def test_a_highlight_serialises_for_the_app():
    payload = Highlight(10.0, 45.5, 0.812, ["loud"]).to_dict()
    assert payload["duration"] == 35.5
    assert payload["score"] == 0.812
    assert payload["reasons"] == ["loud"]


# ------------------------------------------------- cutting the window out ---

def test_the_window_is_cut_on_the_frame_asked_for(long_source, tmp_path):
    """A stream copy would snap the start back to the previous keyframe, which
    on a film is up to ten seconds of the wrong shot."""
    from jobs import _cut_window
    from video_renderer import probe_clip

    destination = tmp_path / "window.mp4"
    assert _cut_window(long_source, destination, 100.0, 130.0)

    info = probe_clip(destination)
    assert info["duration"] == pytest.approx(30.0, abs=0.6)
    assert info["has_audio"] is True


def test_cutting_an_unreadable_source_reports_instead_of_raising(tmp_path):
    from jobs import _cut_window

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert _cut_window(broken, tmp_path / "out.mp4", 0.0, 5.0) is False


# ------------------------------------------------------------ the API -------

def test_the_highlights_endpoint_returns_the_candidates(long_source):
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as client:
        with long_source.open("rb") as handle:
            response = client.post(
                "/api/v1/highlights",
                files={"video": ("movie.mp4", handle, "video/mp4")},
                data={"target_seconds": "25", "count": "2"},
            )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["searched"] is True
    assert payload["highlights"]
    top = payload["highlights"][0]
    assert 90.0 <= top["start"] <= 125.0, payload
    assert top["reasons"]


def test_the_sfx_catalogue_endpoint_lists_the_library():
    from fastapi.testclient import TestClient

    import main
    import sfx as sfx_module

    with TestClient(main.app) as client:
        payload = client.get("/api/v1/sfx").json()
    assert {item["key"] for item in payload["effects"]} == set(sfx_module.LIBRARY)
    assert "ae_hype" in payload["themes"]
