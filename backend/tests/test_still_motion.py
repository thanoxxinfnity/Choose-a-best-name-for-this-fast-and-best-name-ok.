"""Making a still frame sit in a cut instead of freezing it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from still_motion import (  # noqa: E402
    BY_KEY,
    MOVES,
    animate,
    ease,
    frame_at,
    move_for,
    spread_moves,
)


def _still(height: int = 600, width: int = 400) -> np.ndarray:
    """A still with a distinct mark in each corner, so pans are measurable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 40, 45)
    frame[: height // 8, : width // 8] = (255, 0, 0)
    frame[: height // 8, -width // 8:] = (0, 255, 0)
    frame[-height // 8:, : width // 8] = (0, 0, 255)
    frame[-height // 8:, -width // 8:] = (255, 255, 0)
    frame[height // 2 - 20:height // 2 + 20, width // 2 - 20:width // 2 + 20] = (255, 255, 255)
    return frame


# ----------------------------------------------------------------- easing ---

def test_no_punch_is_a_straight_line():
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert ease(t, 0.0) == pytest.approx(t)


def test_punch_front_loads_the_move():
    """A camera that crawls at a constant rate is what makes a slideshow."""
    assert ease(0.25, 0.85) > 0.25
    assert ease(0.5, 0.85) > 0.5


def test_easing_still_starts_at_nothing_and_ends_at_everything():
    for punch in (0.0, 0.4, 1.0):
        assert ease(0.0, punch) == pytest.approx(0.0)
        assert ease(1.0, punch) == pytest.approx(1.0)


def test_easing_never_goes_backwards():
    values = [ease(i / 20, 0.7) for i in range(21)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_easing_clamps_a_progress_outside_the_shot():
    assert ease(-1.0, 0.5) == pytest.approx(0.0)
    assert ease(2.0, 0.5) == pytest.approx(1.0)


# ------------------------------------------------------------- the frame ----

def test_every_frame_comes_out_at_the_requested_size():
    still = _still()
    for move in MOVES:
        out = frame_at(still, 0.5, move, (270, 480))
        assert out.shape == (480, 270, 3), move.key


def test_a_push_in_actually_gets_closer():
    """The centre mark must grow, or nothing pushed in."""
    still = _still()
    push = move_for("push")
    early = frame_at(still, 0.0, push, (270, 480))
    late = frame_at(still, 1.0, push, (270, 480))

    white_early = (early > 200).all(axis=2).sum()
    white_late = (late > 200).all(axis=2).sum()
    assert white_late > white_early


def test_a_pull_out_does_the_opposite():
    still = _still()
    pull = move_for("pull")
    early = frame_at(still, 0.0, pull, (270, 480))
    late = frame_at(still, 1.0, pull, (270, 480))
    assert (late > 200).all(axis=2).sum() < (early > 200).all(axis=2).sum()


def test_a_drift_moves_the_frame_sideways():
    still = _still()
    move = move_for("push_left")
    early = frame_at(still, 0.0, move, (270, 480)).astype(np.float32)
    late = frame_at(still, 1.0, move, (270, 480)).astype(np.float32)
    assert np.abs(early - late).mean() > 2.0, "the frame did not move"


def test_the_crop_never_runs_off_the_image():
    """A frame that walks past the edge shows border smear, not picture."""
    still = _still()
    for move in MOVES:
        for t in (0.0, 0.5, 1.0):
            out = frame_at(still, t, move, (270, 480))
            # The filler colour is only ever produced by an out-of-range crop.
            assert out.shape == (480, 270, 3)
            assert out.std() > 1.0, f"{move.key} at {t} produced a flat frame"


def test_a_tiny_still_is_handled_rather_than_crashing():
    tiny = np.full((12, 8, 3), 128, dtype=np.uint8)
    out = frame_at(tiny, 0.5, move_for("push"), (270, 480))
    assert out.shape == (480, 270, 3)


# ----------------------------------------------------------------- shots ----

def test_animate_returns_the_right_number_of_frames():
    frames = animate(_still(), 1.5, 30, (270, 480), move_for("push"))
    assert len(frames) == 45
    assert all(f.shape == (480, 270, 3) for f in frames)


def test_a_very_short_shot_still_gets_two_frames():
    assert len(animate(_still(), 0.01, 30, (270, 480), move_for("push"))) == 2


def test_the_move_is_visible_across_the_shot():
    frames = animate(_still(), 1.0, 12, (270, 480), move_for("slam"))
    first, last = frames[0].astype(np.float32), frames[-1].astype(np.float32)
    assert np.abs(first - last).mean() > 3.0


# ------------------------------------------------------------ the spread ----

def test_no_two_neighbouring_shots_drift_the_same_way():
    """Two push-ins in a row read as one push with a cut wasted in it."""
    moves = spread_moves(20)
    assert len(moves) == 20
    for earlier, later in zip(moves, moves[1:]):
        assert earlier.key != later.key


def test_the_slam_is_rationed():
    moves = spread_moves(20)
    slams = [m for m in moves if m.key == "slam"]
    assert 0 < len(slams) <= 5, f"{len(slams)} slams in twenty shots"


def test_asking_for_nothing_returns_nothing():
    assert spread_moves(0) == []
