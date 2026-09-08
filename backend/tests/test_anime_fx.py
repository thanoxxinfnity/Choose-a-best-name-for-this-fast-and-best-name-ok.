"""The effects an anime edit is actually made of."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anime_fx import (  # noqa: E402
    Stutter,
    colour_flash,
    echo_trail,
    impact_frame,
    light_leak,
    pixel_sort,
    speed_lines,
    stutter_indices,
)


def _frame(value: int = 60, height: int = 200, width: int = 120) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _noisy(height: int = 120, width: int = 90) -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


# ------------------------------------------------------- impact frames ------

@pytest.mark.parametrize("style", ["burst", "crack", "slash"])
def test_an_impact_frame_is_drawn_not_blank(style):
    """A white card reads as a dropped frame; a drawn shape reads as force."""
    out = impact_frame((120, 200), seed=1, style=style)
    assert out.shape == (200, 120, 3)
    assert out.std() > 20, f"'{style}' came out flat"


def test_the_same_seed_draws_the_same_impact_frame():
    a = impact_frame((120, 200), seed=7, style="burst")
    b = impact_frame((120, 200), seed=7, style="burst")
    assert np.array_equal(a, b)


def test_different_seeds_draw_different_impact_frames():
    a = impact_frame((120, 200), seed=1, style="crack")
    b = impact_frame((120, 200), seed=2, style="crack")
    assert not np.array_equal(a, b)


# --------------------------------------------------------- speed lines -----

def test_speed_lines_brighten_the_frame():
    frame = _frame()
    assert speed_lines(frame, 0.8).mean() > frame.mean() + 10


def test_more_amount_means_more_lines():
    frame = _frame()
    light = speed_lines(frame, 0.2, seed=3).mean()
    heavy = speed_lines(frame, 0.9, seed=3).mean()
    assert heavy > light


def test_speed_lines_leave_the_middle_alone():
    """Lines that reach the centre cover the subject they exist to point at."""
    frame = _frame()
    out = speed_lines(frame, 0.9, seed=5)
    h, w = frame.shape[:2]
    core = out[int(h * 0.45):int(h * 0.55), int(w * 0.45):int(w * 0.55)]
    edge = out[:int(h * 0.12)]
    assert core.mean() < edge.mean(), "the lines closed over the centre"


def test_no_amount_is_a_no_op():
    frame = _frame()
    assert np.array_equal(speed_lines(frame, 0.0), frame)


# --------------------------------------------------------- echo trail ------

def test_a_trail_only_ever_adds_light():
    """Averaging would smear the background, which is not moving."""
    frame, ghost = _frame(60), _frame(200)
    assert (echo_trail(frame, [ghost], 0.5) >= frame).all()


def test_each_ghost_leaves_its_own_mark_and_older_ones_are_fainter():
    """Identical ghosts overlap exactly, so the test has to move the subject.

    A real trail is the subject in a different place each frame; stacking the
    same frame three times correctly changes nothing, because the brightest
    copy already covers the others.
    """
    frame = _frame(30)
    ghosts = []
    for offset in (10, 30, 50):                 # the subject travelling right
        ghost = _frame(30)
        ghost[:, offset:offset + 8] = 240
        ghosts.append(ghost)

    out = echo_trail(frame, ghosts, 0.9)
    marks = [out[:, offset + 4, 0].mean() for offset in (10, 30, 50)]
    assert all(mark > 30 for mark in marks), f"a ghost vanished: {marks}"
    # Newest last in history, so the last mark is the brightest.
    assert marks[2] > marks[1] > marks[0], f"ages are not fading: {marks}"


def test_no_history_changes_nothing():
    frame = _frame()
    assert np.array_equal(echo_trail(frame, [], 0.9), frame)


def test_a_mismatched_ghost_is_resized_not_refused():
    frame = _frame(60, 200, 120)
    odd = np.full((50, 30, 3), 200, dtype=np.uint8)
    assert echo_trail(frame, [odd], 0.5).shape == frame.shape


# -------------------------------------------------------- colour flash -----

def test_a_flash_pushes_toward_the_accent():
    frame = _frame(60)
    out = colour_flash(frame, 0.7, colour=(200, 20, 20))
    assert out[:, :, 0].mean() > out[:, :, 1].mean() + 20


def test_a_flash_keeps_the_picture_underneath():
    """Screened, not replaced - a flat card of the accent is not a flash."""
    frame = _noisy()
    out = colour_flash(frame, 0.6)
    assert out.std() > frame.std() * 0.4, "the picture was flattened away"


def test_no_flash_is_a_no_op():
    frame = _frame()
    assert np.array_equal(colour_flash(frame, 0.0), frame)


# ---------------------------------------------------------- pixel sort -----

def test_pixel_sort_changes_the_frame_but_keeps_its_shape():
    frame = _noisy()
    out = pixel_sort(frame, 0.6)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_pixel_sort_moves_pixels_rather_than_inventing_them():
    """Every row is a permutation of itself, so the histogram is unchanged."""
    frame = _noisy()
    out = pixel_sort(frame, 1.0, threshold=0)
    assert np.array_equal(np.sort(out.ravel()), np.sort(frame.ravel()))


def test_pixel_sort_can_run_vertically():
    frame = _noisy()
    assert pixel_sort(frame, 0.5, vertical=True).shape == frame.shape


def test_no_sort_is_a_no_op():
    frame = _noisy()
    assert np.array_equal(pixel_sort(frame, 0.0), frame)


# ---------------------------------------------------------- light leak -----

def _leak_centre(frame, progress, seed=2):
    """Where the leak's brightest column sits, 0..1 across the frame."""
    lit = light_leak(frame, progress, seed=seed).astype(np.float32).mean(axis=(0, 2))
    return float(np.argmax(lit)) / max(len(lit) - 1, 1)


def test_a_leak_travels_across_the_shot():
    """Comparing the two ends is the wrong test - at both the band is off frame.

    What has to be true is that the bright band MOVES, so its position is what
    gets measured rather than the frame's average brightness.
    """
    frame = _frame()
    positions = [_leak_centre(frame, p) for p in (0.3, 0.5, 0.7)]
    assert positions[0] != positions[2], f"the leak did not move: {positions}"
    assert max(positions) - min(positions) > 0.1


def test_a_leak_is_brightest_partway_through_not_at_the_ends():
    frame = _frame()
    middle = light_leak(frame, 0.5, seed=2).mean()
    assert middle > light_leak(frame, 0.0, seed=2).mean() + 5
    assert middle > light_leak(frame, 1.0, seed=2).mean() + 5


def test_a_leak_adds_light_rather_than_removing_it():
    frame = _frame(60)
    assert (light_leak(frame, 0.5, seed=1) >= frame - 1).all()


# ------------------------------------------------------------- stutter -----

def test_a_stutter_holds_then_jumps():
    assert stutter_indices(30, Stutter(hold=2, jump=4, repeats=3)) == [0, 0, 4, 4, 8, 8]


def test_a_stutter_never_indexes_past_the_clip():
    indices = stutter_indices(5, Stutter(hold=1, jump=9, repeats=6))
    assert indices and max(indices) <= 4 and min(indices) >= 0


def test_a_stutter_can_start_partway_in():
    assert stutter_indices(30, Stutter(hold=1, jump=3, repeats=2), start=10) == [10, 13]


def test_an_empty_clip_stutters_into_nothing():
    assert stutter_indices(0, Stutter()) == []
