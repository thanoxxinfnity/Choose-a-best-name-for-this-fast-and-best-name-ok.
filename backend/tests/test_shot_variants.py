"""Multiplying a short roll of footage into a shot list.

The rules here are not cosmetic. Cutting between two crops of one take back to
back reads as a glitch, and wearing the same disguise twice reads as a
mistake - both are worse than simply having fewer shots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot_variants import (  # noqa: E402
    TREATMENTS,
    apply_treatment,
    describe,
    plan_variants,
    treatment_for,
)


def _frame(height: int = 160, width: int = 90) -> np.ndarray:
    """A frame where every region is visibly its own thing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[: height // 2, : width // 2] = (220, 30, 30)
    frame[: height // 2, width // 2:] = (30, 220, 30)
    frame[height // 2:, : width // 2] = (30, 30, 220)
    frame[height // 2:, width // 2:] = (220, 220, 30)
    return frame


# ------------------------------------------------------------- reframing ----

def test_a_treatment_keeps_the_frame_size():
    """A crop that changed shape would have to be letterboxed back in."""
    frame = _frame()
    for treatment in TREATMENTS:
        out = apply_treatment(frame, treatment)
        assert out.shape == frame.shape, treatment.key


def test_the_full_frame_treatment_changes_nothing():
    frame = _frame()
    assert np.array_equal(apply_treatment(frame, treatment_for("full")), frame)


def test_mirroring_flips_left_and_right():
    frame = _frame()
    out = apply_treatment(frame, treatment_for("mirror"))
    assert np.array_equal(out, frame[:, ::-1])


def test_every_treatment_actually_looks_different():
    """A disguise nobody can see is a wasted slot in the running order."""
    frame = _frame()
    seen = {}
    for treatment in TREATMENTS:
        out = apply_treatment(frame, treatment)
        digest = out.tobytes()
        clash = seen.get(digest)
        assert clash is None, f"'{treatment.key}' renders identically to '{clash}'"
        seen[digest] = treatment.key


def test_a_tight_crop_reads_further_from_the_original_than_a_loose_one():
    frame = _frame()
    original = frame.astype(np.float32)
    loose = np.abs(apply_treatment(frame, treatment_for("push")).astype(np.float32)
                   - original).mean()
    tight = np.abs(apply_treatment(frame, treatment_for("tight")).astype(np.float32)
                   - original).mean()
    assert tight > loose
    assert treatment_for("tight").distance > treatment_for("push").distance


def test_a_degenerate_region_falls_back_to_the_whole_frame():
    from shot_variants import Treatment

    frame = _frame()
    out = apply_treatment(frame, Treatment("nil", "Nil", region=(1.0, 1.0, 0.0, 0.0)))
    assert out.shape == frame.shape


# ------------------------------------------------------- the running order ---

def test_one_source_never_appears_twice_in_a_row():
    shots = plan_variants([4.1] * 7, 24)
    for earlier, later in zip(shots, shots[1:]):
        assert earlier.source_index != later.source_index


def test_two_shots_from_one_source_are_kept_apart():
    shots = plan_variants([4.1] * 7, 24, separation=2)
    for index, shot in enumerate(shots):
        window = shots[max(0, index - 2):index]
        assert shot.source_index not in {s.source_index for s in window}


def test_no_framing_is_reused_on_a_source_while_fresh_ones_remain():
    shots = plan_variants([4.1] * 6, 18)
    pairs = [(shot.source_index, shot.treatment) for shot in shots]
    assert len(pairs) == len(set(pairs)), "a source wore the same disguise twice"


def test_every_shot_lands_inside_its_source():
    durations = [4.1, 2.0, 6.5, 3.0]
    shots = plan_variants(durations, 16)
    for shot in shots:
        assert 0.0 <= shot.start < shot.end <= durations[shot.source_index] + 1e-6


def test_revisiting_a_source_shows_a_different_moment():
    shots = plan_variants([8.0] * 4, 12)
    by_source = {}
    for shot in shots:
        by_source.setdefault(shot.source_index, []).append(shot.start)
    moved = [starts for starts in by_source.values() if len(starts) > 1]
    assert moved, "the fixture should revisit at least one source"
    for starts in moved:
        assert len(set(starts)) > 1, "every visit opened at the same timecode"


def test_a_single_source_still_produces_a_shot_list():
    """One clip is the worst case, not an error case."""
    shots = plan_variants([4.1], 6)
    assert len(shots) == 6
    assert len({shot.treatment for shot in shots}) == 6


def test_nothing_in_means_nothing_out():
    assert plan_variants([], 10) == []
    assert plan_variants([4.0], 0) == []
    assert plan_variants([0.01], 5) == []


def test_the_summary_counts_what_was_actually_built():
    shots = plan_variants([4.1] * 5, 15)
    line = describe(shots)
    assert "15 shots" in line and "5 source" in line
