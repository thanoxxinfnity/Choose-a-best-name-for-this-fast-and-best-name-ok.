"""Blood-red captions with runs that grow as the line is spoken."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_renderer import render_caption_phrase  # noqa: E402

WORDS = ["KNOW", "YOUR", "PLACE"]
RED = "#C1121F"


def _band(highlight: int, drip: float, bleeds: bool = True) -> np.ndarray:
    return render_caption_phrase(
        WORDS, highlight, max_width=640, font_size=58,
        color="#FFFFFF", highlight_color=RED, bleeds=bleeds, drip=drip,
    )


def _ink_below(band: np.ndarray, line: float = 0.62) -> float:
    """How much is drawn in the lower part of the band - i.e. the runs."""
    start = int(band.shape[0] * line)
    return float((band[start:, :, 3] > 40).mean())


# -------------------------------------------------------------- geometry ----

def test_the_band_never_changes_size_while_the_blood_runs():
    """A phrase whose height moves mid-line makes the whole caption jump.

    Headroom therefore comes from the style, not from this frame's progress.
    """
    sizes = {_band(index, (index + 1) / 3).shape for index in range(3)}
    assert len(sizes) == 1, f"the band resized mid-phrase: {sizes}"


def test_a_bleeding_band_reserves_room_a_plain_one_does_not():
    assert _band(0, 0.5).shape[0] > _band(0, 0.0, bleeds=False).shape[0]


# ------------------------------------------------------------- the blood ----

def test_the_runs_grow_as_the_line_is_spoken():
    early, late = _ink_below(_band(0, 0.15)), _ink_below(_band(2, 1.0))
    assert late > early * 1.5, f"blood did not run: {early:.4f} -> {late:.4f}"


def test_no_progress_means_no_blood_yet():
    assert _ink_below(_band(0, 0.0)) == pytest.approx(0.0, abs=0.002)


def test_the_runs_are_the_same_runs_each_time():
    """Re-rolling them per frame boils the band instead of running blood."""
    first = _band(1, 0.6)
    again = _band(1, 0.6)
    assert np.array_equal(first, again)


def test_a_growing_run_keeps_what_it_already_drew():
    """Blood extends downward; it does not move sideways as it grows."""
    early = _band(0, 0.35)[:, :, 3] > 40
    later = _band(2, 1.0)[:, :, 3] > 40
    columns_early = set(np.where(early.any(axis=0))[0])
    columns_later = set(np.where(later.any(axis=0))[0])
    missing = columns_early - columns_later
    assert not missing, f"{len(missing)} column(s) of blood vanished as it grew"


def test_the_blood_is_the_highlight_colour_not_some_other_red():
    band = _band(2, 1.0)
    below = band[int(band.shape[0] * 0.7):]
    lit = below[:, :, 3] > 120
    assert lit.any(), "nothing was drawn to sample"
    mean = below[:, :, :3][lit].mean(axis=0)
    assert mean[0] > mean[1] + 40 and mean[0] > mean[2] + 40, f"not red: {mean}"


def test_the_words_stay_readable_under_the_blood():
    """The runs go behind the text; a caption you cannot read is not a caption."""
    plain = render_caption_phrase(WORDS, 2, max_width=640, font_size=58,
                                  color="#FFFFFF", highlight_color=RED)
    bloody = _band(2, 1.0)
    top = plain.shape[0]
    # The glyph rows of the bloody band must still match the plain one closely.
    overlap = (plain[:top, :, 3] > 128) & (bloody[:top, :, 3] > 128)
    assert overlap.sum() > (plain[:top, :, 3] > 128).sum() * 0.95


def test_a_one_word_phrase_bleeds_without_dividing_by_zero():
    band = render_caption_phrase(["BLEED"], 0, max_width=640, font_size=58,
                                 highlight_color=RED, bleeds=True, drip=1.0)
    assert band.shape[0] > 0 and _ink_below(band) > 0.0
