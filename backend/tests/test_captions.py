"""Word-level captions: a phrase at a time, with the spoken word lit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_renderer import CaptionWord, render_caption_phrase  # noqa: E402


def _colour_count(array: np.ndarray, rgb, tolerance: int = 26) -> int:
    """How many solid pixels are within tolerance of a colour."""
    opaque = array[..., 3] > 200
    close = np.all(np.abs(array[..., :3].astype(int) - np.array(rgb)) <= tolerance, axis=-1)
    return int((opaque & close).sum())


RED = (255, 0, 0)
BLUE = (0, 0, 255)


# ------------------------------------------------------------- the render --

def test_the_highlighted_word_is_the_only_one_in_the_accent_colour():
    array = render_caption_phrase(
        ["EK", "DO", "TEEN"], highlight=1, max_width=900, font_size=64,
        color="#0000FF", highlight_color="#FF0000",
    )
    assert _colour_count(array, RED) > 100
    assert _colour_count(array, BLUE) > 100


def test_moving_the_highlight_does_not_move_the_band():
    """The layout must depend on the words, never on which one is lit.

    If it did, the caption would twitch sideways every time the highlight
    advanced - the single most visible way this style goes wrong.
    """
    frames = [
        render_caption_phrase(["EK", "DO", "TEEN"], highlight=index,
                              max_width=900, font_size=64)
        for index in range(3)
    ]
    assert len({frame.shape for frame in frames}) == 1

    # And the ink sits in the same places: only the colours differ.
    masks = [frame[..., 3] > 200 for frame in frames]
    for other in masks[1:]:
        assert np.array_equal(masks[0], other)


def test_every_word_gets_its_turn():
    counts = []
    for index in range(3):
        array = render_caption_phrase(
            ["EK", "DO", "TEEN"], highlight=index, max_width=900, font_size=64,
            color="#0000FF", highlight_color="#FF0000",
        )
        counts.append(_colour_count(array, RED))
    assert all(count > 50 for count in counts)
    # Different words are different widths, so the lit pixel counts differ.
    assert len(set(counts)) > 1


def test_a_phrase_too_wide_for_the_frame_wraps_instead_of_overflowing():
    narrow = render_caption_phrase(
        ["BAHUT", "LAMBA", "PHRASE", "HAI"], highlight=0, max_width=320, font_size=64,
    )
    wide = render_caption_phrase(
        ["BAHUT", "LAMBA", "PHRASE", "HAI"], highlight=0, max_width=1600, font_size=64,
    )
    assert narrow.shape[1] <= 320 + 40
    assert narrow.shape[0] > wide.shape[0]  # it wrapped onto more lines


def test_a_single_word_phrase_still_renders():
    array = render_caption_phrase(["SOLO"], highlight=0, max_width=900, font_size=64,
                                  highlight_color="#FF0000")
    assert _colour_count(array, RED) > 50


def test_a_highlight_out_of_range_simply_lights_nothing():
    """A caption is never worth crashing a render over."""
    array = render_caption_phrase(["EK", "DO"], highlight=9, max_width=900, font_size=64,
                                  color="#0000FF", highlight_color="#FF0000")
    assert _colour_count(array, RED) == 0
    assert _colour_count(array, BLUE) > 100


def test_devanagari_text_renders():
    array = render_caption_phrase(["नमस्ते", "भाई"], highlight=0, max_width=900, font_size=64)
    assert array.shape[0] > 10 and array.shape[1] > 10
    assert int((array[..., 3] > 200).sum()) > 50


# --------------------------------------------------- the layer scheduling ---

class _Renderer:
    """Just enough of VideoRenderer to exercise the caption scheduling."""

    from video_renderer import VideoRenderer

    _caption_layers = VideoRenderer._caption_layers

    def __init__(self, plan, width=1080, height=1920, fps=30):
        self.plan, self.width, self.height, self.fps = plan, width, height, fps


def _plan(max_words=3):
    from schemas import CaptionSpec, EditPlan

    return EditPlan(captions=CaptionSpec(enabled=True, max_words_on_screen=max_words))


def test_words_are_grouped_into_phrases_not_flashed_one_at_a_time():
    """Every word still gets a frame - the grouping is about what is shown."""
    words = [CaptionWord(f"W{i}", i * 0.4, i * 0.4 + 0.3) for i in range(6)]
    layers = _Renderer(_plan(max_words=3))._caption_layers(words, duration=10.0)
    assert len(layers) == 6


def test_the_phrase_size_the_plan_asks_for_is_honoured():
    """This field existed and did nothing before - the plan could not change it."""
    words = [CaptionWord(f"W{i}", i * 0.4, i * 0.4 + 0.3) for i in range(4)]
    one = _Renderer(_plan(max_words=1))._caption_layers(words, duration=10.0)
    four = _Renderer(_plan(max_words=4))._caption_layers(words, duration=10.0)
    # Same number of layers, but a wider band when four words share it.
    assert len(one) == len(four) == 4
    assert four[0].w > one[0].w


def test_a_highlight_holds_until_the_next_word_starts():
    """Otherwise the band blinks out in the gap between two words."""
    words = [CaptionWord("EK", 0.0, 0.2), CaptionWord("DO", 1.0, 1.2)]
    layers = _Renderer(_plan(max_words=3))._caption_layers(words, duration=5.0)
    assert layers[0].duration == pytest.approx(1.0, abs=0.01)


def test_words_past_the_end_of_the_edit_are_dropped():
    words = [CaptionWord("EK", 0.5, 0.9), CaptionWord("DO", 40.0, 40.4)]
    layers = _Renderer(_plan())._caption_layers(words, duration=5.0)
    assert len(layers) == 1


def test_blank_words_are_skipped():
    words = [CaptionWord("EK", 0.0, 0.4), CaptionWord("   ", 0.5, 0.9),
             CaptionWord("DO", 1.0, 1.4)]
    layers = _Renderer(_plan())._caption_layers(words, duration=5.0)
    assert len(layers) == 2


def test_no_words_means_no_layers():
    assert _Renderer(_plan())._caption_layers([], duration=5.0) == []
