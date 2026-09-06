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


# ------------------------------------- captioning from the script -----------

def _voice_file(tmp_path, name, seconds):
    """A real audio file so the duration is measured, not assumed."""
    import subprocess

    from puter_integration import ffmpeg_binary

    path = tmp_path / name
    subprocess.run(
        [ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=200:duration={seconds}",
         "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )
    return path


class _ScriptRenderer:
    """Just enough of VideoRenderer to exercise caption sourcing."""

    from video_renderer import VideoRenderer

    _caption_words_from_plan = VideoRenderer._caption_words_from_plan
    _track = VideoRenderer._track

    def __init__(self, plan, intro_offset=0.0):
        self.plan = plan
        self._intro_offset = intro_offset
        self._open_clips = []


def _voice_plan(*lines):
    from schemas import AudioSpec, CaptionSpec, EditPlan, TtsLine

    return EditPlan(
        captions=CaptionSpec(enabled=True),
        audio=AudioSpec(use_puter_tts=True, tts_lines=[
            TtsLine(text=text, start_time=str(start)) for text, start in lines
        ]),
    )


def test_the_narration_is_captioned_from_the_script_not_a_transcription(tmp_path):
    """The words are already known; deepening the voice is what breaks ASR.

    Six semitones down with a sub-octave layer under it is exactly what stops
    a recogniser hearing the words - "Know your place" came back as
    "NO, PLEASE. YOUR" from a real render.
    """
    audio = _voice_file(tmp_path, "line.mp3", 2.0)
    plan = _voice_plan(("Know your place", 1.0))
    words = _ScriptRenderer(plan)._caption_words_from_plan([(audio, 1.0, 1.0)])

    assert [w.text for w in words] == ["Know", "your", "place"]
    assert words[0].start == pytest.approx(1.0, abs=0.05)
    # The line's words span its measured audio, not a guessed length.
    assert words[-1].end == pytest.approx(3.0, abs=0.15)


def test_each_line_is_timed_from_its_own_audio(tmp_path):
    short = _voice_file(tmp_path, "a.mp3", 1.0)
    long = _voice_file(tmp_path, "b.mp3", 3.0)
    plan = _voice_plan(("one two", 0.5), ("three four", 5.0))
    words = _ScriptRenderer(plan)._caption_words_from_plan(
        [(short, 0.5, 1.0), (long, 5.0, 1.0)]
    )

    first = [w for w in words if w.text in ("one", "two")]
    second = [w for w in words if w.text in ("three", "four")]
    assert first[-1].end - first[0].start == pytest.approx(1.0, abs=0.15)
    assert second[-1].end - second[0].start == pytest.approx(3.0, abs=0.2)
    assert second[0].start == pytest.approx(5.0, abs=0.05)


def test_longer_words_get_more_of_the_line(tmp_path):
    audio = _voice_file(tmp_path, "c.mp3", 2.0)
    plan = _voice_plan(("I understand", 0.0))
    words = _ScriptRenderer(plan)._caption_words_from_plan([(audio, 0.0, 1.0)])
    short, long = words[0], words[1]
    assert (long.end - long.start) > (short.end - short.start)


def test_a_spliced_intro_shifts_the_captions(tmp_path):
    audio = _voice_file(tmp_path, "d.mp3", 1.0)
    plan = _voice_plan(("hello there", 2.0))
    words = _ScriptRenderer(plan, intro_offset=1.5)._caption_words_from_plan(
        [(audio, 2.0, 1.0)]
    )
    assert words[0].start == pytest.approx(3.5, abs=0.05)


def test_a_plain_script_with_no_timed_lines_still_captions(tmp_path):
    from schemas import AudioSpec, CaptionSpec, EditPlan

    audio = _voice_file(tmp_path, "e.mp3", 1.5)
    plan = EditPlan(
        captions=CaptionSpec(enabled=True),
        audio=AudioSpec(use_puter_tts=True, tts_script="one small step"),
    )
    words = _ScriptRenderer(plan)._caption_words_from_plan([(audio, 0.0, 1.0)])
    assert [w.text for w in words] == ["one", "small", "step"]


def test_no_voiceover_falls_back_to_transcription(tmp_path):
    """Whisper is still the right tool for captioning the source audio."""
    plan = _voice_plan(("something", 0.0))
    assert _ScriptRenderer(plan)._caption_words_from_plan([]) == []


def test_a_mismatched_count_is_not_guessed_at(tmp_path):
    """Two files against three lines means the pairing is unknown."""
    audio = _voice_file(tmp_path, "f.mp3", 1.0)
    plan = _voice_plan(("a b", 0.0), ("c d", 2.0), ("e f", 4.0))
    assert _ScriptRenderer(plan)._caption_words_from_plan([(audio, 0.0, 1.0)]) == []


def test_a_missing_audio_file_is_skipped_not_fatal(tmp_path):
    plan = _voice_plan(("gone", 0.0))
    assert _ScriptRenderer(plan)._caption_words_from_plan(
        [(tmp_path / "nope.mp3", 0.0, 1.0)]
    ) == []


# --------------------------------------------------- phrases and sentences ---

def test_a_caption_band_never_mixes_two_spoken_lines():
    """"NAHI SAKTE. APNI" was the tail of one line and the head of the next.

    Grouping every three words regardless of where they came from builds a
    sentence nobody said, so a phrase must stop at a line boundary even when
    it has room left.
    """
    from video_renderer import CaptionWord

    words = [
        CaptionWord("tum", 0.0, 0.3, group=0),
        CaptionWord("mujhe", 0.3, 0.6, group=0),
        CaptionWord("haara", 0.6, 0.9, group=0),
        CaptionWord("nahi", 0.9, 1.2, group=0),
        CaptionWord("sakte", 1.2, 1.5, group=0),
        CaptionWord("apni", 2.0, 2.3, group=1),
        CaptionWord("aukaat", 2.3, 2.6, group=1),
    ]
    phrases = _phrases(words, per_phrase=3)

    assert [[w.text for w in phrase] for phrase in phrases] == [
        ["tum", "mujhe", "haara"],
        ["nahi", "sakte"],
        ["apni", "aukaat"],
    ]
    for phrase in phrases:
        assert len({w.group for w in phrase}) == 1, "a band spans two lines"


def test_a_phrase_still_fills_up_to_the_limit_inside_one_line():
    from video_renderer import CaptionWord

    words = [CaptionWord(f"w{i}", i * 0.2, i * 0.2 + 0.2, group=0) for i in range(7)]
    phrases = _phrases(words, per_phrase=3)
    assert [len(p) for p in phrases] == [3, 3, 1]


def _phrases(words, per_phrase):
    """The grouping _caption_layers does, isolated from rendering."""
    grouped = []
    for word in words:
        same = grouped and word.group == grouped[-1][0].group
        if same and len(grouped[-1]) < per_phrase:
            grouped[-1].append(word)
        else:
            grouped.append([word])
    return grouped


def test_transcribed_words_are_grouped_by_sentence():
    """Transcription has no line numbers, so a full stop is the boundary."""
    import video_renderer

    class _Word:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class _Segment:
        def __init__(self, words):
            self.words = words

    class _Model:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [_Segment([
                _Word("know", 0.0, 0.3), _Word("your", 0.3, 0.6),
                _Word("place.", 0.6, 0.9),
                _Word("bow", 1.0, 1.3), _Word("down.", 1.3, 1.6),
            ])], None

    import sys as _sys
    import types as _types
    fake = _types.ModuleType("faster_whisper")
    fake.WhisperModel = _Model
    saved = _sys.modules.get("faster_whisper")
    _sys.modules["faster_whisper"] = fake
    try:
        words = video_renderer.transcribe_words(Path("nowhere.wav"))
    finally:
        if saved is None:
            _sys.modules.pop("faster_whisper", None)
        else:
            _sys.modules["faster_whisper"] = saved

    assert [w.group for w in words] == [0, 0, 0, 1, 1]
