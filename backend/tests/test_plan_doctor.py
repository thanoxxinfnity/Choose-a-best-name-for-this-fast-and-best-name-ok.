"""Reading the cut back before it is rendered."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan_doctor import BLOCKING, SERIOUS, diagnose, review  # noqa: E402
from schemas import (  # noqa: E402
    AudioSpec,
    EditPlan,
    PuterSticker,
    TextOverlay,
    TimelineSegment,
    TtsLine,
)


def _plan(*lengths: float, cut: str = "jump_cut") -> EditPlan:
    timeline = []
    at = 0.0
    for length in lengths:
        timeline.append(TimelineSegment(
            start_time=str(round(at, 3)), end_time=str(round(at + length, 3)),
            cut_type=cut, source_index=0,
        ))
        at += length
    return EditPlan(edit_timeline=timeline)


def _rules(diagnosis) -> set:
    return {finding.rule for finding in diagnosis.findings}


# ------------------------------------------------------- footage reality ----

def test_a_segment_past_the_end_of_its_clip_is_blocking_and_pulled_back():
    """A timeline referencing footage that does not exist cannot be rendered."""
    plan = _plan(3.0, 3.0)
    plan.edit_timeline[1].end_time = "00:00:40"
    plan, diagnosis = review(plan, clip_durations=[10.0])

    assert "past_the_end" in _rules(diagnosis)
    assert any(f.severity == BLOCKING for f in diagnosis.findings)
    assert plan.edit_timeline[1].end_seconds <= 10.0


def test_a_segment_pointing_at_a_clip_that_was_never_uploaded_is_remapped():
    plan = _plan(2.0, 2.0)
    plan.edit_timeline[1].source_index = 7
    plan, diagnosis = review(plan, clip_durations=[30.0, 30.0])

    assert "missing_source" in _rules(diagnosis)
    assert plan.edit_timeline[1].source_index < 2


def test_footage_that_fits_raises_nothing():
    plan = _plan(2.0, 2.0, 3.0)
    _, diagnosis = review(plan, clip_durations=[60.0])
    assert "past_the_end" not in _rules(diagnosis)
    assert "missing_source" not in _rules(diagnosis)


# ------------------------------------------------------------- the hook -----

def test_opening_on_a_crossfade_is_caught():
    """Short form has no room to fade in - that is long-form grammar."""
    plan = _plan(1.0, 1.4, 2.2, 1.1)
    plan.edit_timeline[0].cut_type = "crossfade"
    plan, diagnosis = review(plan)

    assert "soft_open" in _rules(diagnosis)
    assert plan.edit_timeline[0].cut_type == "hard_cut"


def test_a_slow_opening_shot_is_shortened():
    plan = _plan(6.0, 1.2, 2.0, 1.5)
    plan, diagnosis = review(plan)

    assert "slow_open" in _rules(diagnosis)
    assert plan.edit_timeline[0].end_seconds - plan.edit_timeline[0].start_seconds < 2.6


def test_a_punchy_open_is_left_alone():
    plan = _plan(1.1, 1.9, 2.6, 1.3)
    plan, diagnosis = review(plan)
    assert "slow_open" not in _rules(diagnosis)
    assert "soft_open" not in _rules(diagnosis)


# -------------------------------------------------------------- rhythm ------

def test_a_metronome_edit_is_caught_and_broken_up():
    """A run of identical shot lengths goes numb however good each shot is."""
    plan = _plan(1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
    plan, diagnosis = review(plan)

    assert "metronome" in _rules(diagnosis)
    lengths = [s.end_seconds - s.start_seconds for s in plan.edit_timeline]
    assert len(set(round(v, 2) for v in lengths)) > 1


def test_an_edit_that_already_varies_is_left_alone():
    plan = _plan(0.8, 2.4, 1.1, 3.2, 1.6)
    _, diagnosis = review(plan)
    assert "metronome" not in _rules(diagnosis)


def test_a_short_timeline_is_not_judged_on_rhythm():
    plan = _plan(1.5, 1.5, 1.5)
    _, diagnosis = review(plan)
    assert "metronome" not in _rules(diagnosis)


# ------------------------------------------------------------- accents ------

def test_punching_every_cut_is_caught_and_thinned():
    """An accent on every cut is a headache, not emphasis."""
    plan = _plan(*([1.2, 2.0] * 4), cut="zoom_punch")
    plan, diagnosis = review(plan)

    assert "accent_inflation" in _rules(diagnosis)
    punches = [s for s in plan.edit_timeline if s.cut_type == "zoom_punch"]
    assert 0 < len(punches) < len(plan.edit_timeline) / 2
    # The payoff always keeps its punch.
    assert plan.edit_timeline[-1].cut_type == "zoom_punch"


def test_a_few_punches_are_left_alone():
    plan = _plan(1.0, 2.0, 1.4, 2.6, 1.1, 2.2)
    plan.edit_timeline[2].cut_type = "zoom_punch"
    plan.edit_timeline[5].cut_type = "zoom_punch"
    _, diagnosis = review(plan)
    assert "accent_inflation" not in _rules(diagnosis)


# ------------------------------------------------------------ stickers ------

def test_back_to_back_stickers_are_thinned():
    """A sticker is punctuation; one every shot and none of them read."""
    plan = _plan(1.0, 2.0, 1.4, 2.6)
    for segment in plan.edit_timeline:
        segment.puter_sticker = PuterSticker(generate_prompt="fire emoji")
    plan, diagnosis = review(plan)

    assert "sticker_crowding" in _rules(diagnosis)
    carrying = [s for s in plan.edit_timeline if s.puter_sticker]
    assert len(carrying) == 2


def test_well_spaced_stickers_survive():
    plan = _plan(1.0, 2.0, 1.4, 2.6, 1.2)
    plan.edit_timeline[0].puter_sticker = PuterSticker(generate_prompt="fire")
    plan.edit_timeline[3].puter_sticker = PuterSticker(generate_prompt="skull")
    plan, diagnosis = review(plan)
    assert "sticker_crowding" not in _rules(diagnosis)
    assert sum(1 for s in plan.edit_timeline if s.puter_sticker) == 2


# --------------------------------------------------------------- text -------

def test_text_and_a_sticker_in_the_same_third_are_separated():
    plan = _plan(1.0, 2.2, 1.4, 2.0)
    segment = plan.edit_timeline[1]
    segment.text_overlay = TextOverlay(text="WATCH THIS", position="bottom_center")
    segment.puter_sticker = PuterSticker(generate_prompt="arrow", position="bottom_right")
    plan, diagnosis = review(plan)

    assert "overlap" in _rules(diagnosis)
    assert segment.puter_sticker.position != "bottom_right"


def test_text_and_a_sticker_in_different_thirds_are_fine():
    plan = _plan(1.0, 2.2, 1.4, 2.0)
    segment = plan.edit_timeline[1]
    segment.text_overlay = TextOverlay(text="WATCH THIS", position="top_center")
    segment.puter_sticker = PuterSticker(generate_prompt="arrow", position="bottom_right")
    _, diagnosis = review(plan)
    assert "overlap" not in _rules(diagnosis)


def test_a_wordy_overlay_is_trimmed():
    plan = _plan(1.0, 2.2, 1.4, 2.0)
    plan.edit_timeline[0].text_overlay = TextOverlay(
        text="this is a very long sentence nobody will ever finish reading")
    plan, diagnosis = review(plan)

    assert "wordy_overlay" in _rules(diagnosis)
    assert len(plan.edit_timeline[0].text_overlay.text.split()) <= 4


def test_a_short_overlay_is_untouched():
    plan = _plan(1.0, 2.2, 1.4, 2.0)
    plan.edit_timeline[0].text_overlay = TextOverlay(text="WAIT FOR IT")
    plan, diagnosis = review(plan)
    assert "wordy_overlay" not in _rules(diagnosis)
    assert plan.edit_timeline[0].text_overlay.text == "WAIT FOR IT"


# --------------------------------------------------------------- voice ------

def test_overlapping_voice_lines_are_pushed_apart():
    plan = _plan(3.0, 4.0, 3.0, 4.0)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text="pehli line jo kaafi lambi hai bhai", start_time="00:00:01"),
        TtsLine(text="doosri line", start_time="00:00:02"),
    ])
    plan, diagnosis = review(plan)

    assert "voice_overlap" in _rules(diagnosis)
    starts = sorted(line.start_seconds for line in plan.audio.tts_lines)
    assert starts[1] > starts[0] + 1.0


def test_a_line_landing_on_a_cut_is_nudged_off_it():
    """A line that starts on the hit fights the hit."""
    plan = _plan(1.8, 4.0, 3.0)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text="ekdum sahi", start_time="00:00:01.800"),
    ])
    plan, diagnosis = review(plan)

    assert "voice_on_the_hit" in _rules(diagnosis)
    assert plan.audio.tts_lines[0].start_seconds > 1.8


def test_a_line_that_runs_past_the_end_is_pulled_back_inside_it():
    """The renderer holds the last frame to let a late line finish speaking.

    So leaving this one reported-but-unfixed did not preserve the script, it
    bought three seconds of frozen frame on the end of a real edit. Moving the
    line is the same repair the two checks above it already make.
    """
    plan = _plan(3.0, 3.0, 3.0)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text="bow down", start_time="00:00:08.500"),
    ])
    plan, diagnosis = review(plan)

    finding = next(f for f in diagnosis.findings if f.rule == "voice_past_the_end")
    assert finding.fixed is True
    assert finding.severity == SERIOUS
    line = plan.audio.tts_lines[0]
    assert line.text == "bow down"
    spoken = max(1.2, len(line.text.split()) / 2.6)
    assert line.start_seconds + spoken <= 9.0


def test_a_line_with_nowhere_left_to_go_is_dropped_rather_than_stretched():
    """An edit two seconds long cannot carry a fifteen-second line."""
    plan = _plan(1.0, 1.0)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text=" ".join(["word"] * 40), start_time="00:00:01.500"),
    ])
    plan, diagnosis = review(plan)

    finding = next(f for f in diagnosis.findings if f.rule == "voice_past_the_end")
    assert finding.fixed is True
    assert plan.audio.timed_lines == []


def test_well_placed_lines_raise_nothing():
    plan = _plan(1.6, 4.0, 3.4, 2.2)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text="pehli baat", start_time="00:00:02.500"),
        TtsLine(text="doosri baat", start_time="00:00:07"),
    ])
    _, diagnosis = review(plan)
    assert not {"voice_overlap", "voice_on_the_hit", "voice_past_the_end"} & _rules(diagnosis)


# -------------------------------------------------------------- ending ------

def test_ending_on_a_stub_is_caught():
    plan = _plan(1.2, 2.4, 1.6, 0.2)
    plan, diagnosis = review(plan)

    assert "stub_ending" in _rules(diagnosis)
    last = plan.edit_timeline[-1]
    assert last.end_seconds - last.start_seconds > 0.45


def test_a_proper_ending_is_left_alone():
    plan = _plan(1.2, 2.4, 1.6, 2.0)
    _, diagnosis = review(plan)
    assert "stub_ending" not in _rules(diagnosis)


# ------------------------------------------------------------- reporting ----

def test_an_empty_timeline_is_blocking_and_stops_the_examination():
    diagnosis = diagnose(EditPlan(edit_timeline=[]))
    assert len(diagnosis.findings) == 1
    assert diagnosis.findings[0].severity == BLOCKING


def test_diagnose_without_treating_changes_nothing():
    """The read-only mode has to be genuinely read-only."""
    plan = _plan(1.5, 1.5, 1.5, 1.5, 1.5)
    before = plan.model_dump(mode="json")
    diagnosis = diagnose(plan, treat=False)

    assert diagnosis.findings
    assert all(not finding.fixed for finding in diagnosis.findings)
    assert plan.model_dump(mode="json") == before


def test_outstanding_findings_become_a_prompt_for_the_model():
    """Findings the doctor cannot repair itself go back to the model.

    Footage that is black end to end is the honest example: there is no
    brighter frame to step onto, so the only fix is a different shot.
    """
    plan = _plan(2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=lambda *_: 0.005)

    block = diagnosis.to_prompt_block()
    assert "REVIEW OF YOUR TIMELINE" in block
    assert "black frame" in block


def test_a_clean_plan_produces_no_prompt_block():
    plan = _plan(1.1, 2.4, 1.5, 3.0, 1.8)
    _, diagnosis = review(plan, clip_durations=[60.0])
    assert diagnosis.to_prompt_block() == ""


def test_findings_serialise_for_the_app():
    plan = _plan(1.5, 1.5, 1.5, 1.5, 1.5)
    _, diagnosis = review(plan)
    payload = diagnosis.to_dict()
    assert payload["fixed"] >= 1
    assert all({"rule", "severity", "detail", "fixed"} <= set(f) for f in payload["findings"])


# ------------------------------------------------ the model's second try ----

def _clips():
    from schemas import ClipInfo

    return [ClipInfo(index=0, filename="a.mp4", path="/tmp/a.mp4", duration=60.0,
                     width=1920, height=1080, fps=30.0, size_bytes=1)]


def test_the_review_goes_back_to_the_model_and_a_better_plan_wins(monkeypatch):
    """The only stage where the model sees a consequence of what it wrote."""
    import json as _json

    from orchestrator import KimiOrchestrator

    orchestrator = KimiOrchestrator(api_key="test-key")
    fixed = _plan(1.1, 2.4, 1.6, 3.0)
    monkeypatch.setattr(
        KimiOrchestrator, "_chat",
        lambda self, *a, **k: _json.dumps(fixed.model_dump(mode="json")),
    )

    broken = _plan(1.5, 1.5, 1.5, 1.5, 1.5)
    _, diagnosis = review(broken)
    revised, warnings = orchestrator.revise_plan(
        broken, diagnosis.to_prompt_block() or "[REVIEW] fix the rhythm", _clips()
    )
    assert revised is not None
    assert len(revised.edit_timeline) == 4


def test_a_revision_that_cannot_be_parsed_keeps_the_repair(monkeypatch):
    """A failed retry costs time and nothing else."""
    from orchestrator import KimiOrchestrator

    monkeypatch.setattr(KimiOrchestrator, "_chat", lambda self, *a, **k: "sorry, no JSON here")
    revised, warnings = KimiOrchestrator(api_key="k").revise_plan(
        _plan(1.0, 2.0), "[REVIEW] something", _clips()
    )
    assert revised is None
    assert any("unusable" in warning for warning in warnings)


def test_a_rate_limited_revision_is_reported_not_raised(monkeypatch):
    from orchestrator import KimiOrchestrator, NimRateLimited

    def _boom(self, *args, **kwargs):
        raise NimRateLimited("rate limited")

    monkeypatch.setattr(KimiOrchestrator, "_chat", _boom)
    revised, warnings = KimiOrchestrator(api_key="k").revise_plan(
        _plan(1.0, 2.0), "[REVIEW] something", _clips()
    )
    assert revised is None
    assert any("rate limited" in warning for warning in warnings)


def test_no_review_means_no_second_call(monkeypatch):
    """An empty review is a clean plan; calling the model again is pure cost."""
    from orchestrator import KimiOrchestrator

    def _should_not_run(self, *args, **kwargs):
        raise AssertionError("the model was called with nothing to fix")

    monkeypatch.setattr(KimiOrchestrator, "_chat", _should_not_run)
    revised, warnings = KimiOrchestrator(api_key="k").revise_plan(
        _plan(1.0, 2.0), "   ", _clips()
    )
    assert revised is None and warnings == []


def test_without_a_key_the_revision_is_simply_skipped():
    from orchestrator import KimiOrchestrator

    revised, warnings = KimiOrchestrator(api_key="").revise_plan(
        _plan(1.0, 2.0), "[REVIEW] fix it", _clips()
    )
    assert revised is None and warnings == []


# ------------------------------------------------- black opens and endings --

def _probe(dark_spans):
    """A stand-in for reading the footage: dark inside the given spans."""
    def probe(_source_index, seconds):
        for low, high in dark_spans:
            if low <= seconds <= high:
                return 0.01
        return 0.4
    return probe


def test_an_edit_ending_on_black_is_caught_and_pulled_back():
    """The last frame is the only one that stays with the viewer."""
    plan = _plan(2.0, 2.0, 3.0)
    # The final segment runs 4-7s and the footage goes black at 6s.
    _, diagnosis = review(plan, frame_probe=_probe([(6.0, 99.0)]))

    finding = next(f for f in diagnosis.findings if f.rule == "dark_edge")
    assert finding.fixed is True
    assert plan.edit_timeline[-1].end_seconds < 7.0


def test_an_edit_opening_on_black_is_caught():
    """Frame one is the whole hook; spending it on nothing wastes the edit."""
    plan = _plan(3.0, 2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=_probe([(0.0, 1.0)]))

    finding = next(f for f in diagnosis.findings if f.rule == "dark_edge")
    assert finding.fixed is True
    assert plan.edit_timeline[0].start_seconds > 0.0


def test_footage_that_is_never_black_raises_nothing():
    plan = _plan(2.0, 2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=_probe([]))
    assert not any(f.rule == "dark_edge" for f in diagnosis.findings)


def test_a_dark_segment_with_no_lit_frame_is_reported_not_guessed_at():
    """Trimming into more blackness would not help, and the footage past the
    end may not exist, so it says so instead of inventing a fix."""
    plan = _plan(2.0, 2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=_probe([(0.0, 99.0)]))

    dark = [f for f in diagnosis.findings if f.rule == "dark_edge"]
    assert dark and all(f.fixed is False for f in dark)


def test_a_graded_night_shot_is_not_mistaken_for_black():
    """Dark footage is a style; an empty frame is a fault."""
    plan = _plan(2.0, 2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=lambda *_: 0.08)
    assert not any(f.rule == "dark_edge" for f in diagnosis.findings)


def test_without_a_probe_the_check_does_not_run():
    """Every other check reads the timeline; this one needs the pixels, and
    without them it makes no claim."""
    plan = _plan(2.0, 2.0, 2.0)
    _, diagnosis = review(plan)
    assert not any(f.rule == "dark_edge" for f in diagnosis.findings)


def test_a_probe_that_fails_is_survived():
    def broken(*_args):
        raise RuntimeError("no such frame")

    plan = _plan(2.0, 2.0, 2.0)
    _, diagnosis = review(plan, frame_probe=broken)
    assert not any(f.rule == "dark_edge" for f in diagnosis.findings)


# ---------------------------------------------------- keys that go nowhere ---

def test_a_key_the_plan_does_not_have_is_named_not_dropped_in_silence():
    """A real plan set 'text_overlays' at the top level and rendered no titles.

    It validated, it dropped the key, and it said nothing - which sends
    whoever notices the missing titles into the renderer after a fault that is
    not there.
    """
    plan = EditPlan.model_validate({
        "edit_timeline": [
            {"start_time": "00:00:00", "end_time": "00:00:02", "source_index": 0},
            {"start_time": "00:00:02", "end_time": "00:00:04", "source_index": 0},
        ],
        "text_overlays": [{"text": "SHINJUKU", "start_time": "00:00:00"}],
    })
    _, diagnosis = review(plan)

    finding = next(f for f in diagnosis.findings if f.rule == "stray_key")
    assert "text_overlays" in finding.detail
    assert "segment" in finding.detail, "it must say where the key belongs"
    assert finding.severity == SERIOUS


def test_a_plan_with_no_stray_keys_says_nothing_about_them():
    plan = _plan(1.6, 2.2, 1.8)
    _, diagnosis = review(plan, clip_durations=[60.0])
    assert "stray_key" not in _rules(diagnosis)


# ------------------------------------------------- how long a line takes -----

def test_the_speaking_estimate_errs_slow_against_real_delivery():
    """The estimate started at 2.6 words a second and under-fired everywhere.

    Five real synthesised lines came back at 1.14, 1.14, 1.29, 1.84 and 2.51
    words a second - a 2.2x spread that no single constant covers exactly. The
    two errors are not equal, so the constant is chosen to err slow: estimating
    short runs lines into each other, estimating long merely spaces them out.
    This pins the direction, and a ceiling so 'slow' cannot become absurd.
    """
    from plan_doctor import spoken_seconds

    measured = [(3, 2.64), (6, 5.28), (5, 1.99), (4, 3.10), (6, 3.26)]
    short = [(w, r) for w, r in measured if spoken_seconds(" ".join(["x"] * w)) < r]
    assert len(short) <= 1, (
        f"the estimate is short on {len(short)} of {len(measured)} real lines: {short}"
    )
    for words, real in measured:
        assert spoken_seconds(" ".join(["x"] * words)) <= real * 2.5, (
            f"{words} words estimated far beyond the {real:.2f}s it really took"
        )


def test_lines_that_only_collide_at_the_real_speaking_rate_are_caught():
    """Two lines 2.4s apart fit at reading pace and collide when spoken."""
    plan = _plan(4.0, 4.0, 4.0)
    plan.audio = AudioSpec(use_puter_tts=True, tts_lines=[
        TtsLine(text="do sabse takatwar shraap aamne saamne", start_time="00:00:00.500"),
        TtsLine(text="tum mujhe haara nahi sakte", start_time="00:00:02.900"),
    ])
    plan, diagnosis = review(plan)

    assert "voice_overlap" in _rules(diagnosis)
    starts = sorted(line.start_seconds for line in plan.audio.tts_lines)
    assert starts[1] - starts[0] > 2.4, "the second line was not pushed clear"
