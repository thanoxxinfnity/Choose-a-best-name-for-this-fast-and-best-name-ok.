"""The AE hype grammar as the renderer enforces it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ae_style import (  # noqa: E402
    apply_velocity_ramps,
    is_velocity_theme,
    promote_hook,
    score_segments,
    select_accent_hits,
)
from schemas import EditPlan, TimelineSegment  # noqa: E402


def _plan(count: int) -> EditPlan:
    return EditPlan(edit_timeline=[
        TimelineSegment(start_time=str(i), end_time=str(i + 1), cut_type="jump_cut")
        for i in range(count)
    ])


# ------------------------------------------------------- accent rationing --

def test_accents_are_spent_sparingly_not_on_every_beat():
    """Flashing every hit is a strobe, not a style."""
    hits = [round(0.4 * i, 2) for i in range(1, 21)]
    chosen = select_accent_hits(hits, every=3)
    assert len(chosen) < len(hits) / 2
    assert set(chosen) <= set(hits)


def test_accents_never_crowd_each_other():
    """Two flashes half a beat apart read as one smeared mistake."""
    hits = [1.0, 1.05, 1.1, 1.15, 1.2, 3.0, 5.0]
    chosen = select_accent_hits(hits, every=2, min_gap=0.55)
    gaps = [b - a for a, b in zip(chosen, chosen[1:])]
    assert all(gap >= 0.55 for gap in gaps), chosen


def test_the_final_hit_is_always_an_accent():
    """The last impact is what the whole edit was building toward."""
    hits = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
    assert select_accent_hits(hits, every=3)[-1] == 9.0


def test_a_lone_hit_still_gets_its_accent():
    assert select_accent_hits([2.5], every=4) == [2.5]


def test_no_hits_means_no_accents():
    assert select_accent_hits([]) == []
    assert select_accent_hits([0.0, -1.0]) == []


def test_every_one_keeps_the_whole_list():
    hits = [1.0, 2.0, 3.0]
    assert select_accent_hits(hits, every=1) == hits


# ----------------------------------------------------------- speed ramps ---

def test_the_shot_before_a_payoff_accelerates_and_the_payoff_does_not():
    """You do not blur past the frame you spent four cuts building to."""
    plan = _plan(8)
    assert apply_velocity_ramps(plan, ramp=1.35, payoff_every=4) > 0

    speeds = [segment.speed for segment in plan.edit_timeline]
    # Index 3 and 7 are payoffs; 2 and 6 run into them.
    assert speeds[2] == pytest.approx(1.35)
    assert speeds[6] == pytest.approx(1.35)
    assert speeds[3] == pytest.approx(1.0)
    assert speeds[7] == pytest.approx(1.0)


def test_the_payoff_becomes_a_punch_in():
    plan = _plan(8)
    apply_velocity_ramps(plan, payoff_every=4)
    assert plan.edit_timeline[3].cut_type == "zoom_punch"
    assert plan.edit_timeline[-1].cut_type == "zoom_punch"


def test_the_edit_always_ends_on_a_payoff():
    """An edit that does not end on a hit has no ending."""
    plan = _plan(6)  # 6 is not a multiple of the payoff stride
    apply_velocity_ramps(plan, payoff_every=4)
    assert plan.edit_timeline[-1].cut_type == "zoom_punch"
    assert plan.edit_timeline[-1].speed == pytest.approx(1.0)


def test_a_deliberate_speed_is_never_overwritten():
    """An explicit slow-motion beat is a decision, not something to stomp on."""
    plan = _plan(8)
    plan.edit_timeline[2].speed = 0.4
    plan.edit_timeline[3].speed = 0.5
    apply_velocity_ramps(plan, ramp=1.35, payoff_every=4)
    assert plan.edit_timeline[2].speed == pytest.approx(0.4)
    assert plan.edit_timeline[3].speed == pytest.approx(0.5)


def test_a_short_timeline_is_left_alone():
    plan = _plan(2)
    assert apply_velocity_ramps(plan) == 0
    assert all(segment.speed == pytest.approx(1.0) for segment in plan.edit_timeline)


def test_only_the_velocity_theme_gets_this_treatment():
    assert is_velocity_theme("ae_hype")
    assert not is_velocity_theme("anime_edits")
    assert not is_velocity_theme("")
    assert not is_velocity_theme(None)


# ---------------------------------------------------------------- the hook --

class _Analysis:
    """Just the motion curve - the only field the hook scorer reads."""

    def __init__(self, curve):
        self.motion_curve = curve


def _curve(*bands):
    """[(start, end, value), ...] -> a motion curve sampled every 0.5s."""
    points = []
    for start, end, value in bands:
        at = start
        while at < end:
            points.append((round(at, 2), value))
            at += 0.5
    return points


def test_the_strongest_shot_is_moved_to_the_front():
    """In short form the opening frame is the whole decision."""
    plan = _plan(5)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index), str(index + 1)
    # Segment 3 (index 2) is where the picture is actually doing something.
    analysis = _Analysis(_curve((0.0, 2.0, 0.1), (2.0, 3.0, 0.9), (3.0, 5.0, 0.1)))

    promoted = promote_hook(plan, [analysis])
    assert promoted == 2
    assert plan.edit_timeline[0].start_seconds == 2.0


def test_an_edit_that_already_opens_strong_is_left_alone():
    plan = _plan(5)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index), str(index + 1)
    analysis = _Analysis(_curve((0.0, 1.0, 0.9), (1.0, 5.0, 0.1)))

    assert promote_hook(plan, [analysis]) is None
    assert plan.edit_timeline[0].start_seconds == 0.0


def test_a_marginal_winner_does_not_churn_the_edit():
    """Swapping two near-identical shots is motion without improvement."""
    plan = _plan(5)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index), str(index + 1)
    analysis = _Analysis(_curve((0.0, 1.0, 0.50), (1.0, 3.0, 0.1), (3.0, 4.0, 0.55),
                                (4.0, 5.0, 0.1)))

    assert promote_hook(plan, [analysis]) is None


def test_the_ending_is_never_stolen_to_open_with():
    """That leaves the edit finishing on its weakest material."""
    plan = _plan(4)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index), str(index + 1)
    analysis = _Analysis(_curve((0.0, 3.0, 0.1), (3.0, 4.0, 0.95)))

    assert promote_hook(plan, [analysis]) is None
    assert plan.edit_timeline[-1].start_seconds == 3.0


def test_a_promoted_opener_stops_being_a_transition():
    plan = _plan(5)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index), str(index + 1)
    plan.edit_timeline[2].cut_type = "crossfade"
    analysis = _Analysis(_curve((0.0, 2.0, 0.1), (2.0, 3.0, 0.9), (3.0, 5.0, 0.1)))

    assert promote_hook(plan, [analysis]) == 2
    assert plan.edit_timeline[0].cut_type == "hard_cut"


def test_without_an_analysis_nothing_is_reordered():
    """Reordering on no evidence is guessing, and it costs the planner's intent."""
    plan = _plan(5)
    assert promote_hook(plan, []) is None
    assert promote_hook(plan, [_Analysis([])]) is None


def test_a_two_shot_edit_is_left_alone():
    assert promote_hook(_plan(2), [_Analysis(_curve((0.0, 5.0, 0.5)))]) is None


def test_segment_scores_come_from_the_window_each_one_uses():
    plan = _plan(3)
    for index, segment in enumerate(plan.edit_timeline):
        segment.start_time, segment.end_time = str(index * 2), str(index * 2 + 2)
    analysis = _Analysis(_curve((0.0, 2.0, 0.2), (2.0, 4.0, 0.8), (4.0, 6.0, 0.4)))

    scores = score_segments(plan, [analysis])
    assert scores[1] > scores[2] > scores[0]
