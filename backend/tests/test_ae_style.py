"""The AE hype grammar as the renderer enforces it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ae_style import (  # noqa: E402
    apply_velocity_ramps,
    is_velocity_theme,
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
