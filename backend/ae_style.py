"""The AE hype grammar, as code rather than as advice.

:mod:`editing_skill` tells Kimi how this school cuts. This module makes the
renderer obey the same rules whether or not the model did, because the three
things that separate a good velocity edit from an amateur one are all
mechanical:

* **Accents are rationed.** A flash and a zoom punch on *every* beat is the
  single loudest tell that an edit was made by a preset. Real ones spend an
  accent every third or fourth hit and let the rest pass clean, so the ones
  that land actually read as impacts.
* **Speed ramps run into a hit, never over it.** The shot *before* the payoff
  accelerates; the payoff itself plays at normal speed, because the frame you
  spent four cuts building to is the one frame you must not blur past.
* **The hits are the music's, not the timeline's.** Cutting near a beat and
  cutting on it are different edits; only one of them feels intentional.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from schemas import EditPlan, TimelineSegment

logger = logging.getLogger(__name__)

# Themes that are cut to this grammar and should get its defaults.
VELOCITY_THEMES = {"ae_hype"}


def is_velocity_theme(theme: Optional[str]) -> bool:
    return (theme or "").strip().lower() in VELOCITY_THEMES


def select_accent_hits(
    hits: Sequence[float],
    every: int = 3,
    min_gap: float = 0.55,
    keep_last: bool = True,
) -> List[float]:
    """Thin a dense hit list down to the ones worth spending an accent on.

    Beat detection happily returns a hit every 400ms. Flashing all of them is
    not a style, it is a strobe: the eye stops registering any single one and
    the edit reads as uniform noise. Taking every ``every``-th hit, with a
    floor on the spacing, leaves the accents far enough apart to land.

    The final hit is always kept - the last impact is the one the whole edit
    was building toward, and dropping it because it fell on the wrong index
    ends the edit on a shrug.
    """
    ordered = sorted(float(hit) for hit in hits if hit is not None and hit > 0)
    if not ordered:
        return []
    if every <= 1:
        return ordered

    chosen: List[float] = []
    for index, hit in enumerate(ordered):
        if index % every:
            continue
        if chosen and hit - chosen[-1] < min_gap:
            continue
        chosen.append(hit)

    if keep_last:
        last = ordered[-1]
        if not chosen:
            chosen = [last]
        elif last - chosen[-1] >= min_gap:
            chosen.append(last)
        else:
            chosen[-1] = last
    return chosen


def apply_velocity_ramps(
    plan: EditPlan,
    ramp: float = 1.35,
    payoff_every: int = 4,
) -> int:
    """Accelerate into the payoff and let the payoff itself play straight.

    Returns how many segments were changed. Segments the planner already gave
    a deliberate speed to are left alone: an explicit slow-motion beat is a
    decision, not something to overwrite.
    """
    timeline = plan.edit_timeline
    if len(timeline) < 3 or ramp <= 1.0:
        return 0

    changed = 0
    # The payoff shots: every N-th, plus the last one - an edit that does not
    # end on a payoff has no ending.
    payoffs = set(range(payoff_every - 1, len(timeline), payoff_every))
    payoffs.add(len(timeline) - 1)

    for index, segment in enumerate(timeline):
        if abs((segment.speed or 1.0) - 1.0) > 0.01:
            continue  # the planner asked for this speed on purpose
        if index in payoffs:
            continue
        if index + 1 in payoffs:
            segment.speed = ramp
            segment.cut_type = "speed_ramp"
            changed += 1

    for index in sorted(payoffs):
        segment = timeline[index]
        if abs((segment.speed or 1.0) - 1.0) > 0.01:
            continue
        # The hit itself: a hard punch in, at normal speed.
        if segment.cut_type in ("jump_cut", "hard_cut", ""):
            segment.cut_type = "zoom_punch"
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


def score_segments(
    plan: EditPlan, analyses: Sequence[Any] = ()
) -> List[float]:
    """How strong each segment's footage is, from the measured motion curve.

    This is the one judgement the planner cannot make from a text description:
    it chose windows from a summary of the clip, but only the analysis knows
    which of those windows the picture is actually doing something in.
    """
    curves: Dict[int, List] = {}
    for index, analysis in enumerate(analyses or ()):
        curve = list(getattr(analysis, "motion_curve", None) or [])
        if curve:
            curves[index] = curve

    scores: List[float] = []
    for segment in plan.edit_timeline:
        curve = curves.get(segment.source_index or 0)
        if not curve:
            scores.append(0.0)
            continue
        low, high = segment.start_seconds, segment.end_seconds
        inside = [value for at, value in curve if low <= at <= high]
        scores.append(sum(inside) / len(inside) if inside else 0.0)
    return scores


def promote_hook(
    plan: EditPlan,
    analyses: Sequence[Any] = (),
    margin: float = 1.35,
) -> Optional[int]:
    """Open on the strongest shot, not on whichever one happens to be first.

    In short form the opening frame is the whole decision - a montage that
    saves its best moment for 0:12 is a montage nobody reaches 0:12 of. A
    montage has no narrative order to protect, so the strongest shot can
    simply be moved to the front.

    Only done when there is a clear winner: ``margin`` is how much better than
    the current opener a shot has to be before reordering is worth it. Swapping
    two near-identical shots churns the edit for nothing, and the planner may
    have opened where it did on purpose.

    Returns the index that was promoted, or ``None`` if nothing was.
    """
    timeline = plan.edit_timeline
    if len(timeline) < 3:
        return None

    scores = score_segments(plan, analyses)
    if not any(scores):
        return None

    best = max(range(len(scores)), key=lambda index: scores[index])
    if best == 0:
        return None
    # The payoff is the other place a strong shot belongs; stealing the ending
    # to open with leaves the edit finishing on its weakest material.
    if best == len(timeline) - 1:
        return None
    if scores[best] < scores[0] * margin:
        return None

    segment = timeline.pop(best)
    timeline.insert(0, segment)
    # An opener is a hard statement, not a transition into one.
    if (segment.cut_type or "").lower() in ("crossfade", "speed_ramp"):
        segment.cut_type = "hard_cut"
    return best
