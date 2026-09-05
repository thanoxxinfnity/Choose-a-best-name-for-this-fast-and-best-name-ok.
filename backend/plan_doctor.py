"""Watching the cut back before it is rendered.

Every other stage of this pipeline is a writer: Kimi writes the timeline, the
analyser writes what is in the footage, the trend research writes what wins.
Nothing was reading. A real editor's second pass is not more ideas, it is
sitting through what they just built and noticing that four shots in a row are
the same length, that a voice line lands on top of a hit, that the last card
is a stub.

That pass is mostly mechanical, which is exactly why a language model is bad
at it and a measurement is good at it: counting shot lengths, spotting an
overlay behind a sticker, checking a timecode against the footage that
actually exists. So this module measures the plan against the same craft rules
:mod:`editing_skill` teaches Kimi, fixes what is safely fixable, and reports
what is not - both to the user and back to the model, which is how the model
gets a second try instead of a second guess.

Nothing here invents content. It moves, trims, demotes and re-times what the
planner already decided; anything that would need a new creative choice is
reported rather than guessed at.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from schemas import EditPlan, TimelineSegment, format_timecode, parse_timecode

logger = logging.getLogger(__name__)

# Severity is about what it costs the viewer, not how odd it looks in JSON.
BLOCKING = "blocking"   # the render will be wrong or will fail
SERIOUS = "serious"     # the edit will land noticeably worse
MINOR = "minor"         # worth saying, not worth stopping for


@dataclass
class Finding:
    rule: str
    severity: str
    detail: str
    where: Optional[int] = None      # segment index, when it is about one
    fixed: bool = False

    def to_line(self) -> str:
        location = f"segment {self.where + 1}: " if self.where is not None else ""
        mark = "fixed" if self.fixed else self.severity
        return f"[{mark}] {location}{self.detail}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "detail": self.detail,
            "segment": self.where,
            "fixed": self.fixed,
        }


@dataclass
class Diagnosis:
    findings: List[Finding] = field(default_factory=list)

    @property
    def fixed(self) -> List[Finding]:
        return [item for item in self.findings if item.fixed]

    @property
    def outstanding(self) -> List[Finding]:
        return [item for item in self.findings if not item.fixed]

    def blocking(self) -> List[Finding]:
        return [item for item in self.outstanding if item.severity == BLOCKING]

    def to_notes(self, limit: int = 8) -> List[str]:
        return [finding.to_line() for finding in self.findings[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [item.to_dict() for item in self.findings],
            "fixed": len(self.fixed),
            "outstanding": len(self.outstanding),
        }

    def to_prompt_block(self, limit: int = 10) -> str:
        """What to hand back to the model when asking it to try again."""
        outstanding = self.outstanding[:limit]
        if not outstanding:
            return ""
        lines = ["[REVIEW OF YOUR TIMELINE - fix each of these and re-emit the JSON]"]
        lines.extend(f"  - {finding.to_line()}" for finding in outstanding)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The examination
# ---------------------------------------------------------------------------


def diagnose(
    plan: EditPlan,
    clip_durations: Sequence[float] = (),
    theme: Optional[Any] = None,
    treat: bool = True,
) -> Diagnosis:
    """Read the plan back and report - and by default repair - what is wrong."""
    diagnosis = Diagnosis()
    timeline = plan.edit_timeline
    if not timeline:
        diagnosis.findings.append(Finding(
            "empty_timeline", BLOCKING, "the timeline has no segments at all."))
        return diagnosis

    _check_footage_exists(plan, clip_durations, diagnosis, treat)
    _check_the_hook(plan, diagnosis, treat)
    _check_rhythm(plan, theme, diagnosis, treat)
    _check_accent_inflation(plan, diagnosis, treat)
    _check_sticker_spacing(plan, diagnosis, treat)
    _check_text_collisions(plan, diagnosis, treat)
    _check_text_length(plan, diagnosis, treat)
    _check_voice_timing(plan, diagnosis, treat)
    _check_the_ending(plan, diagnosis, treat)
    return diagnosis


def _segment_lengths(timeline: Sequence[TimelineSegment]) -> List[float]:
    return [
        max(0.0, segment.end_seconds - segment.start_seconds) / max(segment.speed or 1.0, 0.01)
        for segment in timeline
    ]


def _check_footage_exists(
    plan: EditPlan, clip_durations: Sequence[float], diagnosis: Diagnosis, treat: bool
) -> None:
    """A timeline that references footage that does not exist cannot render."""
    if not clip_durations:
        return
    for index, segment in enumerate(plan.edit_timeline):
        source = segment.source_index or 0
        if source >= len(clip_durations):
            diagnosis.findings.append(Finding(
                "missing_source", BLOCKING,
                f"points at clip {source + 1}, but only {len(clip_durations)} "
                f"{'clip was' if len(clip_durations) == 1 else 'clips were'} uploaded.",
                where=index, fixed=treat,
            ))
            if treat:
                segment.source_index = source % len(clip_durations)
                source = segment.source_index

        available = clip_durations[source]
        if segment.end_seconds <= available + 0.05:
            continue
        diagnosis.findings.append(Finding(
            "past_the_end", BLOCKING,
            f"ends at {segment.end_seconds:.1f}s but clip {source + 1} is only "
            f"{available:.1f}s long.",
            where=index, fixed=treat,
        ))
        if treat:
            length = min(segment.end_seconds - segment.start_seconds, available)
            end = min(segment.end_seconds, available)
            start = max(0.0, end - length)
            segment.start_time = format_timecode(start)
            segment.end_time = format_timecode(end)


def _check_the_hook(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """The hook is frame one. Long-form grammar in the opening loses the viewer."""
    first = plan.edit_timeline[0]
    if (first.cut_type or "").lower() == "crossfade":
        diagnosis.findings.append(Finding(
            "soft_open", SERIOUS,
            "opens on a crossfade - short form has no room to fade in.",
            where=0, fixed=treat,
        ))
        if treat:
            first.cut_type = "hard_cut"

    length = _segment_lengths(plan.edit_timeline)[0]
    if length > 2.6:
        diagnosis.findings.append(Finding(
            "slow_open", SERIOUS,
            f"the opening shot runs {length:.1f}s before the first cut; the promise "
            f"has to land inside the first second.",
            where=0, fixed=treat,
        ))
        if treat:
            speed = first.speed or 1.0
            first.end_time = format_timecode(first.start_seconds + 2.0 * speed)


def _check_rhythm(
    plan: EditPlan, theme: Optional[Any], diagnosis: Diagnosis, treat: bool
) -> None:
    """A run of identically long shots goes numb, however good each one is."""
    lengths = _segment_lengths(plan.edit_timeline)
    if len(lengths) < 4:
        return

    mean = statistics.fmean(lengths)
    if mean <= 0:
        return
    spread = statistics.pstdev(lengths) / mean
    if spread >= 0.18:
        return

    diagnosis.findings.append(Finding(
        "metronome", SERIOUS,
        f"every shot is about {mean:.1f}s - a metronome edit stops registering. "
        f"Shot lengths need to vary before the payoff.",
        fixed=treat,
    ))
    if not treat:
        return

    # Alternate short and long around the mean rather than randomising: the
    # goal is a rhythm, and random lengths are just a different kind of flat.
    for index, segment in enumerate(plan.edit_timeline):
        if index == len(plan.edit_timeline) - 1:
            continue  # leave the payoff its length
        scale = 0.72 if index % 2 else 1.28
        speed = segment.speed or 1.0
        start = segment.start_seconds
        length = (segment.end_seconds - start)
        segment.end_time = format_timecode(start + max(0.35 * speed, length * scale))


def _check_accent_inflation(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """Applied to every cut, an accent stops being an accent."""
    timeline = plan.edit_timeline
    punches = [i for i, s in enumerate(timeline) if (s.cut_type or "").lower() == "zoom_punch"]
    if len(timeline) < 4 or len(punches) <= max(2, len(timeline) // 3):
        return

    diagnosis.findings.append(Finding(
        "accent_inflation", SERIOUS,
        f"{len(punches)} of {len(timeline)} cuts are zoom punches; an accent on "
        f"every cut is a headache, not emphasis.",
        fixed=treat,
    ))
    if not treat:
        return
    # Keep every third punch, and always the payoff. When the payoff is right
    # next to one that survived, it *replaces* it rather than joining it -
    # otherwise thinning leaves the two loudest accents back to back, which is
    # the crowding this check exists to remove.
    keep = set(punches[::3])
    last = punches[-1]
    if last not in keep:
        neighbour = max((index for index in keep if last - index <= 1), default=None)
        if neighbour is not None:
            keep.discard(neighbour)
        keep.add(last)
    for index in punches:
        if index not in keep:
            timeline[index].cut_type = "jump_cut"


def _check_sticker_spacing(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """A sticker is punctuation; one every cut is noise and none of them read."""
    timeline = plan.edit_timeline
    carrying = [i for i, s in enumerate(timeline) if s.puter_sticker and s.puter_sticker.generate_prompt]
    if not carrying:
        return

    dropped: List[int] = []
    previous: Optional[int] = None
    for index in carrying:
        if previous is not None and index - previous < 2:
            dropped.append(index)
            continue
        previous = index

    if dropped:
        diagnosis.findings.append(Finding(
            "sticker_crowding", MINOR,
            f"{len(dropped)} sticker(s) sit on back-to-back shots; they need a "
            f"shot between them to read as punctuation.",
            fixed=treat,
        ))
        if treat:
            for index in dropped:
                timeline[index].puter_sticker = None


def _check_text_collisions(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """Text and a sticker in the same third of the frame fight each other."""
    thirds = {
        "top_left": "top", "top_right": "top", "top_center": "top",
        "center": "middle", "center_left": "middle", "center_right": "middle",
        "bottom_left": "bottom", "bottom_right": "bottom", "bottom_center": "bottom",
    }
    escape = {"top": "bottom_right", "middle": "top_right", "bottom": "top_right"}

    for index, segment in enumerate(plan.edit_timeline):
        overlay, sticker = segment.text_overlay, segment.puter_sticker
        if not (overlay and overlay.active and sticker and sticker.generate_prompt):
            continue
        text_third = thirds.get((overlay.position or "").lower(), "middle")
        sticker_third = thirds.get((sticker.position or "").lower(), "bottom")
        if text_third != sticker_third:
            continue
        diagnosis.findings.append(Finding(
            "overlap", SERIOUS,
            f"the text and the sticker are both in the {text_third} third.",
            where=index, fixed=treat,
        ))
        if treat:
            sticker.position = escape[text_third]


def _check_text_length(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """Text competes with the picture; a sentence loses."""
    for index, segment in enumerate(plan.edit_timeline):
        overlay = segment.text_overlay
        if not (overlay and overlay.active):
            continue
        words = overlay.text.split()
        if len(words) <= 5:
            continue
        diagnosis.findings.append(Finding(
            "wordy_overlay", MINOR,
            f"the overlay is {len(words)} words - on screen for a second, only the "
            f"first few are read at all.",
            where=index, fixed=treat,
        ))
        if treat:
            overlay.text = " ".join(words[:4])


def _check_voice_timing(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """Voice lines go in the gaps between hits, and never over each other."""
    lines = plan.audio.timed_lines
    if not lines:
        return

    # Where the hits actually are, in finished-edit time.
    boundaries: List[float] = []
    running = 0.0
    for length in _segment_lengths(plan.edit_timeline):
        running += length
        boundaries.append(running)
    total = running

    ordered = sorted(lines, key=lambda line: line.start_seconds)
    previous_end = 0.0
    for position, line in enumerate(ordered):
        start = line.start_seconds
        # A rough duration: reading pace is about 2.6 words a second.
        spoken = max(1.2, len(line.text.split()) / 2.6)

        if start < previous_end - 0.05:
            diagnosis.findings.append(Finding(
                "voice_overlap", SERIOUS,
                f"voice line {position + 1} starts at {start:.1f}s while the one "
                f"before it is still speaking.",
                fixed=treat,
            ))
            if treat:
                start = previous_end + 0.25
                line.start_time = format_timecode(start)

        near = [hit for hit in boundaries if abs(hit - start) < 0.3 and hit < total - 0.1]
        if near:
            diagnosis.findings.append(Finding(
                "voice_on_the_hit", MINOR,
                f"voice line {position + 1} starts on a cut at {start:.1f}s; a line "
                f"landing on the hit fights it.",
                fixed=treat,
            ))
            if treat:
                start = near[0] + 0.35
                line.start_time = format_timecode(start)

        if start + spoken > total + 0.5:
            diagnosis.findings.append(Finding(
                "voice_past_the_end", SERIOUS,
                f"voice line {position + 1} would still be speaking {start + spoken - total:.1f}s "
                f"after the edit ends.",
                fixed=False,
            ))
        previous_end = start + spoken


def _check_the_ending(plan: EditPlan, diagnosis: Diagnosis, treat: bool) -> None:
    """End on the strongest remaining moment, not on whatever is left over."""
    timeline = plan.edit_timeline
    if len(timeline) < 2:
        return
    lengths = _segment_lengths(timeline)
    last = lengths[-1]
    if last >= 0.45:
        return

    diagnosis.findings.append(Finding(
        "stub_ending", SERIOUS,
        f"the edit ends on a {last:.2f}s stub, which reads as the video being cut off.",
        where=len(timeline) - 1, fixed=treat,
    ))
    if treat:
        segment = timeline[-1]
        speed = segment.speed or 1.0
        segment.end_time = format_timecode(segment.start_seconds + 1.1 * speed)


# ---------------------------------------------------------------------------
# Entry point used by the pipeline
# ---------------------------------------------------------------------------


def review(
    plan: EditPlan,
    clip_durations: Sequence[float] = (),
    theme: Optional[Any] = None,
) -> Tuple[EditPlan, Diagnosis]:
    """Read the plan back, repair it in place, and report what was found."""
    diagnosis = diagnose(plan, clip_durations, theme, treat=True)
    if diagnosis.findings:
        logger.info(
            "plan doctor: %d finding(s), %d repaired",
            len(diagnosis.findings), len(diagnosis.fixed),
        )
    return plan, diagnosis
