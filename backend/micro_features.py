"""Automatic editing micro-features that need no API key at all.

Everything here runs on the analysis :mod:`video_analyzer` already produced -
silence spans, beat onsets, motion - plus OpenCV on the raw frames.

* :func:`auto_silence_cut` - drop dead air out of the timeline.
* :func:`auto_beat_sync`   - snap every cut onto the nearest musical onset.
* :func:`plan_reframe`     - track the subject so the 9:16 crop follows it
  instead of blindly taking the middle of the frame.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from schemas import EditPlan, TimelineSegment, format_timecode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Auto silence-cut
# ---------------------------------------------------------------------------


def auto_silence_cut(
    plan: EditPlan,
    analyses: Sequence[Any],
    min_silence: float = 0.6,
    keep_padding: float = 0.12,
    min_segment: float = 0.5,
) -> Tuple[EditPlan, List[str]]:
    """Remove dead air from every segment.

    A segment that straddles a silent span is split around it; a segment that
    is entirely silent is dropped.  ``keep_padding`` leaves a breath at each
    edge so cuts do not land hard on the first syllable.
    """
    notes: List[str] = []
    if not plan.edit_timeline:
        return plan, notes

    spans_by_clip: Dict[int, List[Tuple[float, float]]] = {}
    for index, analysis in enumerate(analyses or []):
        spans = [
            (float(a), float(b))
            for a, b in (getattr(analysis, "silence_spans", None) or [])
            if float(b) - float(a) >= min_silence
        ]
        if spans:
            spans_by_clip[index] = sorted(spans)
    if not spans_by_clip:
        return plan, notes

    rebuilt: List[TimelineSegment] = []
    removed = 0.0
    for segment in plan.edit_timeline:
        source = segment.source_index or 0
        spans = spans_by_clip.get(source)
        if not spans:
            rebuilt.append(segment)
            continue

        keep = _subtract_spans(
            segment.start_seconds, segment.end_seconds, spans, keep_padding
        )
        original = segment.duration
        if not keep:
            removed += original
            continue

        for piece_start, piece_end in keep:
            if piece_end - piece_start < min_segment:
                removed += piece_end - piece_start
                continue
            piece = segment.model_copy(deep=True)
            piece.start_time = format_timecode(piece_start)
            piece.end_time = format_timecode(piece_end)
            rebuilt.append(piece)
        removed += original - sum(end - start for start, end in keep)

    if removed > 0.05:
        notes.append(
            f"Auto silence-cut removed {removed:.1f}s of dead air "
            f"({len(plan.edit_timeline)} segments -> {len(rebuilt)})."
        )
    plan.edit_timeline = rebuilt or plan.edit_timeline
    return plan, notes


def _subtract_spans(
    start: float,
    end: float,
    spans: Sequence[Tuple[float, float]],
    padding: float,
) -> List[Tuple[float, float]]:
    """Return the parts of ``[start, end)`` that are not covered by ``spans``."""
    pieces = [(start, end)]
    for span_start, span_end in spans:
        # Shrink the silence by the padding so speech keeps its edges.
        cut_start = span_start + padding
        cut_end = span_end - padding
        if cut_end <= cut_start:
            continue
        next_pieces: List[Tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if cut_end <= piece_start or cut_start >= piece_end:
                next_pieces.append((piece_start, piece_end))
                continue
            if cut_start > piece_start:
                next_pieces.append((piece_start, min(cut_start, piece_end)))
            if cut_end < piece_end:
                next_pieces.append((max(cut_end, piece_start), piece_end))
        pieces = next_pieces
        if not pieces:
            break
    return [(a, b) for a, b in pieces if b - a > 0.01]


# ---------------------------------------------------------------------------
# 2. Auto beat-sync
# ---------------------------------------------------------------------------


def auto_beat_sync(
    plan: EditPlan,
    analyses: Sequence[Any],
    tolerance: float = 0.28,
    min_segment: float = 0.4,
) -> Tuple[EditPlan, List[str]]:
    """Snap every cut onto the nearest beat within ``tolerance`` seconds.

    Only boundaries that are already close to a beat move, so the editorial
    intent survives - this tightens the cut, it does not re-cut the video.
    """
    notes: List[str] = []
    beats_by_clip: Dict[int, List[float]] = {}
    for index, analysis in enumerate(analyses or []):
        beats = sorted(float(beat) for beat in (getattr(analysis, "beats", None) or []))
        if beats:
            beats_by_clip[index] = beats
    if not beats_by_clip or not plan.edit_timeline:
        return plan, notes

    snapped = 0
    for segment in plan.edit_timeline:
        beats = beats_by_clip.get(segment.source_index or 0)
        if not beats:
            continue
        start = _nearest(beats, segment.start_seconds, tolerance)
        end = _nearest(beats, segment.end_seconds, tolerance)
        if end - start < min_segment:
            continue
        if abs(start - segment.start_seconds) > 1e-3:
            segment.start_time = format_timecode(start)
            snapped += 1
        if abs(end - segment.end_seconds) > 1e-3:
            segment.end_time = format_timecode(end)
            snapped += 1

    if snapped:
        bpm = next(
            (getattr(a, "bpm", 0.0) for a in (analyses or []) if getattr(a, "bpm", 0.0)), 0.0
        )
        notes.append(
            f"Auto beat-sync snapped {snapped} cut point(s) to the track"
            + (f" (~{bpm:.0f} BPM)." if bpm else ".")
        )
    return plan, notes


def _nearest(sorted_values: Sequence[float], target: float, tolerance: float) -> float:
    """Closest value to ``target``, or ``target`` itself if none is near enough."""
    if not sorted_values:
        return target
    position = bisect.bisect_left(sorted_values, target)
    candidates = []
    if position < len(sorted_values):
        candidates.append(sorted_values[position])
    if position > 0:
        candidates.append(sorted_values[position - 1])
    best = min(candidates, key=lambda value: abs(value - target))
    return best if abs(best - target) <= tolerance else target


# ---------------------------------------------------------------------------
# 3. Auto-reframe
# ---------------------------------------------------------------------------


@dataclass
class ReframeTrack:
    """Where the crop window should sit over time, in source pixels."""

    times: List[float]
    centres_x: List[float]
    centres_y: List[float]
    width: int
    height: int
    confident: bool = True

    def centre_at(self, t: float) -> Tuple[float, float]:
        if not self.times:
            return self.width / 2.0, self.height / 2.0
        position = bisect.bisect_left(self.times, t)
        if position <= 0:
            return self.centres_x[0], self.centres_y[0]
        if position >= len(self.times):
            return self.centres_x[-1], self.centres_y[-1]
        # Linear interpolation keeps the pan smooth between samples.
        t0, t1 = self.times[position - 1], self.times[position]
        blend = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        x = self.centres_x[position - 1] * (1 - blend) + self.centres_x[position] * blend
        y = self.centres_y[position - 1] * (1 - blend) + self.centres_y[position] * blend
        return x, y


_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(path))
        _FACE_CASCADE = cascade if not cascade.empty() else None
    return _FACE_CASCADE


def plan_reframe(
    video_path: Path,
    target_aspect: float,
    start: float = 0.0,
    end: Optional[float] = None,
    sample_fps: float = 4.0,
    smoothing: float = 0.25,
    max_samples: int = 400,
) -> Optional[ReframeTrack]:
    """Track the subject so a ``target_aspect`` crop can follow it.

    Faces win when the detector finds any; otherwise the centroid of frame-to-
    frame motion is used, falling back to edge density.  The raw trajectory is
    smoothed with an exponential filter and then clamped so the window never
    leaves the frame - a jittery crop looks far worse than a static one.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps else 0.0
        if width <= 0 or height <= 0 or duration <= 0:
            return None

        stop = min(end if end is not None else duration, duration)
        if stop <= start:
            return None

        # Source is already narrower than the target: nothing to pan across.
        if width / height <= target_aspect + 1e-3:
            return None

        crop_width = min(width, int(round(height * target_aspect)))
        step = 1.0 / max(sample_fps, 0.5)
        stamps = [start + index * step for index in range(int((stop - start) / step) + 1)]
        stamps = stamps[:max_samples] or [start]

        times: List[float] = []
        raw_x: List[float] = []
        raw_y: List[float] = []
        previous_gray: Optional[np.ndarray] = None
        hits = 0
        cascade = _face_cascade()
        scale = 240.0 / max(width, 1)

        for stamp in stamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            small = cv2.resize(frame, (240, max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            centre = None
            if cascade is not None:
                faces = cascade.detectMultiScale(gray, 1.15, 4, minSize=(18, 18))
                if len(faces):
                    fx, fy, fw, fh = max(faces, key=lambda face: face[2] * face[3])
                    centre = (fx + fw / 2.0, fy + fh / 2.0)
                    hits += 1

            if centre is None and previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                _ret, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                moments = cv2.moments(mask)
                if moments["m00"] > small.size * 0.01:
                    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                    hits += 1

            if centre is None:
                edges = cv2.Canny(gray, 60, 160)
                moments = cv2.moments(edges)
                if moments["m00"] > 0:
                    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

            previous_gray = gray
            if centre is None:
                continue
            times.append(stamp)
            raw_x.append(centre[0] / scale)
            raw_y.append(centre[1] / scale)

        if len(times) < 2:
            return None

        smooth_x = _smooth(raw_x, smoothing)
        smooth_y = _smooth(raw_y, smoothing)
        half = crop_width / 2.0
        smooth_x = [min(max(value, half), width - half) for value in smooth_x]
        smooth_y = [height / 2.0 for _ in smooth_y]  # vertical fill: no y pan needed

        return ReframeTrack(
            times=times, centres_x=smooth_x, centres_y=smooth_y,
            width=width, height=height,
            confident=hits >= max(2, len(times) // 4),
        )
    finally:
        capture.release()


def _smooth(values: Sequence[float], alpha: float) -> List[float]:
    """Two-pass exponential filter: forward then backward, so there is no lag."""
    if not values:
        return []
    alpha = min(max(alpha, 0.01), 1.0)
    forward: List[float] = [values[0]]
    for value in values[1:]:
        forward.append(forward[-1] * (1 - alpha) + value * alpha)
    backward: List[float] = [forward[-1]]
    for value in reversed(forward[:-1]):
        backward.append(backward[-1] * (1 - alpha) + value * alpha)
    backward.reverse()
    return backward


def apply_micro_features(
    plan: EditPlan,
    analyses: Sequence[Any],
    silence_cut: bool = False,
    beat_sync: bool = False,
) -> Tuple[EditPlan, List[str]]:
    """Run the timeline-level micro-features in the order that makes sense.

    Silence-cut first (it changes which ranges exist), beat-sync second (it
    tightens whatever survived onto the music).
    """
    notes: List[str] = []
    if silence_cut:
        plan, silence_notes = auto_silence_cut(plan, analyses)
        notes.extend(silence_notes)
    if beat_sync:
        plan, beat_notes = auto_beat_sync(plan, analyses)
        notes.extend(beat_notes)
    return plan, notes
