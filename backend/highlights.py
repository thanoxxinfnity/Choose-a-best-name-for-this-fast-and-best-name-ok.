"""Finding the clip inside the movie.

The rest of the pipeline assumes it was handed footage worth editing. This
module handles the other case the app promises: a full episode, a match, a
two hour film - and the question of which forty seconds of it are the short.

It cannot use :mod:`video_analyzer`, which is built for a clip: that samples
densely and spends a vision call, neither of which survives contact with a
two hour source. Instead this makes two cheap streaming passes over the whole
runtime and scores every second of it:

* **Audio** - a loudness envelope, and how sharply loudness *jumps*. A shout,
  a hit, a crowd coming up: the moments a viewer would scrub to look loud
  before they look like anything.
* **Video** - frame difference at a couple of frames a second, which gives
  both motion and, at its spikes, where the shot changes.

Neither signal is trustworthy alone - a loud stretch can be a music bed over
nothing, a busy stretch can be a static camera on a shaking tree - so the
score is the combination, and the boundaries are snapped to real shot changes
so a clip never opens halfway through a sentence or a shot.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from puter_integration import ffmpeg_binary

logger = logging.getLogger(__name__)

# One score per second of runtime: fine enough to find a moment, coarse enough
# that a feature film is a few thousand numbers.
FRAME_RATE = 2.0          # frames per second sampled for motion
FRAME_WIDTH = 96          # motion only needs the shape of the picture
AUDIO_RATE = 8000         # loudness does not need fidelity

# A source shorter than this is already a clip; there is nothing to search.
MIN_SEARCHABLE = 75.0


@dataclass
class Highlight:
    start: float
    end: float
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


@dataclass
class HighlightReport:
    duration: float = 0.0
    highlights: List[Highlight] = field(default_factory=list)
    searched: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": round(self.duration, 2),
            "searched": self.searched,
            "error": self.error,
            "highlights": [item.to_dict() for item in self.highlights],
        }


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def loudness_per_second(path: Path, ffmpeg: Optional[str] = None) -> np.ndarray:
    """RMS loudness for every second of the file, normalised to its own peak."""
    ffmpeg = ffmpeg or ffmpeg_binary()
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"],
            capture_output=True, check=True,
        )
    except Exception as exc:
        logger.debug("highlight audio pass failed for %s: %s", path, exc)
        return np.zeros(0, dtype=np.float32)

    raw = np.frombuffer(result.stdout, dtype=np.int16)
    if raw.size < AUDIO_RATE:
        return np.zeros(0, dtype=np.float32)

    audio = raw.astype(np.float32) / 32768.0
    seconds = audio.size // AUDIO_RATE
    frames = audio[: seconds * AUDIO_RATE].reshape(seconds, AUDIO_RATE)
    envelope = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return _normalise(envelope)


def motion_per_second(
    path: Path, duration: float, ffmpeg: Optional[str] = None
) -> Tuple[np.ndarray, List[float]]:
    """Frame-difference energy per second, plus where the shot changes.

    Sampled at a couple of frames a second and at thumbnail size: a shot
    change is a whole-frame event and survives that, while decoding a feature
    film at full resolution does not.
    """
    ffmpeg = ffmpeg or ffmpeg_binary()
    height = FRAME_WIDTH * 9 // 16
    if height % 2:
        height += 1
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"fps={FRAME_RATE},scale={FRAME_WIDTH}:{height},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    frame_bytes = FRAME_WIDTH * height
    seconds = max(1, int(duration) + 1)
    totals = np.zeros(seconds, dtype=np.float64)
    counts = np.zeros(seconds, dtype=np.float64)
    cuts: List[float] = []

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
    except Exception as exc:
        logger.debug("highlight video pass failed for %s: %s", path, exc)
        return np.zeros(0, dtype=np.float32), []

    previous: Optional[np.ndarray] = None
    index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
            if previous is not None:
                delta = float(np.abs(frame - previous).mean()) / 255.0
                at = index / FRAME_RATE
                slot = min(int(at), seconds - 1)
                totals[slot] += delta
                counts[slot] += 1
                # A whole-frame change of this size is a shot change, not motion.
                if delta > 0.16:
                    cuts.append(round(at, 2))
            previous = frame
            index += 1
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait()

    if counts.sum() == 0:
        return np.zeros(0, dtype=np.float32), []
    motion = np.divide(totals, np.maximum(counts, 1.0))
    return _normalise(motion), cuts


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 against the 97th percentile, so one spike cannot flatten
    everything else into nothing."""
    if values.size == 0:
        return values.astype(np.float32)
    ceiling = float(np.percentile(values, 97))
    if ceiling <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / ceiling, 0.0, 1.0).astype(np.float32)


def _rise(values: np.ndarray, window: int = 8) -> np.ndarray:
    """How much each second is above the recent past.

    A sustained loud passage is a music bed; a sudden one is an event. The
    difference between the two is the whole reason a highlight is a highlight,
    and a plain loudness score cannot see it.
    """
    if values.size == 0:
        return values
    baseline = np.copy(values)
    for index in range(values.size):
        low = max(0, index - window)
        baseline[index] = values[low:index].mean() if index > low else values[index]
    return np.clip(values - baseline, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_seconds(
    loudness: np.ndarray,
    motion: np.ndarray,
    cuts: Sequence[float] = (),
) -> np.ndarray:
    """Blend the signals into one score per second."""
    length = max(loudness.size, motion.size)
    if length == 0:
        return np.zeros(0, dtype=np.float32)

    loud = _fit(loudness, length)
    move = _fit(motion, length)

    cut_density = np.zeros(length, dtype=np.float32)
    for at in cuts:
        slot = int(at)
        if 0 <= slot < length:
            cut_density[slot] += 1.0
    cut_density = _normalise(cut_density)

    # Weights: what a moment sounds like matters most, because audio is the
    # signal that survives a static camera; the rises matter nearly as much,
    # because they are what separates an event from a loud stretch.
    score = (0.34 * loud + 0.26 * _rise(loud) + 0.24 * move + 0.16 * cut_density)
    return _smooth(score.astype(np.float32), 3)


def _fit(values: np.ndarray, length: int) -> np.ndarray:
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    # The two passes can disagree by a second or two on a long file.
    return np.interp(
        np.linspace(0, values.size - 1, length),
        np.arange(values.size), values,
    ).astype(np.float32)


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size <= radius:
        return values
    kernel = np.ones(radius * 2 + 1, dtype=np.float32) / (radius * 2 + 1)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def pick_windows(
    score: np.ndarray,
    window: float,
    count: int,
    min_gap: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """The best non-overlapping windows, strongest first.

    Greedy with suppression rather than exhaustive: the top window is taken,
    everything within it (plus a gap) is struck out, then the next. Two
    highlights that overlap are one highlight reported twice.
    """
    span = max(1, int(round(window)))
    if score.size < span or count <= 0:
        return []

    # Integral image, so every candidate window is one subtraction.
    cumulative = np.concatenate([[0.0], np.cumsum(score.astype(np.float64))])
    totals = cumulative[span:] - cumulative[:-span]
    available = np.copy(totals)
    gap = int(round(min_gap))

    picked: List[Tuple[float, float, float]] = []
    for _ in range(count):
        if not np.isfinite(available).any():
            break
        best = int(np.nanargmax(available))
        if not np.isfinite(available[best]):
            break
        picked.append((float(best), float(best + span), float(totals[best] / span)))
        low = max(0, best - span - gap)
        high = min(available.size, best + span + gap)
        available[low:high] = -np.inf
    return picked


def snap_to_cuts(start: float, end: float, cuts: Sequence[float],
                 tolerance: float = 2.0) -> Tuple[float, float]:
    """Move the edges onto real shot changes when one is close enough.

    A clip that opens three frames into a shot looks like a mistake; one that
    opens on the cut looks chosen.
    """
    if not cuts:
        return start, end
    ordered = sorted(cuts)

    def nearest(value: float) -> float:
        best = min(ordered, key=lambda cut: abs(cut - value))
        return best if abs(best - value) <= tolerance else value

    new_start, new_end = nearest(start), nearest(end)
    if new_end - new_start < 1.0:
        return start, end
    return new_start, new_end


def _explain(loudness: np.ndarray, motion: np.ndarray,
             cuts: Sequence[float], start: float, end: float) -> List[str]:
    reasons: List[str] = []
    low, high = int(start), max(int(start) + 1, int(end))
    if loudness.size:
        window = loudness[low:min(high, loudness.size)]
        if window.size and float(window.mean()) > 0.55:
            reasons.append("loud through the whole window")
        if window.size and float(window.max()) > 0.9:
            reasons.append("carries the loudest moment in it")
    if motion.size:
        window = motion[low:min(high, motion.size)]
        if window.size and float(window.mean()) > 0.5:
            reasons.append("busy on screen")
    inside = [cut for cut in cuts if start <= cut <= end]
    if len(inside) >= 4:
        reasons.append(f"{len(inside)} shot changes inside it")
    return reasons or ["the strongest stretch of the source"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_highlights(
    path: Path,
    duration: Optional[float] = None,
    target_seconds: float = 35.0,
    count: int = 3,
    ffmpeg: Optional[str] = None,
) -> HighlightReport:
    """Find the ``count`` most promising short-form windows in a long source."""
    path = Path(path)
    ffmpeg = ffmpeg or ffmpeg_binary()

    if duration is None:
        from video_renderer import probe_clip
        duration = float(probe_clip(path).get("duration") or 0.0)
    report = HighlightReport(duration=duration)

    if duration <= 0:
        report.error = "The source has no readable duration."
        return report
    target = float(min(max(target_seconds, 5.0), 180.0))
    if duration < MIN_SEARCHABLE or duration < target * 1.5:
        # Already short form: the whole thing is the clip.
        report.highlights = [Highlight(0.0, duration, 1.0, ["the source is already short"])]
        return report

    loudness = loudness_per_second(path, ffmpeg)
    motion, cuts = motion_per_second(path, duration, ffmpeg)
    score = score_seconds(loudness, motion, cuts)
    if score.size == 0:
        report.error = "Neither the audio nor the video could be read."
        return report

    report.searched = True
    windows = pick_windows(score, target, count, min_gap=target * 0.5)
    for start, end, mean in windows:
        start, end = snap_to_cuts(start, min(end, duration), cuts)
        report.highlights.append(
            Highlight(start, end, mean, _explain(loudness, motion, cuts, start, end))
        )
    report.highlights.sort(key=lambda item: item.score, reverse=True)
    return report
