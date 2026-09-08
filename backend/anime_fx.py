"""The effects an anime edit is actually made of.

The existing toolkit covers the plumbing - grading, shake, bloom, a flash on
the hit. What it did not have is the vocabulary: the abstract frame spliced
between two cuts, the speed lines behind a dash, the ghost trail on a fast
move, the stutter that turns one second of footage into a rhythm. Those are
what people mean when they say an edit is good, and none of them need a model
or a key - they are pixels, and they run offline.

Every function here takes a frame and returns a frame, so they compose with
``apply_frame_effect`` like everything else in vfx.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


# ---------------------------------------------------------------------------
# Impact frames
# ---------------------------------------------------------------------------

def impact_frame(
    size: Tuple[int, int],
    seed: int = 0,
    style: str = "burst",
    colour: Tuple[int, int, int] = (255, 255, 255),
    background: Tuple[int, int, int] = (12, 10, 16),
) -> Frame:
    """One abstract frame to splice between two cuts.

    This is the single most recognisable thing in the style and the easiest to
    get wrong: it is not a white card. It is a drawn shape - a burst, a crack,
    a slash - held for two or three frames, which is why it reads as force
    rather than as a dropped frame.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    centre = (width // 2, height // 2)
    ink = tuple(int(c) for c in colour)

    if style == "burst":
        # Rays from the centre, uneven so it does not read as a sunburst clip.
        for _ in range(rng.integers(26, 42)):
            angle = rng.random() * math.tau
            spread = 0.02 + rng.random() * 0.05
            length = max(width, height)
            points = [centre]
            for offset in (-spread, spread):
                points.append((
                    int(centre[0] + math.cos(angle + offset) * length),
                    int(centre[1] + math.sin(angle + offset) * length),
                ))
            cv2.fillPoly(canvas, [np.array(points, dtype=np.int32)], ink)
    elif style == "crack":
        # A jagged split with branches, drawn from one edge to the other.
        start = (int(rng.integers(0, width)), 0)
        end = (int(rng.integers(0, width)), height - 1)
        points = [start]
        steps = 9
        for index in range(1, steps):
            t = index / steps
            x = start[0] + (end[0] - start[0]) * t + rng.normal(0, width * 0.09)
            y = start[1] + (end[1] - start[1]) * t
            points.append((int(x), int(y)))
        points.append(end)
        thickness = max(3, width // 45)
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, ink, thickness, cv2.LINE_AA)
            if rng.random() < 0.55:                      # a branch off the split
                stub = (int(b[0] + rng.normal(0, width * 0.16)),
                        int(b[1] + rng.normal(0, height * 0.05)))
                cv2.line(canvas, b, stub, ink, max(2, thickness // 2), cv2.LINE_AA)
    else:  # "slash"
        for _ in range(rng.integers(2, 5)):
            angle = (rng.random() - 0.5) * 1.2 + math.pi / 4
            offset = rng.integers(-width // 2, width // 2)
            length = max(width, height) * 1.5
            a = (int(centre[0] + offset - math.cos(angle) * length),
                 int(centre[1] - math.sin(angle) * length))
            b = (int(centre[0] + offset + math.cos(angle) * length),
                 int(centre[1] + math.sin(angle) * length))
            cv2.line(canvas, a, b, ink, int(rng.integers(width // 30, width // 12)),
                     cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Speed lines
# ---------------------------------------------------------------------------

def speed_lines(
    frame: Frame,
    amount: float,
    centre: Tuple[float, float] = (0.5, 0.5),
    colour: Tuple[int, int, int] = (255, 255, 255),
    seed: int = 0,
) -> Frame:
    """Radial lines converging on a point, drawn over the frame.

    They stop short of the centre - lines that reach it cover the subject,
    which is the one thing the effect exists to draw attention to.
    """
    amount = float(min(max(amount, 0.0), 1.0))
    if amount <= 0.01:
        return frame
    height, width = frame.shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(frame)
    focus = (centre[0] * width, centre[1] * height)
    reach = math.hypot(width, height)
    hole = reach * (0.34 - 0.16 * amount)     # heavier lines close in further

    for _ in range(int(40 + 150 * amount)):
        angle = rng.random() * math.tau
        inner = hole * (0.85 + rng.random() * 0.5)
        outer = reach * (0.7 + rng.random() * 0.6)
        a = (int(focus[0] + math.cos(angle) * inner),
             int(focus[1] + math.sin(angle) * inner))
        b = (int(focus[0] + math.cos(angle) * outer),
             int(focus[1] + math.sin(angle) * outer))
        cv2.line(layer, a, b, tuple(int(c) for c in colour),
                 int(rng.integers(1, max(2, int(2 + 5 * amount)))), cv2.LINE_AA)

    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(0.6, width * 0.0016))
    return cv2.addWeighted(frame, 1.0, layer, 0.55 + 0.45 * amount, 0.0)


# ---------------------------------------------------------------------------
# Ghost trail
# ---------------------------------------------------------------------------

def echo_trail(
    frame: Frame,
    history: Sequence[Frame],
    strength: float = 0.55,
    tint: Optional[Tuple[int, int, int]] = None,
) -> Frame:
    """Earlier frames laid under this one, fading back in time.

    A real trail comes from the frames that actually happened, not from a
    blur - a blur smears the background too, and the background is not moving.
    """
    strength = float(min(max(strength, 0.0), 1.0))
    if strength <= 0.01 or not history:
        return frame

    out = frame.astype(np.float32)
    for age, earlier in enumerate(reversed(history), start=1):
        if earlier.shape != frame.shape:
            earlier = cv2.resize(earlier, (frame.shape[1], frame.shape[0]))
        ghost = earlier.astype(np.float32)
        if tint is not None:
            ghost = ghost * 0.55 + np.asarray(tint, dtype=np.float32) * 0.45
        # Lighten rather than average: a trail adds light where the subject has
        # been. Averaging would drag the whole shot toward the older frames,
        # including the background, which has not moved and must not smear.
        out = np.maximum(out, ghost * (strength * 0.62 ** age))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Colour flash
# ---------------------------------------------------------------------------

def colour_flash(frame: Frame, amount: float,
                 colour: Tuple[int, int, int] = (193, 18, 31)) -> Frame:
    """Push the whole frame toward an accent colour for a beat.

    Screen-blended rather than mixed, so the picture stays readable underneath
    instead of turning into a flat card of the accent.
    """
    amount = float(min(max(amount, 0.0), 1.0))
    if amount <= 0.005:
        return frame
    tint = np.full_like(frame, np.asarray(colour, dtype=np.uint8))
    source = frame.astype(np.float32) / 255.0
    over = tint.astype(np.float32) / 255.0
    screened = 1.0 - (1.0 - source) * (1.0 - over * amount)
    return np.clip(screened * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pixel sort
# ---------------------------------------------------------------------------

def pixel_sort(frame: Frame, amount: float, threshold: int = 120,
               vertical: bool = False) -> Frame:
    """Sort bright runs of pixels along each row - the datamosh look.

    Only runs brighter than ``threshold`` are sorted, so the effect follows the
    highlights instead of shredding the whole picture.
    """
    amount = float(min(max(amount, 0.0), 1.0))
    if amount <= 0.01:
        return frame
    work = np.rot90(frame).copy() if vertical else frame.copy()
    grey = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    height = work.shape[0]
    rows = np.linspace(0, height - 1, max(1, int(height * amount)), dtype=int)

    for row in np.unique(rows):
        mask = grey[row] > threshold
        if not mask.any():
            continue
        start = None
        for column, lit in enumerate(np.append(mask, False)):
            if lit and start is None:
                start = column
            elif not lit and start is not None:
                if column - start > 3:
                    piece = work[row, start:column]
                    order = np.argsort(piece.sum(axis=1))
                    work[row, start:column] = piece[order]
                start = None
    return np.rot90(work, -1) if vertical else work


# ---------------------------------------------------------------------------
# Light leak
# ---------------------------------------------------------------------------

def light_leak(frame: Frame, progress: float, seed: int = 0,
               colour: Tuple[int, int, int] = (120, 190, 255)) -> Frame:
    """A soft bloom of colour sweeping across the frame.

    Anchored to ``progress`` rather than to time, so it travels across the shot
    once however long the shot is.
    """
    progress = float(min(max(progress, 0.0), 1.0))
    height, width = frame.shape[:2]
    rng = np.random.default_rng(seed)
    angle = rng.random() * math.pi
    travel = -0.35 + 1.7 * progress

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xs = (xs / width - 0.5)
    ys = (ys / height - 0.5)
    along = xs * math.cos(angle) + ys * math.sin(angle)
    band = np.exp(-((along - (travel - 0.5)) ** 2) / (2 * 0.055 ** 2))
    band = band[:, :, None] * np.asarray(colour, dtype=np.float32)[None, None, :] / 255.0

    source = frame.astype(np.float32) / 255.0
    out = 1.0 - (1.0 - source) * (1.0 - band * 0.85)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stutter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stutter:
    """A rhythmic hold-and-jump over a stretch of frames."""

    hold: int = 2        # frames each held position lasts
    jump: int = 4        # frames skipped forward on each jump
    repeats: int = 3     # how many hold/jump pairs


def stutter_indices(count: int, spec: Stutter, start: int = 0) -> list:
    """Frame indices for a stutter, clamped to the clip.

    One second of footage becomes a rhythm rather than a second of footage.
    Every index is inside the clip, so the caller can index straight into it.
    """
    if count <= 0:
        return []
    out: list = []
    cursor = max(0, min(start, count - 1))
    for _ in range(max(spec.repeats, 1)):
        for _hold in range(max(spec.hold, 1)):
            out.append(min(cursor, count - 1))
        cursor = min(cursor + max(spec.jump, 1), count - 1)
    return out
