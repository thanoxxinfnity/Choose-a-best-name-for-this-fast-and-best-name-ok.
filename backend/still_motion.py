"""Giving a still frame enough movement to sit in a cut.

Generated video costs money and generated stills, on the same account, do
not - so an edit that can carry stills is an edit that can be made when the
balance is empty. The catch is that a still dropped into a cut reads as a
freeze, and the usual answer, a slow Ken Burns push, reads as a slideshow.

What sells a still as a shot is that the camera has weight: it accelerates
out of the cut and settles rather than crawling at a constant rate, and it
moves on two axes so the frame is not just growing. That is the whole idea
here - ``ease`` shapes the move, ``MOVES`` gives each shot a different one so
consecutive stills do not drift the same way, and the crop is computed so the
frame never runs past the edge of the image.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


@dataclass(frozen=True)
class Move:
    """A camera move over a still, in fractions of the frame."""

    key: str
    label: str
    zoom_from: float = 1.06
    zoom_to: float = 1.18
    # Where the frame sits at the start and end, -1..1 of the spare margin.
    pan_from: Tuple[float, float] = (0.0, 0.0)
    pan_to: Tuple[float, float] = (0.0, 0.0)
    # 0 = constant speed (the slideshow look), 1 = all the movement up front.
    punch: float = 0.55
    rotate: float = 0.0          # degrees across the whole move


MOVES: Tuple[Move, ...] = (
    Move("push", "Push in", 1.04, 1.20, (0, 0), (0, 0), punch=0.55),
    Move("pull", "Pull out", 1.22, 1.05, (0, 0), (0, 0), punch=0.45),
    Move("push_left", "Push in, drift left", 1.06, 1.22, (0.35, 0), (-0.35, 0), punch=0.5),
    Move("push_right", "Push in, drift right", 1.06, 1.22, (-0.35, 0), (0.35, 0), punch=0.5),
    Move("rise", "Rise", 1.10, 1.20, (0, 0.45), (0, -0.45), punch=0.4),
    Move("fall", "Fall", 1.10, 1.20, (0, -0.45), (0, 0.45), punch=0.4),
    Move("slam", "Slam in", 1.02, 1.30, (0, 0), (0, 0), punch=0.85),
    Move("tilt", "Push in with a tilt", 1.08, 1.24, (0.2, 0.1), (-0.2, -0.1),
         punch=0.5, rotate=1.4),
)

BY_KEY: Dict[str, Move] = {move.key: move for move in MOVES}


def move_for(key: str) -> Move:
    return BY_KEY.get(key, MOVES[0])


def ease(t: float, punch: float) -> float:
    """Shape a 0..1 progress so the move has weight.

    ``punch`` 0 leaves it linear - the slideshow. Higher values spend more of
    the movement in the first part of the shot and settle into the cut, which
    is what a camera actually does and what makes a still stop reading as one.
    """
    t = min(max(float(t), 0.0), 1.0)
    punch = min(max(float(punch), 0.0), 1.0)
    eased = 1.0 - (1.0 - t) ** 3          # cubic settle
    return t * (1.0 - punch) + eased * punch


def frame_at(
    still: Frame,
    progress: float,
    move: Move,
    size: Tuple[int, int],
) -> Frame:
    """One frame of the move, cropped and scaled to ``size`` (width, height)."""
    width, height = size
    source_h, source_w = still.shape[:2]
    t = ease(progress, move.punch)

    zoom = move.zoom_from + (move.zoom_to - move.zoom_from) * t
    zoom = max(zoom, 1.001)

    # The crop that a given zoom implies, in source pixels, matched to the
    # output aspect so nothing is squeezed.
    target_ratio = width / height
    crop_h = source_h / zoom
    crop_w = crop_h * target_ratio
    if crop_w > source_w:
        crop_w = source_w / zoom
        crop_h = crop_w / target_ratio

    # Spare margin on each axis, and where in it this frame sits.
    spare_x = max(source_w - crop_w, 0.0) / 2.0
    spare_y = max(source_h - crop_h, 0.0) / 2.0
    pan_x = move.pan_from[0] + (move.pan_to[0] - move.pan_from[0]) * t
    pan_y = move.pan_from[1] + (move.pan_to[1] - move.pan_from[1]) * t

    centre_x = source_w / 2.0 + pan_x * spare_x
    centre_y = source_h / 2.0 + pan_y * spare_y
    left = int(round(min(max(centre_x - crop_w / 2.0, 0.0), source_w - crop_w)))
    top = int(round(min(max(centre_y - crop_h / 2.0, 0.0), source_h - crop_h)))
    patch = still[top:top + int(round(crop_h)), left:left + int(round(crop_w))]
    if patch.size == 0:
        patch = still

    out = cv2.resize(patch, (width, height), interpolation=cv2.INTER_CUBIC)

    if abs(move.rotate) > 0.01:
        angle = move.rotate * (t - 0.5)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.03)
        out = cv2.warpAffine(out, matrix, (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out


def animate(
    still: Frame,
    seconds: float,
    fps: float,
    size: Tuple[int, int],
    move: Move,
) -> list:
    """Every frame of one still's move, as a list of arrays."""
    count = max(int(round(seconds * fps)), 2)
    return [
        frame_at(still, index / (count - 1), move, size)
        for index in range(count)
    ]


def spread_moves(count: int, seed: int = 3) -> list:
    """Pick a move per shot so no two neighbours drift the same way.

    Two push-ins in a row read as one long push with a cut in it, which throws
    away the cut - so a move is never repeated back to back, and the strongest
    one is rationed rather than used wherever it fits.
    """
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    ordinary = [m for m in MOVES if m.key != "slam"]
    picked: list = []
    for index in range(count):
        # The slam is an accent: every fifth shot at most, never twice running.
        if index and index % 5 == 0 and picked[-1].key != "slam":
            picked.append(BY_KEY["slam"])
            continue
        options = [m for m in ordinary if not picked or m.key != picked[-1].key]
        picked.append(options[int(rng.integers(0, len(options)))])
    return picked
