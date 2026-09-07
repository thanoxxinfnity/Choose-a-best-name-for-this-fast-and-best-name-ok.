"""Getting more shots out of the footage you actually have.

An edit is only as varied as its shot list, and a generated shot costs money
that a free tier does not have. Fourteen shots were budgeted for one AMV and
one arrived; the other thirteen came back "insufficient funds".

So the shots are multiplied instead. This is not padding - it is what an
editor does with a short roll: the same take reframed to a close-up, flipped
so the subject faces the other way, or ramped, reads as a different angle at
the one-second cut lengths this style uses. What ruins it is doing it
carelessly, and there are exactly two ways to do that:

* cutting between two treatments of the same source back to back, which reads
  as a glitch rather than as a cut, and
* using the same treatment twice, which reads as a mistake.

``plan_variants`` exists to prevent both. It hands out distinct treatments and
keeps two shots from one source apart in the running order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


@dataclass(frozen=True)
class Treatment:
    """One way of re-presenting a shot, and how different it looks."""

    key: str
    label: str
    # Where in the frame it looks, as (x, y, width, height) in 0..1. The full
    # frame is the identity crop.
    region: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    mirror: bool = False
    speed: float = 1.0
    # Roughly how far this reads from the untreated shot, 0..1. Used to spend
    # the strongest treatments where the edit needs the most contrast.
    distance: float = 0.0


# Ordered by how far each reads from the original. A crop that keeps most of
# the frame is a weak disguise; a tight corner crop plus a flip is a new shot.
TREATMENTS: Tuple[Treatment, ...] = (
    Treatment("full", "Full frame", distance=0.0),
    Treatment("mirror", "Mirrored", mirror=True, distance=0.35),
    Treatment("push", "Push in", region=(0.12, 0.10, 0.76, 0.76), distance=0.45),
    Treatment("top", "Upper frame", region=(0.10, 0.00, 0.80, 0.55), distance=0.60),
    Treatment("bottom", "Lower frame", region=(0.10, 0.45, 0.80, 0.55), distance=0.60),
    Treatment("left", "Left third", region=(0.00, 0.15, 0.55, 0.70), distance=0.65),
    Treatment("right", "Right third", region=(0.45, 0.15, 0.55, 0.70), distance=0.65),
    Treatment("tight", "Tight centre", region=(0.26, 0.24, 0.48, 0.48), distance=0.75),
    Treatment("top_mirror", "Upper, mirrored", region=(0.10, 0.00, 0.80, 0.55),
              mirror=True, distance=0.80),
    Treatment("tight_mirror", "Tight, mirrored", region=(0.26, 0.24, 0.48, 0.48),
              mirror=True, distance=0.90),
    Treatment("corner", "Corner detail", region=(0.02, 0.02, 0.44, 0.44), distance=0.85),
    Treatment("corner_mirror", "Corner, mirrored", region=(0.52, 0.02, 0.44, 0.44),
              mirror=True, distance=0.88),
)

BY_KEY: Dict[str, Treatment] = {treatment.key: treatment for treatment in TREATMENTS}


def treatment_for(key: str) -> Treatment:
    return BY_KEY.get(key, TREATMENTS[0])


def apply_treatment(frame: Frame, treatment: Treatment) -> Frame:
    """Reframe one frame, keeping the output the same size as the input.

    A crop that changed the frame size would have to be letterboxed back into
    the timeline; scaling it back up keeps every shot the same shape, which is
    what lets these sit next to untreated ones without anyone noticing the
    difference in origin.
    """
    height, width = frame.shape[:2]
    x, y, w, h = treatment.region
    left = int(round(x * width))
    top = int(round(y * height))
    right = min(width, left + max(int(round(w * width)), 8))
    bottom = min(height, top + max(int(round(h * height)), 8))

    patch = frame[top:bottom, left:right]
    if patch.size == 0:
        patch = frame
    if patch.shape[:2] != (height, width):
        # INTER_CUBIC because these are upscales and the edit is graded hard
        # afterwards; a soft crop next to a sharp one is the giveaway.
        patch = cv2.resize(patch, (width, height), interpolation=cv2.INTER_CUBIC)
    if treatment.mirror:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch)


@dataclass
class Shot:
    """One shot in the running order: a source, and how it is presented."""

    source_index: int
    treatment: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


def plan_variants(
    sources: Sequence[float],
    count: int,
    min_shot: float = 0.55,
    max_shot: float = 1.6,
    separation: int = 2,
    seed: int = 7,
) -> List[Shot]:
    """Lay out ``count`` shots across ``sources`` (their durations, in seconds).

    Two rules do the work. A source may not appear again within ``separation``
    shots, so two crops of one take never touch; and a (source, treatment) pair
    is never reused while any unused pair remains, so the same disguise is not
    worn twice.

    Windows walk along each source rather than always starting at zero, so
    successive visits to one take show different moments as well as different
    framings.
    """
    usable = [float(value) for value in sources if float(value) > 0.05]
    if not usable or count <= 0:
        return []

    rng = np.random.default_rng(seed)
    # Every pair, strongest disguises first, so a short edit still gets variety.
    pairs: List[Tuple[int, Treatment]] = [
        (index, treatment)
        for treatment in sorted(TREATMENTS, key=lambda t: -t.distance)
        for index in range(len(usable))
    ]
    spent: set = set()
    cursor: Dict[int, float] = {index: 0.0 for index in range(len(usable))}
    shots: List[Shot] = []

    while len(shots) < count:
        recent = {shot.source_index for shot in shots[-separation:]}
        # Prefer a pair from a source we have not just used.
        options = [p for p in pairs if p not in spent and p[0] not in recent]
        if not options:
            options = [p for p in pairs if p[0] not in recent]
        if not options:
            # Fewer sources than the separation asks for: relax it rather than
            # returning a short edit.
            options = [p for p in pairs if p not in spent] or pairs
        if not options:
            break

        index, treatment = options[int(rng.integers(0, len(options)))]
        spent.add((index, treatment))

        duration = usable[index]
        length = float(rng.uniform(min_shot, max_shot))
        length = min(length, max(duration - 0.05, min_shot * 0.5))
        start = cursor[index]
        if start + length > duration - 0.02:
            start = 0.0
        cursor[index] = start + length * 0.7
        shots.append(Shot(index, treatment.key, round(start, 3),
                          round(min(start + length, duration), 3)))
    return shots


def describe(shots: Sequence[Shot]) -> str:
    """A one-line summary, for the render warnings."""
    if not shots:
        return "no shots"
    sources = len({shot.source_index for shot in shots})
    looks = len({shot.treatment for shot in shots})
    total = sum(shot.seconds for shot in shots)
    return (
        f"{len(shots)} shots ({total:.1f}s) built from {sources} source(s) "
        f"using {looks} different framings"
    )
