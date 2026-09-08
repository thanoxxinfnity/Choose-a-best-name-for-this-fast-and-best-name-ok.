"""Reading the footage itself, rather than the plan that describes it.

plan_doctor checks the timeline. These check the pixels: whether two shots
were graded for the same scene, whether two shots are secretly the same shot,
and which single frame of a finished edit is worth putting on a thumbnail.

All three came out of building one AMV. Mixing generated stills with generated
video gave an edit where consecutive shots were plainly different colours;
multiplying seven sources into twenty-two shots needed a way to prove no two
had come out identical; and the finished file had no obvious frame to lead
with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


# ---------------------------------------------------------------------------
# Colour matching
# ---------------------------------------------------------------------------

def colour_signature(frame: Frame) -> np.ndarray:
    """Mean and spread per channel, in LAB - six numbers describing a grade."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.concatenate([lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))])


def grade_distance(a: Frame, b: Frame) -> float:
    """How far apart two frames are graded. 0 is identical.

    LAB rather than RGB because its axes are close to how the eye separates
    lightness from colour, so the number tracks what a viewer would call a
    mismatch. Lightness is deliberately part of it: a shot that is merely
    darker than the one before it *is* a cut that does not match, and matching
    exposure is half of what makes two sources belong to one edit.
    """
    left, right = colour_signature(a), colour_signature(b)
    return float(np.abs(left - right).mean())


def match_colour(frame: Frame, reference: Frame, strength: float = 1.0) -> Frame:
    """Re-grade ``frame`` toward ``reference``'s look.

    Mean and standard deviation are transferred per LAB channel - the classic
    Reinhard transfer. It is enough to make footage from two sources belong to
    one edit, which is the whole job here, and it does not need either shot to
    contain the same subject.
    """
    strength = float(min(max(strength, 0.0), 1.0))
    if strength <= 0.005:
        return frame

    source = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    out = source.copy()
    for channel in range(3):
        plane = source[:, :, channel]
        s_mean, s_std = plane.mean(), plane.std()
        t_mean, t_std = target[:, :, channel].mean(), target[:, :, channel].std()
        # A flat channel has no spread to rescale, but it still has the wrong
        # mean - skipping it entirely left a flat shot at its original colour,
        # which is exactly the case a solid background hits.
        gain = (t_std / s_std) if s_std >= 1e-3 else 1.0
        shifted = (plane - s_mean) * gain + t_mean
        out[:, :, channel] = plane * (1 - strength) + shifted * strength

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Duplicate shots
# ---------------------------------------------------------------------------

def perceptual_hash(frame: Frame, size: int = 8) -> int:
    """A 64-bit fingerprint of a frame's structure, ignoring colour and scale."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       (size, size), interpolation=cv2.INTER_AREA)
    bits = small > small.mean()
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hash_distance(left: int, right: int) -> int:
    """Bits that differ. 0 is identical, 64 is nothing in common."""
    return bin(left ^ right).count("1")


def find_duplicates(hashes: Sequence[int], tolerance: int = 6) -> List[Tuple[int, int]]:
    """Pairs of shots close enough to read as the same shot twice.

    Six bits of sixty-four is the useful line: below it two shots differ only
    by grade or a small move, which an audience reads as a repeat.
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hash_distance(hashes[i], hashes[j]) <= tolerance:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

@dataclass
class FrameScore:
    seconds: float
    score: float
    sharpness: float
    contrast: float
    colourfulness: float
    faces: int


def score_frame(frame: Frame, faces: int = 0) -> Tuple[float, Dict[str, float]]:
    """How well one frame would work as a thumbnail.

    A thumbnail competes at the size of a fingernail, so what matters is not
    whether the frame is pretty but whether it survives being shrunk: is it
    sharp, does it have contrast, does it have colour, is there a face.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    contrast = float(grey.std())
    channels = frame.astype(np.float32)
    rg = np.abs(channels[:, :, 2] - channels[:, :, 1])
    yb = np.abs(0.5 * (channels[:, :, 2] + channels[:, :, 1]) - channels[:, :, 0])
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # A frame that is nearly black scores nothing however sharp its noise is.
    if grey.mean() < 18:
        return 0.0, {"sharpness": sharpness, "contrast": contrast,
                     "colourfulness": colourfulness, "faces": float(faces)}

    score = (min(sharpness / 400.0, 1.0) * 0.32
             + min(contrast / 70.0, 1.0) * 0.30
             + min(colourfulness / 60.0, 1.0) * 0.23
             + min(faces, 2) * 0.075)
    return float(score), {"sharpness": sharpness, "contrast": contrast,
                          "colourfulness": colourfulness, "faces": float(faces)}


def pick_thumbnail(
    video: Path,
    samples: int = 40,
    detect_faces: bool = True,
    skip_edges: float = 0.06,
) -> Optional[FrameScore]:
    """The frame of a finished edit most worth leading with.

    The first and last few percent are skipped: an edit usually opens on a
    title card and ends on a held frame, and neither is what the video is
    about.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        capture.release()
        return None

    cascade = None
    if detect_faces:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            loaded = cv2.CascadeClassifier(path)
            cascade = loaded if not loaded.empty() else None
        except Exception:
            cascade = None

    low = int(total * skip_edges)
    high = int(total * (1.0 - skip_edges))
    positions = np.linspace(low, max(high, low + 1), max(samples, 2), dtype=int)

    best: Optional[FrameScore] = None
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            continue
        faces = 0
        if cascade is not None:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = len(cascade.detectMultiScale(grey, 1.2, 5))
        score, parts = score_frame(frame, faces)
        if best is None or score > best.score:
            best = FrameScore(seconds=float(position) / fps, score=score,
                              sharpness=parts["sharpness"], contrast=parts["contrast"],
                              colourfulness=parts["colourfulness"], faces=faces)
    capture.release()
    return best
