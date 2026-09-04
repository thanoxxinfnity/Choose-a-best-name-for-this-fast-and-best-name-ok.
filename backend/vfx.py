"""Frame level video effects: shake, RGB split, motion blur, grading, vignette.

Every effect here is a pure ``numpy`` function over an RGB frame plus a small
wrapper that applies it to a MoviePy clip through ``transform``/``fl``, so they
compose freely and work on both MoviePy 1.x and 2.x.

The shake family is the one anime edits live on:

* ``directional`` - a decaying kick along one axis, used on impact frames,
* ``random``      - handheld jitter,
* ``pulse``       - a rhythmic in/out punch locked to beat timestamps,
* ``rgb_split``   - chromatic aberration that ramps with the shake,
* ``motion_blur`` - directional smear that follows the shake vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Frame = np.ndarray


# ---------------------------------------------------------------------------
# Frame primitives
# ---------------------------------------------------------------------------


def shift_frame(frame: Frame, dx: float, dy: float, zoom: float = 1.0) -> Frame:
    """Translate (and optionally scale) a frame, keeping its size."""
    height, width = frame.shape[:2]
    if abs(zoom - 1.0) > 1e-3:
        scaled_w, scaled_h = int(width * zoom), int(height * zoom)
        frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        x0 = max((scaled_w - width) // 2, 0)
        y0 = max((scaled_h - height) // 2, 0)
        frame = frame[y0:y0 + height, x0:x0 + width]
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return frame
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )


def rgb_split(frame: Frame, amount: float, angle: float = 0.0) -> Frame:
    """Chromatic aberration: push R and B apart along ``angle``."""
    if amount < 0.4:
        return frame
    dx = math.cos(angle) * amount
    dy = math.sin(angle) * amount
    red = shift_frame(frame[:, :, 0], dx, dy)
    blue = shift_frame(frame[:, :, 2], -dx, -dy)
    out = frame.copy()
    out[:, :, 0] = red
    out[:, :, 2] = blue
    return out


def directional_blur(frame: Frame, dx: float, dy: float, taps: int = 7) -> Frame:
    """Cheap motion blur: average a few copies along the motion vector."""
    length = math.hypot(dx, dy)
    if length < 0.8:
        return frame
    accumulator = np.zeros_like(frame, dtype=np.float32)
    for tap in range(taps):
        blend = (tap / (taps - 1)) - 0.5
        accumulator += shift_frame(frame, dx * blend, dy * blend).astype(np.float32)
    return np.clip(accumulator / taps, 0, 255).astype(np.uint8)


def colour_grade(
    frame: Frame,
    saturation: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
    tint: Optional[Tuple[float, float, float]] = None,
    gamma: float = 1.0,
) -> Frame:
    out = frame.astype(np.float32)
    if abs(gamma - 1.0) > 1e-3:
        out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), gamma)
    if abs(contrast - 1.0) > 1e-3:
        out = (out - 128.0) * contrast + 128.0
    if abs(brightness) > 1e-3:
        out += brightness * 255.0
    if abs(saturation - 1.0) > 1e-3:
        grey = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = grey[:, :, None] + (out - grey[:, :, None]) * saturation
    if tint:
        out *= np.array(tint, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


_VIGNETTE_CACHE: dict = {}


def vignette(frame: Frame, strength: float = 0.35) -> Frame:
    if strength <= 0.01:
        return frame
    height, width = frame.shape[:2]
    key = (height, width, round(strength, 3))
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        radius = np.sqrt(xs ** 2 + ys ** 2) / math.sqrt(2.0)
        mask = np.clip(1.0 - strength * radius ** 2.2, 0.0, 1.0)[:, :, None]
        if len(_VIGNETTE_CACHE) > 8:
            _VIGNETTE_CACHE.clear()
        _VIGNETTE_CACHE[key] = mask
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def glow_bloom(frame: Frame, strength: float = 0.3, threshold: int = 190) -> Frame:
    """Screen-blend a blurred copy of the highlights back over the frame."""
    if strength <= 0.01:
        return frame
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _ret, mask = cv2.threshold(grey, threshold, 255, cv2.THRESH_TOZERO)
    highlights = cv2.bitwise_and(frame, frame, mask=(mask > 0).astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(highlights, (0, 0), sigmaX=frame.shape[1] * 0.012)
    base = frame.astype(np.float32) / 255.0
    bloom = blurred.astype(np.float32) / 255.0
    screened = 1.0 - (1.0 - base) * (1.0 - bloom * strength)
    return np.clip(screened * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Shake generators - time -> (dx, dy, zoom, split)
# ---------------------------------------------------------------------------


@dataclass
class ShakeSpec:
    kind: str = "none"           # none | random | directional | pulse
    intensity: float = 0.0       # pixels of travel at 1080p
    frequency: float = 9.0       # Hz for random/handheld
    hits: Sequence[float] = ()   # impact/beat timestamps for directional/pulse
    decay: float = 0.16          # seconds an impact takes to settle
    zoom: float = 0.0            # extra punch-in at an impact
    rgb_split: float = 0.0       # chromatic aberration at full intensity
    motion_blur: bool = False


def shake_at(spec: ShakeSpec, t: float, seed: float = 0.0) -> Tuple[float, float, float, float]:
    """Return ``(dx, dy, zoom, split)`` for time ``t``."""
    if spec.kind == "none" or spec.intensity <= 0:
        return 0.0, 0.0, 1.0, 0.0

    if spec.kind == "random":
        phase = (t + seed) * spec.frequency
        dx = math.sin(phase * 2.13) * spec.intensity
        dy = math.cos(phase * 1.71 + 1.3) * spec.intensity * 0.8
        return dx, dy, 1.0, spec.rgb_split * 0.35

    # directional / pulse both key off the nearest preceding hit.
    envelope, index = _hit_envelope(spec, t)
    if envelope <= 0.001:
        return 0.0, 0.0, 1.0, 0.0

    if spec.kind == "pulse":
        return 0.0, 0.0, 1.0 + spec.zoom * envelope, spec.rgb_split * envelope

    angle = (index * 2.399963)  # golden angle -> a different direction each hit
    dx = math.cos(angle) * spec.intensity * envelope
    dy = math.sin(angle) * spec.intensity * envelope
    return dx, dy, 1.0 + spec.zoom * envelope, spec.rgb_split * envelope


def _hit_envelope(spec: ShakeSpec, t: float) -> Tuple[float, int]:
    best = 0.0
    best_index = 0
    for index, hit in enumerate(spec.hits):
        delta = t - hit
        if 0.0 <= delta <= spec.decay:
            # Damped oscillation: strong kick, quick settle.
            envelope = math.exp(-delta / (spec.decay * 0.34)) * abs(
                math.cos(delta / max(spec.decay, 1e-3) * math.pi * 1.5)
            )
            if envelope > best:
                best, best_index = envelope, index
    return best, best_index


# ---------------------------------------------------------------------------
# Clip wrappers
# ---------------------------------------------------------------------------


def apply_frame_effect(clip, effect: Callable[[Frame, float], Frame]):
    """Attach a ``(frame, t) -> frame`` function to a MoviePy clip."""
    def _apply(get_frame, t):
        return effect(get_frame(t), t)

    if hasattr(clip, "fl"):            # MoviePy 1.x
        return clip.fl(_apply, apply_to=[])
    return clip.transform(_apply, apply_to=[])  # MoviePy 2.x


def build_effect_chain(
    shake: Optional[ShakeSpec] = None,
    grade: Optional[dict] = None,
    vignette_strength: float = 0.0,
    bloom: float = 0.0,
) -> Optional[Callable[[Frame, float], Frame]]:
    """Compose the per-frame effects a theme asks for into one callable."""
    if shake is None and not grade and vignette_strength <= 0 and bloom <= 0:
        return None
    seed = 0.0

    def effect(frame: Frame, t: float) -> Frame:
        out = frame
        if shake is not None and shake.kind != "none":
            dx, dy, zoom, split = shake_at(shake, t, seed)
            if shake.motion_blur:
                out = directional_blur(out, dx, dy)
            if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(zoom - 1.0) > 1e-3:
                out = shift_frame(out, dx, dy, zoom)
            if split > 0.4:
                out = rgb_split(out, split, angle=math.atan2(dy, dx) if (dx or dy) else 0.0)
        if grade:
            out = colour_grade(out, **grade)
        if bloom > 0:
            out = glow_bloom(out, bloom)
        if vignette_strength > 0:
            out = vignette(out, vignette_strength)
        return out

    return effect
