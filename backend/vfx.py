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

from motion_graphics import particle_burst

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


def color_pop(frame: Frame, amount: float = 0.5, protect_skin: bool = True) -> Frame:
    """Vibrance, not saturation.

    Plain saturation blows out whatever is already colourful. Vibrance lifts the
    *dull* pixels hardest and leaves saturated ones alone, which is what makes
    a shot pop without turning faces orange - so skin hues are damped further
    when ``protect_skin`` is set.
    """
    if amount <= 0.01:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]

    # Headroom: 0 for an already saturated pixel, 1 for a grey one.
    headroom = 1.0 - (saturation / 255.0)
    boost = 1.0 + amount * headroom

    if protect_skin:
        # OpenCV hue is 0-179; skin sits roughly in 0-25 and 165-179.
        skin = np.clip(1.0 - np.minimum(hue, 179.0 - hue) / 25.0, 0.0, 1.0)
        boost = 1.0 + (boost - 1.0) * (1.0 - 0.6 * skin)

    hsv[:, :, 1] = np.clip(saturation * boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def chroma_key(
    frame: Frame,
    key_rgb: Tuple[int, int, int] = (0, 177, 64),
    tolerance: float = 0.32,
    softness: float = 0.12,
    spill: float = 0.6,
) -> np.ndarray:
    """Green-screen removal. Returns RGBA with the key colour knocked out.

    Keying happens in YCrCb chroma space rather than RGB, so shadows and
    uneven lighting on the screen do not change the match the way brightness
    differences would in RGB.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    key = cv2.cvtColor(
        np.uint8([[list(key_rgb)]]), cv2.COLOR_RGB2YCrCb
    ).astype(np.float32)[0][0]

    distance = np.sqrt(
        (ycrcb[:, :, 1] - key[1]) ** 2 + (ycrcb[:, :, 2] - key[2]) ** 2
    ) / 180.0

    inner = max(tolerance - softness, 0.0)
    alpha = np.clip((distance - inner) / max(tolerance - inner, 1e-3), 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)

    out = frame.astype(np.float32)
    if spill > 0.01:
        # Spill suppression: pull green down to the red/blue average where the
        # subject picked up a colour cast from the screen.
        green = out[:, :, 1]
        limit = (out[:, :, 0] + out[:, :, 2]) / 2.0
        excess = np.maximum(green - limit, 0.0) * spill
        out[:, :, 1] = green - excess * alpha

    rgba = np.dstack([np.clip(out, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return rgba


def composite_over(foreground_rgba: np.ndarray, background: Frame) -> Frame:
    """Alpha-composite an RGBA frame over an opaque background."""
    if background.shape[:2] != foreground_rgba.shape[:2]:
        background = cv2.resize(
            background, (foreground_rgba.shape[1], foreground_rgba.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    alpha = (foreground_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    blended = foreground_rgba[:, :, :3].astype(np.float32) * alpha +         background.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


_VIGNETTE_CACHE: dict = {}


def impact_flash(frame: Frame, amount: float) -> Frame:
    """Lift the whole frame toward white for a hit.

    The AE-school impact frame is not a white card spliced in - that reads as a
    dropped frame. It is the picture itself blown out for two or three frames
    and settling, so the shot underneath stays legible the whole time.
    """
    if amount <= 0.001:
        return frame
    amount = float(min(max(amount, 0.0), 0.9))
    return cv2.addWeighted(frame, 1.0 - amount, np.full_like(frame, 255), amount, 0.0)


def punch_zoom(frame: Frame, scale: float) -> Frame:
    """Scale about the centre and crop back, for a continuous camera push."""
    if abs(scale - 1.0) < 1e-4:
        return frame
    return shift_frame(frame, 0.0, 0.0, scale)


def flash_envelope(hits: Sequence[float], t: float, decay: float = 0.13) -> float:
    """How hard a flash is burning at ``t``, 0..1.

    A flash is asymmetric on purpose: it arrives on the frame of the hit and
    falls away, never ramps up. A symmetric envelope pre-lights the cut and
    gives the hit away a beat early.
    """
    best = 0.0
    for hit in hits or ():
        delta = t - hit
        if 0.0 <= delta <= decay:
            best = max(best, (1.0 - delta / decay) ** 2)
    return best


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
    pop: float = 0.0,
    base_rgb_split: float = 0.0,
    flash_hits: Sequence[float] = (),
    flash_strength: float = 0.0,
    flash_decay: float = 0.13,
    drift_zoom: float = 0.0,
    duration: float = 0.0,
    spark_hits: Sequence[float] = (),
    spark_amount: float = 0.0,
    spark_colour: Tuple[int, int, int] = (255, 236, 190),
    spark_life: float = 0.45,
) -> Optional[Callable[[Frame, float], Frame]]:
    """Compose the per-frame effects a theme asks for into one callable."""
    wants_flash = flash_strength > 0 and bool(flash_hits)
    wants_drift = drift_zoom > 0 and duration > 0
    wants_sparks = spark_amount > 0 and bool(spark_hits)
    if (shake is None and not grade and vignette_strength <= 0 and bloom <= 0
            and pop <= 0 and base_rgb_split <= 0 and not wants_flash
            and not wants_drift and not wants_sparks):
        return None
    seed = 0.0
    hits = tuple(flash_hits or ())
    sparks = tuple(sorted(spark_hits or ()))

    def effect(frame: Frame, t: float) -> Frame:
        out = frame
        # A slow push across the whole shot, before any shake displaces it -
        # the AE camera move that keeps a static shot from feeling frozen.
        if wants_drift:
            progress = min(max(t / duration, 0.0), 1.0)
            out = punch_zoom(out, 1.0 + drift_zoom * progress)
        if shake is not None and shake.kind != "none":
            dx, dy, zoom, split = shake_at(shake, t, seed)
            if shake.motion_blur:
                out = directional_blur(out, dx, dy)
            if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(zoom - 1.0) > 1e-3:
                out = shift_frame(out, dx, dy, zoom)
            if split > 0.4:
                out = rgb_split(out, split, angle=math.atan2(dy, dx) if (dx or dy) else 0.0)
        # Chromatic aberration that never fully goes away: the constant, low
        # level one is what makes the footage read as graded rather than raw,
        # and it is separate from the hard split a hit throws.
        if base_rgb_split > 0.4:
            out = rgb_split(out, base_rgb_split)
        if grade:
            out = colour_grade(out, **grade)
        if pop > 0:
            out = color_pop(out, pop)
        if bloom > 0:
            out = glow_bloom(out, bloom)
        if vignette_strength > 0:
            out = vignette(out, vignette_strength)
        # Sparks are lit objects in the shot, so they are added before the
        # flash blows the frame out - otherwise they read as dirt on the lens
        # that the flash cannot reach.
        if wants_sparks:
            for hit in sparks:
                age = t - hit
                if 0.0 <= age <= spark_life:
                    out = particle_burst(
                        out, age, life=spark_life,
                        count=max(6, int(34 * spark_amount)),
                        seed=int(hit * 1000) & 0xFFFF, colour=spark_colour,
                    )
                    break
        # The flash goes last so it burns the graded picture, not the raw one.
        if wants_flash:
            burn = flash_envelope(hits, t, flash_decay) * flash_strength
            if burn > 0.001:
                out = impact_flash(out, burn)
        return out

    return effect
