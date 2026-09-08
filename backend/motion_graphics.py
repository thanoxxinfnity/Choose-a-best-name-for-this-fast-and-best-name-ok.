"""Motion graphics, rendered rather than generated.

The blueprint expected Puter to generate motion for the edit. Puter serves no
video driver at all, which is now asserted in the live tests - but that turns
out not to be the loss it looks like. The things that make an edit read as
*edited* are not generated footage: they are procedural. A whip pan between
two shots, a glitch that slices the frame on a hit, type that punches in with
real depth - all of them are a function of two frames and a number between
zero and one, and none of them are improved by a diffusion model.

So they are computed here. Every transition takes ``(frame_a, frame_b,
progress)`` and returns a frame, which means the renderer can drop one between
any two segments without knowing what it does, and every one of them is
deterministic and testable frame by frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = np.ndarray


def _ease_out(progress: float) -> float:
    """Fast to start, settling at the end - how a real camera move behaves."""
    return 1.0 - (1.0 - progress) ** 3


def _ease_in_out(progress: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(progress, 0.0), 1.0))


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _match(frame: Frame, like: Frame) -> Frame:
    if frame.shape[:2] != like.shape[:2]:
        return cv2.resize(frame, (like.shape[1], like.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return frame


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def whip_pan(a: Frame, b: Frame, progress: float, direction: str = "left") -> Frame:
    """A hard horizontal sweep with the motion blur the speed would cause.

    The blur is the whole effect. Sliding one frame off and another on without
    it looks like a slideshow; smearing both along the direction of travel is
    what sells it as a camera whipping across.
    """
    b = _match(b, a)
    height, width = a.shape[:2]
    eased = _ease_in_out(_clamp01(progress))
    sign = -1 if direction == "left" else 1
    shift = int(width * eased) * sign

    canvas = np.zeros_like(a)
    # Outgoing frame slides away, incoming slides in behind it.
    _paste(canvas, a, shift)
    _paste(canvas, b, shift - sign * width)

    # Blur peaks in the middle of the move, where the speed does.
    speed = math.sin(math.pi * _clamp01(progress))
    taps = int(1 + speed * max(width // 28, 4))
    if taps > 2:
        kernel = np.zeros((1, taps | 1), dtype=np.float32)
        kernel[0, :] = 1.0 / (taps | 1)
        canvas = cv2.filter2D(canvas, -1, kernel)
    return canvas


def _paste(canvas: Frame, frame: Frame, offset: int) -> None:
    height, width = canvas.shape[:2]
    if offset <= -width or offset >= width:
        return
    if offset >= 0:
        canvas[:, offset:width] = frame[:, 0:width - offset]
    else:
        canvas[:, 0:width + offset] = frame[:, -offset:width]


def glitch_slice(a: Frame, b: Frame, progress: float, seed: int = 7) -> Frame:
    """Horizontal bands tear between the two shots, with the channels split.

    Bands are chosen from a seeded generator so the same cut glitches the same
    way every render - a transition that differs between two runs of the same
    edit is a bug that only shows up as "it looked better last time".
    """
    b = _match(b, a)
    progress = _clamp01(progress)
    height, width = a.shape[:2]
    rng = np.random.default_rng(seed)

    bands = max(6, height // 90)
    edges = np.linspace(0, height, bands + 1).astype(int)
    # Each band flips at its own moment, so the tear sweeps rather than blinks.
    thresholds = rng.uniform(0.15, 0.85, size=bands)
    offsets = rng.integers(-width // 12, width // 12, size=bands)

    out = np.empty_like(a)
    intensity = math.sin(math.pi * progress)
    for index in range(bands):
        top, bottom = edges[index], edges[index + 1]
        source = b if progress >= thresholds[index] else a
        shift = int(offsets[index] * intensity)
        row = np.roll(source[top:bottom], shift, axis=1)
        out[top:bottom] = row

    if intensity > 0.05:
        split = int(intensity * max(width // 90, 3))
        out[:, :, 0] = np.roll(out[:, :, 0], split, axis=1)
        out[:, :, 2] = np.roll(out[:, :, 2], -split, axis=1)
    return out


def zoom_blur(a: Frame, b: Frame, progress: float, strength: float = 0.35) -> Frame:
    """Punch out of one shot and into the next through a radial smear."""
    b = _match(b, a)
    progress = _clamp01(progress)
    height, width = a.shape[:2]

    # First half pushes into A, second half falls out of B.
    if progress < 0.5:
        base, local = a, progress * 2.0
        scale = 1.0 + strength * _ease_out(local)
    else:
        base, local = b, (progress - 0.5) * 2.0
        scale = 1.0 + strength * (1.0 - _ease_out(local))

    frames = []
    steps = 5
    for step in range(steps):
        factor = 1.0 + (scale - 1.0) * (step / max(steps - 1, 1))
        frames.append(_scale_about_centre(base, factor))
    smeared = np.mean(frames, axis=0)

    # Cross-dissolve only across the middle, where both are already smeared.
    if 0.35 < progress < 0.65:
        other = b if progress < 0.5 else a
        mix = (progress - 0.35) / 0.3 if progress < 0.5 else 1.0 - (progress - 0.5) / 0.15
        mix = _clamp01(mix) * 0.5
        smeared = smeared * (1 - mix) + _scale_about_centre(other, scale) * mix
    return np.clip(smeared, 0, 255).astype(np.uint8)


def _scale_about_centre(frame: Frame, factor: float) -> Frame:
    if abs(factor - 1.0) < 1e-4:
        return frame
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, factor)
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def radial_wipe(a: Frame, b: Frame, progress: float, softness: float = 0.12) -> Frame:
    """The next shot opens out of the centre through a soft-edged circle."""
    b = _match(b, a)
    progress = _clamp01(progress)
    height, width = a.shape[:2]

    yy, xx = np.ogrid[:height, :width]
    centre_y, centre_x = height / 2.0, width / 2.0
    distance = np.sqrt((xx - centre_x) ** 2 + (yy - centre_y) ** 2)
    reach = math.hypot(centre_x, centre_y)
    edge = _ease_in_out(progress) * reach * (1.0 + softness)

    feather = max(reach * softness, 1.0)
    alpha = np.clip((edge - distance) / feather, 0.0, 1.0)[:, :, None].astype(np.float32)
    return np.clip(a * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)


def bar_slide(a: Frame, b: Frame, progress: float, bars: int = 7) -> Frame:
    """Vertical bars sweep the new shot in, each one a beat behind the last."""
    b = _match(b, a)
    progress = _clamp01(progress)
    height, width = a.shape[:2]
    out = a.copy()

    edges = np.linspace(0, width, bars + 1).astype(int)
    for index in range(bars):
        left, right = edges[index], edges[index + 1]
        # Stagger: the first bar completes at 0.7, the last at 1.0.
        lag = index / max(bars, 1) * 0.3
        local = _clamp01((progress - lag) / max(1.0 - lag, 1e-3))
        filled = int((right - left) * 0 + height * _ease_out(local))
        if filled > 0:
            out[:filled, left:right] = b[:filled, left:right]
    return out


def light_sweep(a: Frame, b: Frame, progress: float, width_ratio: float = 0.22) -> Frame:
    """A blown highlight crosses the frame and leaves the next shot behind it."""
    b = _match(b, a)
    progress = _clamp01(progress)
    height, width = a.shape[:2]

    position = progress * (1.0 + width_ratio) - width_ratio / 2
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)

    switched = (xx < position)[None, :, None]
    out = np.where(switched, b, a).astype(np.float32)

    band = np.exp(-((xx - position) ** 2) / (2 * (width_ratio / 3) ** 2))
    burn = (band[None, :, None] * 255.0 * math.sin(math.pi * progress))
    return np.clip(out + burn, 0, 255).astype(np.uint8)


TRANSITIONS: Dict[str, Callable[..., Frame]] = {
    "whip_pan": whip_pan,
    "glitch_slice": glitch_slice,
    "zoom_blur": zoom_blur,
    "radial_wipe": radial_wipe,
    "bar_slide": bar_slide,
    "light_sweep": light_sweep,
}

# Which transitions each theme reaches for, in preference order.
THEME_TRANSITIONS: Dict[str, List[str]] = {
    "ae_hype": ["whip_pan", "glitch_slice", "zoom_blur"],
    "anime_edits": ["whip_pan", "zoom_blur", "light_sweep"],
    "haunted": ["glitch_slice", "radial_wipe"],
    "playful": ["bar_slide", "radial_wipe", "light_sweep"],
    "normal": ["radial_wipe"],
}


def transition_names() -> List[str]:
    return list(TRANSITIONS)


def render_transition(a: Frame, b: Frame, progress: float, kind: str,
                      seed: int = 7) -> Frame:
    """One frame of a transition. Unknown kinds cross-dissolve rather than fail."""
    function = TRANSITIONS.get((kind or "").strip().lower())
    if function is None:
        b = _match(b, a)
        alpha = _clamp01(progress)
        return np.clip(a * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)
    if function is glitch_slice:
        return function(a, b, progress, seed=seed)
    return function(a, b, progress)


def transition_frames(a: Frame, b: Frame, kind: str, count: int,
                      seed: int = 7) -> List[Frame]:
    """The whole transition as a list of frames, endpoints excluded.

    Excluded on purpose: frame 0 is the last frame of the outgoing shot and
    frame N is the first of the incoming one, and emitting either would repeat
    a frame the timeline already has, which reads as a stutter on the cut.
    """
    if count <= 0:
        return []
    return [
        render_transition(a, b, (index + 1) / (count + 1), kind, seed=seed)
        for index in range(count)
    ]


def pick_transition(theme: str, index: int = 0) -> str:
    """Rotate through a theme's transitions so a long edit does not repeat one."""
    palette = THEME_TRANSITIONS.get((theme or "").strip().lower())
    if not palette:
        palette = THEME_TRANSITIONS["normal"]
    return palette[index % len(palette)]


# ---------------------------------------------------------------------------
# 3D kinetic type
# ---------------------------------------------------------------------------


@dataclass
class TitleStyle:
    color: Tuple[int, int, int] = (255, 255, 255)
    face_shade: float = 1.0
    # None means "the face colour, unlit". A fixed near-black side wall
    # disappears against a dark shot, which is the same as having no
    # extrusion at all; deriving it from the face keeps the letter reading as
    # one solid object whatever it is sitting on.
    extrude_color: Optional[Tuple[int, int, int]] = None
    outline: Tuple[int, int, int] = (0, 0, 0)
    depth: float = 0.22          # extrusion, as a fraction of the cap height
    yaw: float = 26.0            # degrees the type is turned at rest
    overshoot: float = 1.14      # how far past its size the punch-in goes


def extruded_title(
    text: str,
    size: Tuple[int, int],
    progress: float,
    font_path: Optional[str] = None,
    style: Optional[TitleStyle] = None,
    font_size: Optional[int] = None,
) -> np.ndarray:
    """A word with real extrusion, punching in and settling. RGBA.

    "3D text" in most editors is a drop shadow. This builds an actual
    extrusion: the glyph mask is stamped repeatedly along a depth vector, each
    copy darker than the last, and the face is laid on top - so the side wall
    is visible, it is lit consistently, and it turns with the yaw instead of
    sliding like a shadow.

    The punch overshoots its final size and settles back, because type that
    arrives exactly at its size looks like it was pasted.
    """
    from PIL import Image, ImageDraw

    style = style or TitleStyle()
    width, height = size
    progress = _clamp01(progress)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    letters = (text or "").strip()
    if not letters:
        return np.array(canvas)

    from video_renderer import _has_devanagari, _load_font

    cap = font_size or int(height * 0.19)
    font = _load_font(cap, devanagari=_has_devanagari(letters))

    # Shrink to fit. The size was fixed regardless of how long the word was, so
    # a short title sat comfortably and a long one ran off both edges and got
    # clipped by the canvas - "KING OF CURSES" arrived as "ING OF CURSES". The
    # overshoot has to be in the measurement too, because the punch is at its
    # widest exactly when the type is largest.
    usable = max(width - max(8, cap // 4), 8)
    peak = max(style.overshoot, 1.0)
    for _attempt in range(12):
        measure = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        span = measure.textbbox((0, 0), letters, font=font,
                                stroke_width=max(2, cap // 12))
        needed = (span[2] - span[0]) * peak
        if needed <= usable or cap <= 12:
            break
        cap = max(12, int(cap * min(0.92, usable / max(needed, 1.0))))
        font = _load_font(cap, devanagari=_has_devanagari(letters))

    # Punch: overshoot then settle. Scale is applied to the rendered stamp so
    # the extrusion scales with the face rather than detaching from it.
    if progress < 0.45:
        local = _ease_out(progress / 0.45)
        scale = 0.35 + (style.overshoot - 0.35) * local
        alpha = local
    else:
        local = _ease_out((progress - 0.45) / 0.55)
        scale = style.overshoot + (1.0 - style.overshoot) * local
        alpha = 1.0

    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    box = probe.textbbox((0, 0), letters, font=font, stroke_width=max(2, cap // 12))
    text_width, text_height = box[2] - box[0], box[3] - box[1]

    depth_px = max(2, int(cap * style.depth))
    radians = math.radians(style.yaw)
    step_x, step_y = math.cos(radians), math.sin(radians) * 0.55

    pad = depth_px * 2 + max(4, cap // 8)
    stamp = Image.new("RGBA", (text_width + pad * 2, text_height + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)

    wall = style.extrude_color or tuple(
        max(18, int(channel * 0.42)) for channel in style.color
    )

    # Back to front, so nearer slices cover the ones behind them.
    for layer in range(depth_px, 0, -1):
        # Nearer slices catch more light, which is what gives the wall its
        # curve instead of a flat slab of one colour.
        fade = 0.55 + 0.45 * (1.0 - layer / depth_px)
        shade = tuple(int(channel * fade) for channel in wall)
        draw.text((pad + step_x * layer, pad - step_y * layer), letters,
                  font=font, fill=(*shade, 255))

    face = tuple(int(channel * style.face_shade) for channel in style.color)
    draw.text((pad, pad), letters, font=font, fill=(*face, 255),
              stroke_width=max(2, cap // 12), stroke_fill=(*style.outline, 255))

    new_size = (max(1, int(stamp.width * scale)), max(1, int(stamp.height * scale)))
    stamp = stamp.resize(new_size, Image.LANCZOS)
    if alpha < 1.0:
        faded = stamp.split()[3].point(lambda value: int(value * alpha))
        stamp.putalpha(faded)

    canvas.alpha_composite(
        stamp, ((width - stamp.width) // 2, (height - stamp.height) // 2)
    )
    return np.array(canvas)


# ---------------------------------------------------------------------------
# Impact particles
# ---------------------------------------------------------------------------


def particle_burst(
    frame: Frame,
    age: float,
    life: float = 0.45,
    count: int = 34,
    seed: int = 11,
    colour: Tuple[int, int, int] = (255, 236, 190),
    centre: Optional[Tuple[float, float]] = None,
) -> Frame:
    """Sparks thrown out from a point, added over the frame.

    ``age`` is seconds since the hit. Particles are drawn additively and fade
    on a curve rather than linearly, because a linear fade reads as a fixed
    overlay being turned down rather than as sparks burning out.
    """
    if age < 0 or age > life:
        return frame
    height, width = frame.shape[:2]
    rng = np.random.default_rng(seed)

    centre_x = width * (centre[0] if centre else 0.5)
    centre_y = height * (centre[1] if centre else 0.5)
    reach = min(width, height) * 0.42

    local = age / life
    travel = _ease_out(local)
    fade = (1.0 - local) ** 2.2

    layer = np.zeros_like(frame, dtype=np.float32)
    angles = rng.uniform(0, 2 * math.pi, count)
    speeds = rng.uniform(0.35, 1.0, count)
    sizes = rng.integers(1, max(2, int(min(width, height) * 0.006)) + 1, count)

    for angle, speed, radius in zip(angles, speeds, sizes):
        distance = reach * speed * travel
        x = int(centre_x + math.cos(angle) * distance)
        y = int(centre_y + math.sin(angle) * distance * 0.85)
        if not (0 <= x < width and 0 <= y < height):
            continue
        cv2.circle(layer, (x, y), int(radius), colour, -1, lineType=cv2.LINE_AA)

    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, min(width, height) * 0.0025))
    return np.clip(frame.astype(np.float32) + layer * fade, 0, 255).astype(np.uint8)
