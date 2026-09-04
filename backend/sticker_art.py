"""Procedural sticker artwork - real shapes, not text badges.

When Puter.js txt2img is unavailable the renderer still has to put *something*
on screen, and a pill reading "GOLDEN LOTUS" is not it.  This module draws
actual vector icons - a sword, an explosion, a flame, a lightning bolt, a
sparkle burst, a heart, a play button - with a radial gradient body, an inner
highlight, a dark rim and a soft outer glow, so they read as glossy 3D stickers
against video.

`draw_sticker(prompt, path)` picks the shape from the prompt's keywords and the
palette from its colour words, so a plan asking for "glowing sword slash,
energy trail" gets a sword, not a label.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

RGB = Tuple[int, int, int]
Point = Tuple[float, float]

SUPERSAMPLE = 3  # draw big, downsample -> clean anti-aliased edges


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

PALETTES: Dict[str, Tuple[RGB, RGB, RGB]] = {
    # name: (highlight, body, rim)
    "gold":    ((255, 236, 160), (255, 176, 32), (140, 78, 0)),
    "fire":    ((255, 226, 150), (255, 108, 28), (128, 28, 6)),
    "crimson": ((255, 170, 170), (226, 40, 58), (104, 8, 22)),
    "violet":  ((225, 200, 255), (140, 76, 255), (52, 20, 118)),
    "cyan":    ((198, 246, 255), (44, 194, 240), (10, 78, 118)),
    "green":   ((208, 255, 196), (58, 200, 96), (14, 88, 44)),
    "ice":     ((236, 250, 255), (168, 214, 255), (58, 104, 160)),
    "pink":    ((255, 210, 236), (240, 84, 172), (118, 16, 76)),
    "silver":  ((250, 252, 255), (186, 196, 214), (78, 88, 104)),
    "void":    ((186, 168, 255), (72, 54, 140), (22, 14, 52)),
}

_COLOUR_WORDS: Sequence[Tuple[Tuple[str, ...], str]] = (
    (("gold", "golden", "yellow", "sun", "amber", "honey", "star"), "gold"),
    (("fire", "flame", "burn", "explosion", "blast", "orange", "lava"), "fire"),
    (("red", "blood", "crimson", "danger", "rage"), "crimson"),
    (("purple", "violet", "magic", "cursed", "void", "dark"), "violet"),
    (("blue", "cyan", "water", "ice", "frost", "lightning", "electric", "energy"), "cyan"),
    (("green", "leaf", "nature", "poison", "toxic"), "green"),
    (("white", "silver", "holy", "light", "shine", "chrome", "metal", "sword", "blade"), "silver"),
    (("pink", "love", "heart", "cute", "sakura", "flower"), "pink"),
    (("night", "shadow", "ghost", "haunted", "horror"), "void"),
)


def pick_palette(prompt: str) -> Tuple[RGB, RGB, RGB]:
    lowered = (prompt or "").lower()
    for keywords, name in _COLOUR_WORDS:
        if any(keyword in lowered for keyword in keywords):
            return PALETTES[name]
    return PALETTES["violet"]


# ---------------------------------------------------------------------------
# Shape polygons - all defined in a 0..1 unit square
# ---------------------------------------------------------------------------


def _star(points: int, outer: float, inner: float, rotation: float = -math.pi / 2) -> List[Point]:
    vertices: List[Point] = []
    for index in range(points * 2):
        radius = outer if index % 2 == 0 else inner
        angle = rotation + index * math.pi / points
        vertices.append((0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)))
    return vertices


def _flame() -> List[Point]:
    return [
        (0.50, 0.03), (0.63, 0.24), (0.72, 0.18), (0.74, 0.36), (0.86, 0.52),
        (0.82, 0.72), (0.68, 0.90), (0.50, 0.97), (0.32, 0.90), (0.18, 0.72),
        (0.14, 0.52), (0.26, 0.36), (0.28, 0.18), (0.37, 0.24),
    ]


def _bolt() -> List[Point]:
    return [
        (0.58, 0.02), (0.24, 0.54), (0.45, 0.54), (0.36, 0.98),
        (0.78, 0.42), (0.55, 0.42), (0.70, 0.02),
    ]


def _sword() -> List[Point]:
    return [
        (0.50, 0.02), (0.585, 0.16), (0.585, 0.62), (0.70, 0.62), (0.70, 0.70),
        (0.565, 0.70), (0.565, 0.90), (0.62, 0.98), (0.38, 0.98), (0.435, 0.90),
        (0.435, 0.70), (0.30, 0.70), (0.30, 0.62), (0.415, 0.62), (0.415, 0.16),
    ]


def _heart() -> List[Point]:
    vertices: List[Point] = []
    for step in range(64):
        t = step / 63 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        vertices.append((0.5 + x / 38.0, 0.46 - y / 38.0))
    return vertices


def _shield() -> List[Point]:
    return [
        (0.50, 0.02), (0.90, 0.16), (0.86, 0.58), (0.50, 0.98),
        (0.14, 0.58), (0.10, 0.16),
    ]


def _play() -> List[Point]:
    return [(0.30, 0.10), (0.86, 0.50), (0.30, 0.90)]


def _crown() -> List[Point]:
    return [
        (0.08, 0.82), (0.16, 0.24), (0.32, 0.52), (0.50, 0.14),
        (0.68, 0.52), (0.84, 0.24), (0.92, 0.82),
    ]


def _skull_extra(draw: ImageDraw.ImageDraw, size: int, fill: int = 255) -> None:
    """Eye sockets + nose, punched out of an 8-bit mask."""
    draw.ellipse([0.27 * size, 0.36 * size, 0.44 * size, 0.56 * size], fill=fill)
    draw.ellipse([0.56 * size, 0.36 * size, 0.73 * size, 0.56 * size], fill=fill)
    draw.polygon(
        [(0.50 * size, 0.58 * size), (0.44 * size, 0.70 * size), (0.56 * size, 0.70 * size)],
        fill=fill,
    )


def _skull() -> List[Point]:
    return [
        (0.50, 0.04), (0.78, 0.14), (0.90, 0.40), (0.84, 0.66), (0.72, 0.74),
        (0.72, 0.94), (0.28, 0.94), (0.28, 0.74), (0.16, 0.66), (0.10, 0.40),
        (0.22, 0.14),
    ]


def _speech() -> List[Point]:
    vertices: List[Point] = []
    for step in range(48):
        t = step / 48 * 2 * math.pi
        vertices.append((0.5 + 0.44 * math.cos(t), 0.42 + 0.34 * math.sin(t)))
    vertices += [(0.40, 0.74), (0.30, 0.96), (0.52, 0.75)]
    return vertices


SHAPES: Dict[str, Callable[[], List[Point]]] = {
    "burst": lambda: _star(12, 0.50, 0.20),
    "star": lambda: _star(5, 0.48, 0.20),
    "sparkle": lambda: _star(4, 0.50, 0.11),
    "explosion": lambda: _star(14, 0.50, 0.22),
    "flame": _flame,
    "bolt": _bolt,
    "sword": _sword,
    "heart": _heart,
    "shield": _shield,
    "play": _play,
    "crown": _crown,
    "skull": _skull,
    "speech": _speech,
}

_SHAPE_WORDS: Sequence[Tuple[Tuple[str, ...], str]] = (
    (("sword", "blade", "katana", "slash", "weapon", "knife"), "sword"),
    (("explosion", "blast", "impact", "boom", "smash", "shockwave"), "explosion"),
    (("fire", "flame", "burn", "lava", "heat"), "flame"),
    (("lightning", "bolt", "electric", "thunder", "energy", "power", "speed"), "bolt"),
    (("skull", "death", "ghost", "horror", "haunted", "monster", "demon", "curse"), "skull"),
    (("crown", "king", "queen", "royal", "boss", "legend"), "crown"),
    (("shield", "defend", "guard", "block", "armor", "armour"), "shield"),
    (("heart", "love", "like", "romance"), "heart"),
    (("subscribe", "play", "button", "watch", "video"), "play"),
    (("text", "quote", "talk", "say", "speech", "comment"), "speech"),
    (("sparkle", "shine", "glitter", "magic", "twinkle", "dew"), "sparkle"),
    (("star", "rating", "favourite", "favorite"), "star"),
    (("burst", "glow", "aura", "flash", "light", "sun"), "burst"),
)


def pick_shape(prompt: str) -> str:
    lowered = (prompt or "").lower()
    for keywords, shape in _SHAPE_WORDS:
        if any(keyword in lowered for keyword in keywords):
            return shape
    return "burst"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _scale(points: Sequence[Point], size: int) -> List[Tuple[float, float]]:
    return [(x * size, y * size) for x, y in points]


def _radial_gradient(size: int, inner: RGB, outer: RGB, centre: Point = (0.42, 0.34)) -> Image.Image:
    """Cheap radial gradient by stacking shrinking ellipses."""
    gradient = Image.new("RGB", (size, size), outer)
    draw = ImageDraw.Draw(gradient)
    cx, cy = centre[0] * size, centre[1] * size
    steps = 56
    for step in range(steps, 0, -1):
        blend = step / steps
        radius = size * 0.95 * blend
        colour = tuple(
            int(outer[channel] * blend + inner[channel] * (1 - blend)) for channel in range(3)
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
    return gradient


def draw_sticker(
    prompt: str,
    output_path: Path,
    size: int = 512,
    shape: Optional[str] = None,
    label: Optional[str] = None,
) -> Path:
    """Render a glossy 3D-styled sticker for ``prompt`` as a transparent PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shape_name = shape or pick_shape(prompt)
    highlight, body, rim = pick_palette(prompt)
    big = size * SUPERSAMPLE

    polygon = _scale(SHAPES[shape_name](), big)

    # --- silhouette masks -------------------------------------------------
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    if shape_name == "skull":
        cut = Image.new("L", (big, big), 0)
        _skull_extra(ImageDraw.Draw(cut), big, 255)
        mask = ImageChops.subtract(mask, cut)

    rim_width = max(4, int(big * 0.035))
    rim_mask = Image.new("L", (big, big), 0)
    rim_draw = ImageDraw.Draw(rim_mask)
    rim_draw.polygon(polygon, fill=255)
    rim_mask = rim_mask.filter(ImageFilter.MaxFilter(rim_width * 2 + 1))

    # --- body -------------------------------------------------------------
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))

    rim_layer = Image.new("RGBA", (big, big), rim + (255,))
    rim_layer.putalpha(rim_mask)
    canvas.alpha_composite(rim_layer)

    fill = _radial_gradient(big, highlight, body).convert("RGBA")
    fill.putalpha(mask)
    canvas.alpha_composite(fill)

    # Inner top-left sheen: the body mask, eroded and shifted up.
    sheen_mask = mask.filter(ImageFilter.MinFilter(max(3, int(big * 0.09)) | 1))
    sheen_mask = ImageChops.offset(sheen_mask, 0, -int(big * 0.10))
    sheen_mask = ImageChops.multiply(sheen_mask, mask)
    sheen_mask = sheen_mask.filter(ImageFilter.GaussianBlur(big * 0.03))
    sheen = Image.new("RGBA", (big, big), (255, 255, 255, 0))
    sheen.putalpha(sheen_mask.point(lambda value: int(value * 0.55)))
    canvas.alpha_composite(sheen)

    # Contact shadow along the bottom edge for depth.
    shade_mask = ImageChops.offset(mask, 0, int(big * 0.07))
    shade_mask = ImageChops.subtract(shade_mask, mask)
    shade_mask = ImageChops.multiply(
        shade_mask.filter(ImageFilter.GaussianBlur(big * 0.02)), mask
    )
    shade = Image.new("RGBA", (big, big), rim + (0,))
    shade.putalpha(shade_mask.point(lambda value: int(value * 0.5)))
    canvas.alpha_composite(shade)

    if label:
        _stamp_label(canvas, label, big, rim)

    # --- outer glow -------------------------------------------------------
    pad = int(big * 0.16)
    framed = Image.new("RGBA", (big + pad * 2, big + pad * 2), (0, 0, 0, 0))
    glow_source = Image.new("RGBA", framed.size, body + (0,))
    glow_alpha = Image.new("L", framed.size, 0)
    glow_alpha.paste(rim_mask, (pad, pad))
    glow_source.putalpha(glow_alpha.filter(ImageFilter.GaussianBlur(pad * 0.55)))
    framed.alpha_composite(glow_source)
    framed.alpha_composite(glow_source)
    framed.alpha_composite(canvas, (pad, pad))

    framed = framed.resize(
        (framed.width // SUPERSAMPLE, framed.height // SUPERSAMPLE), Image.LANCZOS
    )
    framed.save(output_path, "PNG")
    return output_path


def _stamp_label(canvas: Image.Image, label: str, size: int, rim: RGB) -> None:
    """Optional short word across the sticker (e.g. a impact word)."""
    text = label.strip().upper()[:10]
    if not text:
        return
    font_size = int(size * 0.20)
    font = _font(font_size)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, size // 90))
    x = (size - (box[2] - box[0])) // 2 - box[0]
    y = int(size * 0.60)
    draw.text(
        (x, y), text, font=font, fill=(255, 255, 255, 255),
        stroke_width=max(2, size // 90), stroke_fill=rim + (255,),
    )


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
