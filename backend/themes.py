"""Editing themes - Haunted, Playful, Normal Edits and Anime Edits.

A theme is a set of real, render-affecting parameters: colour grade, vignette,
bloom, shake behaviour, cut pacing, text and sticker styling.  The renderer
reads them; nothing here is decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vfx import ShakeSpec


@dataclass
class Theme:
    key: str
    name: str
    description: str

    # Per-frame look.
    grade: Dict[str, float] = field(default_factory=dict)
    vignette: float = 0.0
    bloom: float = 0.0

    # Motion.
    shake_kind: str = "none"          # none | random | directional | pulse
    shake_intensity: float = 0.0      # pixels at 1080 wide
    shake_zoom: float = 0.0
    shake_rgb_split: float = 0.0
    shake_decay: float = 0.16
    shake_motion_blur: bool = False
    shake_on: str = "impacts"         # impacts | beats | always
    # Chromatic aberration that is always on, under whatever a hit throws.
    base_rgb_split: float = 0.0
    # A blown-out frame on the hit itself, and how long it takes to fall away.
    flash_strength: float = 0.0
    flash_decay: float = 0.13
    # A slow push across the length of every shot.
    drift_zoom: float = 0.0

    # Pacing hints handed to Kimi and to the deterministic editor.
    segment_seconds: Tuple[float, float] = (2.0, 4.0)
    default_cut: str = "jump_cut"
    speed_ramp: float = 1.0

    # Styling.
    text_style: str = "3d_pop"
    text_colour: str = "#FFFFFF"
    caption_highlight: str = "#FFD400"
    sticker_animation: str = "pop_up"
    sticker_scale: float = 1.0

    def shake_spec(self, hits, width: int = 1080) -> Optional[ShakeSpec]:
        if self.shake_kind == "none" or self.shake_intensity <= 0:
            return None
        scale = width / 1080.0
        return ShakeSpec(
            kind=self.shake_kind,
            intensity=self.shake_intensity * scale,
            hits=tuple(hits or ()),
            decay=self.shake_decay,
            zoom=self.shake_zoom,
            rgb_split=self.shake_rgb_split * scale,
            motion_blur=self.shake_motion_blur,
        )

    def prompt_hint(self) -> str:
        return (
            f"{self.name}: {self.description} Segments should run "
            f"{self.segment_seconds[0]:.1f}-{self.segment_seconds[1]:.1f}s and default to "
            f"'{self.default_cut}' cuts."
        )


THEMES: Dict[str, Theme] = {
    "normal": Theme(
        key="normal",
        name="Normal Edits",
        description="Clean transitions and a cinematic grade - nothing shouts.",
        grade={"saturation": 1.06, "contrast": 1.08, "gamma": 0.98},
        vignette=0.18,
        bloom=0.10,
        shake_kind="none",
        segment_seconds=(2.5, 5.0),
        default_cut="crossfade",
        text_style="3d_pop",
        caption_highlight="#FFD400",
    ),
    "anime_edits": Theme(
        key="anime_edits",
        name="Anime Edits",
        description=(
            "High energy: velocity cuts on the beat, impact shake with RGB split, "
            "punchy contrast and glow on every highlight."
        ),
        grade={"saturation": 1.32, "contrast": 1.22, "gamma": 0.94},
        vignette=0.22,
        bloom=0.42,
        shake_kind="directional",
        shake_intensity=26.0,
        shake_zoom=0.06,
        shake_rgb_split=9.0,
        shake_decay=0.18,
        shake_motion_blur=True,
        shake_on="impacts",
        segment_seconds=(0.9, 2.2),
        default_cut="jump_cut",
        speed_ramp=1.15,
        text_style="3d_pop",
        text_colour="#FFFFFF",
        caption_highlight="#00E5FF",
        sticker_animation="pop_up",
        sticker_scale=1.1,
    ),
    "haunted": Theme(
        key="haunted",
        name="Haunted Mode",
        description=(
            "Dark, desaturated and cold, heavy vignette, slow creeping zoom and a "
            "low uneasy handheld drift."
        ),
        grade={"saturation": 0.55, "contrast": 1.18, "brightness": -0.07,
               "tint": (0.88, 0.94, 1.12), "gamma": 1.14},
        vignette=0.62,
        bloom=0.14,
        shake_kind="random",
        shake_intensity=4.5,
        shake_rgb_split=3.0,
        shake_on="always",
        segment_seconds=(2.6, 5.5),
        default_cut="crossfade",
        speed_ramp=0.92,
        text_style="3d_pop",
        text_colour="#D9D2E9",
        caption_highlight="#9B5CFF",
        sticker_animation="fade_in",
        sticker_scale=0.9,
    ),
    "playful": Theme(
        key="playful",
        name="Playful Mode",
        description=(
            "Vibrant and bouncy: warm saturated grade, quick pops, springy sticker "
            "and text animations."
        ),
        grade={"saturation": 1.42, "contrast": 1.10, "brightness": 0.04,
               "tint": (1.06, 1.0, 0.96)},
        vignette=0.08,
        bloom=0.26,
        shake_kind="pulse",
        shake_intensity=1.0,
        shake_zoom=0.05,
        shake_rgb_split=0.0,
        shake_decay=0.22,
        shake_on="beats",
        segment_seconds=(1.2, 3.0),
        default_cut="zoom_punch",
        text_style="3d_pop",
        text_colour="#FFF6A8",
        caption_highlight="#FF4FA3",
        sticker_animation="pop_up",
        sticker_scale=1.15,
    ),
    "ae_hype": Theme(
        key="ae_hype",
        name="AE Hype Edit",
        description=(
            "The After Effects hype grammar: velocity cuts straight onto the beat, "
            "a blown-out frame on every hit, constant chromatic aberration, motion "
            "blur through the shake and a slow push on every shot."
        ),
        # Crushed blacks, lifted contrast and a cold-highlight / warm-shadow
        # split - the look the whole school grades toward.
        grade={"saturation": 1.18, "contrast": 1.34, "brightness": -0.03,
               "tint": (1.04, 0.99, 1.09), "gamma": 0.90},
        vignette=0.34,
        bloom=0.38,
        shake_kind="directional",
        shake_intensity=19.0,
        shake_zoom=0.075,
        shake_rgb_split=13.0,
        shake_decay=0.14,
        shake_motion_blur=True,
        shake_on="beats",
        base_rgb_split=1.6,
        flash_strength=0.34,
        flash_decay=0.11,
        drift_zoom=0.035,
        segment_seconds=(0.7, 1.8),
        default_cut="zoom_punch",
        speed_ramp=1.2,
        text_style="3d_pop",
        text_colour="#FFFFFF",
        caption_highlight="#B14BFF",
        sticker_animation="pop_up",
        sticker_scale=1.05,
    ),
}

DEFAULT_THEME = "normal"

# Accept a few obvious spellings from users and from the vision model.
_ALIASES = {
    "anime": "anime_edits",
    "anime_edit": "anime_edits",
    "amv": "anime_edits",
    "horror": "haunted",
    "haunted_mode": "haunted",
    "dark": "haunted",
    "fun": "playful",
    "playful_mode": "playful",
    "vibrant": "playful",
    "normal_edits": "normal",
    "cinematic": "normal",
    "clean": "normal",
    # The style is known by its practitioners more than by any label.
    "ae": "ae_hype",
    "ae_edit": "ae_hype",
    "after_effects": "ae_hype",
    "sanchez": "ae_hype",
    "sanchezae": "ae_hype",
    "sanchez_ae": "ae_hype",
    "hype": "ae_hype",
    "velocity": "ae_hype",
    "montage": "ae_hype",
}


def resolve_theme(name: Optional[str]) -> Theme:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    return THEMES.get(key, THEMES[DEFAULT_THEME])


def theme_keys() -> List[str]:
    return list(THEMES)


# Content types that decide the theme on their own: a vision model reading the
# mood of one frame as "cute playful" must not turn an anime fight edit into a
# bouncy pastel edit.
_CONTENT_THEME: Dict[str, str] = {
    # These three are the AE hype school's native material - a fight, a clutch
    # play, a highlight reel are all cut to the same grammar.
    "anime_edit": "ae_hype",
    "gaming": "ae_hype",
    "sports": "ae_hype",
    "dance": "playful",
    "meme": "playful",
    "food": "playful",
    "nature": "normal",
    "tutorial": "normal",
    "product": "normal",
    "vlog": "normal",
}


def choose_theme(analysis: object, requested: Optional[str] = None) -> Theme:
    """Pick the editing mode from every signal, not just the vision model's guess.

    Priority: an explicit user choice, then the content type (a hard fact about
    the footage), then darkness, then the vision model's suggestion, then the
    measured energy.
    """
    if requested and requested.strip().lower() not in ("", "auto"):
        return resolve_theme(requested)

    content = str(getattr(analysis, "content_type", "") or "").strip().lower()
    brightness = float(getattr(analysis, "brightness", 0.5) or 0.5)
    energy = str(getattr(analysis, "energy", "medium") or "medium").lower()
    suggested = str(getattr(analysis, "suggested_theme", "") or "").strip().lower()

    if content in _CONTENT_THEME:
        theme = resolve_theme(_CONTENT_THEME[content])
        # Very dark footage still wins for horror-leaning content.
        if brightness < 0.16 and content not in ("anime_edit", "gaming"):
            return resolve_theme("haunted")
        return theme

    if brightness < 0.22:
        return resolve_theme("haunted")
    if suggested in THEMES or suggested in _ALIASES:
        return resolve_theme(suggested)
    if energy == "high":
        return resolve_theme("ae_hype")
    if brightness > 0.62:
        return resolve_theme("playful")
    return resolve_theme(DEFAULT_THEME)
