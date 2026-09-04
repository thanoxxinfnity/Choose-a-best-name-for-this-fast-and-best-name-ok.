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
}


def resolve_theme(name: Optional[str]) -> Theme:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    return THEMES.get(key, THEMES[DEFAULT_THEME])


def theme_keys() -> List[str]:
    return list(THEMES)
