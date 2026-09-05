"""Export presets, up to 4K 60fps.

The renderer works at the preset's resolution rather than upscaling at the
end, so a 4K export is genuinely 4K wherever the source has the detail for it.
That also means a 4K render moves four times the pixels of a 1080p one, so
:func:`describe_cost` exists to tell the caller what they are asking for, and
:func:`check_source_headroom` says plainly when the footage cannot fill the
frame it is being blown up into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExportPreset:
    key: str
    label: str
    width: int
    height: int
    fps: int
    codec: str = "h264"          # h264 | hevc
    crf: int = 20
    x264_preset: str = "medium"
    audio_bitrate: str = "192k"

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def is_vertical(self) -> bool:
        return self.height >= self.width

    def video_bitrate(self) -> str:
        """Bitrate scaled from pixel rate, not hardcoded per preset."""
        return bitrate_for(self.width, self.height, self.fps, self.codec)

    def encoder(self) -> str:
        return "libx265" if self.codec == "hevc" else "libx264"


def bitrate_for(width: int, height: int, fps: int, codec: str = "h264") -> str:
    """A sane VBV target for the given pixel rate.

    ~0.10 bits per pixel per frame for H.264 at typical short-form motion;
    HEVC gets ~35% less for the same look.
    """
    bits_per_pixel = 0.10 if codec == "h264" else 0.065
    bits = width * height * max(fps, 1) * bits_per_pixel
    megabits = max(4.0, min(bits / 1_000_000.0, 120.0))
    return f"{megabits:.0f}M"


PRESETS: Dict[str, ExportPreset] = {
    "720p60": ExportPreset(
        key="720p60", label="720p 60fps (fast draft)",
        width=720, height=1280, fps=60, crf=22, x264_preset="veryfast",
    ),
    "1080p60": ExportPreset(
        key="1080p60", label="1080p 60fps (default, 1080x1920)",
        width=1080, height=1920, fps=60, crf=20,
    ),
    "1440p60": ExportPreset(
        key="1440p60", label="1440p 60fps (2K vertical)",
        width=1440, height=2560, fps=60, crf=20,
    ),
    "4k60": ExportPreset(
        key="4k60", label="4K 60fps (2160x3840 vertical)",
        width=2160, height=3840, fps=60, crf=21, x264_preset="slow",
        audio_bitrate="256k",
    ),
    "4k60_hevc": ExportPreset(
        key="4k60_hevc", label="4K 60fps HEVC (smaller file)",
        width=2160, height=3840, fps=60, codec="hevc", crf=24,
        x264_preset="medium", audio_bitrate="256k",
    ),
    "4k30": ExportPreset(
        key="4k30", label="4K 30fps (2160x3840 vertical)",
        width=2160, height=3840, fps=30, crf=21, x264_preset="slow",
        audio_bitrate="256k",
    ),
    # Landscape variants for YouTube rather than Shorts/Reels.
    "1080p60_wide": ExportPreset(
        key="1080p60_wide", label="1080p 60fps landscape (1920x1080)",
        width=1920, height=1080, fps=60, crf=20,
    ),
    "4k60_wide": ExportPreset(
        key="4k60_wide", label="4K 60fps landscape (3840x2160)",
        width=3840, height=2160, fps=60, crf=21, x264_preset="slow",
        audio_bitrate="256k",
    ),
}

DEFAULT_PRESET = "1080p60"

_ALIASES = {
    "": DEFAULT_PRESET,
    "auto": DEFAULT_PRESET,
    "default": DEFAULT_PRESET,
    "1080p": "1080p60",
    "1080": "1080p60",
    "fhd": "1080p60",
    "2k": "1440p60",
    "1440p": "1440p60",
    "4k": "4k60",
    "2160p": "4k60",
    "uhd": "4k60",
    "4k_hevc": "4k60_hevc",
    "hevc": "4k60_hevc",
    "720p": "720p60",
    "draft": "720p60",
    "landscape": "1080p60_wide",
    "4k_landscape": "4k60_wide",
}


def resolve_preset(name: Optional[str]) -> ExportPreset:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    return PRESETS.get(key, PRESETS[DEFAULT_PRESET])


def preset_keys() -> List[str]:
    return list(PRESETS)


def describe_cost(preset: ExportPreset, baseline: str = DEFAULT_PRESET) -> float:
    """How much more work this preset is than the default, per second of output."""
    reference = PRESETS[baseline]
    return (preset.pixels * preset.fps) / max(reference.pixels * reference.fps, 1)


def check_source_headroom(
    preset: ExportPreset,
    sources: Sequence[Tuple[int, int]],
) -> Optional[str]:
    """Warn when the footage cannot actually fill the requested frame.

    ``sources`` is ``[(width, height), ...]``.  Upscaling 720p to 4K produces a
    4K file with 720p worth of detail; saying so is more useful than quietly
    shipping a soft render.
    """
    usable = [(w, h) for w, h in sources if w > 0 and h > 0]
    if not usable:
        return None
    # Compare on the short edge, which is what a vertical crop is limited by.
    smallest = min(min(w, h) for w, h in usable)
    target_short = min(preset.width, preset.height)
    if smallest >= target_short * 0.95:
        return None
    ratio = target_short / max(smallest, 1)
    return (
        f"Exporting at {preset.resolution}, but the smallest source is {smallest}px "
        f"on its short edge - a {ratio:.1f}x upscale. The file will be "
        f"{preset.resolution}; the detail in it will not be."
    )
