"""Sound effects, synthesised rather than shipped.

An edit wants whooshes on cuts, impacts on hits and a riser into the payoff.
Shipping a sample pack means licensing, weight and a download; every effect
here is built from ffmpeg's own generators instead - noise through a sweeping
filter, a decaying sine, a pitch ramp - so the library is a few kB of code, is
free of licensing questions, and can be tuned per theme.

:func:`build_sfx_track` places them: whooshes land slightly *before* a cut
(that is how they read as motivating it), impacts land exactly on it, and a
riser is laid in ahead of the biggest moment.
"""

from __future__ import annotations

import hashlib
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from puter_integration import ffmpeg_binary

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class SfxSpec:
    key: str
    label: str
    duration: float
    # An ffmpeg lavfi source plus a filter chain that shapes it.
    source: str
    filters: str
    # Peak the effect is normalised to, 0..1 full scale. A fixed gain cannot
    # work here: a band-pass or a low-pass eats most of the source's amplitude,
    # so the same dB offset lands somewhere different for every effect.
    target_peak: float = 0.5
    # Negative means the effect starts before the event it belongs to.
    lead_in: float = 0.0


LIBRARY: Dict[str, SfxSpec] = {
    "whoosh": SfxSpec(
        key="whoosh", label="Transition whoosh", duration=0.55,
        source="anoisesrc=d=0.55:c=white:a=0.9:s=1337",
        # A band-pass sweeping upward reads as movement past the listener.
        filters=("bandpass=f=900:width_type=h:w=1200,"
                 "volume='0.05+0.95*sin(3.14159*t/0.55)':eval=frame,"
                 "aecho=0.8:0.7:22:0.25"),
        target_peak=0.55, lead_in=-0.22,
    ),
    "whoosh_soft": SfxSpec(
        key="whoosh_soft", label="Soft whoosh", duration=0.7,
        source="anoisesrc=d=0.7:c=pink:a=0.7:s=2024",
        filters=("lowpass=f=2400,volume='0.05+0.8*sin(3.14159*t/0.7)':eval=frame"),
        target_peak=0.40, lead_in=-0.25,
    ),
    "impact": SfxSpec(
        key="impact", label="Impact hit", duration=0.8,
        source="sine=frequency=58:duration=0.8",
        # A sub sine with a fast decay: the body of a cinematic hit.
        filters=("volume='exp(-9*t)':eval=frame,"
                 "aecho=0.9:0.6:55:0.3,lowpass=f=180"),
        target_peak=0.90,
    ),
    "impact_bright": SfxSpec(
        key="impact_bright", label="Bright impact", duration=0.6,
        source="anoisesrc=d=0.6:c=white:a=1.0:s=4242",
        filters=("highpass=f=1800,volume='exp(-14*t)':eval=frame,aecho=0.8:0.6:40:0.25"),
        target_peak=0.45,
    ),
    "riser": SfxSpec(
        key="riser", label="Tension riser", duration=1.8,
        source="sine=frequency=180:duration=1.8",
        # Pitch climbs as the amplitude does - the standard build-up shape.
        filters=("asetrate=44100*1.6,aresample=44100,atempo=0.625,"
                 "volume='pow(t/1.8,2)':eval=frame,highpass=f=140"),
        target_peak=0.40, lead_in=-1.7,
    ),
    "sub_drop": SfxSpec(
        key="sub_drop", label="Sub drop", duration=1.1,
        source="sine=frequency=110:duration=1.1",
        filters=("asetrate=44100*0.55,aresample=44100,"
                 "volume='exp(-3.5*t)':eval=frame,lowpass=f=140"),
        target_peak=0.85,
    ),
    "click": SfxSpec(
        key="click", label="UI click", duration=0.12,
        source="sine=frequency=1400:duration=0.12",
        filters="volume='exp(-30*t)':eval=frame",
        target_peak=0.35,
    ),
    "sparkle": SfxSpec(
        key="sparkle", label="Sparkle", duration=0.9,
        source="sine=frequency=2600:duration=0.9",
        filters=("vibrato=f=14:d=0.6,volume='exp(-5*t)':eval=frame,highpass=f=1500"),
        target_peak=0.30,
    ),
}

# Which effects each theme reaches for, in preference order.
THEME_SFX: Dict[str, Dict[str, str]] = {
    "anime_edits": {"cut": "whoosh", "impact": "impact", "build": "riser"},
    # The AE school hits harder and lower than a plain anime edit: the sub
    # drop is the sound the impact flash is drawn against.
    "ae_hype": {"cut": "whoosh", "impact": "sub_drop", "build": "riser"},
    "haunted": {"cut": "whoosh_soft", "impact": "sub_drop", "build": "riser"},
    "playful": {"cut": "whoosh_soft", "impact": "impact_bright", "build": "sparkle"},
    "normal": {"cut": "whoosh_soft", "impact": "impact", "build": ""},
}


def sfx_keys() -> List[str]:
    return list(LIBRARY)


def describe_library() -> List[Dict[str, object]]:
    """The catalogue the app shows, so the UI never hard-codes effect names."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "duration": round(spec.duration, 3),
            "target_peak": spec.target_peak,
            "lead_in": spec.lead_in,
        }
        for spec in LIBRARY.values()
    ]


def theme_palette(theme: str) -> Dict[str, str]:
    return dict(THEME_SFX.get(theme, THEME_SFX["normal"]))


def cache_name(key: str) -> str:
    """Cache filename for an effect, fingerprinted by its recipe.

    Rendered effects are cached because synthesising one costs three ffmpeg
    passes. Keying the cache on the name alone means a retuned spec keeps
    serving the old wav forever - which is exactly how a stale, badly
    normalised effect survived a level fix - so the recipe goes in the name.
    """
    spec = LIBRARY.get(key)
    if spec is None:
        return f"{key}.wav"
    recipe = f"{spec.source}|{spec.filters}|{spec.target_peak}|{spec.duration}"
    digest = hashlib.sha1(recipe.encode("utf-8")).hexdigest()[:10]
    return f"{key}-{digest}.wav"


def render_sfx(key: str, destination: Path, ffmpeg: Optional[str] = None) -> Optional[Path]:
    """Synthesise one effect to a wav."""
    spec = LIBRARY.get(key)
    if spec is None:
        return None
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = ffmpeg or ffmpeg_binary()

    def render(chain: str, target: Path) -> bool:
        result = subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", spec.source, "-af", chain,
                "-ar", str(SAMPLE_RATE), "-ac", "2", "-t", f"{spec.duration:.3f}",
                "-c:a", "pcm_s16le", str(target),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not target.exists():
            logger.warning("SFX '%s' failed: %s", key, result.stderr[:200])
            return False
        return True

    # Pass one at unity to find out what the filter chain actually left behind.
    probe = destination.with_name(destination.stem + "_probe.wav")
    if not render(spec.filters or "anull", probe):
        return None
    peak = _peak_dbfs(probe, ffmpeg)
    probe.unlink(missing_ok=True)
    if peak is None:
        return None

    target_dbfs = 20.0 * math.log10(max(spec.target_peak, 1e-4))
    gain = target_dbfs - peak

    # Measure, correct, verify. Some chains (an echo tail interacting with a
    # steep decay envelope) do not scale exactly linearly with a pre-gain, so
    # trusting one calculation leaves an effect several dB off. One corrective
    # pass against the real output settles every effect in the library.
    for _attempt in range(2):
        chain = f"{spec.filters},volume={gain:.2f}dB" if spec.filters else f"volume={gain:.2f}dB"
        if not render(chain, destination):
            return None
        achieved = _peak_dbfs(destination, ffmpeg)
        if achieved is None or abs(achieved - target_dbfs) <= 0.5:
            return destination
        gain += target_dbfs - achieved
    return destination


def _peak_dbfs(path: Path, ffmpeg: str) -> Optional[float]:
    """Peak level of a file in dBFS, via ffmpeg's volumedetect."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in (result.stderr or "").splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].replace("dB", "").strip())
            except (IndexError, ValueError):
                return None
    return None


@dataclass
class SfxPlacement:
    key: str
    at: float          # where the effect should be audible, in output time


def plan_placements(
    cut_times: Sequence[float],
    impact_times: Sequence[float] = (),
    theme: str = "normal",
    duration: float = 0.0,
    max_effects: int = 24,
    min_gap: float = 0.35,
) -> List[SfxPlacement]:
    """Decide which effect goes where.

    Impacts win over cuts at the same moment - two effects stacked on one frame
    sound like a mistake - and a riser is laid in ahead of the last big hit.
    """
    palette = THEME_SFX.get(theme, THEME_SFX["normal"])
    placements: List[SfxPlacement] = []

    impacts = sorted(t for t in impact_times if t > 0.05)
    for moment in impacts:
        if palette.get("impact"):
            placements.append(SfxPlacement(palette["impact"], moment))

    for moment in sorted(cut_times):
        if moment <= 0.05:
            continue
        if any(abs(moment - existing.at) < min_gap for existing in placements):
            continue
        if palette.get("cut"):
            placements.append(SfxPlacement(palette["cut"], moment))

    build = palette.get("build")
    if build and impacts:
        # Ahead of the biggest moment we know about: the last impact.
        placements.append(SfxPlacement(build, impacts[-1]))

    placements.sort(key=lambda item: item.at)
    if duration:
        placements = [item for item in placements if item.at < duration]
    return placements[:max_effects]


def build_sfx_track(
    placements: Sequence[SfxPlacement],
    destination: Path,
    duration: float,
    workspace: Optional[Path] = None,
    ffmpeg: Optional[str] = None,
) -> Optional[Path]:
    """Mix the placed effects into one track the renderer can lay under the edit."""
    if not placements or duration <= 0:
        return None

    ffmpeg = ffmpeg or ffmpeg_binary()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace or destination.parent / "sfx_cache")
    workspace.mkdir(parents=True, exist_ok=True)

    inputs: List[str] = []
    filters: List[str] = []
    labels: List[str] = []

    for index, placement in enumerate(placements):
        spec = LIBRARY.get(placement.key)
        if spec is None:
            continue
        source = render_sfx(placement.key, workspace / cache_name(placement.key), ffmpeg=ffmpeg)
        if source is None:
            continue
        # lead_in is negative for effects that must start before their event.
        start = max(0.0, placement.at + spec.lead_in)
        inputs += ["-i", str(source)]
        label = f"s{index}"
        filters.append(f"[{len(labels)}:a]adelay={int(start * 1000)}|{int(start * 1000)}[{label}]")
        labels.append(label)

    if not labels:
        return None

    mix = "".join(f"[{label}]" for label in labels)
    filters.append(
        f"{mix}amix=inputs={len(labels)}:duration=longest:normalize=0[mixed]"
    )
    # Headroom before the limiter: overlapping effects sum well past full scale,
    # and a limiter asked to remove 6dB pumps instead of shaping.
    filters.append(
        f"[mixed]volume=-6dB,apad,atrim=0:{duration:.3f},alimiter=limit=0.89[out]"
    )

    result = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
         "-filter_complex", ";".join(filters), "-map", "[out]",
         "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(destination)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        logger.warning("SFX track failed: %s", result.stderr[:300])
        return None
    return destination
