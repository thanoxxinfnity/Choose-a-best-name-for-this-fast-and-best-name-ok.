"""Voice profiles for the Moja AI voiceover.

A profile is a Puter/Polly voice plus a *shaping* chain applied to the returned
mp3 with ffmpeg.  That second half is what makes a "deep dark mysterious"
narrator possible: Polly will not give you one directly, but pitching a neural
Indian male voice down a few semitones, darkening the timbre with a low-pass and
adding a short reverb tail does exactly that - and it works on whatever voice the
provider actually has.

Shaping runs locally, so it is testable and useful even before a Puter key is
configured.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceProfile:
    key: str
    label: str
    voice_id: str
    language: str
    engine: str = "neural"

    # --- local shaping ----------------------------------------------------
    pitch_semitones: float = 0.0   # negative = deeper
    tempo: float = 1.0             # <1 = slower / more deliberate
    lowpass_hz: int = 0            # 0 = off; darkens the timbre
    highpass_hz: int = 0           # removes rumble after pitching down
    reverb: float = 0.0            # 0..1, cathedral-ish tail
    gain_db: float = 0.0
    # Shown next to the voice when what it is needs explaining - so a label
    # never promises something the synthesiser cannot deliver.
    note: str = ""

    @property
    def needs_shaping(self) -> bool:
        return bool(
            abs(self.pitch_semitones) > 0.01
            or abs(self.tempo - 1.0) > 0.01
            or self.lowpass_hz
            or self.highpass_hz
            or self.reverb > 0.01
            or abs(self.gain_db) > 0.01
        )


# Polly's Indian English catalogue is Kajal (neural), Aditi and Raveena
# (standard) - all female. It ships no male en-IN voice at all, so anything
# claiming to be one here would either fail outright or be a pitched-down
# female voice wearing a label that lies. Every voice_id below is one this
# deployment actually accepts; test_voices_live.py checks that against the
# real API rather than against this comment.
VOICE_PROFILES: Dict[str, VoiceProfile] = {
    # ---- Indian accent (the blueprint default) ---------------------------
    "indian_accent": VoiceProfile(
        key="indian_accent", label="Indian English (female)",
        voice_id="Kajal", language="en-IN", engine="neural",
    ),
    "indian_accent_warm": VoiceProfile(
        key="indian_accent_warm", label="Indian English (warm)",
        voice_id="Raveena", language="en-IN", engine="standard",
    ),
    "indian_accent_low": VoiceProfile(
        key="indian_accent_low", label="Indian English (low)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=-3.0, highpass_hz=70, lowpass_hz=8000,
        note="Polly has no male Indian voice; this is the Indian voice lowered.",
    ),
    "hinglish": VoiceProfile(
        key="hinglish", label="Hindi / Hinglish",
        voice_id="Aditi", language="en-IN", engine="standard",
    ),

    # ---- Deep, dark, mysterious narrator ---------------------------------
    # Two of these on purpose: one keeps the Indian accent and buys its depth
    # with pitch shifting, the other is a genuinely deep male voice that is
    # not Indian. Which trade matters is the user's call, not ours.
    "deep_dark": VoiceProfile(
        key="deep_dark", label="Deep dark mysterious (Indian)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=-6.0, tempo=0.90, lowpass_hz=7000, highpass_hz=70,
        reverb=0.45, gain_db=1.5,
        note="Indian accent kept; the depth comes from pitch shifting.",
    ),
    "deep_dark_male": VoiceProfile(
        key="deep_dark_male", label="Deep dark mysterious (male)",
        voice_id="Gregory", language="en-US", engine="neural",
        pitch_semitones=-3.0, tempo=0.92, lowpass_hz=7200, highpass_hz=65,
        reverb=0.45, gain_db=1.5,
        note="A true deep male voice, but US English - Polly has no male Indian one.",
    ),
    "deep_dark_female": VoiceProfile(
        key="deep_dark_female", label="Deep dark mysterious (Indian female)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=-3.5, tempo=0.94, lowpass_hz=7600, highpass_hz=80,
        reverb=0.40,
    ),
    "horror_whisper": VoiceProfile(
        key="horror_whisper", label="Horror whisper",
        voice_id="Aditi", language="en-IN", engine="standard",
        pitch_semitones=-2.0, tempo=0.88, lowpass_hz=5200, highpass_hz=120,
        reverb=0.65, gain_db=2.0,
    ),
    "hype": VoiceProfile(
        key="hype", label="High energy hype (Indian)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=1.0, tempo=1.08, gain_db=1.0,
    ),

    # ---- Non-Indian fallbacks --------------------------------------------
    "us_accent": VoiceProfile(
        key="us_accent", label="US English (female)",
        voice_id="Joanna", language="en-US", engine="neural",
    ),
    "us_accent_male": VoiceProfile(
        key="us_accent_male", label="US English (male)",
        voice_id="Matthew", language="en-US", engine="neural",
    ),
    "uk_accent": VoiceProfile(
        key="uk_accent", label="British English (female)",
        voice_id="Amy", language="en-GB", engine="neural",
    ),
    "uk_accent_male": VoiceProfile(
        key="uk_accent_male", label="British English (male)",
        voice_id="Brian", language="en-GB", engine="neural",
    ),
}

DEFAULT_VOICE_KEY = "indian_accent"

_ALIASES = {
    "indian": "indian_accent",
    "india": "indian_accent",
    "en_in": "indian_accent",
    "hi": "hinglish",
    "hindi": "hinglish",
    "male": "indian_accent_male",
    "deep": "deep_dark",
    "dark": "deep_dark",
    "mysterious": "deep_dark",
    "deep_dark_mysterious": "deep_dark",
    "deep_and_dark": "deep_dark",
    "narrator": "deep_dark",
    "horror": "horror_whisper",
    "whisper": "horror_whisper",
    "energetic": "hype",
    "hype_male": "hype",
}


def resolve_voice(name: Optional[str]) -> VoiceProfile:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in VOICE_PROFILES:
        return VOICE_PROFILES[key]
    # Anything that mentions India stays Indian; anything dark goes deep.
    if any(word in key for word in ("deep", "dark", "myster", "narrat")):
        return VOICE_PROFILES["deep_dark"]
    if "indian" in key or "hing" in key:
        return VOICE_PROFILES["indian_accent"]
    return VOICE_PROFILES[DEFAULT_VOICE_KEY]


def voice_keys() -> List[str]:
    return list(VOICE_PROFILES)


# ---------------------------------------------------------------------------
# Local shaping
# ---------------------------------------------------------------------------


def build_filter_chain(profile: VoiceProfile, sample_rate: int = 24000) -> str:
    """ffmpeg -af chain that turns a stock voice into the profile's character."""
    filters: List[str] = []

    if abs(profile.pitch_semitones) > 0.01:
        # asetrate is an absolute rate, so it only shifts by the intended amount
        # if the stream is already at `sample_rate` - a 44.1kHz input against a
        # chain written for 24kHz doubles the duration instead of deepening it.
        filters.append(f"aresample={sample_rate}")
        ratio = 2.0 ** (profile.pitch_semitones / 12.0)
        filters.append(f"asetrate={int(sample_rate * ratio)}")
        filters.append(f"aresample={sample_rate}")
        filters.extend(_atempo_chain(1.0 / ratio))

    if abs(profile.tempo - 1.0) > 0.01:
        filters.extend(_atempo_chain(profile.tempo))

    if profile.highpass_hz:
        filters.append(f"highpass=f={profile.highpass_hz}")
    if profile.lowpass_hz:
        filters.append(f"lowpass=f={profile.lowpass_hz}")

    if profile.reverb > 0.01:
        # Two taps: a short room reflection plus a longer tail.
        decay = min(0.9, 0.35 + profile.reverb * 0.55)
        delay_a = int(40 + profile.reverb * 40)
        delay_b = int(90 + profile.reverb * 110)
        filters.append(
            f"aecho=0.85:0.9:{delay_a}|{delay_b}:{decay:.2f}|{decay * 0.6:.2f}"
        )

    if abs(profile.gain_db) > 0.01:
        filters.append(f"volume={profile.gain_db:.2f}dB")

    # Keep the tail from clipping after the echo taps stack up.
    filters.append("alimiter=limit=0.97")
    return ",".join(filters)


def _atempo_chain(tempo: float) -> List[str]:
    """ffmpeg's atempo only accepts 0.5-2.0, so chain it for extreme values."""
    steps: List[str] = []
    remaining = max(0.05, min(tempo, 16.0))
    while remaining < 0.5:
        steps.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        steps.append("atempo=2.0")
        remaining /= 2.0
    if abs(remaining - 1.0) > 0.005:
        steps.append(f"atempo={remaining:.4f}")
    return steps


def shape_voice(
    source: Path,
    destination: Path,
    profile: VoiceProfile,
    ffmpeg: Optional[str] = None,
    audio_bitrate: str = "192k",
) -> Path:
    """Apply the profile's character to a synthesised mp3."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not profile.needs_shaping:
        if source != destination:
            destination.write_bytes(source.read_bytes())
        return destination

    if ffmpeg is None:
        from puter_integration import ffmpeg_binary

        ffmpeg = ffmpeg_binary()

    chain = build_filter_chain(profile)
    result = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-af", chain,
            "-c:a", "libmp3lame", "-b:a", audio_bitrate, str(destination),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning("Voice shaping failed (%s), using the raw voice: %s",
                       profile.key, result.stderr[:200])
        if source != destination:
            destination.write_bytes(source.read_bytes())
    return destination
