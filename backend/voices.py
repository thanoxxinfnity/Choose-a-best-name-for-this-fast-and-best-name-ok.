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

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import settings

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

    # --- layering ---------------------------------------------------------
    # Weight does not come from pitch. Dropping a voice an octave makes it
    # slow and muddy; laying a quiet octave-down copy *under* the original
    # makes it enormous while the words stay crisp, which is the whole trick
    # behind every villain voice that sounds bigger than a person.
    sub_octave: float = 0.0        # level of the octave-down layer, 0..1
    double_detune: float = 0.0     # level of a slightly detuned, delayed copy
    growl: float = 0.0             # soft-clip drive: harmonics read as menace
    body_db: float = 0.0           # low-mid shelf, the chest of the voice
    presence_db: float = 0.0       # 3kHz lift so it stays legible under the sub
    # --- cloning ----------------------------------------------------------
    # A cloned voice does not pick a stock speaker and bend it. It carries a
    # recording of the voice it wants to be, and Magpie synthesises new lines
    # in that voice. The shaping knobs above still apply on top - the clone
    # supplies the timbre, they supply the character.
    clone_reference: str = ""      # prepared reference WAV; empty = stock voice
    clone_language: str = ""       # what Magpie should speak, e.g. "hi-IN"
    clone_quality: int = 20        # 1..40, similarity against synthesis speed
    # Where this character's voice should sit, in Hz, measured after synthesis.
    #
    # A fixed semitone offset cannot hold a cloned voice at a pitch, because
    # the model does not return one: asked for the same line three times it
    # came back at 181, 199 and 122 Hz - most of an octave apart. An offset
    # applied to that lands wherever the model happened to be, which is how
    # two characters who should sound nothing alike ended up three semitones
    # apart. Naming the pitch and correcting to it after the fact is the only
    # thing that keeps a cast distinct take after take.
    target_f0: float = 0.0         # 0 = leave the model's pitch alone

    # Shown next to the voice when what it is needs explaining - so a label
    # never promises something the synthesiser cannot deliver.
    note: str = ""

    @property
    def is_cloned(self) -> bool:
        return bool(self.clone_reference)

    @property
    def needs_shaping(self) -> bool:
        return bool(
            abs(self.pitch_semitones) > 0.01
            or abs(self.tempo - 1.0) > 0.01
            or self.lowpass_hz
            or self.highpass_hz
            or self.reverb > 0.01
            or abs(self.gain_db) > 0.01
            or self.is_layered
        )

    @property
    def is_layered(self) -> bool:
        """Whether this voice needs parallel layers rather than one chain."""
        return bool(
            self.sub_octave > 0.01
            or self.double_detune > 0.01
            or self.growl > 0.01
            or abs(self.body_db) > 0.01
            or abs(self.presence_db) > 0.01
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
    # ---- layered, "bigger than a person" villain voices -------------------
    "demon_king": VoiceProfile(
        key="demon_king", label="Demon king (huge, Indian)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=-5.0, tempo=0.90,
        sub_octave=0.55, double_detune=0.32, growl=0.35,
        body_db=5.0, presence_db=3.0,
        highpass_hz=55, lowpass_hz=7800, reverb=0.50, gain_db=1.0,
        note="Weight comes from an octave-down layer, not from pitching the words down.",
    ),
    "demon_king_male": VoiceProfile(
        key="demon_king_male", label="Demon king (huge, male)",
        voice_id="Gregory", language="en-US", engine="neural",
        pitch_semitones=-3.0, tempo=0.91,
        sub_octave=0.60, double_detune=0.34, growl=0.38,
        body_db=5.5, presence_db=3.0,
        highpass_hz=50, lowpass_hz=7600, reverb=0.50, gain_db=1.0,
        note="A true deep male voice, layered. US English - Polly has no male Indian one.",
    ),
    "ancient_god": VoiceProfile(
        key="ancient_god", label="Ancient god (vast, slow)",
        voice_id="Kajal", language="en-IN", engine="neural",
        pitch_semitones=-6.5, tempo=0.84,
        sub_octave=0.68, double_detune=0.40, growl=0.22,
        body_db=6.0, presence_db=4.0,
        highpass_hz=45, lowpass_hz=7000, reverb=0.78, gain_db=1.5,
        note="Slower and further away; the tail does as much work as the voice.",
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


# ---------------------------------------------------------------------------
# Voice packs: profiles the user builds and keeps
# ---------------------------------------------------------------------------

PACK_FILENAME = "voice_packs.json"
# Every knob a saved pack may set, and the range it is clamped to. Anything
# outside these stops being a voice and starts being a fault report from
# ffmpeg, so the bounds are enforced here rather than discovered at render.
PACK_LIMITS: Dict[str, tuple] = {
    "target_f0": (0.0, 400.0),
    "pitch_semitones": (-14.0, 8.0),
    "tempo": (0.5, 1.6),
    "lowpass_hz": (0, 20000),
    "highpass_hz": (0, 2000),
    "reverb": (0.0, 1.0),
    "gain_db": (-12.0, 12.0),
    "sub_octave": (0.0, 1.0),
    "double_detune": (0.0, 1.0),
    "growl": (0.0, 1.0),
    "body_db": (-12.0, 12.0),
    "presence_db": (-12.0, 12.0),
}

_PACKS: Optional[Dict[str, VoiceProfile]] = None


def pack_path() -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / PACK_FILENAME


def load_packs(refresh: bool = False) -> Dict[str, VoiceProfile]:
    """Custom profiles saved by the user, keyed like the built-in ones."""
    global _PACKS
    if _PACKS is not None and not refresh:
        return _PACKS
    _PACKS = {}
    path = pack_path()
    if path.exists():
        try:
            for key, data in json.loads(path.read_text(encoding="utf-8")).items():
                _PACKS[key] = VoiceProfile(**data)
        except Exception as exc:
            logger.warning("Voice packs unreadable (%s), ignoring them", exc)
            _PACKS = {}
    return _PACKS


def save_pack(
    key: str,
    label: str,
    base: str = DEFAULT_VOICE_KEY,
    note: str = "",
    **knobs: float,
) -> VoiceProfile:
    """Create or replace one custom voice, built on an existing voice id.

    A pack chooses a stock voice to speak and then shapes it. It cannot invent
    a voice id: one Polly does not have is a 400 at render time, and the point
    of saving a pack is that it works later without being re-checked.
    """
    key = re.sub(r"[^a-z0-9_]+", "_", (key or "").strip().lower()).strip("_")
    if not key:
        raise ValueError("a voice pack needs a name")
    if key in VOICE_PROFILES:
        raise ValueError(f"'{key}' is a built-in voice; choose another name")

    parent = resolve_voice(base)
    values: Dict[str, float] = {}
    for field_name, (low, high) in PACK_LIMITS.items():
        if field_name not in knobs or knobs[field_name] is None:
            continue
        value = float(knobs[field_name])
        values[field_name] = min(max(value, low), high)
        if field_name in ("lowpass_hz", "highpass_hz"):
            values[field_name] = int(values[field_name])

    profile = VoiceProfile(
        key=key, label=(label or key.replace("_", " ").title()),
        voice_id=parent.voice_id, language=parent.language, engine=parent.engine,
        note=note or f"Custom pack built on '{parent.key}'.",
        **values,
    )
    packs = load_packs()
    packs[key] = profile
    _write_packs(packs)
    return profile


def clone_dir() -> Path:
    """Where prepared reference recordings live, next to the packs."""
    path = settings.data_dir / "voice_clones"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_clone_pack(
    key: str,
    label: str,
    reference: Path,
    language: str = "en-US",
    quality: int = 20,
    note: str = "",
    ffmpeg: Optional[str] = None,
    **knobs: float,
):
    """Save a voice built from a recording rather than from a stock speaker.

    The reference is copied into the app's own storage in the form the model
    wants, because a pack that points at a file in someone's downloads folder
    stops working the moment they tidy up.

    Returns the profile and the report on the recording, which carries the
    reasons a clone may come back disappointing - those belong in front of
    whoever saved it, not in a log.
    """
    from magpie_tts import prepare_reference  # noqa: PLC0415

    clean = re.sub(r"[^a-z0-9_]+", "_", (key or "").strip().lower()).strip("_")
    if not clean:
        raise ValueError("a voice pack needs a name")
    if clean in VOICE_PROFILES:
        raise ValueError(f"'{clean}' is a built-in voice; choose another name")

    destination = clone_dir() / f"{clean}.wav"
    report = prepare_reference(Path(reference), destination, ffmpeg=ffmpeg)

    values: Dict[str, float] = {}
    for field_name, (low, high) in PACK_LIMITS.items():
        if field_name not in knobs or knobs[field_name] is None:
            continue
        value = min(max(float(knobs[field_name]), low), high)
        values[field_name] = int(value) if field_name.endswith("_hz") else value

    profile = VoiceProfile(
        key=clean, label=(label or clean.replace("_", " ").title()),
        voice_id="", language=language, engine="magpie",
        clone_reference=str(destination), clone_language=language,
        clone_quality=max(1, min(40, int(quality))),
        note=note or "Cloned from a recording with Magpie TTS Zero-Shot.",
        **values,
    )
    packs = load_packs()
    packs[clean] = profile
    _write_packs(packs)
    return profile, report


def delete_pack(key: str) -> bool:
    packs = load_packs()
    if key not in packs:
        return False
    gone = packs.pop(key)
    if gone.clone_reference:
        # The recording is the pack. Leaving it behind means a re-saved pack
        # of the same name silently inherits the old voice.
        Path(gone.clone_reference).unlink(missing_ok=True)
    _write_packs(packs)
    return True


def _write_packs(packs: Dict[str, VoiceProfile]) -> None:
    global _PACKS
    try:
        pack_path().write_text(
            json.dumps({key: asdict(value) for key, value in packs.items()},
                       indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        _PACKS = packs
    except Exception as exc:
        logger.warning("Could not persist the voice packs: %s", exc)


def all_voices() -> Dict[str, VoiceProfile]:
    """Built-ins first, then the user's packs."""
    merged = dict(VOICE_PROFILES)
    merged.update(load_packs())
    return merged


def resolve_voice(name: Optional[str]) -> VoiceProfile:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    packs = load_packs()
    if key in packs:
        return packs[key]
    if key in VOICE_PROFILES:
        return VOICE_PROFILES[key]
    # Anything that mentions India stays Indian; anything dark goes deep.
    if any(word in key for word in ("deep", "dark", "myster", "narrat")):
        return VOICE_PROFILES["deep_dark"]
    if "indian" in key or "hing" in key:
        return VOICE_PROFILES["indian_accent"]
    return VOICE_PROFILES[DEFAULT_VOICE_KEY]


def voice_keys() -> List[str]:
    return list(all_voices())


# ---------------------------------------------------------------------------
# Local shaping
# ---------------------------------------------------------------------------


def measure_f0(path: Path, floor: float = 60.0, ceiling: float = 500.0) -> float:
    """Median fundamental of a recording, in Hz. 0.0 when nothing is voiced.

    Autocorrelation per 40ms frame, taking the median over the voiced ones -
    robust enough for "where does this voice sit", which is all it is for.
    """
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except Exception:
        return 0.0
    if width != 2 or not raw or not rate:
        return 0.0

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    window = int(0.040 * rate)
    hop = int(0.010 * rate)
    low, high = int(rate / ceiling), int(rate / floor)
    if window <= 0 or high <= low or len(samples) < window:
        return 0.0

    found: List[float] = []
    for start in range(0, len(samples) - window, hop):
        frame = samples[start:start + window]
        if float(np.sqrt((frame ** 2).mean())) < 0.02:
            continue
        frame = frame - frame.mean()
        correlation = np.correlate(frame, frame, "full")[window - 1:]
        if correlation[0] <= 0:
            continue
        segment = correlation[low:high]
        if not len(segment):
            continue
        lag = int(np.argmax(segment)) + low
        # Below this the "peak" is noise, not a period.
        if correlation[lag] / correlation[0] < 0.3:
            continue
        found.append(rate / lag)
    return float(np.median(found)) if found else 0.0


def semitones_to(measured: float, target: float, limit: float = 8.0) -> float:
    """The shift that moves ``measured`` onto ``target``, clamped.

    The clamp matters: a line the measurer got wrong - one word, mostly
    breath - must not be transposed into a cartoon. Past the limit it is
    better to be a little off the target than unrecognisable.
    """
    import math  # noqa: PLC0415

    if measured <= 0 or target <= 0:
        return 0.0
    shift = 12.0 * math.log2(target / measured)
    return max(-limit, min(limit, shift))


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


def _pitch_steps(semitones: float, sample_rate: int) -> List[str]:
    """Shift pitch without changing length: resample the rate, then correct it."""
    ratio = 2.0 ** (semitones / 12.0)
    return [
        f"aresample={sample_rate}",
        f"asetrate={int(sample_rate * ratio)}",
        f"aresample={sample_rate}",
        *_atempo_chain(1.0 / ratio),
    ]


def build_complex_chain(profile: VoiceProfile, sample_rate: int = 24000) -> str:
    """An ffmpeg -filter_complex that builds the voice out of stacked layers.

    The dry voice keeps the words. An octave-down copy sits under it carrying
    the weight, low-passed so it is felt rather than heard as a second speaker.
    A third copy, detuned a fraction and delayed twenty milliseconds, thickens
    it into something that does not sound like one throat.

    Levels matter more than any of the effects: the sub has to stay well under
    the dry layer or the words disappear into it, which is the usual way a
    "deep" voice ends up simply unintelligible.
    """
    parts: List[str] = []
    layers: List[str] = []

    parts.append(f"[0:a]aresample={sample_rate},asplit=3[dry0][sub0][dbl0]")

    dry = [*(_pitch_steps(profile.pitch_semitones, sample_rate)
             if abs(profile.pitch_semitones) > 0.01 else [])]
    if abs(profile.tempo - 1.0) > 0.01:
        dry.extend(_atempo_chain(profile.tempo))
    if profile.growl > 0.01:
        # Drive into a tanh curve: the added harmonics are what reads as a
        # snarl. Gain comes back off afterwards so the layer keeps its level.
        drive = 1.0 + profile.growl * 3.0
        dry.append(f"volume={drive:.2f}")
        dry.append(f"asoftclip=type=tanh:param={0.4 + profile.growl:.2f}")
        dry.append(f"volume={1.0 / drive:.3f}")
    if abs(profile.body_db) > 0.01:
        dry.append(f"equalizer=f=180:t=q:w=1.1:g={profile.body_db:.1f}")
    if abs(profile.presence_db) > 0.01:
        dry.append(f"equalizer=f=3000:t=q:w=1.4:g={profile.presence_db:.1f}")
    parts.append("[dry0]" + ",".join(dry or ["anull"]) + "[dry]")
    layers.append("[dry]")

    if profile.sub_octave > 0.01:
        sub = _pitch_steps(profile.pitch_semitones - 12.0, sample_rate)
        # Felt, not heard: everything above the chest is the dry layer's job.
        sub.append("lowpass=f=220")
        if abs(profile.tempo - 1.0) > 0.01:
            sub.extend(_atempo_chain(profile.tempo))
        sub.append(f"volume={profile.sub_octave:.3f}")
        parts.append("[sub0]" + ",".join(sub) + "[sub]")
        layers.append("[sub]")
    else:
        parts.append("[sub0]anullsink")

    if profile.double_detune > 0.01:
        dbl = _pitch_steps(profile.pitch_semitones - 0.35, sample_rate)
        if abs(profile.tempo - 1.0) > 0.01:
            dbl.extend(_atempo_chain(profile.tempo))
        dbl.append("adelay=22:all=1")
        dbl.append(f"volume={profile.double_detune:.3f}")
        parts.append("[dbl0]" + ",".join(dbl) + "[dbl]")
        layers.append("[dbl]")
    else:
        parts.append("[dbl0]anullsink")

    tail: List[str] = []
    if profile.highpass_hz:
        tail.append(f"highpass=f={profile.highpass_hz}")
    if profile.lowpass_hz:
        tail.append(f"lowpass=f={profile.lowpass_hz}")
    if profile.reverb > 0.01:
        decay = min(0.9, 0.35 + profile.reverb * 0.55)
        delay_a = int(40 + profile.reverb * 40)
        delay_b = int(90 + profile.reverb * 110)
        tail.append(f"aecho=0.85:0.9:{delay_a}|{delay_b}:{decay:.2f}|{decay * 0.6:.2f}")
    if abs(profile.gain_db) > 0.01:
        tail.append(f"volume={profile.gain_db:.2f}dB")
    tail.append("alimiter=limit=0.97")

    mix = "".join(layers)
    parts.append(f"{mix}amix=inputs={len(layers)}:normalize=0," + ",".join(tail) + "[out]")
    return ";".join(parts)


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

    if profile.is_layered:
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-filter_complex", build_complex_chain(profile),
            "-map", "[out]",
            "-c:a", "libmp3lame", "-b:a", audio_bitrate, str(destination),
        ]
    else:
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-af", build_filter_chain(profile),
            "-c:a", "libmp3lame", "-b:a", audio_bitrate, str(destination),
        ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Voice shaping failed (%s), using the raw voice: %s",
                       profile.key, result.stderr[:200])
        if source != destination:
            destination.write_bytes(source.read_bytes())
    return destination
