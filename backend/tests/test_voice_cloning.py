"""Cloning a voice from a recording instead of shaping a stock one.

The network half is covered by test_magpie_live.py; everything here runs
offline, because the parts that go wrong most often - a reference that is too
short, a pack whose recording has been deleted, a clone that quietly falls
back to a stock speaker - go wrong without the service being involved at all.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voices  # noqa: E402
from magpie_tts import (  # noqa: E402
    MAX_REFERENCE_SECONDS,
    MIN_REFERENCE_SECONDS,
    REFERENCE_RATE,
    MagpieCloner,
    MagpieUnavailable,
    ReferenceUnusable,
    prepare_reference,
)
from puter_integration import ffmpeg_binary  # noqa: E402
from voices import VoiceProfile, resolve_voice  # noqa: E402

FF = ffmpeg_binary()


def _tone(path: Path, seconds: float, rate: int = 44100, f0: float = 180.0,
          brightness: float = 0.5) -> Path:
    """A voice-ish signal: a fundamental, harmonics, and controllable air."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave_ = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    wave_ += brightness * 0.2 * np.sin(2 * np.pi * 6000 * t)
    # Amplitude that moves, so speech_ratio measures something real.
    wave_ *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((wave_ * 32000).astype(np.int16).tobytes())
    return path


# ------------------------------------------------------- reference checks ---

def test_a_reference_is_converted_to_what_the_model_asks_for(tmp_path):
    source = _tone(tmp_path / "src.wav", 8.0)
    report = prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)

    with wave.open(str(tmp_path / "ref.wav"), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == REFERENCE_RATE
    assert report.seconds == pytest.approx(8.0, abs=0.2)


def test_a_long_reference_is_trimmed_not_refused(tmp_path):
    source = _tone(tmp_path / "src.wav", 30.0)
    prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)

    with wave.open(str(tmp_path / "ref.wav"), "rb") as handle:
        seconds = handle.getnframes() / handle.getframerate()
    assert seconds == pytest.approx(MAX_REFERENCE_SECONDS, abs=0.2)


def test_too_short_a_reference_is_refused_with_the_reason(tmp_path):
    source = _tone(tmp_path / "src.wav", MIN_REFERENCE_SECONDS - 1.0)
    with pytest.raises(ReferenceUnusable, match="pin a voice down"):
        prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)


def test_a_silent_reference_is_refused(tmp_path):
    source = tmp_path / "src.wav"
    _tone(source, 8.0)
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(np.zeros(44100 * 8, dtype=np.int16).tobytes())
    with pytest.raises(ReferenceUnusable, match="silent"):
        prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)


def test_a_low_rate_reference_is_reported_and_still_converted(tmp_path):
    """Resampling raises the header and adds nothing; the report must not lie."""
    bright = _tone(tmp_path / "bright.wav", 8.0, rate=44100, brightness=1.0)
    narrow = tmp_path / "narrow.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(bright), "-ar", "16000",
                    "-ac", "1", str(narrow)], check=True, capture_output=True)

    report = prepare_reference(narrow, tmp_path / "ref.wav", ffmpeg=FF)

    assert report.source_rate == 16000
    assert any("16.0kHz" in note for note in report.notes)
    # The file it writes still claims the rate the model wants.
    with wave.open(str(tmp_path / "ref.wav"), "rb") as handle:
        assert handle.getframerate() == REFERENCE_RATE


def test_a_reference_with_its_air_encoded_away_says_so(tmp_path):
    """The rate is not the measure - a 16kHz file still carries 4-8kHz.

    What actually costs a clone its consonants is a low-bitrate encoder
    throwing the top away, which no header records, so it has to be measured.
    """
    bright = _tone(tmp_path / "bright.wav", 8.0, rate=44100, brightness=1.0)
    dull = tmp_path / "dull.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(bright),
                    "-af", "lowpass=f=3500", "-ac", "1", str(dull)],
                   check=True, capture_output=True)

    dull_report = prepare_reference(dull, tmp_path / "dull-ref.wav", ffmpeg=FF)
    bright_report = prepare_reference(bright, tmp_path / "bright-ref.wav", ffmpeg=FF)

    assert dull_report.brilliance < bright_report.brilliance
    assert any("above" in note and "4kHz" in note for note in dull_report.notes)
    assert not any("4kHz" in note for note in bright_report.notes)


def test_a_bright_reference_raises_no_complaint(tmp_path):
    source = _tone(tmp_path / "src.wav", 8.0, rate=44100, brightness=1.0)
    report = prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)
    assert report.notes == []
    assert report.brilliance > 0.02


# ------------------------------------------------------------- the packs ----

@pytest.fixture
def isolated_packs(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    voices._PACKS = None
    yield
    voices._PACKS = None


def test_a_saved_clone_owns_its_recording(tmp_path, isolated_packs):
    source = _tone(tmp_path / "src.wav", 8.0)
    profile, _ = voices.save_clone_pack(
        "villain", "Villain", source, language="hi-IN", ffmpeg=FF,
        pitch_semitones=3.0,
    )

    kept = Path(profile.clone_reference)
    assert kept.exists()
    # Copied into the app's own storage: deleting the original changes nothing.
    source.unlink()
    assert kept.exists()
    assert profile.is_cloned
    assert profile.clone_language == "hi-IN"
    assert profile.pitch_semitones == 3.0


def test_a_clone_survives_a_reload(tmp_path, isolated_packs):
    _tone(tmp_path / "src.wav", 8.0)
    voices.save_clone_pack("villain", "Villain", tmp_path / "src.wav", ffmpeg=FF)

    voices._PACKS = None
    again = resolve_voice("villain")
    assert again.is_cloned
    assert Path(again.clone_reference).exists()


def test_deleting_a_clone_takes_its_recording_with_it(tmp_path, isolated_packs):
    _tone(tmp_path / "src.wav", 8.0)
    profile, _ = voices.save_clone_pack(
        "villain", "Villain", tmp_path / "src.wav", ffmpeg=FF)
    kept = Path(profile.clone_reference)

    assert voices.delete_pack("villain") is True
    assert not kept.exists(), "a re-saved pack would inherit the old voice"


def test_pack_knobs_are_still_clamped_on_a_clone(tmp_path, isolated_packs):
    _tone(tmp_path / "src.wav", 8.0)
    profile, _ = voices.save_clone_pack(
        "villain", "Villain", tmp_path / "src.wav", ffmpeg=FF,
        pitch_semitones=99.0, growl=5.0,
    )
    low, high = voices.PACK_LIMITS["pitch_semitones"]
    assert profile.pitch_semitones == high
    assert profile.growl == 1.0


def test_a_built_in_name_cannot_be_taken_by_a_clone(tmp_path, isolated_packs):
    _tone(tmp_path / "src.wav", 8.0)
    with pytest.raises(ValueError, match="built-in"):
        voices.save_clone_pack("demon_king", "Nope", tmp_path / "src.wav", ffmpeg=FF)


# ------------------------------------------------------------- the client ---

def test_a_clone_never_quietly_becomes_a_stock_voice(tmp_path):
    """The failure that matters: a missing recording must not render anyway."""
    from puter_integration import PuterClient, PuterError

    profile = VoiceProfile(
        key="ghost", label="Ghost", voice_id="", language="hi-IN",
        engine="magpie", clone_reference=str(tmp_path / "gone.wav"),
    )
    client = PuterClient(api_key="x", nim_api_key="y")
    client.resolve_voice = lambda _accent: profile

    with pytest.raises(PuterError, match="re-save the pack"):
        client.text_to_speech("kuch bhi", tmp_path / "out.mp3", accent="ghost")


def test_a_clone_without_a_key_says_which_key_is_missing(tmp_path):
    from puter_integration import PuterClient, PuterError

    reference = _tone(tmp_path / "ref.wav", 8.0)
    profile = VoiceProfile(
        key="ghost", label="Ghost", voice_id="", language="hi-IN",
        engine="magpie", clone_reference=str(reference),
    )
    client = PuterClient(api_key="puter-key", nim_api_key="")
    client.resolve_voice = lambda _accent: profile

    with pytest.raises(PuterError, match="NVIDIA"):
        client.text_to_speech("kuch bhi", tmp_path / "out.mp3", accent="ghost")


def test_the_cloner_refuses_to_be_built_without_a_key():
    with pytest.raises(MagpieUnavailable, match="NVIDIA API key"):
        MagpieCloner("")


def test_a_stock_voice_is_untouched_by_any_of_this():
    assert resolve_voice("demon_king").is_cloned is False
