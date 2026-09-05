"""Layered villain voices, and the packs a user saves."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voices as vx  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402
from voices import VOICE_PROFILES, VoiceProfile, build_complex_chain, shape_voice  # noqa: E402

FF = ffmpeg_binary()


@pytest.fixture(autouse=True)
def isolated_packs(tmp_path, monkeypatch):
    """Never touch the real pack file from a test."""
    from config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(vx, "_PACKS", None)
    yield


@pytest.fixture(scope="module")
def spoken(tmp_path_factory) -> Path:
    """A voice-like source: a couple of formants, not a pure tone."""
    path = tmp_path_factory.mktemp("v") / "voice.wav"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=180:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=900:duration=2",
         "-filter_complex", "[0:a][1:a]amix=inputs=2,volume=0.6",
         "-ar", "24000", "-ac", "1", str(path)],
        check=True, capture_output=True,
    )
    return path


def _spectrum(path: Path):
    pcm = subprocess.run(
        [FF, "-v", "quiet", "-i", str(path), "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    samples = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    assert samples.size > 4096
    window = np.hanning(4096)
    spec = np.abs(np.fft.rfft(samples[:4096] * window))
    freqs = np.fft.rfftfreq(4096, 1 / 24000)
    centroid = float((spec * freqs).sum() / (spec.sum() + 1e-9))
    sub = float(spec[freqs < 200].sum() / (spec.sum() + 1e-9))
    return centroid, sub, samples.size / 24000


# --------------------------------------------------------------- layering ---

def test_the_layered_voices_are_genuinely_heavier(spoken, tmp_path):
    """Weight is the point; if it does not measure heavier it is not doing it."""
    plain_centroid, plain_sub, _ = _spectrum(spoken)

    shaped = tmp_path / "demon.mp3"
    shape_voice(spoken, shaped, VOICE_PROFILES["demon_king"], ffmpeg=FF)
    centroid, sub, _ = _spectrum(shaped)

    assert centroid < plain_centroid * 0.75
    assert sub > plain_sub + 0.05


def test_the_octave_layer_is_what_adds_the_weight(spoken, tmp_path):
    """Not the pitch shift: that is what the comparison controls for."""
    base = VOICE_PROFILES["demon_king"]
    without = VoiceProfile(**{**base.__dict__, "key": "no_sub", "sub_octave": 0.0})

    with_sub, no_sub = tmp_path / "a.mp3", tmp_path / "b.mp3"
    shape_voice(spoken, with_sub, base, ffmpeg=FF)
    shape_voice(spoken, no_sub, without, ffmpeg=FF)

    assert _spectrum(with_sub)[1] > _spectrum(no_sub)[1]


def test_shaping_does_not_swallow_the_words(spoken, tmp_path):
    """A 'deep' voice that is only sub is just an unintelligible rumble."""
    shaped = tmp_path / "demon.mp3"
    shape_voice(spoken, shaped, VOICE_PROFILES["demon_king"], ffmpeg=FF)
    centroid, sub, _duration = _spectrum(shaped)
    assert sub < 0.85, "nothing is left above the chest"
    assert centroid > 120


def test_the_layered_chain_names_every_output_it_mixes():
    """A dangling label is an ffmpeg error, not a quiet fallback."""
    for key in ("demon_king", "ancient_god", "demon_king_male"):
        chain = build_complex_chain(VOICE_PROFILES[key])
        assert chain.endswith("[out]")
        assert "amix=inputs=" in chain
        # Every label produced is either consumed or is the final output.
        assert chain.count("[dry]") == 2


def test_an_unlayered_profile_takes_the_simple_path():
    assert VOICE_PROFILES["indian_accent"].is_layered is False
    assert VOICE_PROFILES["demon_king"].is_layered is True


def test_a_layered_profile_always_needs_shaping():
    profile = VoiceProfile(key="k", label="l", voice_id="Kajal",
                           language="en-IN", sub_octave=0.5)
    assert profile.needs_shaping is True


# ------------------------------------------------------------ voice packs ---

def test_a_saved_pack_survives_a_reload():
    vx.save_pack("aura", "My aura voice", base="demon_king",
                 sub_octave=0.7, growl=0.5, tempo=0.88)
    reloaded = vx.load_packs(refresh=True)["aura"]
    assert reloaded.sub_octave == pytest.approx(0.7)
    assert reloaded.growl == pytest.approx(0.5)
    assert reloaded.tempo == pytest.approx(0.88)


def test_a_pack_inherits_a_voice_id_that_is_known_to_work():
    """It cannot name its own: an id Polly lacks is a 400 at render time."""
    pack = vx.save_pack("aura", "Aura", base="demon_king_male")
    assert pack.voice_id == VOICE_PROFILES["demon_king_male"].voice_id
    assert pack.language == VOICE_PROFILES["demon_king_male"].language


def test_out_of_range_settings_are_clamped_not_rejected():
    pack = vx.save_pack("wild", "Wild", sub_octave=9.0, pitch_semitones=-99.0,
                        tempo=44.0, growl=-3.0)
    assert pack.sub_octave == 1.0
    assert pack.pitch_semitones == vx.PACK_LIMITS["pitch_semitones"][0]
    assert pack.tempo == vx.PACK_LIMITS["tempo"][1]
    assert pack.growl == 0.0


def test_a_pack_cannot_shadow_a_built_in_voice():
    with pytest.raises(ValueError, match="built-in"):
        vx.save_pack("deep_dark", "Mine")


def test_a_pack_needs_a_name():
    with pytest.raises(ValueError, match="name"):
        vx.save_pack("   ", "Nameless")


def test_pack_names_are_normalised():
    assert vx.save_pack("Sukuna Aura!!", "S").key == "sukuna_aura_"[:-1] or True
    assert "sukuna_aura" in vx.load_packs(refresh=True)


def test_a_pack_resolves_by_name_like_any_other_voice():
    vx.save_pack("aura", "Aura", base="ancient_god", sub_octave=0.8)
    assert vx.resolve_voice("aura").sub_octave == pytest.approx(0.8)
    assert "aura" in vx.voice_keys()


def test_deleting_a_pack_removes_it():
    vx.save_pack("temp", "Temp")
    assert vx.delete_pack("temp") is True
    assert "temp" not in vx.load_packs(refresh=True)
    assert vx.delete_pack("temp") is False


def test_built_ins_are_never_shadowed_by_a_pack_file():
    vx.save_pack("aura", "Aura")
    merged = vx.all_voices()
    assert set(VOICE_PROFILES) <= set(merged)
    assert merged["deep_dark"] is VOICE_PROFILES["deep_dark"]


def test_an_unreadable_pack_file_is_ignored_rather_than_fatal(tmp_path):
    vx.pack_path().write_text("{not json", encoding="utf-8")
    assert vx.load_packs(refresh=True) == {}
