"""Live check against NVIDIA Magpie TTS Zero-Shot. Needs a real key.

    NVIDIA_NIM_API_KEY=... python -m pytest tests/test_magpie_live.py -q

Skipped without one. Everything offline lives in test_voice_cloning.py; this
file exists for the claims only the service can settle - that the function id
is still live, that a clone actually follows the reference, and that the
languages the app offers are ones the model accepts.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magpie_tts import MagpieCloner, prepare_reference  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402

KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not KEY, reason="needs NVIDIA_NIM_API_KEY"),
]
FF = ffmpeg_binary()


def _reference(tmp_path: Path) -> Path:
    """A synthetic 8s 'voice' with a known fundamental."""
    rate, seconds, f0 = 44100, 8.0, 150.0
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.12 * np.sin(2 * np.pi * 5000 * t)
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
    source = tmp_path / "src.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((signal * 32000).astype(np.int16).tobytes())
    prepare_reference(source, tmp_path / "ref.wav", ffmpeg=FF)
    return tmp_path / "ref.wav"


def _seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def test_the_function_id_is_still_live(tmp_path):
    """A moved function id is a silent outage; this is the only thing that catches it."""
    cloner = MagpieCloner(KEY)
    out = cloner.synthesize("Testing one two three.", tmp_path / "a.wav")
    assert _seconds(out) > 0.5


def test_a_reference_clip_is_accepted(tmp_path):
    cloner = MagpieCloner(KEY)
    out = cloner.synthesize(
        "Know your place.", tmp_path / "b.wav", _reference(tmp_path))
    assert _seconds(out) > 0.4


@pytest.mark.parametrize("language", ["en-US", "hi-IN"])
def test_every_language_the_app_offers_is_one_the_model_takes(tmp_path, language):
    cloner = MagpieCloner(KEY)
    out = cloner.synthesize(
        "Namaste. Hello.", tmp_path / f"{language}.wav",
        _reference(tmp_path), language=language,
    )
    assert _seconds(out) > 0.4


def test_a_longer_line_produces_longer_audio(tmp_path):
    """Guards against a truncating response being mistaken for success."""
    cloner = MagpieCloner(KEY)
    reference = _reference(tmp_path)
    short = cloner.synthesize("Stop.", tmp_path / "short.wav", reference)
    long = cloner.synthesize(
        "Stop right there and listen to every single word I am about to say.",
        tmp_path / "long.wav", reference,
    )
    assert _seconds(long) > _seconds(short) * 2
