"""Voice cloning through NVIDIA's Magpie TTS Zero-Shot model.

Puter's TTS reads a line in a stock voice and the local shaping in voices.py
bends it toward a character. That is a costume, not a voice: no amount of
pitch and sub-octave layering makes a stock reader sound like a *particular*
person. Magpie Zero-Shot takes the other route - hand it ten seconds of the
voice you want and it synthesises new lines in that voice, with no training
step.

The model is not on the OpenAI-compatible NIM endpoint. It is a Riva service
reached over gRPC through NVIDIA Cloud Functions, which is why this module
talks a different protocol to every other provider in the project.

    cloner = MagpieCloner(api_key)
    reference = prepare_reference(Path("sample.mp3"), Path("ref.wav"))
    cloner.synthesize("Know your place.", Path("line.wav"), reference)
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ai-magpie-tts-zeroshot, from the NVCF function catalogue. Pinned rather than
# looked up: the listing call needs the same key and the id does not move.
FUNCTION_ID = "55cf67bf-600f-4b04-8eac-12ed39537a08"
GRPC_URI = "grpc.nvcf.nvidia.com:443"

# The model ships two built-in speakers. With a reference clip they are the
# starting point the clone is steered away from, not the output voice.
BUILTIN_VOICES = (
    "Magpie-ZeroShot-Multilingual.Male",
    "Magpie-ZeroShot-Multilingual.Female",
)
DEFAULT_VOICE = BUILTIN_VOICES[0]

# What the model documents as a usable reference: a few seconds is not enough
# to pin a voice down, and past ten it stops paying for the upload.
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0
REFERENCE_RATE = 22050
OUTPUT_RATE = 44100

# Quality trades synthesis speed against how closely the clone follows the
# reference. 20 is the model's own default and sits in the middle of 1..40.
DEFAULT_QUALITY = 20
QUALITY_RANGE = (1, 40)


class MagpieUnavailable(RuntimeError):
    """The service could not be reached, or refused the request."""


class ReferenceUnusable(ValueError):
    """The reference recording cannot be cloned from."""


@dataclass
class ReferenceReport:
    """What the reference recording actually is, measured rather than assumed."""

    path: Path
    seconds: float
    source_rate: int
    peak: float
    speech_ratio: float
    # Share of the energy above 4kHz in the ORIGINAL file. Resampling raises
    # the header and adds nothing, so this is the only honest read on whether
    # the recording carries the detail the model wants.
    brilliance: float
    notes: List[str]

    @property
    def usable(self) -> bool:
        return self.seconds >= MIN_REFERENCE_SECONDS and self.peak > 0.02


def _read_mono(path: Path):
    import numpy as np  # noqa: PLC0415

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _brilliance(samples, rate: int) -> float:
    """Fraction of the energy above 4kHz - the air a phone recording loses."""
    import numpy as np  # noqa: PLC0415

    if len(samples) < 1024:
        return 0.0
    window = samples[: 1 << int(math.log2(len(samples)))]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)))) ** 2
    freqs = np.fft.rfftfreq(len(window), 1.0 / rate)
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    return float(spectrum[freqs >= 4000].sum() / total)


def _speech_ratio(samples, rate: int) -> float:
    """Fraction of the clip loud enough to be someone talking."""
    import numpy as np  # noqa: PLC0415

    window = max(int(0.02 * rate), 1)
    usable = len(samples) - len(samples) % window
    if usable <= 0:
        return 0.0
    frames = samples[:usable].reshape(-1, window)
    loudness = np.sqrt((frames ** 2).mean(axis=1))
    return float((loudness > max(loudness.max() * 0.15, 1e-4)).mean())


def prepare_reference(
    source: Path,
    destination: Path,
    ffmpeg: Optional[str] = None,
    seconds: float = MAX_REFERENCE_SECONDS,
) -> ReferenceReport:
    """Turn any recording into the mono 16-bit WAV the model wants.

    Reports what it found rather than silently accepting it: a reference that
    is mostly silence, or that has had everything above 4kHz thrown away by a
    low-bitrate encoder, clones badly and the caller should be told why before
    it spends a minute finding out.
    """
    import subprocess  # noqa: PLC0415

    from puter_integration import ffmpeg_binary  # noqa: PLC0415

    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise ReferenceUnusable(f"no reference recording at {source}")
    binary = ffmpeg or ffmpeg_binary()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Measure the original before any resampling: that is where the answer to
    # "does this recording carry enough detail" actually lives.
    probe = destination.with_name(destination.stem + "-probe.wav")
    subprocess.run(
        [binary, "-v", "error", "-y", "-i", str(source), "-ac", "1",
         "-sample_fmt", "s16", str(probe)],
        check=True, capture_output=True,
    )
    samples, rate = _read_mono(probe)
    import numpy as np  # noqa: PLC0415

    report = ReferenceReport(
        path=destination,
        seconds=len(samples) / rate if rate else 0.0,
        source_rate=rate,
        peak=float(np.abs(samples).max()) if len(samples) else 0.0,
        speech_ratio=_speech_ratio(samples, rate),
        brilliance=_brilliance(samples, rate),
        notes=[],
    )
    probe.unlink(missing_ok=True)

    if report.seconds < MIN_REFERENCE_SECONDS:
        raise ReferenceUnusable(
            f"the reference is {report.seconds:.1f}s long; the model needs at "
            f"least {MIN_REFERENCE_SECONDS:.0f}s to pin a voice down."
        )
    if report.peak <= 0.02:
        raise ReferenceUnusable("the reference recording is silent.")

    if report.source_rate < REFERENCE_RATE:
        report.notes.append(
            f"the reference was recorded at {report.source_rate / 1000:.1f}kHz, "
            f"below the {REFERENCE_RATE / 1000:.2f}kHz the model asks for; "
            f"resampling raises the number, not the detail."
        )
    if report.brilliance < 0.02:
        report.notes.append(
            f"only {report.brilliance * 100:.1f}% of the reference sits above "
            f"4kHz, so the clone will come back soft on consonants."
        )
    if report.speech_ratio < 0.4:
        report.notes.append(
            f"about {(1 - report.speech_ratio) * 100:.0f}% of the reference is "
            f"silence or room tone; a denser sample clones closer."
        )

    keep = min(max(seconds, MIN_REFERENCE_SECONDS), MAX_REFERENCE_SECONDS)
    subprocess.run(
        [binary, "-v", "error", "-y", "-i", str(source), "-t", f"{keep:.3f}",
         "-ac", "1", "-ar", str(REFERENCE_RATE), "-sample_fmt", "s16",
         "-af", "highpass=f=60,dynaudnorm=f=200:g=5", str(destination)],
        check=True, capture_output=True,
    )
    return report


class MagpieCloner:
    """Synthesises lines in a cloned voice. One channel, opened once."""

    def __init__(self, api_key: str, function_id: str = FUNCTION_ID,
                 uri: str = GRPC_URI) -> None:
        if not api_key:
            raise MagpieUnavailable("no NVIDIA API key for Magpie TTS")
        self.api_key = api_key
        self.function_id = function_id
        self.uri = uri
        self._service = None

    @property
    def service(self):
        if self._service is None:
            try:
                import riva.client  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - install-time only
                raise MagpieUnavailable(
                    "nvidia-riva-client is not installed; Magpie speaks gRPC, "
                    "not REST, so it needs its own client."
                ) from exc
            auth = riva.client.Auth(
                uri=self.uri, use_ssl=True,
                metadata_args=[
                    ["function-id", self.function_id],
                    ["authorization", f"Bearer {self.api_key}"],
                ],
            )
            self._service = riva.client.SpeechSynthesisService(auth)
        return self._service

    def synthesize(
        self,
        text: str,
        destination: Path,
        reference: Optional[Path] = None,
        language: str = "en-US",
        voice: str = DEFAULT_VOICE,
        quality: int = DEFAULT_QUALITY,
        transcript: str = "",
    ) -> Path:
        """Speak one line, in the reference voice when one is given."""
        import riva.client  # noqa: PLC0415

        text = (text or "").strip()
        if not text:
            raise ValueError("nothing to say")
        low, high = QUALITY_RANGE
        quality = max(low, min(high, int(quality)))
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self.service.synthesize(
                text=text,
                voice_name=voice,
                language_code=language,
                sample_rate_hz=OUTPUT_RATE,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                zero_shot_audio_prompt_file=Path(reference) if reference else None,
                zero_shot_quality=quality,
                zero_shot_transcript=transcript or None,
            )
        except Exception as exc:
            raise MagpieUnavailable(f"Magpie TTS refused the line: {exc}") from exc

        audio = getattr(response, "audio", b"")
        if not audio:
            raise MagpieUnavailable("Magpie TTS returned no audio")
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(OUTPUT_RATE)
            handle.writeframes(audio)
        return destination
