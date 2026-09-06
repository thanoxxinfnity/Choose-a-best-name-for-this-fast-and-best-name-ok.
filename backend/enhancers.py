"""Audio enhancement and AI background replacement.

Audio: a loudness-normalising ffmpeg chain per content type, so an edit lands
at the level social platforms expect instead of whatever the source happened to
be.  Normalisation is the part that actually matters - platforms turn loud
uploads down, so mastering louder than the target only costs dynamic range.

Background: two routes to knocking the background out of a shot -
:func:`chroma_key_video` for real green screens (cheap and exact) and
:func:`ai_remove_background` for everything else (``rembg`` per sampled frame,
masks interpolated between them, because a per-frame segmentation of a 60fps
clip is not affordable).
"""

from __future__ import annotations

import bisect
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from puter_integration import ffmpeg_binary
from vfx import chroma_key, composite_over

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio enhancer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioProfile:
    key: str
    label: str
    target_lufs: float = -14.0     # what YouTube/TikTok/Instagram normalise to
    true_peak: float = -1.5
    loudness_range: float = 11.0
    highpass_hz: int = 0
    lowpass_hz: int = 0
    compress: bool = False
    de_ess: bool = False
    denoise: float = 0.0           # 0..1, afftdn noise reduction


AUDIO_PROFILES: Dict[str, AudioProfile] = {
    "off": AudioProfile(key="off", label="No processing"),
    "balanced": AudioProfile(
        key="balanced", label="Balanced (default)",
        highpass_hz=40, compress=True,
    ),
    "voice": AudioProfile(
        key="voice", label="Voice clarity",
        target_lufs=-14.0, loudness_range=7.0,
        highpass_hz=90, lowpass_hz=14000, compress=True, de_ess=True, denoise=0.35,
    ),
    "music": AudioProfile(
        key="music", label="Music punch",
        target_lufs=-13.0, loudness_range=9.0, highpass_hz=25, compress=True,
    ),
    "podcast": AudioProfile(
        key="podcast", label="Spoken word / podcast",
        target_lufs=-16.0, loudness_range=6.0,
        highpass_hz=100, compress=True, de_ess=True, denoise=0.5,
    ),
}

DEFAULT_AUDIO_PROFILE = "balanced"


def resolve_audio_profile(name: Optional[str]) -> AudioProfile:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return AUDIO_PROFILES.get(key, AUDIO_PROFILES[DEFAULT_AUDIO_PROFILE])


def build_audio_chain(profile: AudioProfile) -> str:
    """The ffmpeg -af chain for a profile, ordered the way a mastering chain is."""
    if profile.key == "off":
        return ""
    filters: List[str] = []

    if profile.denoise > 0.01:
        filters.append(f"afftdn=nr={profile.denoise * 24:.1f}:nf=-28")
    if profile.highpass_hz:
        filters.append(f"highpass=f={profile.highpass_hz}")
    if profile.lowpass_hz:
        filters.append(f"lowpass=f={profile.lowpass_hz}")
    if profile.de_ess:
        # Duck 5-8kHz sibilance rather than shelving the whole top end off.
        filters.append("deesser=i=0.4:m=0.5:f=0.5")
    if profile.compress:
        filters.append(
            "acompressor=threshold=-18dB:ratio=3:attack=12:release=180:makeup=2"
        )
    # Loudness last: everything above changes the level it has to hit.
    filters.append(
        f"loudnorm=I={profile.target_lufs}:TP={profile.true_peak}:LRA={profile.loudness_range}"
    )
    filters.append("alimiter=limit=0.98")
    return ",".join(filters)


def enhance_audio(
    source: Path,
    destination: Path,
    profile: AudioProfile,
    ffmpeg: Optional[str] = None,
) -> Path:
    """Run the enhancement chain over a media file, keeping the video stream."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    chain = build_audio_chain(profile)
    if not chain:
        if source != destination:
            destination.write_bytes(source.read_bytes())
        return destination

    ffmpeg = ffmpeg or ffmpeg_binary()
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-af", chain, "-c:v", "copy",
    ]
    command += _audio_codec_args(destination)
    command.append(str(destination))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Audio enhancement failed (%s): %s", profile.key, result.stderr[:200])
        if source != destination:
            destination.write_bytes(source.read_bytes())
    return destination


# Forcing AAC into a container that cannot hold it (a .wav, say) writes a file
# that looks fine and decodes to nothing, so the codec follows the extension.
_CODEC_BY_SUFFIX: Dict[str, List[str]] = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".flac": ["-c:a", "flac"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "256k"],
    ".m4a": ["-c:a", "aac", "-b:a", "256k"],
    ".mp4": ["-c:a", "aac", "-b:a", "256k"],
    ".mov": ["-c:a", "aac", "-b:a", "256k"],
    ".mkv": ["-c:a", "aac", "-b:a", "256k"],
    ".webm": ["-c:a", "libopus", "-b:a", "192k"],
}


def _audio_codec_args(destination: Path) -> List[str]:
    return _CODEC_BY_SUFFIX.get(destination.suffix.lower(), ["-c:a", "aac", "-b:a", "256k"])


def measure_loudness(path: Path, ffmpeg: Optional[str] = None) -> Optional[float]:
    """Integrated loudness in LUFS, or None when it cannot be measured."""
    ffmpeg = ffmpeg or ffmpeg_binary()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning("Loudness measurement failed for %s: %s",
                       path.name, (result.stderr or "")[-200:])
        return None
    for line in reversed((result.stderr or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("I:") and "LUFS" in stripped:
            try:
                value = float(stripped.split("I:")[1].split("LUFS")[0].strip())
            except (IndexError, ValueError):
                continue
            # ebur128 reports its floor for silence; that is not a measurement.
            return None if value <= -69.9 else value
    return None


# ---------------------------------------------------------------------------
# Background removal / replacement
# ---------------------------------------------------------------------------


def chroma_key_video(
    source: Path,
    destination: Path,
    background: Optional[Path] = None,
    key_rgb: Tuple[int, int, int] = (0, 177, 64),
    tolerance: float = 0.32,
    fill: Tuple[int, int, int] = (0, 0, 0),
    ffmpeg: Optional[str] = None,
) -> Path:
    """Remove a green screen, optionally compositing over ``background``."""
    return _process_frames(
        source, destination,
        lambda frame, _t: _key_and_fill(frame, key_rgb, tolerance, background, fill),
        ffmpeg=ffmpeg,
    )


def _matte_keyframes(
    source: Path,
    mask_fps: float,
    matte_width: int,
) -> Tuple[List[Tuple[int, np.ndarray]], float, int]:
    """Segment the subject at intervals, at reduced resolution.

    What makes this affordable is the sampling, not the downscale: the model is
    only asked every few frames, because a silhouette moves far more slowly
    than 60fps. ``mask_fps`` is therefore the knob that controls cost.

    ``matte_width`` is not a speed control, measured rather than assumed - 320
    against 720 on a 720x1080 clip came out 334ms against 352ms a frame, which
    is noise. u2net resizes its input to a fixed size internally, so handing it
    something smaller saves only the resize it would have done itself. The
    downscale stays because it bounds memory on large frames, and because a
    matte is a low-frequency shape that loses nothing by being upscaled.
    """
    from PIL import Image
    from rembg import new_session, remove

    session = new_session("u2net_human_seg")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    keys: List[Tuple[int, np.ndarray]] = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(fps / max(mask_fps, 0.5))))
        total = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            total = index + 1
            if index % stride == 0:
                height, width = frame.shape[:2]
                scale = min(1.0, matte_width / float(max(width, 1)))
                small = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (max(32, int(width * scale)), max(32, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                cut = remove(Image.fromarray(small), session=session,
                             only_mask=True, post_process_mask=True)
                mask = np.array(cut.convert("L") if hasattr(cut, "convert") else cut)
                keys.append((index, mask))
            index += 1
    finally:
        capture.release()
    return keys, fps, total


def _smooth_masks(keys: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, np.ndarray]]:
    """Median-filter the key masks along time.

    A segmenter run independently on each sampled frame occasionally returns
    one bad matte - a dropped limb, a swallowed edge. Held or interpolated,
    that single bad frame becomes a visible pop. A median of three neighbours
    removes it without softening a real change, because a real change is
    present in two frames running and an error is not.
    """
    if len(keys) < 3:
        return keys
    smoothed: List[Tuple[int, np.ndarray]] = [keys[0]]
    for position in range(1, len(keys) - 1):
        window = [keys[position - 1][1], keys[position][1], keys[position + 1][1]]
        shape = window[1].shape
        window = [
            mask if mask.shape == shape
            else cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            for mask in window
        ]
        smoothed.append((keys[position][0], np.median(np.stack(window), axis=0).astype(np.uint8)))
    smoothed.append(keys[-1])
    return smoothed


def ai_remove_background(
    source: Path,
    destination: Path,
    background: Optional[Path] = None,
    fill: Tuple[int, int, int] = (0, 0, 0),
    mask_fps: float = 6.0,
    matte_width: int = 320,
    ffmpeg: Optional[str] = None,
) -> Path:
    """Cut the subject out with ``rembg`` and put it over a new background.

    Done in two passes. The first segments the subject at ``mask_fps`` and at
    ``matte_width`` pixels wide; the second composites every frame, blending
    between the two key mattes that bracket it. Lower ``mask_fps`` to make it
    cheaper - it is the only parameter here that meaningfully does.

    The blend is the point. Holding one matte until the next arrives freezes
    the cut-out edge for several frames and then jumps it, which on a moving
    subject reads as the character sliding inside their own outline. Crossing
    smoothly between them costs nothing and removes that entirely.
    """
    keys, _fps, _total = _matte_keyframes(Path(source), mask_fps, matte_width)
    if not keys:
        raise RuntimeError(f"Could not segment any frame of {source}")
    keys = _smooth_masks(keys)
    indices = [index for index, _mask in keys]

    def alpha_at(frame_index: int, shape: Tuple[int, int]) -> np.ndarray:
        position = bisect.bisect_right(indices, frame_index) - 1
        position = min(max(position, 0), len(keys) - 1)
        left_index, left = keys[position]
        if position + 1 < len(keys) and frame_index > left_index:
            right_index, right = keys[position + 1]
            span = max(right_index - left_index, 1)
            weight = min(max((frame_index - left_index) / span, 0.0), 1.0)
            if right.shape != left.shape:
                right = cv2.resize(right, (left.shape[1], left.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
            blended = left.astype(np.float32) * (1.0 - weight) + right.astype(np.float32) * weight
        else:
            blended = left.astype(np.float32)

        mask = cv2.resize(blended, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        # Feather in proportion to the frame: the same halo has to look right
        # at 720p and at 4K.
        sigma = max(1.0, min(shape) * 0.004)
        return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma), 0, 255).astype(np.uint8)

    state = {"index": 0}

    def process(frame: np.ndarray, _t: float) -> np.ndarray:
        mask = alpha_at(state["index"], frame.shape[:2])
        state["index"] += 1
        rgba = np.dstack([frame, mask])
        return composite_over(rgba, _background_frame(background, frame.shape, fill))

    return _process_frames(source, destination, process, ffmpeg=ffmpeg)


def _key_and_fill(frame, key_rgb, tolerance, background, fill):
    rgba = chroma_key(frame, key_rgb=key_rgb, tolerance=tolerance)
    return composite_over(rgba, _background_frame(background, frame.shape, fill))


_BACKGROUND_CACHE: Dict[str, np.ndarray] = {}


def _background_frame(background: Optional[Path], shape, fill) -> np.ndarray:
    height, width = shape[:2]
    if background is None:
        return np.full((height, width, 3), fill, dtype=np.uint8)
    key = f"{background}:{width}x{height}"
    cached = _BACKGROUND_CACHE.get(key)
    if cached is None:
        image = cv2.imread(str(background), cv2.IMREAD_COLOR)
        if image is None:
            cached = np.full((height, width, 3), fill, dtype=np.uint8)
        else:
            cached = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        if len(_BACKGROUND_CACHE) > 4:
            _BACKGROUND_CACHE.clear()
        _BACKGROUND_CACHE[key] = cached
    return cached


def _process_frames(
    source: Path,
    destination: Path,
    process: Callable[[np.ndarray, float], np.ndarray],
    ffmpeg: Optional[str] = None,
) -> Path:
    """Decode, transform every frame, re-encode - audio carried across."""
    from video_renderer import FfmpegFrameWriter

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg or ffmpeg_binary()

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width -= width % 2
        height -= height % 2

        silent = destination.with_name(destination.stem + "_silent.mp4")
        index = 0
        with FfmpegFrameWriter(silent, width, height, fps) as writer:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame = frame[:height, :width]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out = process(rgb, index / fps)
                writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                index += 1
    finally:
        capture.release()

    merged = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(silent), "-i", str(source),
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
         str(destination)],
        capture_output=True, text=True,
    )
    if merged.returncode != 0:
        silent.replace(destination)
    else:
        silent.unlink(missing_ok=True)
    return destination
