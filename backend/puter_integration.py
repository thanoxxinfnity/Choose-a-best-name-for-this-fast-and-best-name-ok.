"""Puter.js integration over plain REST.

Puter exposes every AI capability behind one endpoint::

    POST https://api.puter.com/drivers/call
    Authorization: Bearer <PUTER_API_KEY>
    {"interface": "...", "driver": "...", "method": "...", "args": {...}}

The response is either

* a raw binary body (``audio/mpeg`` for TTS, ``image/png`` for txt2img), or
* a JSON envelope ``{"success": true, "result": ...}`` where ``result`` is a
  URL, a ``data:`` URI, a base64 blob or a nested object holding one of those.

`PuterClient` normalises all of those shapes into bytes on disk, so the rest of
the pipeline only ever deals with files.

Capabilities implemented here (see the blueprint):

* :meth:`PuterClient.text_to_speech`  - Indian accent voiceover -> ``.mp3``
* :meth:`PuterClient.generate_sticker` - txt2img + ``rembg`` -> transparent PNG
* :meth:`PuterClient.inpaint_image`   - image + mask + prompt -> edited frame
* :func:`extract_frames` / :func:`build_mask` / :func:`frames_to_video` -
  the OpenCV side of the frame inpainting workflow.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice catalogue - Indian accent first, everything else is a graceful fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Voice:
    voice_id: str
    language: str
    engine: str
    label: str


VOICE_CATALOGUE: Dict[str, Voice] = {
    # Indian English (the accent requested by the blueprint).
    "indian_accent": Voice("Kajal", "en-IN", "neural", "Indian English (female)"),
    "indian_accent_male": Voice("Arjun", "en-IN", "neural", "Indian English (male)"),
    "indian_accent_hindi": Voice("Aditi", "en-IN", "standard", "Hindi / Hinglish"),
    "hinglish": Voice("Aditi", "en-IN", "standard", "Hindi / Hinglish"),
    # Non Indian fallbacks.
    "us_accent": Voice("Joanna", "en-US", "neural", "US English (female)"),
    "uk_accent": Voice("Amy", "en-GB", "neural", "British English (female)"),
}

DEFAULT_VOICE = VOICE_CATALOGUE["indian_accent"]

# Polly rejects requests above ~3000 characters, so long scripts are chunked.
TTS_CHUNK_CHARS = 2800

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?;base64,(?P<payload>.+)$", re.S)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


class PuterError(RuntimeError):
    """Raised when the Puter API cannot fulfil a request."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PuterClient:
    """Thin, retrying REST client for the Puter.js driver API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = (api_key or settings.puter_api_key or "").strip()
        self.base_url = (base_url or settings.puter_base_url).rstrip("/")
        self.timeout = timeout or settings.puter_timeout
        self.max_retries = max(1, max_retries)
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ util
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self, accept: str = "*/*") -> Dict[str, str]:
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            "User-Agent": f"ai-video-editor/{settings.app_version}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _require_key(self) -> None:
        if not self.is_configured:
            raise PuterError(
                "No Puter.js API key configured. Save one in the Android Settings "
                "screen or export PUTER_API_KEY on the server."
            )

    # ------------------------------------------------------------ driver call
    def call_driver(
        self,
        interface: str,
        driver: str,
        method: str,
        args: Dict[str, Any],
        accept: str = "*/*",
    ) -> Tuple[bytes, str, Optional[Dict[str, Any]]]:
        """Invoke a Puter driver.

        Returns ``(body, content_type, parsed_json_or_None)``.
        """
        self._require_key()
        url = f"{self.base_url}/drivers/call"
        payload = {
            "interface": interface,
            "driver": driver,
            "method": method,
            "args": args,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    headers=self._headers(accept),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:  # network level failure
                last_error = exc
                logger.warning("Puter %s.%s network error (try %s/%s): %s",
                               interface, method, attempt, self.max_retries, exc)
                self._sleep_backoff(attempt)
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = PuterError(
                    f"Puter {interface}.{method} HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                logger.warning("Puter %s.%s HTTP %s (try %s/%s)", interface, method,
                               response.status_code, attempt, self.max_retries)
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise PuterError(
                    f"Puter {interface}.{method} failed with HTTP "
                    f"{response.status_code}: {response.text[:500]}"
                )

            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
            parsed: Optional[Dict[str, Any]] = None
            if "json" in content_type or response.content[:1] in (b"{", b"["):
                try:
                    parsed = response.json()
                except ValueError:
                    parsed = None

            if isinstance(parsed, dict) and parsed.get("success") is False:
                error = parsed.get("error") or parsed.get("message") or parsed
                raise PuterError(f"Puter {interface}.{method} returned an error: {error}")

            return response.content, content_type, parsed

        raise PuterError(
            f"Puter {interface}.{method} unreachable after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 16))

    # --------------------------------------------------------- media decoding
    def _download(self, url: str) -> bytes:
        headers = {"User-Agent": f"ai-video-editor/{settings.app_version}"}
        if self.api_key and url.startswith(self.base_url):
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def _resolve_media(
        self,
        body: bytes,
        content_type: str,
        parsed: Optional[Any],
        expected: str,
    ) -> bytes:
        """Turn any Puter response shape into raw media bytes.

        ``expected`` is ``"audio"`` or ``"image"`` and is only used for error
        messages and for validating ``data:`` URIs.
        """
        if parsed is None and content_type.startswith(expected):
            return body
        if parsed is None and body and body[:1] not in (b"{", b"["):
            # Unknown content type but clearly not JSON - trust the bytes.
            return body

        candidate = self._find_media_value(parsed)
        if candidate is None:
            snippet = json.dumps(parsed)[:400] if parsed is not None else body[:200]
            raise PuterError(f"No {expected} payload found in Puter response: {snippet}")

        if isinstance(candidate, (bytes, bytearray)):
            return bytes(candidate)

        text = str(candidate).strip()
        data_uri = _DATA_URI_RE.match(text)
        if data_uri:
            return base64.b64decode(data_uri.group("payload"))
        if text.startswith("http://") or text.startswith("https://"):
            return self._download(text)
        if text.startswith("/"):
            return self._download(f"{self.base_url}{text}")
        if len(text) > 64 and _BASE64_RE.match(text):
            try:
                return base64.b64decode(text, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise PuterError(f"Malformed base64 {expected} payload: {exc}") from exc
        raise PuterError(f"Unrecognised {expected} payload from Puter: {text[:200]}")

    @staticmethod
    def _find_media_value(node: Any, depth: int = 0) -> Optional[Any]:
        """Depth first search for the first URL / data URI / base64 blob."""
        if depth > 6 or node is None:
            return None
        if isinstance(node, (bytes, bytearray)):
            return node
        if isinstance(node, str):
            text = node.strip()
            if not text:
                return None
            if _DATA_URI_RE.match(text) or text.startswith(("http://", "https://", "/")):
                return text
            if len(text) > 256 and _BASE64_RE.match(text):
                return text
            return None
        if isinstance(node, dict):
            preferred = (
                "url", "audio_url", "image_url", "signed_url", "download_url",
                "b64_json", "base64", "data", "content", "output", "result",
                "audio", "image", "file", "path", "items",
            )
            for key in preferred:
                if key in node:
                    found = PuterClient._find_media_value(node[key], depth + 1)
                    if found is not None:
                        return found
            for key, value in node.items():
                if key in preferred:
                    continue
                found = PuterClient._find_media_value(value, depth + 1)
                if found is not None:
                    return found
            return None
        if isinstance(node, (list, tuple)):
            for item in node:
                found = PuterClient._find_media_value(item, depth + 1)
                if found is not None:
                    return found
        return None

    # ------------------------------------------------------------------- TTS
    @staticmethod
    def resolve_voice(accent: Optional[str]) -> Voice:
        key = (accent or "").strip().lower().replace("-", "_").replace(" ", "_")
        if key in VOICE_CATALOGUE:
            return VOICE_CATALOGUE[key]
        if "indian" in key or key in ("hi", "hi_in", "en_in", "india"):
            return DEFAULT_VOICE
        return DEFAULT_VOICE

    def text_to_speech(
        self,
        text: str,
        output_path: Path,
        accent: str = "indian_accent",
        speed: float = 1.0,
    ) -> Path:
        """Synthesise ``text`` with an Indian accent and write an ``.mp3``."""
        script = (text or "").strip()
        if not script:
            raise PuterError("Cannot synthesise speech from an empty script.")

        voice = self.resolve_voice(accent)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = _chunk_text(script, TTS_CHUNK_CHARS)
        logger.info("Puter TTS: %s chunk(s), voice=%s (%s)", len(chunks), voice.voice_id, voice.language)

        parts: List[Path] = []
        with tempfile.TemporaryDirectory(prefix="puter-tts-") as tmp:
            tmp_dir = Path(tmp)
            for index, chunk in enumerate(chunks):
                args = {
                    "text": chunk,
                    "voice": voice.voice_id,
                    "language": voice.language,
                    "engine": voice.engine,
                    "accent": "indian_accent",
                    "format": "mp3",
                    "speed": round(float(speed), 2),
                }
                body, content_type, parsed = self.call_driver(
                    settings.puter_tts_interface,
                    settings.puter_tts_driver,
                    settings.puter_tts_method,
                    args,
                    accept="audio/mpeg",
                )
                audio = self._resolve_media(body, content_type, parsed, "audio")
                if not audio:
                    raise PuterError("Puter TTS returned an empty audio payload.")
                part = tmp_dir / f"part_{index:03d}.mp3"
                part.write_bytes(audio)
                parts.append(part)

            if len(parts) == 1:
                shutil.copyfile(parts[0], output_path)
            else:
                _concat_audio(parts, output_path)

        logger.info("Puter TTS written to %s (%s bytes)", output_path, output_path.stat().st_size)
        return output_path

    # -------------------------------------------------------------- txt2img
    def text_to_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
    ) -> Path:
        prompt = (prompt or "").strip()
        if not prompt:
            raise PuterError("Cannot generate an image from an empty prompt.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "n": 1,
            "response_format": "b64_json",
        }
        if style:
            args["style"] = style

        body, content_type, parsed = self.call_driver(
            settings.puter_txt2img_interface,
            settings.puter_txt2img_driver,
            settings.puter_txt2img_method,
            args,
            accept="image/png",
        )
        image = self._resolve_media(body, content_type, parsed, "image")
        output_path.write_bytes(image)
        return output_path

    def generate_sticker(
        self,
        prompt: str,
        output_path: Path,
        remove_background: bool = True,
        size: int = 1024,
    ) -> Path:
        """txt2img -> ``rembg`` -> transparent PNG sticker."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sticker_prompt = (
            f"{prompt.strip()}, sticker art, bold clean outline, vibrant colours, "
            "centered subject, isolated on a plain flat white background, no text, "
            "no watermark, high contrast, product photo lighting"
        )
        with tempfile.TemporaryDirectory(prefix="puter-sticker-") as tmp:
            raw = Path(tmp) / "raw.png"
            self.text_to_image(sticker_prompt, raw, width=size, height=size)
            if remove_background:
                remove_background_to_png(raw, output_path)
            else:
                Image.open(raw).convert("RGBA").save(output_path, "PNG")
        return output_path

    # ------------------------------------------------------------ inpainting
    def inpaint_image(
        self,
        image_path: Path,
        mask_path: Optional[Path],
        prompt: str,
        output_path: Path,
        strength: float = 0.85,
    ) -> Path:
        """Image-to-image / inpainting for a single extracted video frame."""
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as probe:
            width, height = probe.size

        args: Dict[str, Any] = {
            "prompt": prompt,
            "image": _to_data_uri(image_path),
            "width": width,
            "height": height,
            "strength": round(float(strength), 3),
            "n": 1,
            "response_format": "b64_json",
        }
        if mask_path is not None and Path(mask_path).exists():
            args["mask"] = _to_data_uri(Path(mask_path))

        body, content_type, parsed = self.call_driver(
            settings.puter_inpaint_interface,
            settings.puter_inpaint_driver,
            settings.puter_inpaint_method,
            args,
            accept="image/png",
        )
        image = self._resolve_media(body, content_type, parsed, "image")
        output_path.write_bytes(image)

        # Puter may return a square canvas: force the original frame geometry
        # back so the frames still line up when they are muxed into the video.
        with Image.open(output_path) as edited:
            if edited.size != (width, height):
                edited.convert("RGB").resize((width, height), Image.LANCZOS).save(output_path, "PNG")
        return output_path


# ---------------------------------------------------------------------------
# rembg - transparent sticker PNGs
# ---------------------------------------------------------------------------

_REMBG_LOCK = threading.Lock()
_REMBG_SESSION: Any = None


def _rembg_session() -> Any:
    global _REMBG_SESSION
    with _REMBG_LOCK:
        if _REMBG_SESSION is None:
            from rembg import new_session  # imported lazily: downloads a model

            _REMBG_SESSION = new_session(settings.rembg_model)
        return _REMBG_SESSION


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency probe
        return False
    return True


def remove_background_to_png(source: Path, destination: Path) -> Path:
    """Cut the subject out of ``source`` and save a transparent PNG."""
    from rembg import remove

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        rgba = image.convert("RGBA")

    cut = remove(
        rgba,
        session=_rembg_session(),
        post_process_mask=True,
        alpha_matting=False,
    )
    if not isinstance(cut, Image.Image):  # rembg can return bytes
        cut = Image.open(io.BytesIO(cut)).convert("RGBA")

    cut = _trim_transparent(cut.convert("RGBA"))
    cut.save(destination, "PNG")
    return destination


def _trim_transparent(image: Image.Image, padding: int = 8) -> Image.Image:
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return image
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image.width)
    bottom = min(bbox[3] + padding, image.height)
    return image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# OpenCV helpers for the frame inpainting workflow
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    path: Path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 2.0,
    max_frames: int = 24,
    start: float = 0.0,
    end: Optional[float] = None,
) -> List[ExtractedFrame]:
    """Pull evenly spaced frames out of a clip with OpenCV."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PuterError(f"OpenCV could not open {video_path}")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_duration = frame_count / source_fps if source_fps > 0 and frame_count else 0.0
        stop = end if end is not None else (video_duration or start + 5.0)
        stop = max(stop, start + (1.0 / max(sample_fps, 0.1)))

        step = 1.0 / max(sample_fps, 0.05)
        timestamps = []
        current = start
        while current < stop and len(timestamps) < max_frames:
            timestamps.append(current)
            current += step
        if not timestamps:
            timestamps = [start]

        frames: List[ExtractedFrame] = []
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            path = output_dir / f"frame_{index:05d}.png"
            cv2.imwrite(str(path), frame)
            frames.append(ExtractedFrame(index=index, timestamp=timestamp, path=path))
        return frames
    finally:
        capture.release()


def build_mask(
    frame_path: Path,
    mask_path: Path,
    target_object: str = "",
    region: Optional[str] = None,
) -> Path:
    """Create a white-on-black inpainting mask for ``frame_path``.

    White pixels are the area Puter should repaint.  Without a segmentation
    model we use two cheap but effective heuristics:

    * ``sky`` / ``background`` -> HSV threshold for bright blue/grey pixels in
      the upper half of the frame, dilated and feathered.
    * everything else -> a rectangular region derived from ``region`` (or a
      centred box when no hint is available).
    """
    frame_path = Path(frame_path)
    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(frame_path))
    if image is None:
        raise PuterError(f"Could not read frame {frame_path}")
    height, width = image.shape[:2]

    hint = (region or target_object or "").strip().lower()
    mask = np.zeros((height, width), dtype=np.uint8)

    if any(word in hint for word in ("sky", "cloud", "sunset", "heaven")):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 20, 90]), np.array([135, 255, 255]))
        bright = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 60, 255]))
        sky = cv2.bitwise_or(blue, bright)
        sky[int(height * 0.66):, :] = 0  # sky never lives in the bottom third
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        if cv2.countNonZero(sky) > (height * width * 0.02):
            mask = sky
        else:  # nothing sky-like detected: fall back to the top third
            mask[: int(height * 0.35), :] = 255
    elif any(word in hint for word in ("ground", "floor", "road", "bottom", "water", "sea")):
        mask[int(height * 0.6):, :] = 255
    elif "left" in hint:
        mask[:, : int(width * 0.5)] = 255
    elif "right" in hint:
        mask[:, int(width * 0.5):] = 255
    elif "top" in hint:
        mask[: int(height * 0.5), :] = 255
    elif any(word in hint for word in ("full", "whole", "everything", "background")):
        mask[:, :] = 255
    else:
        y0, y1 = int(height * 0.18), int(height * 0.82)
        x0, x1 = int(width * 0.15), int(width * 0.85)
        mask[y0:y1, x0:x1] = 255

    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 96, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(mask_path), mask)
    return mask_path


def frames_to_video(
    frames: Sequence[Path],
    output_path: Path,
    fps: float,
    size: Optional[Tuple[int, int]] = None,
) -> Path:
    """Encode an ordered list of PNG frames into a lossless-ish MP4."""
    if not frames:
        raise PuterError("No frames to encode.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if size is None:
        with Image.open(frames[0]) as first:
            size = first.size
    width, height = size
    width -= width % 2
    height -= height % 2

    with tempfile.TemporaryDirectory(prefix="frames-") as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            shutil.copyfile(frame, tmp_dir / f"f_{index:06d}.png")
        command = [
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps:.6f}",
            "-i", str(tmp_dir / "f_%06d.png"),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run(command)
    return output_path


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------


def ffmpeg_binary() -> str:
    """Resolve an ffmpeg binary (system first, then the imageio bundled one)."""
    binary = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise PuterError("ffmpeg is not installed and imageio-ffmpeg is unavailable.") from exc


def ffmpeg_available() -> bool:
    try:
        ffmpeg_binary()
        return True
    except Exception:
        return False


def _run(command: List[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise PuterError(
            f"Command failed ({' '.join(command[:3])} ...): "
            f"{result.stderr.strip()[:600]}"
        )


def _concat_audio(parts: Sequence[Path], output_path: Path) -> None:
    """Concatenate mp3 chunks without re-encoding when possible."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for part in parts:
            handle.write(f"file '{part.resolve()}'\n")
        list_path = handle.name
    try:
        _run([
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", str(output_path),
        ])
    except PuterError:
        _run([
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:a", "libmp3lame", "-b:a", settings.audio_bitrate, str(output_path),
        ])
    finally:
        os.unlink(list_path)


def _to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _chunk_text(text: str, limit: int) -> List[str]:
    """Split a script on sentence boundaries without exceeding ``limit``."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > limit:  # a single gigantic "sentence"
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:limit])
            sentence = sentence[limit:]
        if len(current) + len(sentence) + 1 > limit:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def cache_key(*parts: Any) -> str:
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8"))
    return digest.hexdigest()[:16]
