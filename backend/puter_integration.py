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
from dataclasses import dataclass, replace
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


# The catalogue moved to voices.py so a profile can carry its shaping chain
# (pitch, timbre, reverb) alongside the provider voice id.
from voices import (
    VoiceProfile,
    measure_f0,
    resolve_voice as resolve_voice_profile,
    semitones_to,
    shape_voice,
)

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
        nim_api_key: Optional[str] = None,
        pollinations_key: Optional[str] = None,
        image_model: Optional[str] = None,
        horde_key: Optional[str] = None,
    ) -> None:
        self.api_key = (api_key or settings.puter_api_key or "").strip()
        # Cloned voices are synthesised by NVIDIA, not by Puter, so the client
        # that serves a TTS request has to be able to reach both.
        self.nim_api_key = (nim_api_key or "").strip()
        # Images can come from either. Puter's image driver spent most of this
        # project out of credit, and the chosen model is a real creative
        # decision - an anime key frame and a photoreal plate are not the same
        # job - so the second route is a first-class one rather than a rescue.
        self.pollinations_key = (pollinations_key or settings.pollinations_api_key or "").strip()
        self.image_model = (image_model or settings.image_model or "").strip()
        self.horde_key = (horde_key or settings.horde_api_key or "").strip()
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
            "User-Agent": f"moja-ai/{settings.app_version}",
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
        headers = {"User-Agent": f"moja-ai/{settings.app_version}"}
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
    def resolve_voice(accent: Optional[str]) -> VoiceProfile:
        return resolve_voice_profile(accent)

    def text_to_speech(
        self,
        text: str,
        output_path: Path,
        accent: str = "indian_accent",
        speed: float = 1.0,
    ) -> Path:
        """Synthesise ``text`` and write a shaped ``.mp3``.

        The provider voice comes from the profile; the profile's character
        (deep / dark / mysterious, whisper, hype) is then applied locally with
        ffmpeg, because Polly has no such voice to ask for.
        """
        script = (text or "").strip()
        if not script:
            raise PuterError("Cannot synthesise speech from an empty script.")

        profile = self.resolve_voice(accent)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if profile.is_cloned:
            return self._clone_to_speech(script, output_path, profile, speed)

        chunks = _chunk_text(script, TTS_CHUNK_CHARS)
        logger.info(
            "Puter TTS: %s chunk(s), profile=%s voice=%s (%s)",
            len(chunks), profile.key, profile.voice_id, profile.language,
        )

        parts: List[Path] = []
        with tempfile.TemporaryDirectory(prefix="puter-tts-") as tmp:
            tmp_dir = Path(tmp)
            for index, chunk in enumerate(chunks):
                args = {
                    "text": chunk,
                    "voice": profile.voice_id,
                    "language": profile.language,
                    "engine": profile.engine,
                    "accent": profile.language,
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

            raw = tmp_dir / "raw.mp3"
            if len(parts) == 1:
                shutil.copyfile(parts[0], raw)
            else:
                _concat_audio(parts, raw)
            shape_voice(raw, output_path, profile, ffmpeg=ffmpeg_binary(),
                        audio_bitrate=settings.audio_bitrate)

        logger.info("Puter TTS written to %s (%s bytes)", output_path, output_path.stat().st_size)
        return output_path

    def _clone_to_speech(
        self, script: str, output_path: Path, profile: VoiceProfile, speed: float
    ) -> Path:
        """Speak a line in a cloned voice, then shape it like any other.

        Magpie supplies the timbre; the profile's knobs still decide the
        character on top of it. Chunking is Puter's problem, not this one -
        the model takes a whole line.
        """
        from magpie_tts import MagpieCloner, MagpieUnavailable  # noqa: PLC0415

        reference = Path(profile.clone_reference)
        if not reference.exists():
            raise PuterError(
                f"The recording behind voice '{profile.key}' is gone "
                f"({reference}); re-save the pack from the original audio."
            )
        try:
            cloner = MagpieCloner(self.nim_api_key)
        except MagpieUnavailable as exc:
            raise PuterError(f"Cloned voice '{profile.key}' unavailable: {exc}") from exc

        logger.info(
            "Magpie TTS: profile=%s language=%s quality=%s",
            profile.key, profile.clone_language or profile.language,
            profile.clone_quality,
        )
        with tempfile.TemporaryDirectory(prefix="magpie-tts-") as tmp:
            raw = Path(tmp) / "raw.wav"
            try:
                cloner.synthesize(
                    script, raw, reference,
                    language=profile.clone_language or profile.language or "en-US",
                    quality=profile.clone_quality,
                )
            except MagpieUnavailable as exc:
                raise PuterError(str(exc)) from exc

            pitch = profile.pitch_semitones
            if profile.target_f0 > 0:
                # The model does not return a consistent pitch, so the shift is
                # computed against what it actually produced this time rather
                # than assumed. Without this two characters drift into each
                # other from one take to the next.
                measured = measure_f0(raw)
                if measured > 0:
                    pitch = semitones_to(measured, profile.target_f0)
                    logger.info(
                        "Magpie pitch: %s came back at %.0fHz, moving %+.2f "
                        "semitones onto its %.0fHz mark",
                        profile.key, measured, pitch, profile.target_f0,
                    )
                else:
                    self_note = (
                        f"Could not measure the pitch of '{profile.key}'; left it "
                        f"where the model put it."
                    )
                    logger.warning(self_note)

            shaped = replace(
                profile,
                pitch_semitones=pitch,
                tempo=profile.tempo * max(float(speed), 0.1),
            )
            shape_voice(raw, output_path, shaped, ffmpeg=ffmpeg_binary(),
                        audio_bitrate=settings.audio_bitrate)
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

        # If the chosen model belongs to another service, go straight there
        # rather than spending a failed Puter call to arrive at it.
        if self._chosen_provider() != "puter":
            return self._draw_from_catalogue(prompt, output_path, width, height)

        try:
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
        except Exception as exc:
            logger.info("Puter image failed (%s); falling back", str(exc)[:90])
            try:
                return self._draw_from_catalogue(prompt, output_path, width, height)
            except Exception:
                raise exc from None

    def _chosen_provider(self) -> str:
        """Which service the selected model belongs to."""
        import image_models  # noqa: PLC0415

        model = (self.image_model or "").strip()
        if not model:
            return "puter"
        return image_models.resolve(model).provider

    def _draw_from_catalogue(self, prompt: str, output_path: Path,
                             width: int, height: int) -> Path:
        """Draw with the picked model, then with whatever is still standing.

        The order is deliberate: the picked model first because it is what was
        asked for, then Pollinations if there is a key, then the horde, which
        is last because it is slow but goes last for the better reason that it
        cannot run out - no key, no balance, no quota. So an image job only
        fails here if every route is down, not merely if the paid ones are.
        """
        import image_models  # noqa: PLC0415

        attempts: list[tuple[str, str]] = []
        picked = (self.image_model or "").strip()
        if picked:
            attempts.append((picked, "the model you picked"))
        if self.pollinations_key:
            attempts.append((image_models.DEFAULT_MODEL, "Pollinations"))
        attempts.append(("horde:AlbedoBase XL (SDXL)", "the AI Horde"))

        failures: list[str] = []
        seen: set[str] = set()
        for model, description in attempts:
            if model in seen:
                continue
            seen.add(model)
            try:
                return image_models.generate(
                    prompt, output_path, self.pollinations_key, model=model,
                    width=width, height=height, horde_key=self.horde_key,
                )
            except Exception as exc:
                logger.info("%s could not draw it: %s", description, str(exc)[:120])
                failures.append(f"{description}: {exc}")

        raise PuterError("Image generation failed. " + " | ".join(failures))

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
        """Replace the masked region of a frame with generated content.

        Puter's image driver exposes generation only - it has no edit or
        inpaint method - so the edit happens here rather than there. The
        replacement is generated at the frame's own aspect ratio and then
        composited into the mask with a feathered edge.

        The tempting shortcut is to post the frame to ``generate`` and hope:
        that returns HTTP 200 and a perfectly good picture of something else
        entirely, because the image argument is ignored. A silent wrong answer
        is worse than an error, so the compositing is explicit.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as probe:
            frame = probe.convert("RGB")
            width, height = frame.size
            frame_array = np.array(frame)

        with tempfile.TemporaryDirectory(prefix="puter-inpaint-") as tmp:
            replacement_path = Path(tmp) / "replacement.png"
            self.text_to_image(
                f"{prompt}, photographic, matching lighting and perspective",
                replacement_path, width=width, height=height,
            )
            with Image.open(replacement_path) as generated:
                replacement = generated.convert("RGB")
                if replacement.size != (width, height):
                    replacement = replacement.resize((width, height), Image.LANCZOS)
                replacement_array = np.array(replacement)

        alpha = _feathered_mask(mask_path, (width, height), strength)
        blended = (
            frame_array.astype(np.float32) * (1.0 - alpha)
            + replacement_array.astype(np.float32) * alpha
        )
        Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(output_path, "PNG")
        return output_path


def _feathered_mask(
    mask_path: Optional[Path], size: Tuple[int, int], strength: float
) -> np.ndarray:
    """A 0..1 blend map with a soft edge, shaped ``(h, w, 1)``.

    A hard mask edge is the thing that makes a composite look pasted on, so the
    boundary is blurred in proportion to the frame rather than by a fixed
    number of pixels - the same seam has to look right at 720p and at 4K.
    """
    width, height = size
    strength = float(min(max(strength, 0.0), 1.0))

    if mask_path is not None and Path(mask_path).exists():
        with Image.open(mask_path) as handle:
            mask = handle.convert("L").resize((width, height), Image.LANCZOS)
        alpha = np.array(mask).astype(np.float32) / 255.0
    else:
        # No mask means "repaint the whole frame", which is what an
        # image-to-image request without one has always meant.
        alpha = np.ones((height, width), dtype=np.float32)

    feather = max(3, int(min(width, height) * 0.02)) | 1
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
    return (alpha * strength)[:, :, None]


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


# ---------------------------------------------------------------------------
# Procedural sticker fallback (no Puter key required)
# ---------------------------------------------------------------------------

_STICKER_FILLERS = {
    "a", "an", "and", "the", "with", "of", "in", "on", "at", "3d", "glowing",
    "glossy", "soft", "light", "lights", "shiny", "sparkling", "serene", "warm",
    "cartoon", "sticker", "icon", "style", "effect", "neon", "bright", "cute",
    "beautiful", "aesthetic", "premium", "modern", "clean", "small", "big",
}

_STICKER_PALETTE = {
    "gold": ((255, 196, 60), (255, 138, 30)),
    "green": ((124, 226, 130), (34, 160, 96)),
    "blue": ((122, 190, 255), (44, 110, 230)),
    "moon": ((168, 168, 255), (86, 76, 200)),
    "fire": ((255, 150, 70), (232, 58, 40)),
    "pink": ((255, 150, 200), (214, 51, 132)),
    "violet": ((186, 150, 255), (109, 40, 217)),
}

_STICKER_COLOUR_HINTS = (
    (("gold", "golden", "sun", "amber", "honey"), "gold"),
    (("leaf", "green", "forest", "plant", "tree", "nature"), "green"),
    (("water", "ocean", "sea", "rain", "sky", "blue", "wave"), "blue"),
    (("moon", "night", "star", "dream", "calm"), "moon"),
    (("fire", "flame", "hot", "explosion", "energy"), "fire"),
    (("flower", "lotus", "rose", "butterfly", "bloom", "love"), "pink"),
)


def sticker_label(prompt: str, max_words: int = 2) -> str:
    """Condense a sticker prompt into a short badge label."""
    words = [
        word.strip(".,!?:;\"'()").upper()
        for word in (prompt or "").split()
        if word.strip(".,!?:;\"'()").lower() not in _STICKER_FILLERS
        and word.strip(".,!?:;\"'()").isalpha()
    ]
    if not words:
        return "HIGHLIGHT"
    return " ".join(words[:max_words])


def _sticker_colours(prompt: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    lowered = (prompt or "").lower()
    for keywords, name in _STICKER_COLOUR_HINTS:
        if any(keyword in lowered for keyword in keywords):
            return _STICKER_PALETTE[name]
    return _STICKER_PALETTE["violet"]


def procedural_sticker(prompt: str, output_path: Path, width: int = 760) -> Path:
    """Render a glowing motion-graphic badge locally.

    Used when no Puter.js key is available (or a txt2img call fails) so the
    timeline still carries the motion graphics Kimi asked for, instead of
    silently dropping them.  It is a stand-in, never a replacement for the real
    AI sticker.
    """
    from PIL import ImageDraw, ImageFilter, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label = sticker_label(prompt)
    top_colour, bottom_colour = _sticker_colours(prompt)

    pad_x, pad_y = int(width * 0.10), int(width * 0.075)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    # Shrink the label until the pill can actually contain it.
    font_size = int(width * 0.115)
    while True:
        font = _sticker_font(font_size)
        text_box = probe.textbbox((0, 0), label, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
        if text_w + pad_x * 2 <= width or font_size <= int(width * 0.055):
            break
        font_size -= 2
    pill_w = min(max(text_w + pad_x * 2, int(width * 0.55)), width)
    pill_h = text_h + pad_y * 2
    margin = int(width * 0.09)  # room for the glow
    canvas = Image.new("RGBA", (pill_w + margin * 2, pill_h + margin * 2), (0, 0, 0, 0))

    # Vertical gradient body.
    gradient = Image.new("RGBA", (pill_w, pill_h))
    for y in range(pill_h):
        blend = y / max(pill_h - 1, 1)
        gradient.paste(
            tuple(
                int(top_colour[channel] * (1 - blend) + bottom_colour[channel] * blend)
                for channel in range(3)
            ) + (255,),
            (0, y, pill_w, y + 1),
        )
    rounded = Image.new("L", (pill_w, pill_h), 0)
    ImageDraw.Draw(rounded).rounded_rectangle(
        [0, 0, pill_w - 1, pill_h - 1], radius=pill_h // 2, fill=255
    )
    gradient.putalpha(rounded)

    # Outer glow.
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow.paste(gradient, (margin, margin), gradient)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=margin * 0.55))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(glow)
    canvas.paste(gradient, (margin, margin), gradient)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [margin, margin, margin + pill_w - 1, margin + pill_h - 1],
        radius=pill_h // 2, outline=(255, 255, 255, 210), width=max(3, pill_h // 26),
    )
    # Glass highlight across the top half. ImageDraw replaces pixels rather
    # than blending them, so the translucent sheen goes on its own layer.
    sheen = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sheen).rounded_rectangle(
        [margin + pill_h // 6, margin + pill_h // 8,
         margin + pill_w - pill_h // 6, margin + pill_h // 2],
        radius=pill_h // 3, fill=(255, 255, 255, 58),
    )
    sheen.putalpha(Image.composite(sheen.split()[-1], Image.new("L", canvas.size, 0), _pill_mask(canvas.size, margin, pill_w, pill_h)))
    canvas.alpha_composite(sheen)
    text_x = margin + (pill_w - text_w) // 2 - text_box[0]
    text_y = margin + (pill_h - text_h) // 2 - text_box[1]
    draw.text((text_x, text_y + 2), label, font=font, fill=(0, 0, 0, 110))
    draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255, 255))

    canvas.save(output_path, "PNG")
    return output_path


def _pill_mask(size: Tuple[int, int], margin: int, pill_w: int, pill_h: int):
    """Mask limiting the sheen to the pill body."""
    from PIL import ImageDraw as _ImageDraw

    mask = Image.new("L", size, 0)
    _ImageDraw.Draw(mask).rounded_rectangle(
        [margin, margin, margin + pill_w - 1, margin + pill_h - 1],
        radius=pill_h // 2, fill=255,
    )
    return mask


def _sticker_font(size: int):
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def cache_key(*parts: Any) -> str:
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8"))
    return digest.hexdigest()[:16]
