"""Puter.js video generation: text-to-video, image-to-video, keyframe animation.

Models (per the product spec):

* ``sora-2`` - text to video, and image to video with a seed frame

Both go through the same driver endpoint the rest of the Puter integration
uses::

    POST {PUTER_BASE_URL}/drivers/call
    {"interface": "puter-video-generation", "driver": "openai-video-generation",
     "method": "generate", "args": {"model": ..., "prompt": ..., ...}}

Video generation is slow, so the driver may answer either with the finished
media or with a job handle; :meth:`PuterVideoClient.generate` handles both and
polls until the clip is ready.

:func:`animate_keyframe` is the "AI keyframe-to-animation insertion" feature:
pull a still out of the source video at a chosen timestamp, send it to i2v with
the surrounding story context, and hand back a rendered clip the timeline can
splice in.

NOTE: Puter has no anonymous tier - ``POST /drivers/call`` answers
``401 token_missing`` without a key - so every function here needs a Puter API
key. Callers get a clear :class:`~puter_integration.PuterError` when one is
missing rather than a silent failure.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import requests

from config import settings
from puter_integration import PuterClient, PuterError, _to_data_uri, ffmpeg_binary

logger = logging.getLogger(__name__)

# Read from settings so there is one place the model is named. The wan2.2
# ids this used to carry were never served here - the account's upstream
# rejects them with a 401, which surfaces as an unhelpful HTTP 500.
TEXT_TO_VIDEO_MODEL = settings.puter_t2v_model
IMAGE_TO_VIDEO_MODEL = settings.puter_i2v_model

# Keys a driver may use to hand back an async job handle.
_JOB_KEYS = ("job_id", "id", "task_id", "request_id", "generation_id")
_STATUS_DONE = {"succeeded", "success", "completed", "complete", "done", "finished", "ready"}
_STATUS_FAILED = {"failed", "error", "cancelled", "canceled", "rejected"}


@dataclass
class VideoGenerationResult:
    path: Path
    model: str
    seconds: float
    prompt: str
    elapsed: float


ProgressHook = Callable[[str, float], None]


class PuterVideoClient(PuterClient):
    """Text-to-video and image-to-video on top of the Puter driver API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Video jobs can take minutes; do not inherit the image timeout.
        self.timeout = max(self.timeout, settings.puter_video_timeout)
        self._home_cache: Optional[str] = None

    # ------------------------------------------------------------------ API
    def text_to_video(
        self,
        prompt: str,
        output_path: Path,
        seconds: float = 5.0,
        resolution: str = "720x1280",
        model: str = TEXT_TO_VIDEO_MODEL,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        on_progress: Optional[ProgressHook] = None,
    ) -> VideoGenerationResult:
        if not (prompt or "").strip():
            raise PuterError("Text-to-video needs a prompt.")
        args: Dict[str, Any] = {
            "model": model,
            "prompt": prompt.strip(),
            "duration": round(float(seconds), 2),
            "resolution": resolution,
            "fps": 24,
        }
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed is not None:
            args["seed"] = int(seed)
        return self._generate(args, output_path, model, seconds, prompt, on_progress)

    def image_to_video(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seconds: float = 5.0,
        model: str = IMAGE_TO_VIDEO_MODEL,
        motion_strength: float = 0.7,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        on_progress: Optional[ProgressHook] = None,
    ) -> VideoGenerationResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise PuterError(f"Image-to-video source missing: {image_path}")
        args: Dict[str, Any] = {
            "model": model,
            "prompt": (prompt or "").strip(),
            "image": _to_data_uri(image_path),
            "duration": round(float(seconds), 2),
            "motion_strength": round(float(motion_strength), 3),
            "fps": 24,
        }
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        if seed is not None:
            args["seed"] = int(seed)
        return self._generate(args, output_path, model, seconds, prompt, on_progress)

    # ------------------------------------------------------------ internals
    def _generate(
        self,
        args: Dict[str, Any],
        output_path: Path,
        model: str,
        seconds: float,
        prompt: str,
        on_progress: Optional[ProgressHook],
    ) -> VideoGenerationResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()

        def report(message: str, fraction: float) -> None:
            logger.info("Puter video: %s (%.0f%%)", message, fraction * 100)
            if on_progress:
                try:
                    on_progress(message, fraction)
                except Exception:
                    logger.debug("video progress hook raised", exc_info=True)

        # The driver does not hand back the bytes. Asked for video/mp4 it
        # returns a JSON-serialised Node stream object - an empty husk with no
        # data in it - so the video is written to the account's own filesystem
        # and read back from there instead. That is what puter_output_path is
        # for, and it is the only route that actually yields a playable file.
        remote = f"/{self._home()}/moja-ai-{uuid.uuid4().hex[:12]}.mp4"
        args = {**args, "puter_output_path": remote}

        report(f"submitting {model}", 0.05)
        body, content_type, parsed = self.call_driver(
            settings.puter_video_interface,
            settings.puter_video_driver,
            settings.puter_video_method,
            args,
            accept="video/mp4",
        )

        job_id = self._job_handle(parsed)
        if job_id:
            report("queued, waiting for the render", 0.15)
            body, content_type, parsed = self._await_job(job_id, report)

        try:
            video = self._resolve_media(body, content_type, parsed, "video")
        except PuterError:
            # Expected on this driver: it answers with an empty stream husk
            # rather than the bytes, so the file on the account is the payload.
            video = b""
        if not video:
            report("fetching the rendered file", 0.9)
            video = self._download(remote)
        if not video:
            raise PuterError("Puter video generation returned an empty payload.")
        output_path.write_bytes(video)
        self._remove_remote(remote)
        report("downloaded", 1.0)

        return VideoGenerationResult(
            path=output_path, model=model, seconds=seconds,
            prompt=prompt, elapsed=time.time() - started,
        )

    def _home(self) -> str:
        """The account's own directory name, which is its username."""
        if self._home_cache is None:
            response = requests.get(
                f"{self.base_url}/whoami",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            self._home_cache = str(response.json().get("username") or "").strip()
            if not self._home_cache:
                raise PuterError("Puter did not report a username to write the video under.")
        return self._home_cache

    def _download(self, remote_path: str) -> bytes:
        """Read a file back off the account's filesystem."""
        response = requests.get(
            f"{self.base_url}/read",
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"path": remote_path},
            timeout=settings.puter_video_timeout,
        )
        if response.status_code != 200:
            raise PuterError(
                f"Could not read the generated video back from {remote_path}: "
                f"HTTP {response.status_code} {response.text[:160]}"
            )
        return response.content

    def _remove_remote(self, remote_path: str) -> None:
        """Tidy up: the render lives in the user's own storage, not ours."""
        try:
            requests.post(
                f"{self.base_url}/delete",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"paths": [remote_path]}, timeout=60,
            )
        except Exception:
            logger.debug("could not remove %s from Puter storage", remote_path)

    @staticmethod
    def _job_handle(parsed: Any) -> Optional[str]:
        if not isinstance(parsed, dict):
            return None
        node = parsed.get("result") if isinstance(parsed.get("result"), dict) else parsed
        if not isinstance(node, dict):
            return None
        # Only treat it as a job when there is no media in the payload already.
        if PuterClient._find_media_value(node) is not None:
            return None
        for key in _JOB_KEYS:
            value = node.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value)
        return None

    def _await_job(
        self,
        job_id: str,
        report: Callable[[str, float], None],
    ) -> Tuple[bytes, str, Optional[Dict[str, Any]]]:
        deadline = time.time() + settings.puter_video_timeout
        interval = 3.0
        attempt = 0
        while time.time() < deadline:
            time.sleep(interval)
            attempt += 1
            interval = min(interval * 1.25, 15.0)
            body, content_type, parsed = self.call_driver(
                settings.puter_video_interface,
                settings.puter_video_driver,
                settings.puter_video_status_method,
                {"job_id": job_id, "id": job_id},
                accept="application/json",
            )
            status = self._status_of(parsed)
            elapsed = settings.puter_video_timeout - (deadline - time.time())
            report(
                f"generating ({status or 'working'}, {elapsed:.0f}s)",
                min(0.15 + 0.75 * (elapsed / settings.puter_video_timeout), 0.9),
            )
            if status in _STATUS_FAILED:
                raise PuterError(f"Puter video job {job_id} {status}.")
            if status in _STATUS_DONE or PuterClient._find_media_value(parsed) is not None:
                return body, content_type, parsed
        raise PuterError(
            f"Puter video job {job_id} did not finish within "
            f"{settings.puter_video_timeout}s."
        )

    @staticmethod
    def _status_of(parsed: Any) -> str:
        if not isinstance(parsed, dict):
            return ""
        node = parsed.get("result") if isinstance(parsed.get("result"), dict) else parsed
        for key in ("status", "state", "phase"):
            value = node.get(key) if isinstance(node, dict) else None
            if isinstance(value, str):
                return value.strip().lower()
        return ""


# ---------------------------------------------------------------------------
# Keyframe -> animation insertion
# ---------------------------------------------------------------------------


def extract_keyframe(video_path: Path, timestamp: float, output_path: Path) -> Path:
    """Grab a single still from ``video_path`` at ``timestamp``."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PuterError(f"Cannot open {video_path} to extract a keyframe.")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:  # seek past the end: fall back to the first frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise PuterError(f"No decodable frame at {timestamp:.2f}s in {video_path.name}.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        return output_path
    finally:
        capture.release()


def build_animation_prompt(
    base_prompt: str,
    story_context: str = "",
    subjects: Optional[List[str]] = None,
    art_style: str = "",
    mood: str = "",
) -> str:
    """Fold the surrounding story context into the i2v prompt."""
    parts = [(base_prompt or "").strip()]
    if subjects:
        parts.append("featuring " + ", ".join(subjects[:3]))
    if art_style:
        parts.append(f"in {art_style} style")
    if mood:
        parts.append(f"{mood} mood")
    if story_context:
        parts.append(f"story context: {story_context.strip()}")
    parts.append("smooth cinematic camera motion, consistent character design, no text")
    return ", ".join(part for part in parts if part)


def animate_keyframe(
    client: PuterVideoClient,
    video_path: Path,
    timestamp: float,
    prompt: str,
    output_path: Path,
    seconds: float = 4.0,
    workspace: Optional[Path] = None,
    story_context: str = "",
    subjects: Optional[List[str]] = None,
    art_style: str = "",
    mood: str = "",
    motion_strength: float = 0.7,
    on_progress: Optional[ProgressHook] = None,
) -> VideoGenerationResult:
    """Extract a keyframe, animate it with i2v, return the generated clip."""
    workspace = Path(workspace or Path(output_path).parent)
    still = extract_keyframe(
        video_path, timestamp, workspace / f"keyframe_{timestamp:.2f}.png".replace(".", "_", 1)
    )
    full_prompt = build_animation_prompt(prompt, story_context, subjects, art_style, mood)
    return client.image_to_video(
        still, full_prompt, output_path,
        seconds=seconds, motion_strength=motion_strength, on_progress=on_progress,
    )


def conform_generated_clip(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Re-frame a generated clip to the project canvas so it splices cleanly."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps={fps}"
    )
    import subprocess

    result = subprocess.run(
        [
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", str(destination),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise PuterError(f"Could not conform the generated clip: {result.stderr[:300]}")
    return destination
