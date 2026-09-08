"""Pluggable AI video-generation providers.

The renderer, the job pipeline and the Android UI only ever talk to
:class:`VideoProvider`.  Puter's wan2.2 is one implementation; adding another
service means writing one adapter class and registering it, not touching the
pipeline.

Every provider reports progress through the same callback shape, so the app's
progress bar, status timer and preview work identically whichever backend is
generating the clip.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import requests

from puter_integration import _to_data_uri

logger = logging.getLogger(__name__)

# (stage, fraction 0..1, elapsed seconds)
ProgressHook = Callable[[str, float, float], None]


@dataclass
class GenerationResult:
    path: Path
    provider: str
    model: str
    seconds: float
    prompt: str
    elapsed: float
    width: int = 0
    height: int = 0


@dataclass
class ProviderInfo:
    key: str
    label: str
    text_to_video: bool
    image_to_video: bool
    models: List[str] = field(default_factory=list)
    requires_key: bool = True
    configured: bool = False
    notes: str = ""


class ProviderError(RuntimeError):
    """A provider could not fulfil the request."""


class VideoProvider(ABC):
    """What every AI video backend has to be able to do."""

    key: str = "base"
    label: str = "Base provider"
    supports_text_to_video: bool = False
    supports_image_to_video: bool = False
    requires_key: bool = True

    def __init__(self, api_key: str = "", **options: object) -> None:
        self.api_key = (api_key or "").strip()
        self.options = options

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or not self.requires_key

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            key=self.key,
            label=self.label,
            text_to_video=self.supports_text_to_video,
            image_to_video=self.supports_image_to_video,
            models=self.models(),
            requires_key=self.requires_key,
            configured=self.is_configured,
            notes=self.notes(),
        )

    def models(self) -> List[str]:
        return []

    def notes(self) -> str:
        return ""

    @abstractmethod
    def text_to_video(
        self,
        prompt: str,
        output_path: Path,
        seconds: float = 5.0,
        resolution: str = "720x1280",
        on_progress: Optional[ProgressHook] = None,
        **kwargs: object,
    ) -> GenerationResult:
        ...

    @abstractmethod
    def image_to_video(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seconds: float = 5.0,
        motion_strength: float = 0.7,
        on_progress: Optional[ProgressHook] = None,
        **kwargs: object,
    ) -> GenerationResult:
        ...

    # -- helper for subclasses ------------------------------------------------
    @staticmethod
    def _reporter(on_progress: Optional[ProgressHook], started: float) -> Callable[[str, float], None]:
        def report(stage: str, fraction: float) -> None:
            if on_progress:
                try:
                    on_progress(stage, max(0.0, min(fraction, 1.0)), time.time() - started)
                except Exception:
                    logger.debug("progress hook raised", exc_info=True)
        return report


# ---------------------------------------------------------------------------
# Puter.js wan2.2
# ---------------------------------------------------------------------------


class PuterVideoProvider(VideoProvider):
    key = "puter"
    label = "Puter.js (wan2.2)"
    supports_text_to_video = True
    supports_image_to_video = True
    requires_key = True

    def models(self) -> List[str]:
        from puter_video import IMAGE_TO_VIDEO_MODEL, TEXT_TO_VIDEO_MODEL

        return [TEXT_TO_VIDEO_MODEL, IMAGE_TO_VIDEO_MODEL]

    def notes(self) -> str:
        return (
            "Puter has no anonymous tier: POST /drivers/call answers 401 "
            "token_missing without a key."
        )

    def _client(self):
        from puter_video import PuterVideoClient

        return PuterVideoClient(api_key=self.api_key)

    def text_to_video(
        self, prompt, output_path, seconds=5.0, resolution="720x1280",
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        from config import settings
        from puter_integration import PuterError

        started = time.time()
        report = self._reporter(on_progress, started)
        report("submitting", 0.05)
        try:
            result = self._client().text_to_video(
                prompt, output_path, seconds=seconds, resolution=resolution,
                model=str(kwargs.get("model") or settings.puter_t2v_model),
                on_progress=lambda message, fraction: report(message, fraction),
            )
        except PuterError as exc:
            raise ProviderError(str(exc)) from exc
        width, height = _parse_resolution(resolution)
        return GenerationResult(
            path=Path(result.path), provider=self.key, model=result.model,
            seconds=seconds, prompt=prompt, elapsed=time.time() - started,
            width=width, height=height,
        )

    def image_to_video(
        self, image_path, prompt, output_path, seconds=5.0, motion_strength=0.7,
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        from config import settings
        from puter_integration import PuterError

        started = time.time()
        report = self._reporter(on_progress, started)
        report("submitting", 0.05)
        try:
            result = self._client().image_to_video(
                image_path, prompt, output_path, seconds=seconds,
                model=str(kwargs.get("model") or settings.puter_i2v_model),
                motion_strength=motion_strength,
                on_progress=lambda message, fraction: report(message, fraction),
            )
        except PuterError as exc:
            raise ProviderError(str(exc)) from exc
        return GenerationResult(
            path=Path(result.path), provider=self.key, model=result.model,
            seconds=seconds, prompt=prompt, elapsed=time.time() - started,
        )


# ---------------------------------------------------------------------------
# Local fallback: no service, no key, no network
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VideoForge - a self-hosted Z.ai endpoint
# ---------------------------------------------------------------------------


class VideoForgeProvider(VideoProvider):
    """A keyless Z.ai (Zhipu) video endpoint that queues and paces work.

    It is the opposite trade to Puter: nothing is billed per clip, but nothing
    is instant either. Acceptance is unlimited, throughput is not - the service
    guarantees ten requests a minute and renders two at a time, so a batch of
    keyframe animations has to be spaced out rather than fired at once.

    Submission is asynchronous even though the API offers ``?wait=true``. A
    synchronous call holds one connection open for the whole render, and long
    held connections are exactly what fails through a proxy; polling a task id
    survives that.
    """

    key = "videoforge"
    label = "VideoForge (Z.ai)"
    supports_text_to_video = True
    supports_image_to_video = True
    requires_key = False

    # Shared across instances: the rate limit belongs to the service, not to
    # whichever object happens to be making this call.
    _last_submit = 0.0
    _submit_lock = threading.Lock()

    def models(self) -> List[str]:
        return ["z-ai/text2video", "z-ai/image2video"]

    def notes(self) -> str:
        from config import settings

        return (
            f"No key and no per-clip cost; paced at about "
            f"{settings.videoforge_rpm} requests a minute with two renders in "
            f"flight, so a batch is spaced out rather than queued all at once."
        )

    # -- plumbing ------------------------------------------------------------
    def _base(self) -> str:
        from config import settings

        return str(self.options.get("base_url") or settings.videoforge_base_url).rstrip("/")

    def _pace(self) -> None:
        """Hold each submission back so a burst cannot trip the limit."""
        from config import settings

        gap = 60.0 / max(int(settings.videoforge_rpm), 1)
        with VideoForgeProvider._submit_lock:
            wait = gap - (time.time() - VideoForgeProvider._last_submit)
            if wait > 0:
                time.sleep(wait)
            VideoForgeProvider._last_submit = time.time()

    def _submit(self, payload: Dict[str, object]) -> str:
        from config import settings

        self._pace()
        response = requests.post(
            f"{self._base()}/api/v1/generate", json=payload,
            timeout=min(120, settings.videoforge_timeout),
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"VideoForge rejected the request: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        body = response.json()
        task_id = str(
            body.get("task_id")
            or body.get("id")
            or (body.get("data") or {}).get("id", "")
        ).strip()
        if not task_id:
            raise ProviderError(f"VideoForge returned no task id: {response.text[:200]}")
        return task_id

    def _await(self, task_id: str, report: Callable[[str, float], None]) -> None:
        from config import settings

        deadline = time.time() + settings.videoforge_timeout
        seen = ""
        observed = False
        started_upstream = False
        while time.time() < deadline:
            time.sleep(settings.videoforge_poll_seconds)
            response = requests.get(f"{self._base()}/api/v1/tasks/{task_id}", timeout=60)
            if response.status_code >= 400:
                raise ProviderError(
                    f"VideoForge task {task_id}: HTTP {response.status_code}"
                )
            node = response.json()
            node = node.get("data", node) if isinstance(node, dict) else {}
            status = str(node.get("status") or "").upper()
            observed = True
            # The service stamps this when it hands the job to the model. A task
            # that never gets one is not a slow render, it is a queue that is
            # not draining - a different problem with a different owner, and
            # worth saying so rather than reporting a flat timeout.
            started_upstream = started_upstream or bool(node.get("submitted_at"))
            if status != seen:
                seen = status
                elapsed = settings.videoforge_timeout - (deadline - time.time())
                report(f"{status.lower() or 'working'} ({elapsed:.0f}s)",
                       0.25 if status == "QUEUED" else 0.6)
            if status == "SUCCESS":
                return
            if status in ("FAIL", "FAILED", "ERROR"):
                raise ProviderError(
                    f"VideoForge failed: {node.get('error') or node.get('message') or status}"
                )
        if observed and not started_upstream:
            raise ProviderError(
                f"VideoForge accepted task {task_id} but never started it: it sat "
                f"queued for {settings.videoforge_timeout}s without being handed to "
                f"the model. The endpoint is up but its render queue is not "
                f"draining - check {self._base()}/api/v1/stats for how many tasks "
                f"are queued against how many are processing."
            )
        # Either it was seen rendering, or the budget was gone before a single
        # poll - and in that case nothing was observed, so nothing is claimed.
        raise ProviderError(
            f"VideoForge task {task_id} did not finish within "
            f"{settings.videoforge_timeout}s."
        )

    def _download(self, task_id: str, output_path: Path) -> None:
        from config import settings

        # The task's video_url points at a CDN whose links expire; the service
        # keeps this route working after they do, so it is the one to use.
        response = requests.get(
            f"{self._base()}/api/v1/file/{task_id}", timeout=settings.videoforge_timeout
        )
        if response.status_code >= 400 or len(response.content) < 1024:
            raise ProviderError(
                f"VideoForge produced no file for {task_id}: "
                f"HTTP {response.status_code}, {len(response.content)} bytes"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

    def _run(self, payload: Dict[str, object], output_path: Path, seconds: float,
             prompt: str, model: str, on_progress) -> GenerationResult:
        started = time.time()
        report = self._reporter(on_progress, started)
        report("submitting", 0.05)
        task_id = self._submit(payload)
        self._await(task_id, report)
        report("downloading", 0.9)
        self._download(task_id, Path(output_path))
        report("done", 1.0)
        return GenerationResult(
            path=Path(output_path), provider=self.key, model=model,
            seconds=seconds, prompt=prompt, elapsed=time.time() - started,
        )

    # The service takes two clip lengths and rejects anything else outright,
    # so a 4-second request has to become a 5-second one here rather than a
    # 400 at submit time.
    ALLOWED_SECONDS = (5, 10)

    @classmethod
    def _clamp_seconds(cls, seconds: float) -> int:
        return min(cls.ALLOWED_SECONDS, key=lambda option: abs(option - float(seconds)))

    # -- API -----------------------------------------------------------------
    def text_to_video(
        self, prompt, output_path, seconds=5.0, resolution="720x1280",
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        width, height = _parse_resolution(resolution)
        seconds = self._clamp_seconds(seconds)
        payload = {
            "prompt": prompt,
            "duration": seconds,
            "size": f"{width}x{height}",
            "fps": int(kwargs.get("fps") or 30),
            "quality": str(kwargs.get("quality") or "high"),
            "with_audio": bool(kwargs.get("with_audio", False)),
            "watermark": False,
            "client": "moja-ai",
        }
        return self._run(payload, output_path, seconds, prompt,
                         "z-ai/text2video", on_progress)

    def image_to_video(
        self, image_path, prompt, output_path, seconds=5.0, motion_strength=0.7,
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        seconds = self._clamp_seconds(seconds)
        payload = {
            "image_url": _to_data_uri(Path(image_path)),
            "prompt": prompt,
            "duration": seconds,
            "fps": int(kwargs.get("fps") or 30),
            "quality": str(kwargs.get("quality") or "high"),
            "with_audio": bool(kwargs.get("with_audio", False)),
            "watermark": False,
            "client": "moja-ai",
        }
        return self._run(payload, output_path, seconds, prompt,
                         "z-ai/image2video", on_progress)


class ModelScopeProvider(VideoProvider):
    """Alibaba's ModelScope, which serves Wan and friends on a free daily quota.

    The reason it is here: of everything checked, this is the only route that
    is free at a volume worth having. NVIDIA NIM has no video model at all -
    its catalogue carries text-to-image, video *understanding* and an AI-video
    *detector*, and nothing that generates. Google's Veo has no free API tier.
    Hugging Face routes to the same open models but gives a free account $0.10
    of credit a month, which is about one clip. ModelScope's free tier is
    counted in requests per day rather than cents per month.

    Submission is asynchronous: the POST returns a task id, and the task is
    polled on a header-gated endpoint until the video URL appears.
    """

    key = "modelscope"
    label = "ModelScope (Wan 2.2)"
    supports_text_to_video = True
    supports_image_to_video = True
    requires_key = True

    BASE = "https://api-inference.modelscope.cn"
    DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
    DEFAULT_I2V_MODEL = "Wan-AI/Wan2.2-I2V-A14B"

    def models(self) -> List[str]:
        return [self.DEFAULT_MODEL, self.DEFAULT_I2V_MODEL]

    def notes(self) -> str:
        return (
            "Free daily quota rather than a credit balance. Needs a ModelScope "
            "account token (free) from modelscope.cn; renders are queued, so a "
            "clip takes minutes rather than seconds."
        )

    def _headers(self, asynchronous: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if asynchronous:
            # Without this the request is served synchronously and a long
            # render dies holding the connection open.
            headers["X-ModelScope-Async-Mode"] = "true"
        return headers

    def _submit(self, payload: Dict[str, object]) -> str:
        response = requests.post(
            f"{self.BASE}/v1/videos/generations",
            headers=self._headers(), json=payload, timeout=120,
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"ModelScope rejected the request: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        body = response.json()
        task_id = str(body.get("task_id") or body.get("id") or "").strip()
        if not task_id:
            raise ProviderError(f"ModelScope returned no task id: {response.text[:200]}")
        return task_id

    def _await(self, task_id: str, report: Callable[[str, float], None],
               timeout: float) -> str:
        deadline = time.time() + timeout
        seen = ""
        while time.time() < deadline:
            time.sleep(5.0)
            response = requests.get(
                f"{self.BASE}/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "X-ModelScope-Task-Type": "image_generation"},
                timeout=60,
            )
            if response.status_code >= 400:
                raise ProviderError(
                    f"ModelScope task {task_id}: HTTP {response.status_code} "
                    f"{response.text[:200]}"
                )
            node = response.json()
            status = str(node.get("task_status") or node.get("status") or "").upper()
            if status != seen:
                seen = status
                report(f"{status.lower() or 'working'}", 0.3 if status == "PENDING" else 0.6)
            if status in ("SUCCEED", "SUCCEEDED", "SUCCESS"):
                urls = node.get("output_video_url") or node.get("output_videos") or []
                if isinstance(urls, str):
                    return urls
                if urls:
                    return str(urls[0])
                raise ProviderError(
                    f"ModelScope finished task {task_id} with no video url: "
                    f"{str(node)[:200]}"
                )
            if status in ("FAILED", "FAIL", "ERROR"):
                raise ProviderError(
                    f"ModelScope failed: {node.get('message') or node.get('errors') or status}"
                )
        raise ProviderError(
            f"ModelScope task {task_id} did not finish within {timeout:.0f}s."
        )

    def _run(self, payload, output_path, seconds, prompt, model, on_progress,
             timeout: float = 900.0) -> GenerationResult:
        if not self.api_key:
            raise ProviderError(
                "ModelScope needs a free account token; add it in Settings."
            )
        started = time.time()
        report = self._reporter(on_progress, started)
        report("submitting", 0.05)
        task_id = self._submit(payload)
        url = self._await(task_id, report, timeout)

        report("downloading", 0.9)
        video = requests.get(url, timeout=300)
        if video.status_code >= 400 or len(video.content) < 1024:
            raise ProviderError(
                f"ModelScope produced no file: HTTP {video.status_code}, "
                f"{len(video.content)} bytes"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video.content)
        report("done", 1.0)
        return GenerationResult(
            path=output_path, provider=self.key, model=model, seconds=seconds,
            prompt=prompt, elapsed=time.time() - started,
        )

    def text_to_video(
        self, prompt, output_path, seconds=5.0, resolution="720x1280",
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        model = str(kwargs.get("model") or self.DEFAULT_MODEL)
        return self._run(
            {"model": model, "prompt": prompt},
            output_path, seconds, prompt, model, on_progress,
        )

    def image_to_video(
        self, image_path, prompt, output_path, seconds=5.0, motion_strength=0.7,
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        model = str(kwargs.get("model") or self.DEFAULT_I2V_MODEL)
        return self._run(
            {"model": model, "prompt": prompt,
             "image_url": _to_data_uri(Path(image_path))},
            output_path, seconds, prompt, model, on_progress,
        )


class PollinationsProvider(VideoProvider):
    """Pollinations, which fronts nineteen video models behind one key.

    The cheapest route found: MiniMax H3 Turbo bills 0.00625 "pollen" per
    generated second at 480p and a pollen is about a dollar, so a five second
    clip costs roughly three cents - against forty cents for the same length
    through Puter's Sora-2, which is what the last batch was being billed at.
    It also carries Veo, Wan 3.0 and Seedance, so the model choice is not a
    consolation prize for the price.

    The API is a GET with the prompt in the path, like their image endpoint,
    and it answers with the video bytes rather than a task id - so there is no
    polling, just a long read.
    """

    key = "pollinations"
    label = "Pollinations (Wan / Veo / Seedance)"
    supports_text_to_video = True
    supports_image_to_video = True
    requires_key = True

    BASE = "https://gen.pollinations.ai"
    DEFAULT_MODEL = "wan-fast"

    def models(self) -> List[str]:
        return ["wan-fast", "wan", "wan-pro", "wan-3.0", "seedance-2.0-fast",
                "seedance-2.5", "minimax-h3", "veo"]

    def notes(self) -> str:
        return (
            "One key, nineteen video models. Around three cents a clip at 480p "
            "- the cheapest of everything checked. Needs a free key from "
            "enter.pollinations.ai/keys."
        )

    def _fetch(self, prompt: str, params: Dict[str, object], output_path: Path,
               seconds: float, model: str, on_progress) -> GenerationResult:
        from urllib.parse import quote  # noqa: PLC0415

        if not self.api_key:
            raise ProviderError(
                "Pollinations needs a free key from enter.pollinations.ai/keys; "
                "add it in Settings."
            )
        started = time.time()
        report = self._reporter(on_progress, started)
        report("generating", 0.15)

        try:
            response = requests.get(
                f"{self.BASE}/video/{quote(prompt.strip()[:1500], safe='')}",
                params={k: v for k, v in params.items() if v not in (None, "")},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=900,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Pollinations was unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Pollinations refused the request: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        # A JSON body here is an error wearing a 200, not a video.
        if "video" not in (response.headers.get("content-type") or ""):
            raise ProviderError(
                f"Pollinations returned {response.headers.get('content-type')} "
                f"instead of video: {response.text[:200]}"
            )
        if len(response.content) < 1024:
            raise ProviderError(
                f"Pollinations returned only {len(response.content)} bytes"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        report("done", 1.0)
        return GenerationResult(
            path=output_path, provider=self.key, model=model, seconds=seconds,
            prompt=prompt, elapsed=time.time() - started,
        )

    def text_to_video(
        self, prompt, output_path, seconds=5.0, resolution="720x1280",
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        model = str(kwargs.get("model") or self.DEFAULT_MODEL)
        width, height = _parse_resolution(resolution)
        return self._fetch(prompt, {
            "model": model,
            "duration": int(round(seconds)),
            "aspectRatio": "9:16" if height >= width else "16:9",
            "audio": "true" if kwargs.get("with_audio") else None,
            "seed": kwargs.get("seed"),
        }, output_path, seconds, model, on_progress)

    # Models that accept reference media, and so can hold a character's design
    # steady from shot to shot. The plain ones cannot: each generation is
    # independent, so the same prompt gives a differently designed character
    # every time, which is what makes AI fight scenes fall apart.
    REFERENCE_MODELS = ("wan-3.0", "wan-pro", "seedance-2.5", "seedance-2.0")

    @classmethod
    def reference_url(cls, prompt: str, seed: int = 7, model: str = "zimage",
                      width: int = 768, height: int = 1344) -> str:
        """A stable public URL for a generated still, usable as a reference.

        The service documents an /upload route for reference media, but it
        answers 404 - so a local file cannot be handed over. What works
        instead is that image generation is itself addressed by URL and cached
        immutably: the same prompt and seed give back the same picture
        forever, so that URL *is* the reference and nothing has to be hosted.

        One catch, measured rather than assumed: the URL is NOT public until
        the picture behind it has been made once. Fetched before that it
        answers 401; fetched after, it serves the identical bytes to anyone,
        with no key. So ``warm`` it before handing it to a video model, or the
        model fetches a 401 instead of a character. ``prime_reference`` does
        exactly that.
        """
        from urllib.parse import quote, urlencode  # noqa: PLC0415

        query = urlencode({"model": model, "width": width, "height": height,
                           "seed": int(seed)})
        return f"{cls.BASE}/image/{quote(prompt.strip()[:1200], safe='')}?{query}"

    def prime_reference(self, url: str, timeout: int = 240) -> bool:
        """Make the picture behind a reference URL, so the URL becomes public.

        Returns whether the URL now serves an image. A video model fetches
        references anonymously, so an unprimed URL reaches it as a 401 and the
        clip comes back with no character in it - a failure that looks like a
        bad prompt rather than a missing fetch.
        """
        try:
            response = requests.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Could not prime reference %s: %s", url[:80], exc)
            return False
        ok = response.status_code < 400 and len(response.content) > 1024
        if not ok:
            logger.warning("Reference %s did not prime: HTTP %s",
                           url[:80], response.status_code)
        return ok

    def image_to_video(
        self, image_path, prompt, output_path, seconds=5.0, motion_strength=0.7,
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        model = str(kwargs.get("model") or "wan-3.0")
        if model not in self.REFERENCE_MODELS:
            raise ProviderError(
                f"'{model}' does not take reference media; use one of "
                f"{', '.join(self.REFERENCE_MODELS)} to hold a character steady."
            )
        # reference_images takes public URLs only. A local file has nowhere to
        # live, so the caller passes a URL - reference_url() builds one out of
        # the image generator, which needs no hosting.
        reference = kwargs.get("reference_image_url")
        if not reference:
            raise ProviderError(
                "Pollinations image-to-video needs a public image URL. Build one "
                "with PollinationsProvider.reference_url(prompt, seed), or pass "
                "reference_image_url= for a still you already host."
            )
        # A reference the model cannot fetch is worse than none: the clip comes
        # back without the character and nothing says why.
        if str(reference).startswith(self.BASE) and not kwargs.get("primed"):
            if not self.prime_reference(str(reference)):
                raise ProviderError(
                    "The reference image could not be made public; the video "
                    "model would fetch a 401 instead of the character."
                )
        return self._fetch(prompt, {
            "model": model,
            "duration": int(round(seconds)),
            "reference_images": str(reference),
            "aspectRatio": "9:16",
        }, output_path, seconds, model, on_progress)


class LocalMotionProvider(VideoProvider):
    """Ken Burns style motion from a still - not AI, but never a dead button.

    Image-to-video only. It exists so the i2v UI has something real to do
    before a generation key is configured, and so a failed provider call can
    degrade to *a clip* instead of nothing.
    """

    key = "local_motion"
    label = "Local motion (no AI, no key)"
    supports_text_to_video = False
    supports_image_to_video = True
    requires_key = False

    def notes(self) -> str:
        return "Animates a still with a slow push-in and drift. Runs offline."

    def text_to_video(self, *args, **kwargs) -> GenerationResult:  # pragma: no cover
        raise ProviderError(
            "Local motion cannot invent footage from text - it only animates an image."
        )

    def image_to_video(
        self, image_path, prompt, output_path, seconds=5.0, motion_strength=0.7,
        on_progress=None, **kwargs,
    ) -> GenerationResult:
        import subprocess

        from puter_integration import ffmpeg_binary

        started = time.time()
        report = self._reporter(on_progress, started)
        report("rendering motion", 0.2)

        image_path, output_path = Path(image_path), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 30
        frames = max(int(seconds * fps), 1)
        zoom = 1.0 + 0.18 * max(0.1, min(motion_strength, 1.5))

        # zoompan needs an oversized source to pan inside without softening.
        vf = (
            f"scale=iw*2:ih*2,"
            f"zoompan=z='min(1+({zoom - 1.0})*on/{frames},{zoom})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s=1080x1920:fps={fps}"
        )
        result = subprocess.run(
            [
                ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(image_path), "-vf", vf,
                "-t", f"{seconds:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", str(output_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ProviderError(f"Local motion render failed: {result.stderr[:200]}")
        report("done", 1.0)
        return GenerationResult(
            path=output_path, provider=self.key, model="ken-burns",
            seconds=seconds, prompt=prompt, elapsed=time.time() - started,
            width=1080, height=1920,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[VideoProvider]] = {}


def register_provider(provider: Type[VideoProvider]) -> Type[VideoProvider]:
    _REGISTRY[provider.key] = provider
    return provider


register_provider(PuterVideoProvider)
register_provider(VideoForgeProvider)
register_provider(ModelScopeProvider)
register_provider(PollinationsProvider)
register_provider(LocalMotionProvider)

# VideoForge first: it needs no key and bills nothing, so it is the one
# that works out of the box.
DEFAULT_PROVIDER = VideoForgeProvider.key


def provider_keys() -> List[str]:
    return list(_REGISTRY)


def get_provider(key: Optional[str], api_key: str = "", **options: object) -> VideoProvider:
    """Build a provider by key, falling back to the default."""
    resolved = (key or DEFAULT_PROVIDER).strip().lower()
    factory = _REGISTRY.get(resolved, _REGISTRY[DEFAULT_PROVIDER])
    return factory(api_key=api_key, **options)


def describe_providers(api_keys: Optional[Dict[str, str]] = None) -> List[ProviderInfo]:
    keys = api_keys or {}
    return [
        factory(api_key=keys.get(key, "")).info() for key, factory in _REGISTRY.items()
    ]


def best_available(
    api_keys: Optional[Dict[str, str]] = None,
    need_text_to_video: bool = False,
    **options: object,
) -> Optional[VideoProvider]:
    """The first configured provider that can do what is being asked.

    ``options`` reach every candidate, so a per-request setting such as the
    user's own endpoint applies whichever provider ends up being chosen.
    """
    keys = api_keys or {}
    for key, factory in _REGISTRY.items():
        provider = factory(api_key=keys.get(key, ""), **options)
        if not provider.is_configured:
            continue
        if need_text_to_video and not provider.supports_text_to_video:
            continue
        if not need_text_to_video and not provider.supports_image_to_video:
            continue
        return provider
    return None


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width, height = resolution.lower().split("x")
        return int(width), int(height)
    except (AttributeError, ValueError):
        return 0, 0
