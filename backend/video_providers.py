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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

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
register_provider(LocalMotionProvider)

DEFAULT_PROVIDER = PuterVideoProvider.key


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
    api_keys: Optional[Dict[str, str]] = None, need_text_to_video: bool = False
) -> Optional[VideoProvider]:
    """The first configured provider that can do what is being asked."""
    keys = api_keys or {}
    for key, factory in _REGISTRY.items():
        provider = factory(api_key=keys.get(key, ""))
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
