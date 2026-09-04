"""Rendering pipeline: FFmpeg + MoviePy + Puter.js assets + Faster-Whisper.

Given an :class:`~schemas.EditPlan` produced by Kimi K3 and the clips uploaded
from the Android app this module:

1. cuts the timeline (jump cuts, zoom punches, speed ramps),
2. re-frames every segment to a 1080x1920 vertical canvas,
3. round trips selected frames through Puter inpainting with OpenCV masks,
4. overlays the transparent Puter stickers (``rembg`` output) and 3D text,
5. lays the Indian accent Puter TTS voiceover over the original audio bed,
6. burns word level animated captions produced by Faster-Whisper,
7. encodes a 1080x1920 60 fps H.264 MP4.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import settings
from puter_integration import (
    ExtractedFrame,
    PuterClient,
    PuterError,
    build_mask,
    extract_frames,
    ffmpeg_binary,
)
from sticker_art import draw_sticker
from themes import Theme, resolve_theme
from vfx import apply_frame_effect, build_effect_chain
from schemas import (
    EditPlan,
    JobStage,
    PuterSticker,
    StickerAnimation,
    TextOverlay,
    TimelineSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MoviePy 1.x / 2.x compatibility shim
# ---------------------------------------------------------------------------

try:  # MoviePy 1.0.3 (pinned in requirements.txt)
    from moviepy.editor import (  # type: ignore
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
        vfx,
    )
except ImportError:  # MoviePy >= 2.0
    from moviepy import (  # type: ignore
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
        vfx,
    )


def _sub(clip, start: float, end: float):
    if hasattr(clip, "subclip"):
        return clip.subclip(start, end)
    return clip.subclipped(start, end)


def _with_start(clip, value: float):
    return clip.set_start(value) if hasattr(clip, "set_start") else clip.with_start(value)


def _with_duration(clip, value: float):
    return clip.set_duration(value) if hasattr(clip, "set_duration") else clip.with_duration(value)


def _with_position(clip, value):
    return clip.set_position(value) if hasattr(clip, "set_position") else clip.with_position(value)


def _with_audio(clip, audio):
    return clip.set_audio(audio) if hasattr(clip, "set_audio") else clip.with_audio(audio)


def _with_fps(clip, fps: float):
    return clip.set_fps(fps) if hasattr(clip, "set_fps") else clip.with_fps(fps)


def _with_mask(clip, mask):
    return clip.set_mask(mask) if hasattr(clip, "set_mask") else clip.with_mask(mask)


def _volume(clip, factor: float):
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    from moviepy.audio.fx import MultiplyVolume  # type: ignore

    return clip.with_effects([MultiplyVolume(factor)])


def _resize(clip, newsize):
    if hasattr(clip, "resize"):
        return clip.resize(newsize)
    return clip.resized(newsize)


def _speed(clip, factor: float):
    if hasattr(clip, "fx"):
        return clip.fx(vfx.speedx, factor)
    return clip.with_effects([vfx.MultiplySpeed(factor)])


def _crossfadein(clip, duration: float):
    if hasattr(clip, "crossfadein"):
        return clip.crossfadein(duration)
    return clip.with_effects([vfx.CrossFadeIn(duration)])


ProgressCallback = Callable[[JobStage, float, str], None]


class RenderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_DEVANAGARI_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
]


def _load_font(size: int, devanagari: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (_DEVANAGARI_CANDIDATES if devanagari else []) + _FONT_CANDIDATES
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    custom = settings.data_dir / "fonts"
    if custom.is_dir():
        for path in sorted(custom.glob("*.tt[fc]")):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


# ---------------------------------------------------------------------------
# Raw frame -> ffmpeg pipe (used by the inpainting compositor)
# ---------------------------------------------------------------------------


class FfmpegFrameWriter:
    """Stream BGR frames straight into an ffmpeg H.264 encoder."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", f"{fps:.6f}", "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        # communicate() flushes and closes stdin itself; closing it first makes
        # the flush fail with "flush of closed file".
        _, stderr = self.process.communicate()
        if self.process.returncode != 0:
            raise RenderError(f"ffmpeg frame writer failed: {stderr.decode('utf-8', 'ignore')[:500]}")

    def __enter__(self) -> "FfmpegFrameWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.close()
        else:  # pragma: no cover - propagate the original error
            self.process.kill()


# ---------------------------------------------------------------------------
# Word level captions (Faster-Whisper)
# ---------------------------------------------------------------------------


@dataclass
class CaptionWord:
    text: str
    start: float
    end: float


def transcribe_words(audio_path: Path, language: Optional[str] = None) -> List[CaptionWord]:
    """Word level transcription with Faster-Whisper (empty list on failure)."""
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("faster-whisper unavailable, skipping captions: %s", exc)
        return []

    try:
        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.cache_dir / "whisper"),
        )
        segments, _info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            language=language,
            beam_size=5,
        )
        words: List[CaptionWord] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                text = (word.word or "").strip()
                if not text:
                    continue
                start = float(word.start or 0.0)
                end = float(word.end or start + 0.25)
                if end <= start:
                    end = start + 0.2
                words.append(CaptionWord(text=text, start=start, end=end))
        logger.info("Faster-Whisper produced %s caption words", len(words))
        return words
    except Exception as exc:
        logger.warning("Transcription failed, continuing without captions: %s", exc)
        return []


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# PIL text rendering (no ImageMagick dependency)
# ---------------------------------------------------------------------------


def render_text_rgba(
    text: str,
    max_width: int,
    font_size: int,
    color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: Optional[int] = None,
    shadow: bool = True,
    three_d: bool = False,
) -> np.ndarray:
    """Render ``text`` to an RGBA numpy array with an outline + drop shadow."""
    font = _load_font(font_size, devanagari=_has_devanagari(text))
    stroke = stroke_width if stroke_width is not None else max(3, font_size // 10)

    lines = _wrap_text(text, font, max_width - 4 * stroke)
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)

    line_sizes = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        line_sizes.append((box[2] - box[0], box[3] - box[1]))

    line_gap = int(font_size * 0.22)
    width = max((size[0] for size in line_sizes), default=1) + 6 * stroke
    height = sum(size[1] for size in line_sizes) + line_gap * max(len(lines) - 1, 0) + 6 * stroke
    depth = max(4, font_size // 12) if three_d else 0
    canvas = Image.new("RGBA", (width + depth * 2, height + depth * 2 + int(font_size * 0.3)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    y = 3 * stroke
    for line, (line_width, line_height) in zip(lines, line_sizes):
        x = (canvas.width - line_width) // 2
        if three_d:
            for offset in range(depth, 0, -1):
                shade = int(60 + 90 * (1 - offset / max(depth, 1)))
                draw.text(
                    (x + offset, y + offset), line, font=font,
                    fill=(shade, shade, shade, 255),
                    stroke_width=stroke, stroke_fill=(0, 0, 0, 255),
                )
        draw.text(
            (x, y), line, font=font, fill=color,
            stroke_width=stroke, stroke_fill=stroke_color,
        )
        y += line_height + line_gap

    if shadow:
        glow = canvas.filter(ImageFilter.GaussianBlur(radius=max(2, font_size // 14)))
        backdrop = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        backdrop.alpha_composite(glow)
        backdrop.alpha_composite(canvas)
        canvas = backdrop

    return np.array(canvas)


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        box = probe.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _clip_from_rgba(array: np.ndarray):
    """Build a MoviePy clip (with a proper alpha mask) from an RGBA array."""
    if array.ndim != 3 or array.shape[2] != 4:
        return ImageClip(array)
    rgb = np.ascontiguousarray(array[:, :, :3])
    alpha = np.ascontiguousarray(array[:, :, 3].astype(np.float64) / 255.0)
    clip = ImageClip(rgb)
    mask = ImageClip(alpha, ismask=True) if _accepts_ismask() else ImageClip(alpha, is_mask=True)
    return _with_mask(clip, mask)


def _accepts_ismask() -> bool:
    import inspect

    return "ismask" in inspect.signature(ImageClip.__init__).parameters


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_POSITION_ANCHORS: Dict[str, Tuple[float, float]] = {
    "top_left": (0.16, 0.14),
    "top_center": (0.50, 0.14),
    "top_right": (0.84, 0.14),
    "center_left": (0.18, 0.50),
    "center": (0.50, 0.50),
    "center_right": (0.82, 0.50),
    "bottom_left": (0.18, 0.82),
    "bottom_center": (0.50, 0.80),
    "bottom_right": (0.82, 0.82),
}


def anchor_for(position: str) -> Tuple[float, float]:
    key = (position or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _POSITION_ANCHORS.get(key, _POSITION_ANCHORS["bottom_center"])


def fit_vertical(clip, width: int, height: int):
    """Scale-to-cover + centre crop so any clip fills the 1080x1920 canvas."""
    source_width, source_height = clip.size
    if source_width <= 0 or source_height <= 0:
        raise RenderError("Clip has an invalid size.")
    scale = max(width / source_width, height / source_height)
    new_width = int(math.ceil(source_width * scale))
    new_height = int(math.ceil(source_height * scale))
    new_width += new_width % 2
    new_height += new_height % 2
    resized = _resize(clip, (new_width, new_height))

    x_center = new_width / 2
    y_center = new_height / 2
    if hasattr(resized, "crop"):
        cropped = resized.crop(
            x_center=x_center, y_center=y_center, width=width, height=height
        )
    else:  # MoviePy 2.x
        cropped = resized.cropped(
            x_center=x_center, y_center=y_center, width=width, height=height
        )
    return cropped


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class VideoRenderer:
    """Executes an :class:`EditPlan` end to end and writes the final MP4."""

    def __init__(
        self,
        plan: EditPlan,
        clip_paths: Sequence[Path],
        workspace: Path,
        puter: Optional[PuterClient] = None,
        progress: Optional[ProgressCallback] = None,
        theme: Optional[str] = None,
        analyses: Optional[Sequence[Any]] = None,
    ) -> None:
        self.theme: Theme = resolve_theme(theme)
        self.analyses = list(analyses or [])
        self.plan = plan
        self.clip_paths = [Path(path) for path in clip_paths]
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.puter = puter
        self._progress = progress
        self.warnings: List[str] = []

        self.width, self.height = plan.project_meta.size
        self.fps = plan.project_meta.fps or settings.output_fps
        self.assets_dir = self.workspace / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._open_clips: List[Any] = []

        # The theme drives caption colours and sticker animation unless the
        # plan asked for something specific.
        if self.plan.captions.highlight_color in ("", "#FFD400"):
            self.plan.captions.highlight_color = self.theme.caption_highlight

    # ------------------------------------------------------------- plumbing
    def report(self, stage: JobStage, progress: float, message: str) -> None:
        logger.info("[%s] %.0f%% %s", stage.value, progress * 100, message)
        if self._progress:
            try:
                self._progress(stage, progress, message)
            except RenderError:
                # The job manager signals cancellation through this callback.
                raise
            except Exception:  # never let plain reporting break a render
                logger.debug("progress callback raised", exc_info=True)

    def warn(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    def _track(self, clip):
        self._open_clips.append(clip)
        return clip

    def close(self) -> None:
        for clip in self._open_clips:
            try:
                clip.close()
            except Exception:
                pass
        self._open_clips.clear()

    # ----------------------------------------------------------------- main
    def render(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            voiceover = self._generate_voiceover()
            stickers = self._generate_stickers()
            base_video = self._render_base_video()
            return self._compose_final(base_video, voiceover, stickers, output_path)
        finally:
            self.close()

    # ------------------------------------------------------------------ TTS
    def _generate_voiceover(self) -> Optional[Path]:
        audio = self.plan.audio
        if not audio.has_voiceover:
            return None
        if self.puter is None or not self.puter.is_configured:
            self.warn("Puter.js key missing - the Indian accent voiceover was skipped.")
            return None

        self.report(JobStage.TTS, 0.05, "Synthesising the Indian accent voiceover with Puter.js")
        destination = self.assets_dir / "voiceover.mp3"
        try:
            self.puter.text_to_speech(audio.tts_script, destination, accent=audio.voice_accent)
            return destination
        except PuterError as exc:
            self.warn(f"Puter TTS failed: {exc}")
            return None

    # ------------------------------------------------------------- stickers
    def _generate_stickers(self) -> Dict[int, Path]:
        requests_: List[Tuple[int, PuterSticker]] = [
            (index, segment.puter_sticker)
            for index, segment in enumerate(self.plan.edit_timeline)
            if segment.puter_sticker and segment.puter_sticker.active
        ]
        if not requests_:
            return {}

        have_puter = self.puter is not None and self.puter.is_configured
        if not have_puter:
            self.warn(
                "Puter.js key missing - the AI stickers were drawn locally as vector "
                "icons (sword / explosion / bolt / skull ...) instead."
            )

        results: Dict[int, Path] = {}
        for order, (index, sticker) in enumerate(requests_):
            self.report(
                JobStage.STICKERS,
                0.10 + 0.10 * (order / max(len(requests_), 1)),
                f"{'Generating' if have_puter else 'Drawing'} sticker "
                f"{order + 1}/{len(requests_)}: {sticker.generate_prompt[:48]}",
            )
            destination = self.assets_dir / f"sticker_{index:03d}.png"
            if have_puter:
                try:
                    self.puter.generate_sticker(sticker.generate_prompt, destination)
                    results[index] = destination
                    continue
                except Exception as exc:
                    self.warn(
                        f"Puter sticker '{sticker.generate_prompt[:40]}' failed ({exc}) - "
                        "drew a vector icon instead."
                    )
            try:
                draw_sticker(sticker.generate_prompt, destination)
                results[index] = destination
            except Exception as exc:
                self.warn(f"Sticker '{sticker.generate_prompt[:40]}' could not be drawn: {exc}")
        return results

    # --------------------------------------------------------- base timeline
    def _render_base_video(self) -> Path:
        timeline = self.plan.edit_timeline
        if not timeline:
            raise RenderError("The edit timeline is empty - nothing to render.")

        self.report(JobStage.RENDERING, 0.22, f"Cutting {len(timeline)} segments")
        segment_clips = []
        for index, segment in enumerate(timeline):
            progress = 0.22 + 0.38 * (index / max(len(timeline), 1))
            self.report(
                JobStage.RENDERING, progress,
                f"Segment {index + 1}/{len(timeline)} ({segment.cut_type})",
            )
            clip = self._build_segment(index, segment)
            if clip is not None:
                segment_clips.append(clip)

        if not segment_clips:
            raise RenderError("Every timeline segment failed to build.")

        self.report(JobStage.RENDERING, 0.62, "Concatenating the timeline")
        base = concatenate_videoclips(segment_clips, method="compose")
        base = _with_fps(base, self.fps)
        self._track(base)

        base_path = self.workspace / "base.mp4"
        self._write_video(base, base_path, with_audio=True)
        return base_path

    def _build_segment(self, index: int, segment: TimelineSegment):
        source_index = segment.source_index if segment.source_index is not None else index % len(self.clip_paths)
        if not 0 <= source_index < len(self.clip_paths):
            source_index = index % len(self.clip_paths)
        source = self.clip_paths[source_index]

        start = segment.start_seconds
        end = segment.end_seconds
        if end <= start:
            end = start + 2.0

        working_source = source
        working_start, working_end = start, end
        if segment.puter_inpaint and segment.puter_inpaint.is_enabled:
            inpainted = self._inpaint_segment(index, segment, source, start, end)
            if inpainted is not None:
                working_source = inpainted
                working_start, working_end = 0.0, end - start

        try:
            clip = self._track(VideoFileClip(str(working_source)))
        except Exception as exc:
            self.warn(f"Segment {index}: could not open {working_source.name} ({exc}).")
            return None

        duration = clip.duration or 0.0
        working_start = max(0.0, min(working_start, max(duration - 0.1, 0.0)))
        working_end = min(working_end, duration) if duration else working_end
        if working_end - working_start < 0.15:
            self.warn(f"Segment {index}: window {working_start:.2f}-{working_end:.2f}s is too short.")
            return None

        piece = _sub(clip, working_start, working_end)
        piece = fit_vertical(piece, self.width, self.height)

        speed = segment.speed if segment.speed and segment.speed > 0 else 1.0
        cut_type = (segment.cut_type or "").lower()
        if cut_type == "speed_ramp" and speed == 1.0:
            speed = 1.5
        if abs(speed - 1.0) > 0.01:
            piece = _speed(piece, speed)

        if cut_type == "zoom_punch":
            piece = self._apply_zoom_punch(piece)
        if cut_type == "crossfade":
            piece = _crossfadein(piece, min(0.35, (piece.duration or 1.0) / 3))

        piece = _with_fps(piece, self.fps)
        return self._track(piece)

    def _apply_zoom_punch(self, clip):
        duration = clip.duration or 1.0
        width, height = self.width, self.height

        def zoom(get_frame, t):
            frame = get_frame(t)
            factor = 1.0 + 0.09 * (t / duration)
            new_width = int(width * factor)
            new_height = int(height * factor)
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            x0 = (new_width - width) // 2
            y0 = (new_height - height) // 2
            return resized[y0:y0 + height, x0:x0 + width]

        if hasattr(clip, "fl"):
            return clip.fl(zoom, apply_to=[])
        return clip.transform(zoom, apply_to=[])

    # ------------------------------------------------------------ inpainting
    def _inpaint_segment(
        self,
        index: int,
        segment: TimelineSegment,
        source: Path,
        start: float,
        end: float,
    ) -> Optional[Path]:
        inpaint = segment.puter_inpaint
        if inpaint is None or self.puter is None or not self.puter.is_configured:
            if inpaint is not None:
                self.warn("Puter.js key missing - inpainting was skipped.")
            return None

        self.report(
            JobStage.INPAINTING,
            0.20,
            f"Inpainting '{inpaint.target_object or 'region'}' -> '{inpaint.replace_prompt[:40]}'",
        )
        work_dir = self.workspace / f"inpaint_{index:03d}"
        frames_dir = work_dir / "frames"
        edited_dir = work_dir / "edited"
        edited_dir.mkdir(parents=True, exist_ok=True)

        try:
            keyframes: List[ExtractedFrame] = extract_frames(
                source,
                frames_dir,
                sample_fps=settings.inpaint_fps,
                max_frames=settings.inpaint_max_frames_per_segment,
                start=start,
                end=end,
            )
        except Exception as exc:
            self.warn(f"Frame extraction failed for segment {index}: {exc}")
            return None
        if not keyframes:
            self.warn(f"No frames could be extracted for segment {index}.")
            return None

        edited: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for order, frame in enumerate(keyframes):
            try:
                mask_path = build_mask(
                    frame.path,
                    work_dir / "masks" / f"mask_{order:05d}.png",
                    target_object=inpaint.target_object,
                    region=inpaint.mask_region,
                )
                output = self.puter.inpaint_image(
                    frame.path,
                    mask_path,
                    inpaint.replace_prompt,
                    edited_dir / f"edit_{order:05d}.png",
                    strength=inpaint.strength,
                )
            except Exception as exc:
                self.warn(f"Puter inpainting failed on frame {order} of segment {index}: {exc}")
                continue

            edited_image = cv2.imread(str(output), cv2.IMREAD_COLOR)
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if edited_image is None or mask_image is None:
                continue
            soft_mask = cv2.GaussianBlur(mask_image, (31, 31), 0).astype(np.float32) / 255.0
            edited.append((frame.timestamp, edited_image, soft_mask))

        if not edited:
            self.warn(f"Segment {index}: no frame survived inpainting, using the original footage.")
            return None

        return self._composite_inpainted_segment(index, source, start, end, edited)

    def _composite_inpainted_segment(
        self,
        index: int,
        source: Path,
        start: float,
        end: float,
        edited: List[Tuple[float, np.ndarray, np.ndarray]],
    ) -> Optional[Path]:
        """Blend the repainted key frames back over every frame of the window."""
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            self.warn(f"OpenCV could not reopen {source.name} for inpaint compositing.")
            return None

        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                return None
            width -= width % 2
            height -= height % 2

            timestamps = np.array([item[0] for item in edited], dtype=np.float32)
            silent = self.workspace / f"inpaint_{index:03d}_silent.mp4"
            capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

            written = 0
            max_frames = int(math.ceil((end - start) * source_fps)) + 2
            with FfmpegFrameWriter(silent, width, height, source_fps) as writer:
                while written < max_frames:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if position > end + (1.0 / source_fps):
                        break
                    frame = frame[:height, :width]

                    nearest = int(np.argmin(np.abs(timestamps - position)))
                    _, edited_frame, soft_mask = edited[nearest]
                    if edited_frame.shape[:2] != (height, width):
                        edited_frame = cv2.resize(edited_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    if soft_mask.shape[:2] != (height, width):
                        soft_mask = cv2.resize(soft_mask, (width, height), interpolation=cv2.INTER_LINEAR)

                    alpha = soft_mask[:, :, None]
                    blended = frame.astype(np.float32) * (1.0 - alpha) + edited_frame.astype(np.float32) * alpha
                    writer.write(np.clip(blended, 0, 255).astype(np.uint8))
                    written += 1
        finally:
            capture.release()

        if written == 0:
            return None

        # Re-attach the original audio for this window.
        final = self.workspace / f"inpaint_{index:03d}.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(silent),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", settings.audio_bitrate,
                    # No -shortest: the extracted audio is often a few
                    # milliseconds shorter than the repainted picture and would
                    # truncate the last video frames of the segment.
                    str(final),
                ],
                check=True, capture_output=True,
            )
            return final
        except subprocess.CalledProcessError:
            return silent

    # ------------------------------------------------------------- compositing
    def _compose_final(
        self,
        base_video: Path,
        voiceover: Optional[Path],
        stickers: Dict[int, Path],
        output_path: Path,
    ) -> Path:
        self.report(JobStage.RENDERING, 0.66, "Building the overlay composite")
        base = self._track(VideoFileClip(str(base_video)))
        base = _with_fps(base, self.fps)
        duration = base.duration or 0.0

        audio_clip, audio_duration = self._build_audio(base, voiceover, duration)
        if audio_duration > duration + 0.25:
            base = self._extend_to(base, audio_duration)
            duration = audio_duration

        layers = [base]
        layers.extend(self._sticker_layers(stickers, duration))
        layers.extend(self._text_layers(duration))

        caption_source = self._export_audio_for_captions(audio_clip, voiceover)
        if self.plan.captions.enabled and settings.enable_captions and caption_source:
            self.report(JobStage.CAPTIONS, 0.74, "Transcribing word level captions with Faster-Whisper")
            words = transcribe_words(caption_source)
            if words:
                layers.extend(self._caption_layers(words, duration))
            else:
                self.warn("No captions were produced (silent audio or Whisper unavailable).")

        composite = CompositeVideoClip(layers, size=(self.width, self.height))
        composite = _with_duration(composite, duration)
        composite = _with_fps(composite, self.fps)
        composite = self._apply_theme_look(composite, duration)
        if audio_clip is not None:
            composite = _with_audio(composite, audio_clip)
        self._track(composite)

        self.report(JobStage.ENCODING, 0.82, f"Encoding {self.width}x{self.height} @ {self.fps}fps")
        self._write_video(composite, output_path, with_audio=True)
        self.report(JobStage.ENCODING, 0.98, "Encode finished")
        return output_path

    def _theme_hits(self, duration: float) -> List[float]:
        """Timestamps in OUTPUT time where the theme should kick.

        ``impacts`` fires on every cut - which is what an anime edit does -
        while ``beats`` remaps the source-clip beat onsets of each segment into
        the re-cut timeline.
        """
        mode = self.theme.shake_on
        offsets = self._segment_offsets()
        if mode == "impacts":
            return [start for start, _length in offsets if 0.05 < start < duration]

        if mode == "beats":
            beats_by_clip: Dict[int, List[float]] = {}
            for index, analysis in enumerate(self.analyses):
                beats_by_clip[index] = list(getattr(analysis, "beats", None) or [])
            if not any(beats_by_clip.values()):
                return [start for start, _length in offsets if 0.05 < start < duration]

            hits: List[float] = []
            for position, segment in enumerate(self.plan.edit_timeline):
                if position >= len(offsets):
                    break
                start_out, length = offsets[position]
                source = segment.source_index or 0
                speed = segment.speed or 1.0
                for beat in beats_by_clip.get(source, []):
                    if segment.start_seconds <= beat <= segment.end_seconds:
                        mapped = start_out + (beat - segment.start_seconds) / speed
                        if 0.05 < mapped < duration:
                            hits.append(round(mapped, 3))
            return sorted(hits)

        return []

    def _apply_theme_look(self, clip, duration: float):
        """Grade, bloom, vignette and shake, as one per-frame pass."""
        hits = self._theme_hits(duration)
        shake = self.theme.shake_spec(hits, width=self.width)
        effect = build_effect_chain(
            shake=shake,
            grade=self.theme.grade or None,
            vignette_strength=self.theme.vignette,
            bloom=self.theme.bloom,
        )
        if effect is None:
            return clip
        self.report(
            JobStage.RENDERING, 0.78,
            f"Applying '{self.theme.name}' look"
            + (f" ({self.theme.shake_kind} shake on {len(hits)} hits)" if shake else ""),
        )
        return apply_frame_effect(clip, effect)

    def _extend_to(self, clip, target_duration: float):
        """Hold the last frame so a long voiceover is never cut off."""
        extra = target_duration - (clip.duration or 0.0)
        if extra <= 0.05:
            return clip
        last_time = max((clip.duration or 0.0) - (1.0 / max(self.fps, 1)), 0.0)
        frame = clip.get_frame(last_time)
        tail = _with_duration(ImageClip(frame), extra)
        tail = _with_fps(tail, self.fps)
        extended = concatenate_videoclips([clip, tail], method="compose")
        return self._track(_with_fps(extended, self.fps))

    def _build_audio(self, base, voiceover: Optional[Path], duration: float):
        tracks = []
        keep_original = self.plan.audio.keep_original_audio
        gain = self.plan.audio.background_music_gain
        if gain is None:
            gain = settings.background_audio_gain if voiceover else 1.0

        if base.audio is not None and keep_original:
            tracks.append(_volume(base.audio, max(0.0, float(gain))))

        audio_duration = duration
        if voiceover and Path(voiceover).exists():
            narration = self._track(AudioFileClip(str(voiceover)))
            narration = _volume(narration, settings.tts_audio_gain)
            tracks.append(narration)
            audio_duration = max(duration, narration.duration or duration)

        if not tracks:
            return None, duration
        if len(tracks) == 1:
            return _with_duration(tracks[0], audio_duration), audio_duration
        return _with_duration(CompositeAudioClip(tracks), audio_duration), audio_duration

    def _export_audio_for_captions(self, audio_clip, voiceover: Optional[Path]) -> Optional[Path]:
        if voiceover and Path(voiceover).exists():
            return Path(voiceover)  # clean speech transcribes far better
        if audio_clip is None:
            return None
        destination = self.workspace / "caption_source.wav"
        try:
            audio_clip.write_audiofile(
                str(destination), fps=16000, nbytes=2, codec="pcm_s16le", logger=None
            )
            return destination
        except Exception as exc:
            self.warn(f"Could not export audio for captioning: {exc}")
            return None

    # --------------------------------------------------------------- layers
    def _segment_offsets(self) -> List[Tuple[float, float]]:
        offsets: List[Tuple[float, float]] = []
        cursor = 0.0
        for segment in self.plan.edit_timeline:
            length = segment.duration / (segment.speed or 1.0)
            offsets.append((cursor, length))
            cursor += length
        return offsets

    def _sticker_layers(self, stickers: Dict[int, Path], duration: float) -> List[Any]:
        if not stickers:
            return []
        layers: List[Any] = []
        offsets = self._segment_offsets()
        for index, path in sorted(stickers.items()):
            if index >= len(offsets):
                continue
            segment = self.plan.edit_timeline[index]
            sticker = segment.puter_sticker
            if sticker is None:
                continue
            start, length = offsets[index]
            start = min(start + max(sticker.delay, 0.0), max(duration - 0.5, 0.0))
            show_for = sticker.duration or max(min(length, 4.0), 1.0)
            show_for = min(show_for, max(duration - start, 0.3))
            if show_for <= 0.1:
                continue
            layer = self._sticker_clip(path, sticker, start, show_for)
            if layer is not None:
                layers.append(layer)
        return layers

    def _sticker_clip(self, path: Path, sticker: PuterSticker, start: float, duration: float):
        try:
            with Image.open(path) as image:
                rgba = np.array(image.convert("RGBA"))
        except Exception as exc:
            self.warn(f"Could not load sticker {path.name}: {exc}")
            return None

        scale = max(sticker.scale, 0.2) * self.theme.sticker_scale
        target_width = int(self.width * settings.sticker_max_width_ratio * scale)
        target_width = max(120, min(target_width, self.width))
        ratio = target_width / rgba.shape[1]
        target_height = max(60, int(rgba.shape[0] * ratio))
        resized = cv2.resize(rgba, (target_width, target_height), interpolation=cv2.INTER_AREA)

        clip = _clip_from_rgba(resized)
        clip = _with_duration(clip, duration)
        clip = _with_start(clip, start)
        clip = _with_fps(clip, self.fps)
        animation = sticker.animation or self.theme.sticker_animation
        return self._animate(clip, animation, sticker.position, duration,
                             (target_width, target_height))

    def _text_layers(self, duration: float) -> List[Any]:
        layers: List[Any] = []
        offsets = self._segment_offsets()
        for index, segment in enumerate(self.plan.edit_timeline):
            overlay = segment.text_overlay
            if overlay is None or not overlay.active or index >= len(offsets):
                continue
            start, length = offsets[index]
            start = min(start + max(overlay.delay, 0.0), max(duration - 0.4, 0.0))
            show_for = overlay.duration or max(min(length, 3.5), 1.0)
            show_for = min(show_for, max(duration - start, 0.3))
            if show_for <= 0.1:
                continue
            layer = self._text_clip(overlay, start, show_for)
            if layer is not None:
                layers.append(layer)
        return layers

    def _text_clip(self, overlay: TextOverlay, start: float, duration: float):
        try:
            array = render_text_rgba(
                overlay.text.upper(),
                max_width=int(self.width * 0.86),
                font_size=int(self.height * 0.052),
                color=overlay.color or "#FFFFFF",
                three_d=("3d" in (overlay.style or "").lower()),
            )
        except Exception as exc:
            self.warn(f"Could not render the text overlay '{overlay.text[:30]}': {exc}")
            return None

        clip = _clip_from_rgba(array)
        clip = _with_duration(clip, duration)
        clip = _with_start(clip, start)
        clip = _with_fps(clip, self.fps)
        return self._animate(clip, overlay.animation, overlay.position, duration,
                             (array.shape[1], array.shape[0]))

    def _caption_layers(self, words: List[Any], duration: float) -> List[Any]:
        style = self.plan.captions
        anchor_x, anchor_y = anchor_for(style.position or "bottom_center")
        # Lift the caption band clear of a bottom anchored sticker.
        if anchor_y > 0.7:
            anchor_y = 0.72
        font_size = int(self.height * 0.046)
        layers: List[Any] = []

        for word in words:
            start = max(0.0, float(word.start))
            if start >= duration:
                break
            end = min(float(word.end), duration)
            show_for = max(end - start, 0.18)
            text = word.text.strip().upper()
            if not text:
                continue
            try:
                array = render_text_rgba(
                    text,
                    max_width=int(self.width * 0.9),
                    font_size=font_size,
                    color=style.highlight_color or "#FFD400",
                    three_d=False,
                )
            except Exception:
                continue

            clip = _clip_from_rgba(array)
            clip = _with_duration(clip, show_for)
            clip = _with_start(clip, start)
            clip = _with_fps(clip, self.fps)

            width, height = array.shape[1], array.shape[0]
            x = int(self.width * anchor_x - width / 2)
            y = int(self.height * anchor_y - height / 2)
            pop = min(0.12, show_for * 0.4)

            def position(t, x=x, y=y, pop=pop):
                if t < pop and pop > 0:
                    lift = int(18 * (1 - t / pop))
                    return (x, y + lift)
                return (x, y)

            layers.append(_with_position(clip, position))
        return layers

    def _animate(self, clip, animation: str, position: str, duration: float, size: Tuple[int, int]):
        width, height = size
        anchor_x, anchor_y = anchor_for(position)
        base_x = int(self.width * anchor_x - width / 2)
        base_y = int(self.height * anchor_y - height / 2)
        base_x = max(min(base_x, self.width - width), 0) if width < self.width else base_x
        base_y = max(min(base_y, self.height - height), 0) if height < self.height else base_y

        name = (animation or "").strip().lower()
        intro = min(0.45, duration * 0.5)

        if name == StickerAnimation.SLIDE_UP.value:
            def position_fn(t):
                if t >= intro:
                    return (base_x, base_y)
                travel = int(self.height * 0.25 * (1 - t / intro))
                return (base_x, base_y + travel)
            return _with_position(clip, position_fn)

        if name == StickerAnimation.SLIDE_DOWN.value:
            def position_fn(t):
                if t >= intro:
                    return (base_x, base_y)
                travel = int(self.height * 0.25 * (1 - t / intro))
                return (base_x, base_y - travel)
            return _with_position(clip, position_fn)

        if name == StickerAnimation.SHAKE.value:
            def position_fn(t):
                offset = int(12 * math.sin(t * 22))
                return (base_x + offset, base_y)
            return _with_position(clip, position_fn)

        if name == StickerAnimation.FADE_IN.value:
            faded = _crossfadein(clip, intro)
            return _with_position(faded, (base_x, base_y))

        if name in (StickerAnimation.POP_UP.value, StickerAnimation.ZOOM_OUT.value, ""):
            # Ease-out-back scale, applied around the anchor point.
            zoom_out = name == StickerAnimation.ZOOM_OUT.value

            def scale_at(t: float) -> float:
                if t >= intro or intro <= 0:
                    return 1.0
                progress = t / intro
                if zoom_out:
                    return 1.35 - 0.35 * progress
                overshoot = 1.70158
                p = progress - 1
                return 1 + (p * p * ((overshoot + 1) * p + overshoot))

            def resizer(t):
                return max(0.12, scale_at(t))

            def position_fn(t):
                scale = max(0.12, scale_at(t))
                return (
                    int(self.width * anchor_x - width * scale / 2),
                    int(self.height * anchor_y - height * scale / 2),
                )

            try:
                scaled = _resize(clip, resizer)
                return _with_position(scaled, position_fn)
            except Exception:
                return _with_position(clip, (base_x, base_y))

        return _with_position(clip, (base_x, base_y))

    # ------------------------------------------------------------- encoding
    def _write_video(self, clip, destination: Path, with_audio: bool) -> None:
        destination = Path(destination)
        temp_audio = self.workspace / f"{destination.stem}-audio.m4a"
        ffmpeg_params = [
            "-pix_fmt", "yuv420p",
            "-crf", str(settings.x264_crf),
            "-profile:v", "high",
            "-level", "4.2",
            "-movflags", "+faststart",
        ]
        kwargs: Dict[str, Any] = {
            "fps": self.fps,
            "codec": "libx264",
            "preset": settings.x264_preset,
            "bitrate": settings.video_bitrate,
            "ffmpeg_params": ffmpeg_params,
            "logger": None,
            "temp_audiofile": str(temp_audio),
            "remove_temp": True,
        }
        if settings.ffmpeg_threads:
            kwargs["threads"] = settings.ffmpeg_threads
        if with_audio and clip.audio is not None:
            kwargs["audio_codec"] = "aac"
            kwargs["audio_bitrate"] = settings.audio_bitrate
        else:
            kwargs["audio"] = False

        clip.write_videofile(str(destination), **kwargs)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def probe_clip(path: Path) -> Dict[str, Any]:
    """Duration / geometry / audio presence via ffprobe, OpenCV as a fallback."""
    path = Path(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            import json as _json

            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            data = _json.loads(result.stdout)
            streams = data.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), None)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            duration = float(data.get("format", {}).get("duration") or 0.0)
            if video:
                fps = _parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
                return {
                    "duration": duration or float(video.get("duration") or 0.0),
                    "width": int(video.get("width") or 0),
                    "height": int(video.get("height") or 0),
                    "fps": fps,
                    "has_audio": has_audio,
                }
        except Exception as exc:
            logger.debug("ffprobe failed for %s: %s", path, exc)

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": False}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return {
            "duration": (frames / fps) if fps else 0.0,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(fps),
            # OpenCV cannot see audio streams, so ask ffmpeg instead.
            "has_audio": _has_audio_stream(path),
        }
    finally:
        capture.release()


def _has_audio_stream(path: Path) -> bool:
    """Detect an audio track with ffmpeg when ffprobe is not installed."""
    try:
        result = subprocess.run(
            [ffmpeg_binary(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True,
        )
    except Exception:
        return False
    return "Audio:" in (result.stderr or "")


def _parse_fraction(value: str) -> float:
    try:
        if "/" in value:
            numerator, denominator = value.split("/")
            denominator = float(denominator)
            return float(numerator) / denominator if denominator else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
