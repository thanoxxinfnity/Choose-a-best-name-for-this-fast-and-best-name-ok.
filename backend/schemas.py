"""Pydantic models for the Kimi K3 editing timeline and the public REST API.

The `EditPlan` model is a superset of the JSON contract declared in the
orchestrator system prompt: every field the model is *required* to emit is
mandatory here, everything the model *may* emit is optional with a sane
default, so a slightly creative LLM response never crashes the renderer.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

_TIMECODE_RE = re.compile(
    r"^(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?$"
)


def parse_timecode(value: Any) -> float:
    """Convert `"00:00:05"`, `"01:02.500"`, `12`, `"12.5"` -> seconds (float)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return 0.0
    match = _TIMECODE_RE.match(text)
    if match:
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
        millis = int((match.group("ms") or "0").ljust(3, "0"))
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    try:
        return max(0.0, float(text))
    except ValueError:
        return 0.0


def format_timecode(seconds: float, keep_millis: bool = True) -> str:
    """Seconds -> ``HH:MM:SS`` (or ``HH:MM:SS.mmm`` when it would lose precision).

    Voiceover lines are pinned to sub-second positions, so truncating to whole
    seconds here would silently drag every line up to 999ms out of sync.
    """
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:  # rounded up into the next second
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if keep_millis and millis:
        return f"{base}.{millis:03d}"
    return base


# ---------------------------------------------------------------------------
# Enums (kept permissive: unknown values degrade to a default at render time)
# ---------------------------------------------------------------------------


class StickerPosition(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class StickerAnimation(str, Enum):
    POP_UP = "pop_up"
    FADE_IN = "fade_in"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ZOOM_OUT = "zoom_out"
    SHAKE = "shake"
    NONE = "none"


class CutType(str, Enum):
    JUMP_CUT = "jump_cut"
    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"
    SPEED_RAMP = "speed_ramp"
    ZOOM_PUNCH = "zoom_punch"


# ---------------------------------------------------------------------------
# Timeline models
# ---------------------------------------------------------------------------


class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    resolution: str = "1080x1920"
    fps: int = 60

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: str) -> str:
        if not re.match(r"^\d{2,5}x\d{2,5}$", str(value)):
            return "1080x1920"
        return str(value)

    @field_validator("fps", mode="before")
    @classmethod
    def _validate_fps(cls, value: Any) -> int:
        try:
            fps = int(float(value))
        except (TypeError, ValueError):
            return 60
        return fps if 1 <= fps <= 240 else 60

    @property
    def size(self) -> tuple[int, int]:
        width, height = self.resolution.lower().split("x")
        return int(width), int(height)


class TtsLine(BaseModel):
    """One spoken line, placed at an exact point in the finished edit."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    # Where the line starts in the FINAL timeline (not in a source clip).
    start_time: str = "00:00:00"
    # Optional per-line voice override, e.g. a deep narrator for the hook and a
    # hype voice for the payoff.
    voice: Optional[str] = None
    gain: float = 1.0

    @field_validator("start_time", mode="before")
    @classmethod
    def _coerce_start(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            return format_timecode(float(value))
        return str(value)

    @property
    def start_seconds(self) -> float:
        return parse_timecode(self.start_time)

    @property
    def active(self) -> bool:
        return bool(self.text.strip())


class AudioSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    use_puter_tts: bool = False
    # Voice profile key: indian_accent, indian_accent_male, hinglish,
    # deep_dark, deep_dark_female, horror_whisper, hype, ...
    voice_accent: str = "indian_accent"
    # A single block of narration, laid over the whole edit.
    tts_script: str = ""
    # Or individual lines pinned to timeline positions. When both are given the
    # lines win, because they carry timing.
    tts_lines: List[TtsLine] = Field(default_factory=list)
    keep_original_audio: bool = True
    background_music_gain: Optional[float] = None
    # Duck the bed further while a line is actually speaking.
    duck_under_voice: float = 0.08

    @property
    def timed_lines(self) -> List[TtsLine]:
        return [line for line in self.tts_lines if line.active]

    @property
    def has_voiceover(self) -> bool:
        return bool(self.use_puter_tts and (self.tts_script.strip() or self.timed_lines))


class PuterSticker(BaseModel):
    model_config = ConfigDict(extra="allow")

    generate_prompt: str = ""
    position: str = StickerPosition.BOTTOM_CENTER.value
    animation: str = StickerAnimation.POP_UP.value
    scale: float = 1.0
    # Optional offsets relative to the segment start, in seconds.
    delay: float = 0.0
    duration: Optional[float] = None

    @property
    def active(self) -> bool:
        return bool(self.generate_prompt.strip())


class PuterInpaint(BaseModel):
    model_config = ConfigDict(extra="allow")

    active: bool = False
    target_object: str = ""
    replace_prompt: str = ""
    # Region of the frame to mask when the target object cannot be segmented
    # automatically: sky/top, ground/bottom, center, left, right, full.
    mask_region: Optional[str] = None
    strength: float = 0.85

    @property
    def is_enabled(self) -> bool:
        return bool(self.active and self.replace_prompt.strip())


class PuterAnimate(BaseModel):
    """AI keyframe-to-animation insertion (Puter image-to-video)."""

    model_config = ConfigDict(extra="allow")

    active: bool = False
    # Timestamp inside the SOURCE clip to lift the still from. None -> the
    # midpoint of the segment.
    keyframe_time: Optional[float] = None
    # A second keyframe may be animated for the same segment (1-2 per spec).
    second_keyframe_time: Optional[float] = None
    prompt: str = ""
    story_context: str = ""
    seconds: float = 4.0
    motion_strength: float = 0.7
    # replace: the generated clip stands in for the segment.
    # insert:  it is spliced in directly after the segment.
    mode: str = "insert"

    @property
    def is_enabled(self) -> bool:
        return bool(self.active and self.prompt.strip())


class TextOverlay(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = ""
    style: str = "3d_pop"
    position: str = StickerPosition.CENTER.value
    animation: str = StickerAnimation.POP_UP.value
    color: str = "#FFFFFF"
    delay: float = 0.0
    duration: Optional[float] = None

    @property
    def active(self) -> bool:
        return bool(self.text.strip())


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="allow")

    start_time: str = "00:00:00"
    end_time: str = "00:00:05"
    cut_type: str = CutType.JUMP_CUT.value
    # Which uploaded clip this segment is taken from (0 based).  Kimi may omit
    # it, in which case the renderer round robins across the uploads.
    source_index: Optional[int] = None
    speed: float = 1.0
    puter_sticker: Optional[PuterSticker] = None
    puter_inpaint: Optional[PuterInpaint] = None
    puter_animate: Optional[PuterAnimate] = None
    text_overlay: Optional[TextOverlay] = None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _coerce_timecode(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            return format_timecode(float(value))
        return str(value)

    @field_validator("speed", mode="before")
    @classmethod
    def _coerce_speed(cls, value: Any) -> float:
        try:
            speed = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(speed) or speed <= 0:
            return 1.0
        return min(max(speed, 0.25), 4.0)

    @property
    def start_seconds(self) -> float:
        return parse_timecode(self.start_time)

    @property
    def end_seconds(self) -> float:
        return parse_timecode(self.end_time)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


class GeneratedSequence(BaseModel):
    """An AI generated intro or outro spliced onto the edit."""

    model_config = ConfigDict(extra="allow")

    active: bool = False
    prompt: str = ""
    seconds: float = 3.0
    # "t2v"  -> generate from the prompt alone
    # "i2v"  -> animate a frame of the edit (first frame for an intro, last for
    #           an outro) so the sequence matches the footage
    mode: str = "i2v"
    motion_strength: float = 0.7
    # Optional title burnt over the generated sequence.
    text: str = ""
    text_animation: str = "pop_up"

    @property
    def is_enabled(self) -> bool:
        return bool(self.active and (self.prompt.strip() or self.text.strip()))


class CaptionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    style: str = "word_pop"
    color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    position: str = "bottom_center"
    max_words_on_screen: int = 3


class EditPlan(BaseModel):
    """The structured editing timeline produced by Kimi K3."""

    model_config = ConfigDict(extra="allow")

    project_meta: ProjectMeta = Field(default_factory=ProjectMeta)
    audio: AudioSpec = Field(default_factory=AudioSpec)
    edit_timeline: List[TimelineSegment] = Field(default_factory=list)
    captions: CaptionSpec = Field(default_factory=CaptionSpec)
    intro: GeneratedSequence = Field(default_factory=GeneratedSequence)
    outro: GeneratedSequence = Field(default_factory=GeneratedSequence)
    theme: str = "normal"
    notes: Optional[str] = None

    @property
    def total_duration(self) -> float:
        return sum(segment.duration for segment in self.edit_timeline)

    def sticker_prompts(self) -> List[str]:
        prompts: List[str] = []
        for segment in self.edit_timeline:
            if segment.puter_sticker and segment.puter_sticker.active:
                prompts.append(segment.puter_sticker.generate_prompt)
        return prompts


# ---------------------------------------------------------------------------
# REST API models
# ---------------------------------------------------------------------------


class JobStage(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    ANIMATING = "animating"
    PLANNING = "planning"
    TTS = "tts"
    STICKERS = "stickers"
    INPAINTING = "inpainting"
    RENDERING = "rendering"
    CAPTIONS = "captions"
    ENCODING = "encoding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipInfo(BaseModel):
    filename: str
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False


class JobStatus(BaseModel):
    job_id: str
    stage: JobStage = JobStage.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    finished_at: Optional[float] = None
    error: Optional[str] = None
    prompt: str = ""
    youtube_url: Optional[str] = None
    theme: str = "normal"
    clips: List[ClipInfo] = Field(default_factory=list)
    analysis: List[Dict[str, Any]] = Field(default_factory=list)
    plan: Optional[EditPlan] = None
    export_preset: str = "1080p60"
    output_filename: Optional[str] = None
    output_size_bytes: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    output_fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.stage in (JobStage.COMPLETED, JobStage.FAILED, JobStage.CANCELLED)


class JobCreatedResponse(BaseModel):
    job_id: str
    stage: JobStage
    status_url: str
    message: str = "Job accepted"


class PlanRequest(BaseModel):
    prompt: str
    youtube_url: Optional[str] = None
    clip_durations: List[float] = Field(default_factory=list)
    target_duration: Optional[float] = None


class TtsRequest(BaseModel):
    text: str
    accent: str = "indian_accent"


class StickerRequest(BaseModel):
    prompt: str
    remove_background: bool = True


class VoiceInfo(BaseModel):
    key: str
    label: str
    voice_id: str
    language: str
    deep: bool = False


class VideoProviderInfo(BaseModel):
    key: str
    label: str
    text_to_video: bool
    image_to_video: bool
    models: List[str] = Field(default_factory=list)
    requires_key: bool = True
    configured: bool = False
    notes: str = ""


class ExportPresetInfo(BaseModel):
    key: str
    label: str
    width: int
    height: int
    fps: int
    codec: str
    bitrate: str
    vertical: bool
    relative_cost: float


class ThemeInfo(BaseModel):
    key: str
    name: str
    description: str
    default_cut: str
    shake: str


class TextToVideoRequest(BaseModel):
    prompt: str
    seconds: float = 5.0
    resolution: str = "720x1280"
    negative_prompt: str = ""
    seed: Optional[int] = None
    provider: Optional[str] = None


class ImageToVideoRequest(BaseModel):
    prompt: str = ""
    seconds: float = 5.0
    motion_strength: float = 0.7
    negative_prompt: str = ""
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    ffmpeg: bool
    rembg: bool
    whisper: bool
    puter_configured: bool
    nim_configured: bool
    active_jobs: int
    details: Dict[str, Any] = Field(default_factory=dict)
