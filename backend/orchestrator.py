"""AI orchestrator: NVIDIA NIM (Kimi K3) -> structured editing timeline.

The user prompt, the metadata of the uploaded clips and an optional YouTube
reference video are handed to Kimi K3 through the NVIDIA NIM OpenAI compatible
chat completions endpoint.  Kimi answers with the strict JSON timeline defined
in :mod:`schemas` (jump cuts, Puter TTS timings, Puter stickers, Puter
inpainting instructions).

If NIM is unavailable - no key, model not enabled on the account, malformed
answer - :func:`build_fallback_plan` produces a deterministic timeline so the
rendering pipeline always has something valid to work with.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import requests

from config import settings
from schemas import (
    AudioSpec,
    CaptionSpec,
    ClipInfo,
    EditPlan,
    ProjectMeta,
    PuterInpaint,
    PuterSticker,
    TextOverlay,
    TimelineSegment,
    format_timecode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt - verbatim contract from the blueprint
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """[ROLE]
You are a Master AI Video Editor. Your job is to analyze raw clips, a user prompt, and an optional YouTube reference URL. You control FFmpeg, Puter.js (for TTS, Stickers, Inpainting), and Whisper.

[PIPELINE INSTRUCTIONS]
1. AUDIO & TTS (Puter.js): If the user wants an AI voiceover, output instructions to use Puter.js TTS with an "Indian Accent".
2. STICKERS (Puter.js txt2img): Identify exciting moments in the script. Instruct the system to generate custom stickers, remove their background (using rembg), and overlay them using MoviePy.
3. FRAME INPAINTING (Puter.js Image-to-Image): If the user requests editing an object in the video or matching a reference video background, specify exact timestamps for frame extraction and Puter.js inpainting.
4. CUTS & TEXT: Make fast-paced jump cuts and place 3D animated text overlays.

[OUTPUT FORMAT]
You MUST output strictly in this JSON format. No extra text.

{
  "project_meta": { "resolution": "1080x1920", "fps": 60 },
  "audio": {
    "use_puter_tts": true,
    "voice_accent": "indian_accent",
    "tts_script": "Doston, aaj hum dekhenge..."
  },
  "edit_timeline": [
    {
      "start_time": "00:00:00",
      "end_time": "00:00:05",
      "cut_type": "jump_cut",
      "puter_sticker": {
        "generate_prompt": "3D glowing subscribe button",
        "position": "bottom_center",
        "animation": "pop_up"
      },
      "puter_inpaint": {
        "active": true,
        "target_object": "sky",
        "replace_prompt": "dark stormy sky with lightning"
      }
    }
  ]
}"""

# Extra machine readable constraints appended to the user turn so the plan can
# actually be rendered (clip indices, legal enum values, duration budget).
PLAN_CONSTRAINTS = """
[HARD CONSTRAINTS - the renderer rejects anything else]
- start_time / end_time use "HH:MM:SS" and are timestamps INSIDE the source clip named by "source_index".
- "source_index" is a 0 based index into the uploaded clips listed above and is REQUIRED on every segment.
- Every segment must be between 0.8 and 8 seconds long. Produce fast paced jump cuts.
- Allowed "cut_type": jump_cut, hard_cut, crossfade, speed_ramp, zoom_punch.
- Allowed sticker "position": top_left, top_center, top_right, center_left, center, center_right, bottom_left, bottom_center, bottom_right.
- Allowed sticker "animation": pop_up, fade_in, slide_up, slide_down, zoom_out, shake, none.
- Use at most {max_stickers} stickers and at most {max_inpaints} inpainted segments in the whole timeline (they are expensive).
- Omit "puter_sticker" / "puter_inpaint" entirely on segments that do not need them.
- AUDIO: set "audio.use_puter_tts" true when narration helps. Choose "audio.voice_accent" from: indian_accent, indian_accent_male, hinglish, deep_dark (deep dark mysterious narrator), deep_dark_female, horror_whisper, hype.
- Prefer "audio.tts_lines" over a single "tts_script": a list of {{"text": "...", "start_time": "00:00:02.500", "voice": "deep_dark"}} pinned to moments in the FINAL edit, so each line lands on the beat it belongs to. Give each line 2-6 seconds of room and never overlap two lines. "voice" is optional and overrides the default for that line.
- Any "tts_script" you emit instead must be natural Indian English / Hinglish and readable in about {tts_seconds} seconds ({tts_words} words maximum).
- INTRO / OUTRO: you may set "intro" and "outro" to {{"active": true, "prompt": "...", "seconds": 2-4, "mode": "i2v" or "t2v", "text": "SHORT TITLE"}}. "i2v" animates a real frame of this footage (use it to stay on-style); "t2v" invents a shot from the prompt alone.
- KEYFRAME ANIMATION: on at most {max_animations} segment(s) you may add "puter_animate": {{"active": true, "prompt": "what should move", "story_context": "what is happening around it", "keyframe_time": <seconds inside the source clip>, "second_keyframe_time": <optional>, "seconds": 2-4, "mode": "insert" or "replace"}}. Use it on the most dramatic moment, not on filler.
- You may add an optional "text_overlay": {{"text": "...", "position": "...", "animation": "...", "style": "3d_pop"}} per segment.
- The total timeline should be close to {target_duration} seconds.
- Output raw JSON only. No markdown fences, no commentary.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/|/v/)([A-Za-z0-9_-]{11})")

WORDS_PER_SECOND = 2.6  # comfortable Indian English narration pace

# NIM rate limits are enforced per minute, so short exponential backoff never
# clears them; these waits (seconds) do.
RATE_LIMIT_BACKOFF = (5, 15, 30, 60, 60)


_PINNED_PARAM_RE = re.compile(
    r"`?(?P<name>[a-z_]+)`?\s+is immutable[^.]*?must be\s+(?P<value>-?\d+(?:\.\d+)?)",
    re.I,
)


def _pinned_parameter(body: str) -> Optional[tuple[str, float]]:
    """Extract a sampling parameter NIM refuses to let us change."""
    match = _PINNED_PARAM_RE.search(body or "")
    if not match:
        return None
    try:
        return match.group("name"), float(match.group("value"))
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(1.0, min(float(raw), 120.0))
    except (TypeError, ValueError):
        return None


class OrchestratorError(RuntimeError):
    """Raised when the plan cannot be produced at all."""


class NimRateLimited(OrchestratorError):
    """NIM kept answering 429 - the account's request budget is exhausted.

    This is explicitly *not* a reason to fall back to another model id: the
    same limit applies there, and a second model burns another request.
    """


# ---------------------------------------------------------------------------
# YouTube reference ingestion
# ---------------------------------------------------------------------------


@dataclass
class YouTubeReference:
    url: str
    video_id: Optional[str] = None
    title: str = ""
    channel: str = ""
    description: str = ""
    duration: float = 0.0
    tags: List[str] = field(default_factory=list)
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "none"

    def to_prompt_block(self) -> str:
        if self.source == "none":
            return f"Reference URL (metadata unavailable): {self.url}"
        lines = [
            f"Reference video: {self.title or 'unknown title'}",
            f"Channel: {self.channel or 'unknown'}",
            f"Duration: {self.duration:.0f}s" if self.duration else "Duration: unknown",
        ]
        if self.tags:
            lines.append("Tags: " + ", ".join(self.tags[:15]))
        if self.chapters:
            chapter_text = "; ".join(
                f"{format_timecode(chapter.get('start_time', 0))} {chapter.get('title', '')}"
                for chapter in self.chapters[:12]
            )
            lines.append(f"Chapters: {chapter_text}")
        if self.description:
            lines.append("Description: " + " ".join(self.description.split())[:900])
        return "\n".join(lines)


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = _YOUTUBE_ID_RE.search(url)
    if match:
        return match.group(1)
    parsed = urlparse(url)
    candidate = parse_qs(parsed.query).get("v", [None])[0]
    if candidate and len(candidate) == 11:
        return candidate
    return None


def fetch_youtube_reference(url: Optional[str], api_token: Optional[str] = None) -> Optional[YouTubeReference]:
    """Best effort metadata lookup: YouTube Data API v3 first, then yt-dlp."""
    if not url or not url.strip():
        return None
    url = url.strip()
    reference = YouTubeReference(url=url, video_id=extract_youtube_id(url))

    token = (api_token or settings.youtube_api_key or "").strip()
    if token and reference.video_id:
        try:
            response = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "id": reference.video_id,
                    "part": "snippet,contentDetails,statistics",
                    "key": token,
                },
                timeout=settings.youtube_timeout,
            )
            if response.status_code == 200:
                items = response.json().get("items") or []
                if items:
                    snippet = items[0].get("snippet", {})
                    reference.title = snippet.get("title", "")
                    reference.channel = snippet.get("channelTitle", "")
                    reference.description = snippet.get("description", "")
                    reference.tags = list(snippet.get("tags") or [])
                    reference.duration = _parse_iso8601_duration(
                        items[0].get("contentDetails", {}).get("duration", "")
                    )
                    reference.source = "youtube_data_api"
                    return reference
            else:
                logger.warning("YouTube Data API returned HTTP %s", response.status_code)
        except requests.RequestException as exc:
            logger.warning("YouTube Data API lookup failed: %s", exc)

    try:
        import yt_dlp

        options = {"quiet": True, "skip_download": True, "no_warnings": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        reference.title = info.get("title") or ""
        reference.channel = info.get("uploader") or info.get("channel") or ""
        reference.description = info.get("description") or ""
        reference.duration = float(info.get("duration") or 0.0)
        reference.tags = list(info.get("tags") or [])[:25]
        reference.chapters = list(info.get("chapters") or [])
        reference.source = "yt_dlp"
    except Exception as exc:  # yt-dlp missing, geo blocked, age gated, ...
        logger.warning("yt-dlp reference lookup failed for %s: %s", url, exc)

    return reference


def _parse_iso8601_duration(value: str) -> float:
    match = re.match(r"^P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", value or "")
    if not match:
        return 0.0
    hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


# ---------------------------------------------------------------------------
# Kimi K3 orchestrator
# ---------------------------------------------------------------------------


class KimiOrchestrator:
    """Calls Kimi K3 on NVIDIA NIM and returns a validated :class:`EditPlan`."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        session: Optional[requests.Session] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = (api_key or settings.nim_api_key or "").strip()
        self.base_url = (base_url or settings.nim_base_url).rstrip("/")
        self.model = model or settings.nim_model
        self.timeout = timeout or settings.nim_timeout
        self.session = session or requests.Session()
        self.max_retries = max(1, max_retries or settings.nim_max_retries)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------ API
    def build_plan(
        self,
        prompt: str,
        clips: Sequence[ClipInfo],
        youtube_reference: Optional[YouTubeReference] = None,
        target_duration: Optional[float] = None,
        max_stickers: int = 4,
        max_inpaints: int = 2,
        analyses: Optional[Sequence[Any]] = None,
        theme: Optional[str] = None,
        max_animations: int = 0,
    ) -> tuple[EditPlan, List[str]]:
        """Return ``(plan, warnings)``; never raises for recoverable failures."""
        warnings: List[str] = []
        available = sum(max(clip.duration, 0.0) for clip in clips)
        target = target_duration or min(max(available * 0.55, 12.0), 75.0)

        if not self.is_configured:
            warnings.append(
                "No NVIDIA NIM API key supplied - used the built-in deterministic editor "
                "instead of Kimi K3."
            )
            return build_fallback_plan(prompt, clips, youtube_reference, target, analyses), warnings

        user_message = self._compose_user_message(
            prompt, clips, youtube_reference, target, max_stickers, max_inpaints,
            analyses=analyses, theme=theme, max_animations=max_animations,
        )

        raw: Optional[str] = None
        deadline = time.monotonic() + settings.nim_plan_budget_seconds
        for model in self._model_candidates():
            if time.monotonic() >= deadline:
                warnings.append(
                    f"Kimi K3 did not answer within "
                    f"{settings.nim_plan_budget_seconds}s - used the deterministic editor."
                )
                break
            try:
                raw = self._chat(model, user_message, deadline=deadline)
                if model != self.model:
                    warnings.append(f"Kimi model '{self.model}' unavailable - used '{model}'.")
                    self.model = model
                break
            except NimRateLimited as exc:
                # A second model id would hit the same account-wide limit.
                logger.warning("NIM rate limited on %s: %s", model, exc)
                warnings.append(str(exc))
                break
            except OrchestratorError as exc:
                logger.warning("NIM model %s failed: %s", model, exc)
                warnings.append(str(exc))

        if not raw:
            warnings.append("Kimi K3 did not return a plan - used the deterministic editor.")
            return build_fallback_plan(prompt, clips, youtube_reference, target, analyses), warnings

        try:
            payload = extract_json(raw)
            plan = EditPlan.model_validate(payload)
        except Exception as exc:
            logger.warning("Kimi returned an unusable timeline: %s", exc)
            warnings.append(f"Kimi returned an unusable timeline ({exc}) - used the deterministic editor.")
            return build_fallback_plan(prompt, clips, youtube_reference, target, analyses), warnings

        plan, sanitise_warnings = sanitise_plan(
            plan, clips, max_stickers, max_inpaints, max_animations
        )
        warnings.extend(sanitise_warnings)
        if not plan.edit_timeline:
            warnings.append("Kimi produced an empty timeline - used the deterministic editor.")
            return build_fallback_plan(prompt, clips, youtube_reference, target, analyses), warnings
        return plan, warnings

    # ------------------------------------------------------------- internals
    def _model_candidates(self) -> List[str]:
        candidates = [self.model]
        if settings.nim_fallback_model and settings.nim_fallback_model not in candidates:
            candidates.append(settings.nim_fallback_model)
        return candidates

    def _chat(self, model: str, user_message: str, deadline: Optional[float] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": settings.nim_temperature,
            "top_p": settings.nim_top_p,
            "max_tokens": settings.nim_max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        rate_limited = False
        for attempt in range(1, self.max_retries + 1):
            remaining = (deadline - time.monotonic()) if deadline else None
            if remaining is not None and remaining <= 1.0:
                raise OrchestratorError(
                    f"NVIDIA NIM ran out of the planning time budget "
                    f"({settings.nim_plan_budget_seconds}s)."
                )
            request_timeout = min(self.timeout, remaining) if remaining else self.timeout
            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=request_timeout
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(min(2 ** attempt, 8))
                continue

            if response.status_code == 400 and "response_format" in response.text:
                # Older NIM deployments reject the JSON mode flag.
                payload.pop("response_format", None)
                continue
            if response.status_code == 400:
                # e.g. "`top_p` is immutable for this model and must be 0.95".
                pinned = _pinned_parameter(response.text)
                if pinned and payload.get(pinned[0]) != pinned[1]:
                    logger.info("NIM pins %s=%s for %s, retrying", pinned[0], pinned[1], model)
                    payload[pinned[0]] = pinned[1]
                    continue
            if response.status_code == 429:
                rate_limited = True
                last_error = f"HTTP 429: {response.text[:200]}"
                delay = _retry_after_seconds(response) or RATE_LIMIT_BACKOFF[
                    min(attempt - 1, len(RATE_LIMIT_BACKOFF) - 1)
                ]
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - time.monotonic()))
                    if delay <= 0.5:
                        raise NimRateLimited(
                            "NVIDIA NIM kept rate limiting this key until the planning "
                            "budget ran out."
                        )
                logger.info(
                    "NIM rate limited on %s, waiting %ss (attempt %s/%s)",
                    model, delay, attempt, self.max_retries,
                )
                time.sleep(delay)
                continue
            if response.status_code in (500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(min(2 ** attempt, 8))
                continue
            if response.status_code >= 400:
                raise OrchestratorError(
                    f"NVIDIA NIM rejected the request (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise OrchestratorError("NVIDIA NIM returned no choices.")
            content = (choices[0].get("message") or {}).get("content") or ""
            if not content.strip():
                raise OrchestratorError("NVIDIA NIM returned an empty message.")
            return content

        if rate_limited:
            raise NimRateLimited(
                "NVIDIA NIM is rate limiting this API key (HTTP 429). Wait a minute "
                "and retry, or use an account with a higher request budget."
            )
        raise OrchestratorError(f"NVIDIA NIM unreachable: {last_error}")

    @staticmethod
    def _compose_user_message(
        prompt: str,
        clips: Sequence[ClipInfo],
        reference: Optional[YouTubeReference],
        target_duration: float,
        max_stickers: int,
        max_inpaints: int,
        analyses: Optional[Sequence[Any]] = None,
        theme: Optional[str] = None,
        max_animations: int = 0,
    ) -> str:
        clip_lines = [
            f"  [{index}] {clip.filename} - {clip.duration:.2f}s, "
            f"{clip.width}x{clip.height} @ {clip.fps:.2f}fps, "
            f"audio={'yes' if clip.has_audio else 'no'}"
            for index, clip in enumerate(clips)
        ] or ["  (no clips uploaded)"]

        tts_seconds = max(6.0, target_duration * 0.9)
        blocks = [
            "[USER PROMPT]",
            prompt.strip() or "Make an engaging fast paced vertical short.",
            "",
            "[UPLOADED CLIPS]",
            "\n".join(clip_lines),
            "",
            f"[TARGET] resolution 1080x1920, 60 fps, about {target_duration:.0f} seconds total.",
        ]
        if analyses:
            blocks += ["", "[WHAT IS ACTUALLY IN THE FOOTAGE - from a vision pass over "
                           "sampled frames plus motion, scene-cut and audio analysis]"]
            for index, analysis in enumerate(analyses):
                blocks.append(f"  clip [{index}]:")
                blocks.append(
                    "\n".join(f"    {line}" for line in analysis.to_prompt_block().splitlines())
                )
            blocks += [
                "",
                "Base every sticker, text overlay and cut on THAT description - it is what "
                "the viewer will actually see. Do not invent a subject that is not listed. "
                "Prefer cutting on the listed scene cuts and landing stickers or text on the "
                "listed high-motion moments and beats.",
            ]
        if theme:
            blocks += ["", f"[THEME PRESET] {theme} - match its look and pacing."]
        if reference is not None:
            blocks += ["", "[YOUTUBE REFERENCE - the style the user wants to match]",
                       reference.to_prompt_block()]
        blocks += [
            "",
            PLAN_CONSTRAINTS.format(
                max_stickers=max_stickers,
                max_inpaints=max_inpaints,
                max_animations=max_animations,
                target_duration=int(target_duration),
                tts_seconds=int(tts_seconds),
                tts_words=int(tts_seconds * WORDS_PER_SECOND),
            ),
        ]
        return "\n".join(blocks)


# ---------------------------------------------------------------------------
# JSON extraction + sanitising
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of an LLM answer (fences, prose, trailing text)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in the response")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("unbalanced JSON object in the response")


def sanitise_plan(
    plan: EditPlan,
    clips: Sequence[ClipInfo],
    max_stickers: int = 4,
    max_inpaints: int = 2,
    max_animations: int = 0,
) -> tuple[EditPlan, List[str]]:
    """Clamp an LLM plan to something the renderer can actually execute."""
    warnings: List[str] = []
    if not clips:
        return plan, ["No clips available for the timeline."]

    plan.project_meta.resolution = settings.resolution
    plan.project_meta.fps = settings.output_fps

    cleaned: List[TimelineSegment] = []
    sticker_count = 0
    inpaint_count = 0
    animation_count = 0

    for position, segment in enumerate(plan.edit_timeline):
        index = segment.source_index if segment.source_index is not None else position % len(clips)
        if not isinstance(index, int) or not 0 <= index < len(clips):
            index = position % len(clips)
        clip = clips[index]
        segment.source_index = index

        start = segment.start_seconds
        end = segment.end_seconds
        duration = clip.duration if clip.duration > 0 else 5.0

        if start >= duration:
            start = max(0.0, (position * 3.0) % max(duration - 1.0, 1.0))
        if end <= start:
            end = start + 3.0
        end = min(end, duration)
        if end - start < 0.8:
            end = min(start + 1.5, duration)
        if end - start < 0.4:
            warnings.append(
                f"Dropped segment {position} - clip {clip.filename} is too short for it."
            )
            continue
        if end - start > 8.0:
            end = start + 8.0

        segment.start_time = format_timecode(start)
        segment.end_time = format_timecode(end)

        if segment.puter_sticker is not None:
            if not segment.puter_sticker.active or sticker_count >= max_stickers:
                segment.puter_sticker = None
            else:
                sticker_count += 1
        if segment.puter_inpaint is not None:
            if not segment.puter_inpaint.is_enabled or inpaint_count >= max_inpaints:
                segment.puter_inpaint = None
            else:
                inpaint_count += 1
        if segment.text_overlay is not None and not segment.text_overlay.active:
            segment.text_overlay = None
        if segment.puter_animate is not None:
            if not segment.puter_animate.is_enabled or animation_count >= max_animations:
                segment.puter_animate = None
            else:
                animation_count += 1
                animate = segment.puter_animate
                animate.seconds = min(max(animate.seconds, 1.0), 6.0)
                if animate.mode not in ("insert", "replace"):
                    animate.mode = "insert"
                for attribute in ("keyframe_time", "second_keyframe_time"):
                    stamp = getattr(animate, attribute)
                    if stamp is not None and not 0.0 <= stamp <= clip.duration:
                        setattr(animate, attribute, None)

        cleaned.append(segment)

    plan.edit_timeline = cleaned

    # Voice lines must land inside the edit and must not stack on each other.
    total = sum(segment.duration / (segment.speed or 1.0) for segment in cleaned)
    lines = sorted(plan.audio.timed_lines, key=lambda line: line.start_seconds)
    kept: List[Any] = []
    previous_end = 0.0
    for line in lines:
        start = max(line.start_seconds, previous_end)
        if total and start >= total:
            warnings.append(f"Dropped voice line past the end of the edit: {line.text[:40]}")
            continue
        line.start_time = format_timecode(start)
        # A rough read-back estimate keeps the next line from overlapping.
        previous_end = start + max(1.2, len(line.text.split()) / WORDS_PER_SECOND)
        kept.append(line)
    plan.audio.tts_lines = kept

    if plan.audio.use_puter_tts and not (plan.audio.tts_script.strip() or kept):
        plan.audio.use_puter_tts = False
        warnings.append("TTS requested without a script - voiceover disabled.")

    for spec, name in ((plan.intro, "intro"), (plan.outro, "outro")):
        if spec.active:
            spec.seconds = min(max(spec.seconds, 1.0), 8.0)
            if spec.mode not in ("i2v", "t2v"):
                spec.mode = "i2v"
            if not spec.is_enabled:
                spec.active = False
                warnings.append(f"Dropped an empty {name} sequence.")
    return plan, warnings


# ---------------------------------------------------------------------------
# Deterministic fallback editor
# ---------------------------------------------------------------------------

_STICKER_IDEAS = [
    "3D glowing subscribe button with a red arrow",
    "cartoon fire emoji explosion, glossy 3D",
    "3D golden trophy with sparkles",
    "neon 100 percent emoji, glossy 3D",
]

_HOOK_LINES = [
    "Doston, aaj hum dekhenge kuch bilkul alag.",
    "Yeh part dhyaan se dekhna, sabse mazedaar hai.",
    "Aur ab aata hai asli twist.",
    "Agar yeh pasand aaya toh follow zaroor karna.",
]


def build_fallback_plan(
    prompt: str,
    clips: Sequence[ClipInfo],
    reference: Optional[YouTubeReference] = None,
    target_duration: float = 30.0,
    analyses: Optional[Sequence[Any]] = None,
) -> EditPlan:
    """A sensible fast-cut vertical edit produced without any LLM."""
    plan = EditPlan(
        project_meta=ProjectMeta(resolution=settings.resolution, fps=settings.output_fps),
        audio=AudioSpec(
            use_puter_tts=_wants_voiceover(prompt),
            voice_accent="indian_accent",
            tts_script="",
        ),
        captions=CaptionSpec(enabled=settings.enable_captions),
        notes="Generated by the deterministic fallback editor (Kimi K3 unavailable).",
    )

    if not clips:
        return plan

    # When the footage was analysed, borrow its ideas and its natural cuts so
    # even the no-LLM path matches what is on screen.
    analysis = analyses[0] if analyses else None
    sticker_ideas = list(getattr(analysis, "sticker_ideas", None) or _STICKER_IDEAS)
    text_ideas = list(getattr(analysis, "text_ideas", None) or [])

    segment_length = 3.0
    if analysis is not None and getattr(analysis, "energy", "") == "high":
        segment_length = 1.6
    segments: List[TimelineSegment] = []
    remaining = max(target_duration, segment_length)
    cursors: Dict[int, float] = {index: 0.0 for index in range(len(clips))}
    position = 0

    while remaining > 0.5 and position < 40:
        index = position % len(clips)
        clip = clips[index]
        duration = clip.duration if clip.duration > 0 else 5.0
        start = cursors[index]
        if start + segment_length > duration:
            start = 0.0 if duration <= segment_length else max(0.0, duration - segment_length)
            if cursors[index] >= duration:
                cursors[index] = 0.0
                start = 0.0
        end = min(start + segment_length, duration)
        if end - start < 0.6:
            position += 1
            if position >= len(clips) * 8:
                break
            continue
        cursors[index] = end + 0.35  # skip a beat -> jump cut feel

        segment = TimelineSegment(
            start_time=format_timecode(start),
            end_time=format_timecode(end),
            cut_type="jump_cut" if position % 3 else "zoom_punch",
            source_index=index,
        )
        if position in (1, 4, 7) and position // 3 < len(sticker_ideas):
            segment.puter_sticker = PuterSticker(
                generate_prompt=sticker_ideas[position // 3],
                position="bottom_center" if position % 2 else "top_center",
                animation="pop_up",
            )
        if position % 3 == 0 and text_ideas:
            segment.text_overlay = TextOverlay(
                text=text_ideas[(position // 3) % len(text_ideas)],
                position="center", animation="pop_up", style="3d_pop",
            )
        elif position == 0:
            segment.text_overlay = TextOverlay(
                text=_headline(prompt, reference),
                position="center", animation="pop_up", style="3d_pop",
            )
        segments.append(segment)
        remaining -= end - start
        position += 1

    plan.edit_timeline = segments

    if plan.audio.use_puter_tts:
        total = sum(segment.duration for segment in segments) or target_duration
        lines: List[str] = []
        budget = int(total * WORDS_PER_SECOND)
        for line in _HOOK_LINES:
            if len(" ".join(lines).split()) + len(line.split()) > budget:
                break
            lines.append(line)
        plan.audio.tts_script = " ".join(lines) or _HOOK_LINES[0]
    return plan


def _wants_voiceover(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    keywords = (
        "voice", "voiceover", "voice over", "narrat", "tts", "commentary",
        "speak", "audio", "hindi", "hinglish", "awaaz", "bolo",
    )
    return any(keyword in lowered for keyword in keywords)


def _headline(prompt: str, reference: Optional[YouTubeReference]) -> str:
    if reference is not None and reference.title:
        words = reference.title.split()
        return " ".join(words[:5]).upper()
    words = (prompt or "WATCH THIS").split()
    return " ".join(words[:5]).upper() or "WATCH THIS"
