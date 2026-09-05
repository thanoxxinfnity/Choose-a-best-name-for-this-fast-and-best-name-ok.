"""Look at the footage before planning the edit.

The orchestrator used to see nothing but a filename, a duration and a
resolution, so Kimi planned a "calm nature" edit for what was actually a
Jujutsu Kaisen anime edit.  This module fixes that: it extracts what the video
*actually* contains and hands it to Kimi as part of the prompt.

Two layers:

* **Local CV / DSP** (always available, no API key): scene cuts, per-second
  motion energy, brightness, a k-means colour palette, audio loudness, silence
  spans and beat onsets.
* **Vision model** (NVIDIA NIM, e.g. ``meta/llama-3.2-90b-vision-instruct``):
  a 2x2 contact sheet of key frames is sent in a single request and the model
  returns structured JSON describing subject, art style, mood and energy, plus
  sticker/text ideas that actually match the footage.

If the vision call is unavailable the local layer alone still produces a usable
description, and every consumer degrades gracefully.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from config import settings
from puter_integration import ffmpeg_binary

logger = logging.getLogger(__name__)

# NIM inlines images as data URIs and rejects payloads over ~180 kB.
MAX_INLINE_IMAGE_BYTES = 170_000
# Vision models bill roughly 0.10 image tokens per pixel and cap the context at
# 32k, so the contact sheet has a pixel budget, not just a byte budget: a 2x2
# sheet of 448px cells costs ~84k tokens and is rejected outright.
MAX_SHEET_PIXELS = 190_000
CONTACT_SHEET_CELL = 224
VISION_MAX_RETRIES = 3
VISION_BACKOFF = (5, 15, 30)

VISION_SYSTEM_PROMPT = (
    "You are a video content analyst for a short-form video editor. You are shown a "
    "contact sheet of frames sampled in order from one vertical video. Identify what "
    "the video actually contains and answer with strict JSON only."
)

VISION_USER_PROMPT = """This contact sheet holds {count} frames sampled evenly from a {duration:.0f} second vertical video, in reading order (left to right, top to bottom).

Answer with this JSON and nothing else:
{{
  "content_type": "one of: anime_edit, gaming, vlog, nature, music_video, sports, food, product, dance, meme, tutorial, other",
  "subjects": ["the main characters/objects actually visible, up to 4"],
  "art_style": "e.g. 2D anime, 3D render, live action cinematic, screen recording",
  "mood": "3-5 words",
  "energy": "low | medium | high",
  "recognisable": "named franchise/character if you are confident, else empty string",
  "suggested_theme": "one of: anime_edits, haunted, playful, normal",
  "sticker_ideas": ["4 sticker prompts that FIT this footage, each a short visual description"],
  "text_ideas": ["4 punchy on-screen text lines, 1-3 words each, that fit this footage"],
  "summary": "one sentence describing the video"
}}"""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class VideoAnalysis:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0

    # --- local CV / DSP ---
    scene_cuts: List[float] = field(default_factory=list)
    motion_curve: List[Tuple[float, float]] = field(default_factory=list)
    high_motion_moments: List[float] = field(default_factory=list)
    brightness: float = 0.0
    palette: List[str] = field(default_factory=list)
    silence_spans: List[Tuple[float, float]] = field(default_factory=list)
    speech_ratio: float = 0.0
    beats: List[float] = field(default_factory=list)
    bpm: float = 0.0

    # --- vision model ---
    content_type: str = "unknown"
    subjects: List[str] = field(default_factory=list)
    art_style: str = ""
    mood: str = ""
    energy: str = "medium"
    recognisable: str = ""
    suggested_theme: str = "normal"
    sticker_ideas: List[str] = field(default_factory=list)
    text_ideas: List[str] = field(default_factory=list)
    summary: str = ""
    vision_model: str = ""
    vision_error: str = ""
    # False when the description came from the fallback model. Its invented
    # nouns must not reach the stickers and the on-screen text.
    vision_trusted: bool = True

    def to_prompt_block(self) -> str:
        """Everything Kimi needs to plan an edit that matches the footage."""
        lines = [f"Content type: {self.content_type}"]
        if self.recognisable:
            lines.append(f"Recognised: {self.recognisable}")
        if self.subjects:
            lines.append("Subjects on screen: " + ", ".join(self.subjects[:4]))
        if self.art_style:
            lines.append(f"Art style: {self.art_style}")
        if self.mood:
            lines.append(f"Mood: {self.mood} (energy: {self.energy})")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.palette:
            lines.append("Dominant colours: " + ", ".join(self.palette[:5]))
        if self.scene_cuts:
            cuts = ", ".join(f"{cut:.1f}s" for cut in self.scene_cuts[:14])
            lines.append(f"Natural scene cuts at: {cuts}")
        if self.high_motion_moments:
            hits = ", ".join(f"{moment:.1f}s" for moment in self.high_motion_moments[:10])
            lines.append(f"High-motion / impact moments at: {hits}")
        if self.beats:
            lines.append(
                f"Music beats at about {self.bpm:.0f} BPM, first beats: "
                + ", ".join(f"{beat:.2f}s" for beat in self.beats[:8])
            )
        if self.silence_spans:
            spans = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in self.silence_spans[:6])
            lines.append(f"Silent / dead spans worth cutting: {spans}")
        if self.sticker_ideas:
            lines.append("Stickers that would suit this footage: " + "; ".join(self.sticker_ideas[:4]))
        if self.text_ideas:
            lines.append("On-screen text that would suit it: " + "; ".join(self.text_ideas[:4]))
        lines.append(f"Recommended theme preset: {self.suggested_theme}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["motion_curve"] = [[round(t, 2), round(v, 4)] for t, v in self.motion_curve]
        data["vision_trusted"] = self.vision_trusted
        data["silence_spans"] = [[round(a, 2), round(b, 2)] for a, b in self.silence_spans]
        return data


# ---------------------------------------------------------------------------
# Local visual analysis
# ---------------------------------------------------------------------------


def _sample_frames(path: Path, count: int) -> List[Tuple[float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps else 0.0
        if duration <= 0:
            return []
        # Skip the very first and last frames: they are often black.
        timestamps = np.linspace(duration * 0.06, duration * 0.94, count)
        frames: List[Tuple[float, np.ndarray]] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append((float(timestamp), frame))
        return frames
    finally:
        capture.release()


def scan_visual_timeline(
    path: Path,
    sample_fps: float = 6.0,
    cut_threshold: float = 0.42,
) -> Dict[str, Any]:
    """Scene cuts, motion energy and brightness in a single decode pass."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"scene_cuts": [], "motion_curve": [], "brightness": 0.0, "palette": []}

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / max(sample_fps, 0.5))))

        previous_hist: Optional[np.ndarray] = None
        previous_gray: Optional[np.ndarray] = None
        scene_cuts: List[float] = []
        motion_curve: List[Tuple[float, float]] = []
        brightness_samples: List[float] = []
        swatch_pixels: List[np.ndarray] = []

        index = 0
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    timestamp = index / fps
                    small = cv2.resize(frame, (160, 284), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    brightness_samples.append(float(gray.mean()) / 255.0)

                    hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
                    cv2.normalize(hist, hist)
                    hist = hist.flatten()
                    if previous_hist is not None:
                        # Bhattacharyya-style distance: 0 identical, 1 unrelated.
                        distance = 1.0 - float(np.minimum(hist, previous_hist).sum())
                        if distance > cut_threshold:
                            scene_cuts.append(round(timestamp, 2))
                    previous_hist = hist

                    if previous_gray is not None:
                        diff = cv2.absdiff(gray, previous_gray)
                        motion_curve.append((timestamp, float(diff.mean()) / 255.0))
                    previous_gray = gray

                    if len(swatch_pixels) < 24:
                        swatch_pixels.append(
                            cv2.resize(small, (24, 42), interpolation=cv2.INTER_AREA).reshape(-1, 3)
                        )
            index += 1

        motion_values = np.array([value for _, value in motion_curve], dtype=np.float32)
        high_motion: List[float] = []
        if motion_values.size:
            threshold = float(motion_values.mean() + motion_values.std())
            for (timestamp, value) in motion_curve:
                if value >= threshold and (not high_motion or timestamp - high_motion[-1] > 0.8):
                    high_motion.append(round(timestamp, 2))

        return {
            "scene_cuts": scene_cuts,
            "motion_curve": motion_curve,
            "high_motion_moments": high_motion,
            "brightness": float(np.mean(brightness_samples)) if brightness_samples else 0.0,
            "palette": _palette(np.vstack(swatch_pixels)) if swatch_pixels else [],
        }
    finally:
        capture.release()


def _palette(pixels: np.ndarray, colours: int = 5) -> List[str]:
    """k-means dominant colours, returned as hex, brightest first."""
    if pixels.size == 0:
        return []
    sample = pixels.astype(np.float32)
    if sample.shape[0] > 20000:
        sample = sample[np.random.choice(sample.shape[0], 20000, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _compactness, labels, centers = cv2.kmeans(
            sample, colours, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        return []
    counts = np.bincount(labels.flatten(), minlength=colours)
    order = np.argsort(-counts)
    return [
        "#{:02X}{:02X}{:02X}".format(
            int(centers[i][2]), int(centers[i][1]), int(centers[i][0])
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Audio analysis (silence spans + beat onsets)
# ---------------------------------------------------------------------------


def analyse_audio(path: Path, sample_rate: int = 22050) -> Dict[str, Any]:
    """Loudness envelope -> silence spans, beat onsets and a BPM estimate."""
    try:
        result = subprocess.run(
            [
                ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
                "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-f", "s16le", "-",
            ],
            capture_output=True, check=True,
        )
    except Exception as exc:
        logger.debug("audio analysis failed for %s: %s", path, exc)
        return {"silence_spans": [], "beats": [], "bpm": 0.0, "speech_ratio": 0.0}

    raw = np.frombuffer(result.stdout, dtype=np.int16)
    if raw.size == 0:
        return {"silence_spans": [], "beats": [], "bpm": 0.0, "speech_ratio": 0.0}

    audio = raw.astype(np.float32) / 32768.0
    hop = max(1, sample_rate // 100)  # 10 ms frames
    frames = audio[: (audio.size // hop) * hop].reshape(-1, hop)
    envelope = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    times = np.arange(envelope.size) * (hop / sample_rate)

    peak = float(np.percentile(envelope, 97)) or 1.0
    normalised = envelope / peak
    loud = normalised > 0.12

    silence_spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for index, is_loud in enumerate(loud):
        if not is_loud and start is None:
            start = float(times[index])
        elif is_loud and start is not None:
            if times[index] - start >= 0.45:
                silence_spans.append((round(start, 2), round(float(times[index]), 2)))
            start = None
    if start is not None and times[-1] - start >= 0.45:
        silence_spans.append((round(start, 2), round(float(times[-1]), 2)))

    # Onsets: positive jumps in the envelope, peak-picked with a refractory gap.
    flux = np.diff(normalised, prepend=normalised[:1])
    flux[flux < 0] = 0.0
    if flux.max() > 0:
        flux = flux / flux.max()
    onset_threshold = max(0.16, float(np.percentile(flux, 92)))
    beats: List[float] = []
    for index, value in enumerate(flux):
        if value >= onset_threshold:
            timestamp = float(times[index])
            if not beats or timestamp - beats[-1] > 0.22:
                beats.append(round(timestamp, 3))

    bpm = 0.0
    if len(beats) > 4:
        intervals = np.diff(np.array(beats))
        intervals = intervals[(intervals > 0.25) & (intervals < 1.6)]
        if intervals.size:
            bpm = float(60.0 / np.median(intervals))

    return {
        "silence_spans": silence_spans,
        "beats": beats,
        "bpm": round(bpm, 1),
        "speech_ratio": round(float(loud.mean()), 3),
    }


# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------


def build_contact_sheet(
    frames: Sequence[Tuple[float, np.ndarray]],
    cell: int = CONTACT_SHEET_CELL,
    max_pixels: int = MAX_SHEET_PIXELS,
) -> Optional[bytes]:
    """Pack sampled frames into one JPEG grid small enough to inline."""
    if not frames:
        return None
    count = len(frames)
    height, width = frames[0][1].shape[:2]
    aspect = (width / height) if height else 1.0

    # Cells are shaped like the footage and the column count is chosen to keep
    # the whole sheet near square. Fitting portrait frames into square cells
    # spends about half the model's pixel budget on black bars, and an odd
    # frame count in a two-column grid leaves an empty cell for it to reason
    # about - both of which cost accuracy the analysis cannot afford.
    def penalty(columns: int) -> tuple:
        rows = int(math.ceil(count / columns))
        squareness = abs(math.log((columns * aspect) / rows))
        return (columns * rows != count, squareness)

    columns = min(range(1, count + 1), key=penalty)
    rows = int(math.ceil(count / columns))

    cell_height = cell
    cell_width = max(16, int(round(cell * aspect)))
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (12, 12, 16))

    for index, (_timestamp, frame) in enumerate(frames):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((cell_width, cell_height), Image.LANCZOS)
        sheet.paste(image, ((index % columns) * cell_width,
                            (index // columns) * cell_height))

    # Stay inside the model's image-token budget.
    pixels = sheet.width * sheet.height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        sheet = sheet.resize(
            (max(64, int(sheet.width * scale)), max(64, int(sheet.height * scale))),
            Image.LANCZOS,
        )

    payload = b""
    for quality in (82, 70, 58, 45, 32):
        buffer = io.BytesIO()
        sheet.save(buffer, "JPEG", quality=quality, optimize=True)
        payload = buffer.getvalue()
        if len(base64.b64encode(payload)) <= MAX_INLINE_IMAGE_BYTES:
            return payload
    return payload


def describe_with_vision(
    sheet_jpeg: bytes,
    frame_count: int,
    duration: float,
    api_key: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Ask a NIM vision model what the footage actually shows."""
    if not api_key:
        raise RuntimeError("no NVIDIA NIM key for the vision pass")

    base_url = (base_url or settings.nim_base_url).rstrip("/")
    encoded = base64.b64encode(sheet_jpeg).decode("ascii")
    prompt = VISION_USER_PROMPT.format(count=frame_count, duration=duration)

    candidates = [model or settings.nim_vision_model]
    if settings.nim_vision_fallback_model not in candidates:
        candidates.append(settings.nim_vision_fallback_model)

    body = {
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'{prompt}\n<img src="data:image/jpeg;base64,{encoded}" />',
            },
        ],
        "max_tokens": 900,
        "temperature": 0.2,
    }

    last_error = ""
    deadline = time.monotonic() + settings.vision_budget_seconds
    for candidate in candidates:
        for attempt in range(1, VISION_MAX_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                raise RuntimeError(
                    last_error or f"vision pass ran out of its "
                    f"{settings.vision_budget_seconds}s budget"
                )
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                    json={"model": candidate, **body},
                    timeout=min(timeout, remaining),
                )
            except requests.RequestException as exc:
                last_error = f"vision model {candidate} network error: {exc}"
                logger.warning(last_error)
                break

            # Rate limits here are per minute and account wide, exactly as for
            # the planning call, so back off rather than dropping the pass.
            if response.status_code == 429:
                delay = min(VISION_BACKOFF[min(attempt - 1, len(VISION_BACKOFF) - 1)],
                            max(0.0, deadline - time.monotonic()))
                last_error = f"vision model {candidate} HTTP 429"
                if delay <= 0.5:
                    break
                logger.info("Vision pass rate limited, waiting %ss (attempt %s)", delay, attempt)
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                last_error = (
                    f"vision model {candidate} HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                logger.warning(last_error)
                break

            content = (response.json()["choices"][0]["message"].get("content") or "").strip()
            from orchestrator import extract_json  # local import avoids a cycle

            payload = extract_json(content)
            payload["_model"] = candidate
            return payload

    raise RuntimeError(last_error or "no vision model answered")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyse_video(
    path: Path,
    nim_api_key: str = "",
    frame_count: int = 4,
    use_vision: bool = True,
) -> VideoAnalysis:
    """Full analysis of one clip.  Never raises: missing pieces stay empty."""
    path = Path(path)
    analysis = VideoAnalysis()

    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        analysis.fps = float(fps)
        analysis.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        analysis.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        analysis.duration = float(total / fps) if fps else 0.0
    capture.release()

    visual = scan_visual_timeline(path)
    analysis.scene_cuts = visual.get("scene_cuts", [])
    analysis.motion_curve = visual.get("motion_curve", [])
    analysis.high_motion_moments = visual.get("high_motion_moments", [])
    analysis.brightness = visual.get("brightness", 0.0)
    analysis.palette = visual.get("palette", [])

    audio = analyse_audio(path)
    analysis.silence_spans = [tuple(span) for span in audio.get("silence_spans", [])]
    analysis.beats = audio.get("beats", [])
    analysis.bpm = audio.get("bpm", 0.0)
    analysis.speech_ratio = audio.get("speech_ratio", 0.0)

    if not use_vision or not nim_api_key:
        analysis.vision_error = "vision pass skipped (no NVIDIA NIM key)"
        _apply_local_fallback(analysis)
        return analysis

    try:
        frames = _sample_frames(path, frame_count)
        sheet = build_contact_sheet(frames)
        if sheet is None:
            raise RuntimeError("could not sample any frames")
        payload = describe_with_vision(sheet, len(frames), analysis.duration, nim_api_key)
        analysis.content_type = str(payload.get("content_type") or "unknown").strip()
        analysis.subjects = [str(item) for item in (payload.get("subjects") or [])][:4]
        analysis.art_style = str(payload.get("art_style") or "").strip()
        analysis.mood = str(payload.get("mood") or "").strip()
        analysis.energy = str(payload.get("energy") or "medium").strip().lower()
        analysis.recognisable = str(payload.get("recognisable") or "").strip()
        analysis.suggested_theme = str(payload.get("suggested_theme") or "normal").strip().lower()
        analysis.sticker_ideas = [str(item) for item in (payload.get("sticker_ideas") or [])][:6]
        analysis.text_ideas = [str(item) for item in (payload.get("text_ideas") or [])][:6]
        analysis.summary = str(payload.get("summary") or "").strip()
        analysis.vision_model = str(payload.get("_model") or "")

        # A fallback answer is not a weaker answer, it is an unchecked one.
        # Asked to describe an anime edit, the small model reported a gaming
        # screen recording once and a city vlog the next time, both with
        # complete confidence - and the invented nouns went straight into the
        # stickers and the on-screen text, which is how an anime edit ends up
        # captioned LEVEL UP. Measured facts survive; invented ones do not.
        analysis.vision_trusted = analysis.vision_model == settings.nim_vision_model
        if not analysis.vision_trusted:
            _discard_invented_detail(analysis)
    except Exception as exc:
        logger.warning("vision analysis failed for %s: %s", path.name, exc)
        analysis.vision_error = str(exc)[:300]
        _apply_local_fallback(analysis)

    return analysis


def _discard_invented_detail(analysis: VideoAnalysis) -> None:
    """Drop everything the model said and re-derive it from the file.

    Subjects, recognisable names, sticker prompts and caption copy are exactly
    what a model confabulates, and each one becomes something the viewer sees.
    Energy, mood and the suggested theme go too - not because they are as
    damaging, but because there is no reason to keep a guess from a reading
    already judged unreliable when motion, brightness and cut density measure
    the same three things directly.

    What is left is the file itself, and the planner then writes from that and
    the user's prompt - which fails far more gracefully than confident
    nonsense does.
    """
    analysis.subjects = []
    analysis.sticker_ideas = []
    analysis.text_ideas = []
    analysis.recognisable = ""
    analysis.art_style = ""
    analysis.summary = ""
    analysis.content_type = "unknown"
    analysis.mood = ""
    analysis.energy = ""
    analysis.suggested_theme = ""
    _apply_local_fallback(analysis)
    analysis.vision_error = (
        f"the primary vision model did not answer; '{analysis.vision_model}' "
        f"stood in and its description of the footage is not reliable enough "
        f"to name subjects or write on-screen text from"
    )


def _apply_local_fallback(analysis: VideoAnalysis) -> None:
    """Infer what we can from motion, brightness and beats alone."""
    motion = np.array([value for _, value in analysis.motion_curve], dtype=np.float32)
    average_motion = float(motion.mean()) if motion.size else 0.0
    cuts_per_second = len(analysis.scene_cuts) / max(analysis.duration, 1.0)

    if average_motion > 0.10 or cuts_per_second > 0.55:
        analysis.energy = "high"
    elif average_motion > 0.045:
        analysis.energy = "medium"
    else:
        analysis.energy = "low"

    if analysis.brightness < 0.28:
        analysis.suggested_theme = "haunted"
        analysis.mood = analysis.mood or "dark, moody"
    elif analysis.energy == "high":
        analysis.suggested_theme = "anime_edits"
        analysis.mood = analysis.mood or "fast, punchy"
    elif analysis.brightness > 0.6:
        analysis.suggested_theme = "playful"
        analysis.mood = analysis.mood or "bright, upbeat"
    else:
        analysis.suggested_theme = "normal"
        analysis.mood = analysis.mood or "calm, cinematic"

    if not analysis.summary:
        analysis.summary = (
            f"{analysis.duration:.0f}s vertical clip, {analysis.energy} energy, "
            f"{len(analysis.scene_cuts)} scene cuts"
            + (f", ~{analysis.bpm:.0f} BPM" if analysis.bpm else "")
        )
