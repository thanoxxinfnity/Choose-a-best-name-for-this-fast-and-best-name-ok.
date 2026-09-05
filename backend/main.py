"""FastAPI entrypoint for the Moja AI backend.

Endpoints
---------
``GET  /health``                       service + dependency probe
``POST /api/v1/render``                upload clips + prompt -> render job
``GET  /api/v1/jobs``                  recent jobs
``GET  /api/v1/jobs/{id}``             job status (Android polls this)
``GET  /api/v1/jobs/{id}/plan``        the Kimi K3 editing timeline
``GET  /api/v1/jobs/{id}/stream``      Range enabled MP4 for ExoPlayer preview
``GET  /api/v1/jobs/{id}/download``    high quality MP4 download
``DELETE /api/v1/jobs/{id}``           cancel + delete a job
``POST /api/v1/plan``                  timeline only (no rendering)
``POST /api/v1/puter/tts``             Indian accent TTS probe
``POST /api/v1/puter/sticker``         txt2img + rembg sticker probe

API keys travel in headers so they never have to be persisted server side:
``X-Puter-Key``, ``X-NIM-Key``, ``X-YouTube-Token``.
"""

from __future__ import annotations

import logging
import mimetypes
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from config import settings
from jobs import JobCredentials, JobRequest, job_manager
from orchestrator import KimiOrchestrator, fetch_youtube_reference
from puter_integration import PuterClient, PuterError, ffmpeg_available, rembg_available
from puter_video import (
    IMAGE_TO_VIDEO_MODEL,
    TEXT_TO_VIDEO_MODEL,
    PuterVideoClient,
    animate_keyframe,
)
from schemas import (
    ClipInfo,
    EditPlan,
    ExportPresetInfo,
    HealthResponse,
    ImageToVideoRequest,
    JobCreatedResponse,
    JobStage,
    JobStatus,
    PlanRequest,
    StickerRequest,
    TextToVideoRequest,
    ThemeInfo,
    TtsRequest,
    VideoProviderInfo,
    VoiceInfo,
)
from export_presets import PRESETS, describe_cost, resolve_preset
from style_library import all_exemplars, validate_library
from themes import THEMES, resolve_theme
from trend_research import derive_style, research_trends
from video_providers import (
    ProviderError,
    best_available,
    describe_providers,
    get_provider,
)
from voices import VOICE_PROFILES, resolve_voice
from video_analyzer import analyse_video
from video_renderer import probe_clip, whisper_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("mojaai")

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
CHUNK_SIZE = 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    removed = job_manager.purge_expired()
    logger.info(
        "%s v%s ready - output %s @ %sfps (ffmpeg=%s, rembg=%s, whisper=%s), purged %s old job(s)",
        settings.app_name, settings.app_version, settings.resolution, settings.output_fps,
        ffmpeg_available(), rembg_available(), whisper_available(), removed,
    )
    try:
        yield
    finally:
        job_manager.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Kimi K3 orchestration + Puter.js AI assets + FFmpeg/MoviePy rendering.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_credentials(
    x_puter_key: Optional[str] = Header(default=None, alias="X-Puter-Key"),
    x_nim_key: Optional[str] = Header(default=None, alias="X-NIM-Key"),
    x_youtube_token: Optional[str] = Header(default=None, alias="X-YouTube-Token"),
) -> JobCredentials:
    return JobCredentials(
        puter_key=(x_puter_key or "").strip(),
        nim_key=(x_nim_key or "").strip(),
        youtube_token=(x_youtube_token or "").strip(),
    )


def get_job_or_404(job_id: str) -> JobStatus:
    status = job_manager.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return status


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    })


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        ffmpeg=ffmpeg_available(),
        rembg=rembg_available(),
        whisper=whisper_available(),
        puter_configured=bool(settings.puter_api_key),
        nim_configured=bool(settings.nim_api_key),
        active_jobs=job_manager.active_count(),
        details={
            "output": f"{settings.resolution} @ {settings.output_fps}fps",
            "nim_model": settings.nim_model,
            "whisper_model": settings.whisper_model,
            "rembg_model": settings.rembg_model,
            "vision_model": settings.nim_vision_model,
            "t2v_model": TEXT_TO_VIDEO_MODEL,
            "i2v_model": IMAGE_TO_VIDEO_MODEL,
            "themes": list(THEMES),
            "voices": list(VOICE_PROFILES),
            "export_presets": list(PRESETS),
            "max_export": "3840x2160 / 2160x3840 @ 60fps",
            "max_upload_mb": settings.max_upload_mb,
        },
    )


# ---------------------------------------------------------------------------
# Render pipeline
# ---------------------------------------------------------------------------


@app.post("/api/v1/render", response_model=JobCreatedResponse, status_code=202)
async def create_render_job(
    videos: List[UploadFile] = File(..., description="One or more source clips"),
    prompt: str = Form(..., description="What the editor should do"),
    youtube_url: Optional[str] = Form(default=None),
    target_duration: Optional[float] = Form(default=None),
    enable_captions: bool = Form(default=True),
    voice_accent: str = Form(default="indian_accent"),
    max_stickers: int = Form(default=4),
    max_inpaints: int = Form(default=2),
    theme: str = Form(default="auto"),
    enable_voiceover: Optional[bool] = Form(default=None),
    enable_animation: bool = Form(default=False),
    max_animations: int = Form(default=1),
    enable_intro: bool = Form(default=False),
    enable_outro: bool = Form(default=False),
    auto_silence_cut: bool = Form(default=False),
    auto_beat_sync: bool = Form(default=False),
    auto_reframe: bool = Form(default=False),
    export_preset: str = Form(default="1080p60"),
    research_trends_flag: bool = Form(default=False, alias="research_trends"),
    trend_query: str = Form(default=""),
    use_exemplars: bool = Form(default=True),
    credentials: JobCredentials = Depends(get_credentials),
) -> JobCreatedResponse:
    if not videos:
        raise HTTPException(status_code=422, detail="Upload at least one video clip.")
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="The prompt cannot be empty.")

    upload_id = uuid.uuid4().hex[:12]
    upload_dir = settings.uploads_dir / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    total_bytes = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        for index, upload in enumerate(videos):
            suffix = Path(upload.filename or "clip.mp4").suffix.lower() or ".mp4"
            if suffix not in ALLOWED_VIDEO_SUFFIXES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported video format '{suffix}'. Allowed: "
                           f"{', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
                )
            destination = upload_dir / f"clip_{index:02d}{suffix}"
            with destination.open("wb") as handle:
                while chunk := await upload.read(CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                        )
                    handle.write(chunk)
            await upload.close()
            if destination.stat().st_size == 0:
                raise HTTPException(status_code=422, detail=f"'{upload.filename}' is empty.")
            saved.append(destination)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    status = job_manager.create(
        JobRequest(
            prompt=prompt.strip(),
            youtube_url=(youtube_url or "").strip() or None,
            clip_paths=saved,
            credentials=credentials,
            target_duration=target_duration,
            enable_captions=enable_captions,
            voice_accent=voice_accent,
            max_stickers=max(0, min(int(max_stickers), 10)),
            max_inpaints=max(0, min(int(max_inpaints), 6)),
            theme=theme,
            enable_voiceover=enable_voiceover,
            enable_animation=enable_animation,
            max_animations=max(0, min(int(max_animations), 4)),
            enable_intro=enable_intro,
            enable_outro=enable_outro,
            auto_silence_cut=auto_silence_cut,
            auto_beat_sync=auto_beat_sync,
            auto_reframe=auto_reframe,
            export_preset=export_preset,
            research_trends=research_trends_flag,
            trend_query=trend_query,
            use_exemplars=use_exemplars,
        )
    )
    export = resolve_preset(export_preset)
    return JobCreatedResponse(
        job_id=status.job_id,
        stage=status.stage,
        status_url=f"/api/v1/jobs/{status.job_id}",
        message=f"Rendering {len(saved)} clip(s) at {export.label}",
    )


@app.get("/api/v1/jobs", response_model=List[JobStatus])
def list_jobs(limit: int = 25) -> List[JobStatus]:
    return job_manager.list_jobs(limit=max(1, min(limit, 200)))


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(status: JobStatus = Depends(get_job_or_404)) -> JobStatus:
    return status


@app.get("/api/v1/jobs/{job_id}/plan", response_model=EditPlan)
def get_job_plan(status: JobStatus = Depends(get_job_or_404)) -> EditPlan:
    if status.plan is None:
        raise HTTPException(status_code=404, detail="The timeline has not been generated yet.")
    return status.plan


@app.delete("/api/v1/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    get_job_or_404(job_id)
    job_manager.delete(job_id)
    return JSONResponse({"job_id": job_id, "deleted": True})


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str) -> JobStatus:
    status = get_job_or_404(job_id)
    if status.is_terminal:
        raise HTTPException(status_code=409, detail=f"Job is already {status.stage.value}.")
    job_manager.cancel(job_id)
    return get_job_or_404(job_id)


# ---------------------------------------------------------------------------
# Playback + download
# ---------------------------------------------------------------------------


def _completed_output(job_id: str) -> Path:
    status = get_job_or_404(job_id)
    if status.stage != JobStage.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{status.stage.value}', the video is not ready yet.",
        )
    path = job_manager.output_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=410, detail="The rendered file is no longer on disk.")
    return path


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/api/v1/jobs/{job_id}/stream")
def stream_video(job_id: str, request: Request):
    """Range enabled progressive streaming so ExoPlayer can seek/scrub."""
    path = _completed_output(job_id)
    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})

    match = _RANGE_RE.match(range_header.strip())
    if not match:
        raise HTTPException(status_code=416, detail="Malformed Range header.")
    start_text, end_text = match.groups()
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:  # suffix range: bytes=-N
        length = int(end_text or 0)
        start = max(file_size - length, 0)
        end = file_size - 1
    if start >= file_size:
        return JSONResponse(
            status_code=416,
            content={"detail": "Requested range not satisfiable"},
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)

    return StreamingResponse(
        _iter_file(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/v1/jobs/{job_id}/download")
def download_video(job_id: str):
    path = _completed_output(job_id)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"moja_ai_{job_id}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )


# ---------------------------------------------------------------------------
# Direct AI endpoints (planning / debugging / Android "test key" buttons)
# ---------------------------------------------------------------------------


@app.post("/api/v1/plan", response_model=EditPlan)
def create_plan(
    payload: PlanRequest = Body(...),
    credentials: JobCredentials = Depends(get_credentials),
) -> EditPlan:
    clips = [
        ClipInfo(filename=f"clip_{index}.mp4", path="", duration=float(duration))
        for index, duration in enumerate(payload.clip_durations or [12.0])
    ]
    reference = fetch_youtube_reference(payload.youtube_url, credentials.resolved_youtube_token())
    orchestrator = KimiOrchestrator(api_key=credentials.resolved_nim_key())
    plan, warnings = orchestrator.build_plan(
        prompt=payload.prompt,
        clips=clips,
        youtube_reference=reference,
        target_duration=payload.target_duration,
    )
    if warnings:
        plan.notes = " | ".join(filter(None, [plan.notes, *warnings]))
    return plan


@app.post("/api/v1/puter/tts")
def puter_tts(
    payload: TtsRequest = Body(...),
    credentials: JobCredentials = Depends(get_credentials),
):
    client = PuterClient(api_key=credentials.resolved_puter_key())
    profile = resolve_voice(payload.accent)
    destination = Path(tempfile.mkdtemp(prefix="tts-", dir=settings.cache_dir)) / "voice.mp3"
    try:
        client.text_to_speech(payload.text, destination, accent=payload.accent)
    except PuterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(
        destination, media_type="audio/mpeg", filename="voice.mp3",
        headers={"X-Moja-Voice": profile.key, "X-Moja-Voice-Id": profile.voice_id},
    )


@app.post("/api/v1/puter/sticker")
def puter_sticker(
    payload: StickerRequest = Body(...),
    credentials: JobCredentials = Depends(get_credentials),
):
    client = PuterClient(api_key=credentials.resolved_puter_key())
    destination = Path(tempfile.mkdtemp(prefix="sticker-", dir=settings.cache_dir)) / "sticker.png"
    try:
        client.generate_sticker(
            payload.prompt, destination, remove_background=payload.remove_background
        )
    except PuterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sticker generation failed: {exc}") from exc
    return FileResponse(destination, media_type="image/png", filename="sticker.png")


@app.get("/api/v1/themes", response_model=List[ThemeInfo])
def list_themes() -> List[ThemeInfo]:
    """The editing modes the renderer actually implements."""
    return [
        ThemeInfo(
            key=theme.key,
            name=theme.name,
            description=theme.description,
            default_cut=theme.default_cut,
            shake=theme.shake_kind,
        )
        for theme in THEMES.values()
    ]


@app.get("/api/v1/export-presets", response_model=List[ExportPresetInfo])
def list_export_presets() -> List[ExportPresetInfo]:
    """Everything the encoder can actually output, up to 4K 60fps."""
    return [
        ExportPresetInfo(
            key=preset.key,
            label=preset.label,
            width=preset.width,
            height=preset.height,
            fps=preset.fps,
            codec=preset.codec,
            bitrate=preset.video_bitrate(),
            vertical=preset.is_vertical,
            relative_cost=round(describe_cost(preset), 2),
        )
        for preset in PRESETS.values()
    ]


@app.get("/api/v1/voices", response_model=List[VoiceInfo])
def list_voices() -> List[VoiceInfo]:
    """Voice profiles the voiceover can use, shaping included."""
    return [
        VoiceInfo(
            key=profile.key,
            label=profile.label,
            voice_id=profile.voice_id,
            language=profile.language,
            deep=profile.pitch_semitones < -1.0,
        )
        for profile in VOICE_PROFILES.values()
    ]


@app.post("/api/v1/analyze")
async def analyze_upload(
    video: UploadFile = File(...),
    use_vision: bool = Form(default=True),
    credentials: JobCredentials = Depends(get_credentials),
):
    """What is actually in this clip: subjects, style, cuts, beats, silence."""
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=settings.cache_dir) as handle:
        while chunk := await video.read(CHUNK_SIZE):
            handle.write(chunk)
        path = Path(handle.name)
    try:
        analysis = analyse_video(
            path,
            nim_api_key=credentials.resolved_nim_key(),
            use_vision=use_vision,
        )
        payload = analysis.to_dict()
        payload["theme"] = resolve_theme(analysis.suggested_theme).key
        payload["prompt_block"] = analysis.to_prompt_block()
        return payload
    finally:
        path.unlink(missing_ok=True)


@app.get("/api/v1/trends")
def get_trends(
    query: str,
    limit: int = 25,
    credentials: JobCredentials = Depends(get_credentials),
):
    """What is winning on YouTube for a niche, and the editing style it implies."""
    report = research_trends(
        query, api_key=credentials.resolved_youtube_token(), limit=max(5, min(limit, 50)),
    )
    if report.error:
        raise HTTPException(status_code=502, detail=report.error)
    payload = report.to_dict()
    payload["derived_style"] = derive_style(report)
    payload["prompt_block"] = report.to_prompt_block()
    return payload


@app.get("/api/v1/styles")
def list_styles():
    """The exemplar edits the planner learns from, curated plus learned."""
    problems = validate_library()
    return {
        "exemplars": [
            {
                "key": item.key, "content_type": item.content_type, "energy": item.energy,
                "theme": item.theme, "why": item.why, "source": item.source,
                "score": item.score, "segments": len(item.plan.get("edit_timeline", [])),
            }
            for item in all_exemplars()
        ],
        "invalid": problems,
    }


@app.get("/api/v1/video/providers", response_model=List[VideoProviderInfo])
def list_video_providers(credentials: JobCredentials = Depends(get_credentials)):
    """Which generation backends exist and which are usable right now."""
    return [
        VideoProviderInfo(**vars(info))
        for info in describe_providers({"puter": credentials.resolved_puter_key()})
    ]


@app.post("/api/v1/video/text-to-video")
def text_to_video(
    payload: TextToVideoRequest = Body(...),
    credentials: JobCredentials = Depends(get_credentials),
):
    """Generate a clip from a prompt (Puter wan2.2-t2v-a14b by default)."""
    keys = {"puter": credentials.resolved_puter_key()}
    provider = (
        get_provider(payload.provider, api_key=keys.get(payload.provider or "puter", ""))
        if payload.provider
        else best_available(keys, need_text_to_video=True)
    )
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="No text-to-video provider is configured. Add a Puter.js API key "
                   "in Settings, or use image-to-video, which works offline.",
        )
    destination = Path(tempfile.mkdtemp(prefix="t2v-", dir=settings.cache_dir)) / "generated.mp4"
    try:
        result = provider.text_to_video(
            payload.prompt, destination,
            seconds=payload.seconds, resolution=payload.resolution,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(
        result.path, media_type="video/mp4", filename="moja_ai_t2v.mp4",
        headers={
            "X-Moja-Provider": result.provider,
            "X-Moja-Model": result.model,
            "X-Moja-Elapsed": f"{result.elapsed:.1f}",
        },
    )


@app.post("/api/v1/video/image-to-video")
async def image_to_video(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    seconds: float = Form(default=5.0),
    motion_strength: float = Form(default=0.7),
    provider_key: Optional[str] = Form(default=None),
    credentials: JobCredentials = Depends(get_credentials),
):
    """Animate a still. Falls back to offline motion when no AI key is set."""
    workspace = Path(tempfile.mkdtemp(prefix="i2v-", dir=settings.cache_dir))
    source = workspace / (Path(image.filename or "frame.png").name or "frame.png")
    with source.open("wb") as handle:
        while chunk := await image.read(CHUNK_SIZE):
            handle.write(chunk)

    keys = {"puter": credentials.resolved_puter_key()}
    provider = (
        get_provider(provider_key, api_key=keys.get(provider_key or "puter", ""))
        if provider_key
        else best_available(keys)
    )
    if provider is None:
        raise HTTPException(status_code=503, detail="No image-to-video provider available.")
    try:
        result = provider.image_to_video(
            source, prompt, workspace / "generated.mp4",
            seconds=seconds, motion_strength=motion_strength,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(
        result.path, media_type="video/mp4", filename="moja_ai_i2v.mp4",
        headers={
            "X-Moja-Provider": result.provider,
            "X-Moja-Model": result.model,
            "X-Moja-Elapsed": f"{result.elapsed:.1f}",
        },
    )


@app.post("/api/v1/video/animate-keyframe")
async def animate_keyframe_endpoint(
    video: UploadFile = File(...),
    timestamp: float = Form(...),
    prompt: str = Form(...),
    story_context: str = Form(default=""),
    seconds: float = Form(default=4.0),
    motion_strength: float = Form(default=0.7),
    credentials: JobCredentials = Depends(get_credentials),
):
    """Lift a keyframe out of a clip and animate it with image-to-video."""
    workspace = Path(tempfile.mkdtemp(prefix="anim-", dir=settings.cache_dir))
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    source = workspace / f"source{suffix}"
    with source.open("wb") as handle:
        while chunk := await video.read(CHUNK_SIZE):
            handle.write(chunk)

    client = PuterVideoClient(api_key=credentials.resolved_puter_key())
    try:
        result = animate_keyframe(
            client, source, timestamp, prompt, workspace / "animated.mp4",
            seconds=seconds, workspace=workspace, story_context=story_context,
            motion_strength=motion_strength,
        )
    except PuterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(
        result.path, media_type="video/mp4", filename="moja_ai_animated.mp4",
        headers={"X-Moja-Model": result.model, "X-Moja-Elapsed": f"{result.elapsed:.1f}"},
    )


@app.post("/api/v1/probe")
async def probe_upload(video: UploadFile = File(...)):
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=settings.cache_dir) as handle:
        while chunk := await video.read(CHUNK_SIZE):
            handle.write(chunk)
        path = Path(handle.name)
    try:
        return probe_clip(path)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
