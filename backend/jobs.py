"""Background job manager for the render pipeline.

A job owns a workspace directory under ``data/jobs/<job_id>/`` holding the
uploaded clips, the Kimi K3 plan, the Puter assets and the final MP4.  Status is
kept in memory *and* mirrored to ``status.json`` so the Android client can keep
polling after a backend restart.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from orchestrator import KimiOrchestrator, fetch_youtube_reference
from export_presets import check_source_headroom, describe_cost, resolve_preset
from micro_features import apply_micro_features
from themes import choose_theme, resolve_theme
from video_analyzer import VideoAnalysis, analyse_video
from puter_integration import PuterClient
from schemas import ClipInfo, EditPlan, JobStage, JobStatus
from video_renderer import RenderError, VideoRenderer, probe_clip

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "moja_ai_final.mp4"


@dataclass
class JobCredentials:
    """Per-request keys sent by the Android app (EncryptedSharedPreferences)."""

    puter_key: str = ""
    nim_key: str = ""
    youtube_token: str = ""

    def resolved_puter_key(self) -> str:
        return self.puter_key or settings.puter_api_key

    def resolved_nim_key(self) -> str:
        return self.nim_key or settings.nim_api_key

    def resolved_youtube_token(self) -> str:
        return self.youtube_token or settings.youtube_api_key


@dataclass
class JobRequest:
    prompt: str
    youtube_url: Optional[str]
    clip_paths: List[Path]
    credentials: JobCredentials
    target_duration: Optional[float] = None
    enable_captions: bool = True
    voice_accent: str = "indian_accent"
    theme: str = "auto"
    # Tri-state on purpose: None = let Kimi decide from the prompt, True/False =
    # the user pressed the button and their choice wins.
    enable_voiceover: Optional[bool] = None
    enable_animation: bool = False
    max_animations: int = 1
    # --- micro-features -----------------------------------------------------
    auto_silence_cut: bool = False
    auto_beat_sync: bool = False
    auto_reframe: bool = False
    export_preset: str = "1080p60"
    enable_intro: bool = False
    enable_outro: bool = False
    max_stickers: int = 4
    max_inpaints: int = 2
    plan_override: Optional[EditPlan] = None
    extra: Dict[str, str] = field(default_factory=dict)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobStatus] = {}
        self._futures: Dict[str, Future] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs), thread_name_prefix="render"
        )
        self._load_from_disk()

    # ------------------------------------------------------------- lifecycle
    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def workspace(self, job_id: str) -> Path:
        return settings.jobs_dir / job_id

    def output_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / OUTPUT_FILENAME

    # ----------------------------------------------------------------- CRUD
    def create(self, request: JobRequest) -> JobStatus:
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        clips: List[ClipInfo] = []
        for path in request.clip_paths:
            info = probe_clip(path)
            clips.append(
                ClipInfo(
                    filename=path.name,
                    path=str(path),
                    duration=float(info.get("duration") or 0.0),
                    width=int(info.get("width") or 0),
                    height=int(info.get("height") or 0),
                    fps=float(info.get("fps") or 0.0),
                    has_audio=bool(info.get("has_audio")),
                )
            )

        status = JobStatus(
            job_id=job_id,
            stage=JobStage.QUEUED,
            progress=0.0,
            message="Queued",
            created_at=now,
            updated_at=now,
            prompt=request.prompt,
            youtube_url=request.youtube_url,
            clips=clips,
            preview_url=f"/api/v1/jobs/{job_id}/stream",
            download_url=f"/api/v1/jobs/{job_id}/download",
        )
        with self._lock:
            self._jobs[job_id] = status
        self._persist(status)

        future = self._executor.submit(self._run, job_id, request)
        with self._lock:
            self._futures[job_id] = future
        return status

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            status = self._jobs.get(job_id)
        if status is not None:
            return status
        return self._load_one(job_id)

    def list_jobs(self, limit: int = 50) -> List[JobStatus]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if not job.is_terminal)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            status = self._jobs.get(job_id)
            if status is None or status.is_terminal:
                return False
            self._cancelled.add(job_id)
            future = self._futures.get(job_id)
        if future is not None and future.cancel():
            self._update(job_id, stage=JobStage.CANCELLED, message="Cancelled before it started")
        else:
            self._update(job_id, message="Cancellation requested")
        return True

    def delete(self, job_id: str) -> bool:
        self.cancel(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
            self._cancelled.discard(job_id)
        workspace = self.workspace(job_id)
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
            return True
        return False

    def purge_expired(self) -> int:
        cutoff = time.time() - settings.job_retention_hours * 3600
        removed = 0
        for status in self.list_jobs(limit=10_000):
            if status.is_terminal and status.updated_at < cutoff:
                self.delete(status.job_id)
                removed += 1
        return removed

    # -------------------------------------------------------------- updates
    def _update(self, job_id: str, **fields) -> Optional[JobStatus]:
        with self._lock:
            status = self._jobs.get(job_id)
            if status is None:
                return None
            for key, value in fields.items():
                setattr(status, key, value)
            status.updated_at = time.time()
            snapshot = status.model_copy(deep=True)
        self._persist(snapshot)
        return snapshot

    def _append_warnings(self, job_id: str, warnings: List[str]) -> None:
        if not warnings:
            return
        with self._lock:
            status = self._jobs.get(job_id)
            if status is None:
                return
            for warning in warnings:
                if warning and warning not in status.warnings:
                    status.warnings.append(warning)
            snapshot = status.model_copy(deep=True)
        self._persist(snapshot)

    def _check_cancelled(self, job_id: str) -> None:
        with self._lock:
            cancelled = job_id in self._cancelled
        if cancelled:
            raise RenderError("Job cancelled by the client.")

    # ------------------------------------------------------------ pipeline
    def _run(self, job_id: str, request: JobRequest) -> None:
        workspace = self.workspace(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        started = time.time()

        try:
            self._check_cancelled(job_id)
            status = self.get(job_id)
            if status is None:
                raise RenderError(f"Job {job_id} disappeared before it started.")

            puter = PuterClient(api_key=request.credentials.resolved_puter_key())

            # -------------------------------------- 1. look at the footage
            self._update(job_id, stage=JobStage.ANALYZING, progress=0.02,
                         message="Analysing what is actually in the footage")
            nim_key = request.credentials.resolved_nim_key()
            analyses: List[VideoAnalysis] = []
            for index, clip_path in enumerate(request.clip_paths):
                self._update(
                    job_id, progress=0.02 + 0.02 * index,
                    message=f"Analysing clip {index + 1}/{len(request.clip_paths)}",
                )
                analyses.append(analyse_video(clip_path, nim_api_key=nim_key))
            if analyses and analyses[0].vision_error:
                self._append_warnings(job_id, [f"Vision pass: {analyses[0].vision_error}"])

            theme = (
                choose_theme(analyses[0], request.theme)
                if analyses else resolve_theme(request.theme)
            )

            export = resolve_preset(request.export_preset)
            headroom = check_source_headroom(
                export, [(clip.width, clip.height) for clip in status.clips]
            )
            notes = [headroom] if headroom else []
            cost = describe_cost(export)
            if cost > 1.5:
                notes.append(
                    f"{export.label} moves {cost:.1f}x the pixels of a 1080p60 render, "
                    f"so expect it to take roughly that much longer."
                )
            self._append_warnings(job_id, notes)
            self._update(
                job_id,
                theme=theme.key,
                analysis=[analysis.to_dict() for analysis in analyses],
                message=f"Footage: {analyses[0].summary if analyses else 'unknown'} "
                        f"- theme '{theme.name}'",
            )

            reference = None
            if request.youtube_url:
                reference = fetch_youtube_reference(
                    request.youtube_url, request.credentials.resolved_youtube_token()
                )
                if reference is not None and reference.source == "none":
                    self._append_warnings(job_id, ["Could not read the YouTube reference metadata."])

            # ----------------------------------------------------- 2. plan
            self._check_cancelled(job_id)
            self._update(job_id, stage=JobStage.PLANNING, progress=0.06,
                         message="Kimi K3 is writing the editing timeline")

            if request.plan_override is not None:
                plan: EditPlan = request.plan_override
                warnings: List[str] = []
            else:
                orchestrator = KimiOrchestrator(api_key=nim_key)
                plan, warnings = orchestrator.build_plan(
                    prompt=request.prompt,
                    clips=status.clips,
                    youtube_reference=reference,
                    target_duration=request.target_duration,
                    max_stickers=request.max_stickers,
                    max_inpaints=request.max_inpaints,
                    analyses=analyses,
                    theme=theme.prompt_hint(),
                    max_animations=request.max_animations if request.enable_animation else 0,
                )
            plan.captions.enabled = plan.captions.enabled and request.enable_captions
            plan.theme = theme.key
            if request.voice_accent:
                plan.audio.voice_accent = request.voice_accent

            if request.enable_voiceover is False:
                plan.audio.use_puter_tts = False
                plan.audio.tts_lines = []
                plan.audio.tts_script = ""
                # With nobody speaking there is nothing to duck under.
                plan.audio.background_music_gain = 1.0
            elif request.enable_voiceover is True and not plan.audio.has_voiceover:
                self._append_warnings(job_id, [
                    "Voiceover was switched on but Kimi wrote no narration for this "
                    "edit - rendering without it."
                ])
            if not request.enable_animation:
                for segment in plan.edit_timeline:
                    segment.puter_animate = None
            if not request.enable_intro:
                plan.intro.active = False
            if not request.enable_outro:
                plan.outro.active = False

            # ------------------------------------------ 2b. micro-features
            if request.auto_silence_cut or request.auto_beat_sync:
                plan, micro_notes = apply_micro_features(
                    plan, analyses,
                    silence_cut=request.auto_silence_cut,
                    beat_sync=request.auto_beat_sync,
                )
                warnings.extend(micro_notes)

            self._append_warnings(job_id, warnings)
            self._update(job_id, plan=plan, message=f"Timeline ready: {len(plan.edit_timeline)} segments")
            (workspace / "plan.json").write_text(
                json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # --------------------------------------------------- 3. render
            self._check_cancelled(job_id)

            def on_progress(stage: JobStage, progress: float, message: str) -> None:
                self._check_cancelled(job_id)
                self._update(job_id, stage=stage, progress=round(float(progress), 4), message=message)

            renderer = VideoRenderer(
                plan=plan,
                clip_paths=request.clip_paths,
                workspace=workspace,
                puter=puter,
                progress=on_progress,
                theme=theme.key,
                analyses=analyses,
                auto_reframe=request.auto_reframe,
                export=export.key,
            )
            output = self.output_path(job_id)
            renderer.render(output)
            self._append_warnings(job_id, renderer.warnings)

            if not output.exists() or output.stat().st_size == 0:
                raise RenderError("The renderer produced no output file.")

            info = probe_clip(output)
            self._update(
                job_id,
                export_preset=export.key,
                output_width=int(info.get("width") or 0),
                output_height=int(info.get("height") or 0),
                output_fps=float(info.get("fps") or 0.0),
                stage=JobStage.COMPLETED,
                progress=1.0,
                message=f"Done in {time.time() - started:.0f}s ({export.resolution} @ {export.fps}fps)",
                finished_at=time.time(),
                output_filename=output.name,
                output_size_bytes=output.stat().st_size,
                duration_seconds=float(info.get("duration") or 0.0),
            )
            self._cleanup_intermediates(workspace)
            logger.info("Job %s completed in %.1fs", job_id, time.time() - started)

        except Exception as exc:
            with self._lock:
                cancelled = job_id in self._cancelled
            if cancelled:
                self._update(job_id, stage=JobStage.CANCELLED, message="Cancelled",
                             finished_at=time.time())
                return
            logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            self._update(
                job_id,
                stage=JobStage.FAILED,
                message="Render failed",
                error=str(exc),
                finished_at=time.time(),
            )

        finally:
            # The originals have been cut into the output (or the job died);
            # either way they are not needed again.
            self._cleanup_uploads(request.clip_paths)

    @staticmethod
    def _cleanup_uploads(clip_paths: List[Path]) -> None:
        """Remove the uploaded originals once the render is over."""
        directories = {
            path.parent for path in clip_paths
            if path.parent.parent == settings.uploads_dir
        }
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _cleanup_intermediates(workspace: Path) -> None:
        for name in ("base.mp4", "caption_source.wav"):
            path = workspace / name
            if path.exists():
                path.unlink(missing_ok=True)
        for directory in workspace.glob("inpaint_*"):
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)

    # ---------------------------------------------------------- persistence
    def _status_file(self, job_id: str) -> Path:
        return self.workspace(job_id) / "status.json"

    def _persist(self, status: JobStatus) -> None:
        try:
            path = self._status_file(status.job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = status.model_dump(mode="json")
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.debug("Could not persist job %s", status.job_id, exc_info=True)

    def _load_one(self, job_id: str) -> Optional[JobStatus]:
        path = self._status_file(job_id)
        if not path.exists():
            return None
        try:
            status = JobStatus.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None
        if not status.is_terminal:  # a job cannot survive a restart mid-render
            status.stage = JobStage.FAILED
            status.error = "The backend restarted while this job was rendering."
        with self._lock:
            self._jobs.setdefault(job_id, status)
        return status

    def _load_from_disk(self) -> None:
        if not settings.jobs_dir.exists():
            return
        for directory in settings.jobs_dir.iterdir():
            if directory.is_dir():
                self._load_one(directory.name)


job_manager = JobManager()
