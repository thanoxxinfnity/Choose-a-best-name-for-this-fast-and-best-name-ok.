"""Central configuration for the AI Video Editor backend.

Every value can be overridden with an environment variable (or a `.env` file
sitting next to this module).  API keys are optional here: the Android client
stores them in `EncryptedSharedPreferences` and sends them per request via the
`X-Puter-Key` / `X-NIM-Key` / `X-YouTube-Token` headers.  The values below are
only used as a server side fallback (useful for curl / CI / self hosting).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("AIVE_ENV_FILE", str(BASE_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app --
    app_name: str = "Moja AI"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]

    # Directory layout ------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    max_upload_mb: int = 2048
    max_concurrent_jobs: int = 1
    job_retention_hours: int = 24

    # ------------------------------------------------------------- puter.js --
    # Puter exposes every AI capability through a single driver endpoint:
    #   POST {puter_base_url}/drivers/call
    #   { "interface": ..., "driver": ..., "method": ..., "args": {...} }
    puter_base_url: str = "https://api.puter.com"
    puter_api_key: str = ""
    puter_timeout: int = 180

    puter_tts_interface: str = "puter-tts"
    puter_tts_driver: str = "aws-polly"
    puter_tts_method: str = "synthesize"

    puter_txt2img_interface: str = "puter-image-generation"
    puter_txt2img_driver: str = "openai-image-generation"
    puter_txt2img_method: str = "generate"

    # --- video generation (wan 2.2 text-to-video / image-to-video) ---------
    puter_video_interface: str = "puter-video-generation"
    puter_video_driver: str = "wan-ai"
    puter_video_method: str = "generate"
    puter_video_status_method: str = "status"
    puter_video_timeout: int = 900
    puter_t2v_model: str = "wan-ai/wan2.2-t2v-a14b"
    puter_i2v_model: str = "wan-ai/wan2.2-i2v-a14b"

    puter_inpaint_interface: str = "puter-image-generation"
    puter_inpaint_driver: str = "openai-image-generation"
    # Puter's image driver has no edit/inpaint method - only generation.
    # The replacement is generated and composited locally; see
    # PuterClient.inpaint_image.
    puter_inpaint_method: str = "generate"

    # ----------------------------------------------------- nvidia nim / kimi --
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_api_key: str = ""
    # Kimi K3 on NVIDIA NIM (verified against the live /v1/models catalogue).
    # If your account does not expose K3, set NIM_MODEL to another chat model.
    nim_model: str = "moonshotai/kimi-k3"
    nim_fallback_model: str = "moonshotai/kimi-k2.6"
    nim_timeout: int = 120
    # NIM enforces its limit per minute; see RATE_LIMIT_BACKOFF in orchestrator.
    nim_max_retries: int = 5
    # Hard ceiling on the whole planning step. Without it, 5 retries x a 120s
    # request timeout x 2 model candidates is a 30 minute worst case and the
    # Android client just sees "planning" forever.
    nim_plan_budget_seconds: int = 240
    # The second pass has a draft to work from, so it is much shorter than
    # the first - and it is optional, so it must never dominate the job.
    nim_revise_budget_seconds: int = 120
    enable_plan_review: bool = True
    # Vision pass: what is actually in the footage (subjects, style, mood).
    nim_vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    nim_vision_fallback_model: str = "meta/llama-3.2-11b-vision-instruct"
    analysis_frame_count: int = 4
    enable_vision_analysis: bool = True
    # Ceiling on the whole vision pass, retries included.
    vision_budget_seconds: int = 150
    nim_temperature: float = 0.6
    # Kimi K3 on NIM pins top_p at 0.95 and rejects anything else.
    nim_top_p: float = 0.95
    nim_max_tokens: int = 4096

    # ------------------------------------------------------------- youtube ---
    youtube_api_key: str = ""
    youtube_timeout: int = 30

    # ------------------------------------------------------------ rendering --
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 60
    video_bitrate: str = "12M"
    audio_bitrate: str = "192k"
    x264_preset: str = "medium"
    x264_crf: int = 20
    ffmpeg_threads: int = 0  # 0 => let ffmpeg decide

    # Original audio level once the Indian accent TTS voiceover is laid on top.
    background_audio_gain: float = 0.18
    tts_audio_gain: float = 1.0
    # SFX sit under the narration, not beside it: loud enough to punch the
    # cut, quiet enough that a word is never lost behind a whoosh.
    sfx_audio_gain: float = 0.45
    enable_sfx: bool = True

    # ---------------------------------------------------- motion graphics --
    enable_transitions: bool = True
    # A transition on every cut is the same as one on none, so they are
    # rationed the way the impact accents are.
    transition_every: int = 4
    transition_seconds: float = 0.22

    # ------------------------------------------------------------- captions --
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    enable_captions: bool = True

    # ------------------------------------------------------------- stickers --
    rembg_model: str = "isnet-general-use"
    sticker_max_width_ratio: float = 0.42

    # ----------------------------------------------------------- inpainting --
    # Inpainting is expensive: only this many frames per segment are round
    # tripped through Puter, the rest are interpolated by holding the last
    # rendered frame (keeps a 5 s segment at ~10 API calls instead of 300).
    inpaint_fps: float = 2.0
    inpaint_max_frames_per_segment: int = 24

    # ----------------------------------------------------------- properties --
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def resolution(self) -> str:
        return f"{self.output_width}x{self.output_height}"

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.uploads_dir, self.jobs_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
