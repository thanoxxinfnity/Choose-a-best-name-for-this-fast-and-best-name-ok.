# REST API

Base URL: whatever you set as **Backend URL** in the Android Settings screen
(e.g. `http://10.0.2.2:8000`). Interactive docs live at `/docs`.

## Authentication

There is no account system. The three credentials are sent per request as
headers, so nothing is persisted on the server:

| Header | Value | Required for |
|---|---|---|
| `X-Puter-Key` | Puter.js API key | TTS, stickers, inpainting |
| `X-NIM-Key` | NVIDIA NIM key (`nvapi-...`) | the Kimi K3 timeline |
| `X-YouTube-Token` | YouTube Data API v3 key | reference metadata (optional) |

If a header is absent the corresponding `.env` value is used as a fallback; if
that is empty too, the affected stage is skipped and a warning is attached to
the job instead of failing the render.

---

## `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "ffmpeg": true,
  "rembg": true,
  "whisper": true,
  "puter_configured": false,
  "nim_configured": false,
  "active_jobs": 0,
  "details": { "output": "1080x1920 @ 60fps", "nim_model": "moonshotai/kimi-k3" }
}
```

## `POST /api/v1/render`

`multipart/form-data`, returns **202**.

| Field | Type | Notes |
|---|---|---|
| `videos` | file (repeatable) | `.mp4 .mov .m4v .mkv .webm .avi .3gp` |
| `prompt` | string | required |
| `youtube_url` | string | optional reference |
| `target_duration` | float | seconds, optional |
| `enable_captions` | bool | default `true` |
| `voice_accent` | string | `indian_accent`, `indian_accent_male`, `hinglish`, ... |
| `max_stickers` | int | default 4 |
| `max_inpaints` | int | default 2 |

```bash
curl -X POST http://localhost:8000/api/v1/render \
  -H "X-Puter-Key: $PUTER_API_KEY" \
  -H "X-NIM-Key: $NIM_API_KEY" \
  -F "videos=@clip_a.mp4" \
  -F "videos=@clip_b.mp4" \
  -F "prompt=Fast paced Hinglish reel with an Indian voiceover and a glowing subscribe sticker" \
  -F "youtube_url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

```json
{
  "job_id": "9f1c2d3e4b5a6789",
  "stage": "queued",
  "status_url": "/api/v1/jobs/9f1c2d3e4b5a6789",
  "message": "Rendering 2 clip(s) at 1080x1920 60fps"
}
```

## `GET /api/v1/jobs/{job_id}`

Poll this (the app polls every 2 s).

```json
{
  "job_id": "9f1c2d3e4b5a6789",
  "stage": "rendering",
  "progress": 0.42,
  "message": "Segment 3/7 (jump_cut)",
  "clips": [{ "filename": "clip_00.mp4", "duration": 18.4, "width": 1920, "height": 1080, "fps": 30.0, "has_audio": true }],
  "plan": { "project_meta": { "resolution": "1080x1920", "fps": 60 }, "...": "..." },
  "warnings": [],
  "preview_url": "/api/v1/jobs/9f1c2d3e4b5a6789/stream",
  "download_url": "/api/v1/jobs/9f1c2d3e4b5a6789/download"
}
```

`stage` is one of `queued`, `analyzing`, `planning`, `tts`, `stickers`,
`inpainting`, `rendering`, `captions`, `encoding`, `completed`, `failed`,
`cancelled`.

## Other job endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/jobs?limit=25` | recent jobs |
| `GET` | `/api/v1/jobs/{id}/plan` | just the Kimi K3 timeline |
| `POST` | `/api/v1/jobs/{id}/cancel` | stop a running render |
| `DELETE` | `/api/v1/jobs/{id}` | cancel and delete the workspace |
| `GET` | `/api/v1/jobs/{id}/stream` | **Range-enabled** MP4 for ExoPlayer |
| `GET` | `/api/v1/jobs/{id}/download` | `Content-Disposition: attachment` MP4 |

`/stream` answers `Range: bytes=...` with `206 Partial Content` plus
`Accept-Ranges`/`Content-Range`, which is what lets ExoPlayer seek and scrub
without downloading the whole file.

## Analysis and editing modes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/themes` | the four editing modes and their pacing/shake |
| `GET` | `/api/v1/voices` | voice profiles, `deep: true` marks the deep/dark narrators |
| `POST` | `/api/v1/analyze` | multipart `video` -> subjects, art style, mood, energy, scene cuts, motion peaks, palette, beats, BPM, silence spans, recommended theme |

`POST /api/v1/render` also accepts:

| Field | Default | Meaning |
|---|---|---|
| `theme` | `auto` | `auto`, `anime_edits`, `haunted`, `playful`, `normal` |
| `voice_accent` | `indian_accent` | any key from `/api/v1/voices` |
| `enable_animation` | `false` | keyframe-to-animation insertion |
| `max_animations` | `1` | how many moments may be animated (0-4) |
| `enable_intro` | `false` | generate an AI intro |
| `enable_outro` | `false` | generate an AI outro |
| `enable_voiceover` | unset | unset lets Kimi decide; `true`/`false` is the user's switch and wins |
| `auto_silence_cut` | `false` | drop detected dead air from the timeline |
| `auto_beat_sync` | `false` | snap cuts onto the nearest musical onset |
| `auto_reframe` | `false` | track the subject so the 9:16 crop follows it |

## AI video generation (Puter.js wan2.2)

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/v1/video/text-to-video` | `{"prompt": "...", "seconds": 5, "resolution": "720x1280"}` | `video/mp4` from `wan-ai/wan2.2-t2v-a14b` |
| `POST` | `/api/v1/video/image-to-video` | multipart `image` + `prompt`, `seconds`, `motion_strength` | `video/mp4` from `wan-ai/wan2.2-i2v-a14b` |
| `POST` | `/api/v1/video/animate-keyframe` | multipart `video` + `timestamp`, `prompt`, `story_context` | the keyframe animated and conformed to the canvas |

All three require `X-Puter-Key`; generation is long-running, so the client polls
the driver's job handle internally and answers once the clip is ready.

## Direct AI endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/v1/plan` | `{"prompt": "...", "youtube_url": "...", "clip_durations": [12.0]}` | the JSON timeline, no rendering |
| `POST` | `/api/v1/puter/tts` | `{"text": "...", "accent": "indian_accent"}` | `audio/mpeg` |
| `POST` | `/api/v1/puter/sticker` | `{"prompt": "3D glowing subscribe button"}` | transparent `image/png` |
| `POST` | `/api/v1/probe` | multipart `video` | duration / size / fps / audio |

## Errors

Standard FastAPI shape: `{"detail": "..."}`.

| Code | Meaning |
|---|---|
| 409 | the video is not ready yet (job still running) |
| 410 | the rendered file was purged (`JOB_RETENTION_HOURS`) |
| 413 | upload larger than `MAX_UPLOAD_MB` |
| 415 | unsupported container |
| 416 | malformed `Range` header |
| 502 | Puter.js rejected the request |
