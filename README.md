# AI Video Editor - Native Kotlin Android app + Python FastAPI backend

A complete, production-shaped AI short-form video editor:

* **Android client** - native Kotlin / Jetpack Compose, `EncryptedSharedPreferences`
  for the API keys, multi-clip picker, native **ExoPlayer (Media3)** preview and a
  high-quality MP4 download straight into the gallery. **No Flutter anywhere.**
* **FastAPI backend** - Kimi K3 on NVIDIA NIM writes a structured editing
  timeline; **Puter.js** provides the Indian-accent TTS, the AI stickers and the
  frame inpainting; FFmpeg + MoviePy + Faster-Whisper render a
  **1080x1920 60 fps** MP4.

```
┌──────────────────────────┐        multipart upload         ┌─────────────────────────────┐
│  Android (Kotlin)        │  ─────────────────────────────► │  FastAPI backend            │
│  • MainActivity.kt       │   X-Puter-Key / X-NIM-Key /     │  • main.py                  │
│  • Settings.kt (crypto)  │   X-YouTube-Token headers       │  • orchestrator.py  (Kimi)  │
│  • ApiClient.kt (OkHttp) │                                 │  • puter_integration.py     │
│  • ExoPlayer preview     │ ◄───────────────────────────────│  • video_renderer.py        │
└──────────────────────────┘   Range-enabled MP4 stream      └─────────────────────────────┘
                                                                │        │           │
                                                    Puter.js ◄──┘        │           └──► FFmpeg / MoviePy
                                            (TTS, txt2img, inpaint)      │                Faster-Whisper
                                                                         └──► NVIDIA NIM (Kimi K3)
```

---

## Repository layout

```
android/                         Native Kotlin app (Gradle, AGP 8.7, Kotlin 2.0)
  app/src/main/java/com/aivideo/editor/
    MainActivity.kt              Editor UI, ExoPlayer preview, download button
    Settings.kt                  SecureStore (EncryptedSharedPreferences) + settings screen
    ApiClient.kt                 OkHttp client, streaming upload/download, DTOs
    EditorViewModel.kt           UI state, job polling
    EditorApplication.kt         Warms up the encrypted store
    ui/Theme.kt                  Material 3 theme
backend/
  main.py                        FastAPI app + endpoints
  orchestrator.py                Kimi K3 (NVIDIA NIM) -> JSON editing timeline
  puter_integration.py           Puter.js TTS / txt2img+rembg / inpainting + OpenCV
  video_renderer.py              FFmpeg + MoviePy + Faster-Whisper pipeline
  jobs.py                        Background job manager
  schemas.py                     Pydantic models (the JSON timeline contract)
  config.py                      Settings (env / .env)
  requirements.txt               rembg, moviepy, opencv-python, fastapi, uvicorn, requests, ...
  Dockerfile, docker-compose.yml
docs/API.md                      REST reference
docs/PIPELINE.md                 How a render actually runs, stage by stage
```

---

## 1. Run the backend

### With Docker (recommended - ffmpeg and fonts are included)

```bash
cd backend
cp .env.example .env      # optional: server-side fallback keys
docker compose up --build
```

### Locally

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ffmpeg + ffprobe must be on PATH
sudo apt-get install -y ffmpeg          # or: brew install ffmpeg

cp .env.example .env                     # optional
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check it: `curl http://localhost:8000/health`

```json
{"status":"ok","version":"1.0.0","ffmpeg":true,"rembg":true,"whisper":true,...}
```

Interactive API docs: <http://localhost:8000/docs>

> **First run downloads models.** `rembg` fetches its segmentation model and
> `faster-whisper` fetches the Whisper weights (~500 MB for `small`). Both are
> cached under `backend/data/cache/`.

---

## 2. Build the Android app

```bash
cd android
cp local.properties.example local.properties   # point sdk.dir at your Android SDK
./gradlew :app:assembleDebug                   # or open the folder in Android Studio
adb install app/build/outputs/apk/debug/app-debug.apk
```

Requirements: Android Studio Ladybug+, JDK 17, `compileSdk 35`, `minSdk 24`.

### Point the app at your backend

Open **Settings** in the app and set:

| Field | Example | Used for |
|---|---|---|
| Backend URL | `http://10.0.2.2:8000` (emulator) or `http://192.168.1.20:8000` | everything |
| NVIDIA NIM API Key | `nvapi-...` | Kimi K3 editing timeline |
| Puter.js API Key | `sk-...` | TTS (Indian accent), stickers, inpainting |
| YouTube Data Token | `AIza...` | richer reference-video metadata (optional) |

Tap **Test connection** to verify.

All four values live in `EncryptedSharedPreferences` (AES-256-GCM values,
AES-256-SIV keys, master key in the Android Keystore) and are excluded from
cloud backup and device transfer. They are sent only as request headers to the
backend URL you configured; the backend never writes them to disk.

> Cleartext HTTP is allowed only for `10.0.2.2`, `localhost` and the common
> private LAN ranges (see `res/xml/network_security_config.xml`). A public
> backend must be HTTPS.

---

## 3. Make an edit

1. **Pick videos** - one or more clips (system photo picker, or "Browse files").
2. **Write the prompt** - e.g. *"Fast paced Hinglish reel with an Indian voiceover,
   glowing subscribe sticker at the hook and a stormy sky in the drone shot."*
3. Optionally paste a **YouTube reference URL** - its title, tags, chapters and
   description are fed to Kimi K3 so the pacing and style can be matched.
4. Choose the accent, captions and target length, then **Generate AI edit**.
5. Watch the stage-by-stage progress, preview the result in ExoPlayer, and tap
   **Download high quality MP4** to save it to `Movies/AI Video Editor/`.

---

## What each AI service does

| Stage | Service | Implementation |
|---|---|---|
| Editing timeline | **Kimi K3** on NVIDIA NIM | `orchestrator.py` - the blueprint system prompt, strict JSON out, deterministic fallback editor when NIM is unavailable |
| Voiceover | **Puter.js TTS** | `puter_integration.py::text_to_speech` - `en-IN` neural voice (`Kajal` / `Arjun` / `Aditi`), long scripts chunked and concatenated with ffmpeg, saved as `.mp3` |
| Stickers | **Puter.js txt2img** + `rembg` | `generate_sticker` - image generated, background removed with `rembg`, alpha-trimmed and saved as a transparent PNG |
| Inpainting | **Puter.js image-to-image** + OpenCV | frames extracted with OpenCV, masks built by heuristic (HSV sky detection / region hints), repainted frames blended back over every frame of the window |
| Captions | **Faster-Whisper** | word-level timestamps, rendered with Pillow (no ImageMagick needed) and animated per word |
| Render | **FFmpeg + MoviePy** | jump cuts, zoom punches, speed ramps, scale-to-cover vertical reframing, sticker/text overlays, audio ducking, H.264 `yuv420p` + `faststart` |

The JSON contract Kimi K3 must emit is documented in `docs/PIPELINE.md` and
enforced by `schemas.EditPlan`.

---

## Notes and assumptions

* **Kimi K3 model id.** `NIM_MODEL` defaults to `moonshotai/kimi-k3-instruct`. If
  your NVIDIA NIM account does not expose that id yet, the orchestrator
  automatically retries with `NIM_FALLBACK_MODEL`
  (`moonshotai/kimi-k2-instruct`) and records a warning on the job.
* **Puter driver ids.** Puter routes every AI capability through
  `POST /drivers/call` with an `interface` / `driver` / `method` triple. The
  defaults (`puter-tts` + `aws-polly`, `puter-image-generation` +
  `openai-image-generation`) are configurable in `config.py` / `.env`, and the
  response decoder accepts raw bytes, `data:` URIs, base64 and URLs so a change
  in Puter's envelope shape does not break the pipeline.
* **Graceful degradation.** A missing Puter key skips TTS/stickers/inpainting, a
  missing NIM key uses the deterministic editor, and a missing Whisper model
  skips captions. Every skip is reported back to the app as a job warning
  instead of failing the render.
* Inpainting is sampled (`INPAINT_FPS`, default 2 fps, max 24 frames per
  segment) and blended back with a feathered mask - a full 60 fps round trip
  through an image API would be prohibitively slow and expensive.

## License

MIT - see [LICENSE](LICENSE).
