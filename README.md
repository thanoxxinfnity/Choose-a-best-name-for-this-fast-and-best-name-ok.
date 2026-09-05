# Moja AI - Native Kotlin Android app + Python FastAPI backend

A complete, production-shaped AI short-form video editor that **looks at your
footage before it plans the edit**:

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
│  • Settings.kt (crypto)  │   X-YouTube-Token headers       │  • video_analyzer.py  ◄─ 1  │
│  • ApiClient.kt (OkHttp) │                                 │  • orchestrator.py    ◄─ 2  │
│  • ExoPlayer preview     │ ◄───────────────────────────────│  • video_renderer.py  ◄─ 3  │
└──────────────────────────┘   Range-enabled MP4 stream      └─────────────────────────────┘

  1. SEE     scene cuts, motion peaks, palette, beats, silence  (OpenCV + DSP)
             subjects / art style / mood / energy               (NIM vision model)
  2. PLAN    a JSON edit timeline that matches what is on screen (Kimi K3 on NIM)
  3. RENDER  theme grade + impact shake + stickers + captions    (FFmpeg / MoviePy)
                                                                 + Puter.js assets
```

**Why the analysis pass matters.** Without it the planner only sees a filename.
A Jujutsu Kaisen fight edit whose file happened to be named
`...Quiet_nature_ideas...` got a calm nature plan - "GOLDEN LOTUS" stickers and
a *"Just Breathe"* title over a sword fight. With the vision pass the same clip
is reported as `anime_edit / 2D anime / dark fantasy, high energy, 172 BPM`, and
Kimi returns `FIGHT` / `POWER` / `UNSTOPPABLE` with sword-slash and magic-circle
stickers on 1-2 second jump cuts.

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
  video_analyzer.py              What is in the footage: CV/DSP + NIM vision pass
  orchestrator.py                Kimi K3 (NVIDIA NIM) -> JSON editing timeline
  themes.py                      Haunted / Playful / Normal / Anime editing modes
  vfx.py                         Shake, RGB split, motion blur, grade, vignette, bloom
  sticker_art.py                 Procedural vector sticker artwork
  puter_video.py                 wan2.2 text-to-video, image-to-video, keyframe animation
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
| Footage analysis | OpenCV + **NIM vision** | `video_analyzer.py` - scene cuts by histogram distance, motion peaks, k-means palette, audio envelope (silence spans, onset beats, BPM); then a contact sheet of key frames goes to `meta/llama-3.2-90b-vision-instruct` for subjects, art style, mood, energy and footage-appropriate sticker/text ideas |
| Editing timeline | **Kimi K3** on NVIDIA NIM | `orchestrator.py` - the blueprint system prompt, strict JSON out, deterministic fallback editor when NIM is unavailable |
| Voiceover | **Puter.js TTS** | `puter_integration.py::text_to_speech` - `en-IN` neural voice (`Kajal` / `Arjun` / `Aditi`), long scripts chunked and concatenated with ffmpeg, saved as `.mp3` |
| Stickers | **Puter.js txt2img** + `rembg` | `generate_sticker` - image generated, background removed with `rembg`, alpha-trimmed and saved as a transparent PNG |
| Inpainting | **Puter.js image-to-image** + OpenCV | frames extracted with OpenCV, masks built by heuristic (HSV sky detection / region hints), repainted frames blended back over every frame of the window |
| Captions | **Faster-Whisper** | word-level timestamps, rendered with Pillow (no ImageMagick needed) and animated per word |
| Render | **FFmpeg + MoviePy** | jump cuts, zoom punches, speed ramps, scale-to-cover vertical reframing, sticker/text overlays, audio ducking, H.264 `yuv420p` + `faststart` |

| Editing modes | local | `themes.py` - **Haunted** (desaturated, cold tint, heavy vignette, handheld drift), **Playful** (vibrant, beat-locked pulse zoom), **Normal Edits** (cinematic grade), **Anime Edits** (punched contrast, bloom, directional impact shake with RGB split and motion blur on every cut). Selected by `choose_theme()` from content type, brightness and energy, or forced with the `theme` form field |
| Video generation | **Puter.js wan2.2** | `puter_video.py` - text-to-video (`wan-ai/wan2.2-t2v-a14b`), image-to-video (`wan-ai/wan2.2-i2v-a14b`), and keyframe-to-animation insertion: 1-2 stills are lifted from a critical moment, described with the surrounding story context (subjects, art style, mood from the analysis pass), animated, conformed to the canvas and either spliced in after the segment (`insert`) or used in its place (`replace`) |
| AI intro / outro | **Puter.js wan2.2** | `intro` / `outro` on the plan. `i2v` animates the edit's own first or last frame so the bookend matches the footage; `t2v` invents a shot from the prompt. A title can be burnt over it, and without a Puter key it degrades to a rendered title card rather than vanishing |
| Voiceover | **Puter TTS + local shaping** | `voices.py` - the provider voice (Polly `Kajal`/`Arjun`/`Aditi`, `en-IN`) plus an ffmpeg shaping chain, which is how a **deep dark mysterious** narrator exists at all: pitch down 4.5 semitones, slow to 0.92, high-pass 70 Hz, low-pass 7.2 kHz, two-tap reverb, limiter. Lines are pinned to timestamps in the finished edit and shift automatically when a generated intro is spliced in front |

The JSON contract Kimi K3 must emit is documented in `docs/PIPELINE.md` and
enforced by `schemas.EditPlan`.

### Export presets

| Key | Output | Codec | Bitrate | Cost |
|---|---|---|---|---|
| `720p60` | 720x1280 60fps | H.264 | 6M | 0.4x |
| `1080p60` | **1080x1920 60fps** (default) | H.264 | 12M | 1.0x |
| `1440p60` | 1440x2560 60fps | H.264 | 22M | 1.8x |
| `4k60` | **2160x3840 60fps** | H.264 | 50M | 4.0x |
| `4k60_hevc` | 2160x3840 60fps | HEVC | 32M | 4.0x |
| `4k30` | 2160x3840 30fps | H.264 | 25M | 2.0x |
| `1080p60_wide` / `4k60_wide` | landscape 1920x1080 / 3840x2160 | H.264 | 12M / 50M | 1.0x / 4.0x |

The timeline is rendered *at* the preset's resolution rather than upscaled at
the end, so a 4K export is 4K wherever the source has the detail. Bitrate is
derived from the pixel rate, not hardcoded; H.264 gets level 5.2 above 1080p60
because 4.2 cannot carry 4K, and HEVC is tagged `hvc1` so Apple players will
open it. Two honest warnings come back on the job: the relative render cost,
and an upscale notice when the smallest source cannot fill the requested frame.

### Audio enhancer

| Profile | Target | Chain |
|---|---|---|
| `voice` | -14 LUFS | denoise, 90Hz-14kHz, de-esser, compressor, loudnorm |
| `podcast` | -16 LUFS | heavier denoise, 100Hz high-pass, de-esser, compressor |
| `music` | -13 LUFS | 25Hz high-pass, compressor |
| `balanced` | -14 LUFS | 40Hz high-pass, compressor |
| `off` | - | untouched |

Loudness normalisation runs last, because everything before it changes the
level it has to hit. Verified end to end: a -48.6 LUFS source lands within
0.2 LU of every target.

### Colour and background tools

* **Colour pop** is vibrance, not saturation - the lift is proportional to each
  pixel's remaining headroom, and skin hues are damped, so a shot pops without
  faces going orange.
* **Green-screen removal** keys in YCrCb chroma space, so shadows and uneven
  lighting on the screen do not break the match, with green-spill suppression
  on the subject edge.
* **AI background removal** uses `rembg` at a sampled rate with masks
  interpolated between key frames - segmenting every frame of a 60fps clip is
  not affordable, and a silhouette moves far slower than the frame rate.

### AI video providers

Generation is behind a provider interface, so adding a service is one adapter
class rather than a pipeline change:

| Provider | t2v | i2v | Key |
|---|---|---|---|
| `puter` | yes (`wan2.2-t2v-a14b`) | yes (`wan2.2-i2v-a14b`) | required |
| `local_motion` | no | yes (Ken Burns) | **none - works offline** |

`local_motion` exists so the image-to-video button is never dead: with no key
configured it still returns a real 1080x1920 clip, and a failed provider call
degrades to one instead of to nothing.

### Auto edit micro-features

All three run locally on the analysis pass - no API key, no model download.

| Feature | What it does | How |
|---|---|---|
| **Auto silence-cut** | drops dead air out of the timeline | the audio envelope's silence spans are subtracted from each segment, splitting it around a hole or dropping it entirely; `keep_padding` leaves a breath so cuts never land on the first syllable |
| **Auto beat-sync** | snaps every cut onto the music | each boundary moves to the nearest onset **within a tolerance**, so a deliberate cut that sits nowhere near a beat is left alone, and a snap that would collapse a segment is refused |
| **Auto-reframe** | the 9:16 crop follows the subject | faces first (Haar cascade), then the centroid of frame-to-frame motion, then edge density; the trajectory is smoothed forward-and-backward so there is no lag, and clamped inside the frame |

Silence-cut runs before beat-sync: the first decides which ranges exist, the
second tightens whatever survived. Already-vertical footage is skipped by
auto-reframe - there is nothing to pan across.

Enable them per render with `auto_silence_cut`, `auto_beat_sync` and
`auto_reframe`, or from the **Auto edit** switches in the app.

### Voice profiles

| Key | Voice | Character |
|---|---|---|
| `indian_accent` | Kajal, en-IN neural | Indian English, female, unshaped |
| `indian_accent_male` | Arjun, en-IN neural | Indian English, male, unshaped |
| `hinglish` | Aditi, en-IN | Hindi / Hinglish |
| `deep_dark` | Arjun + shaping | **deep dark mysterious** narrator: -4.5 st, 0.92x, darkened, reverb |
| `deep_dark_female` | Kajal + shaping | the same character, female |
| `horror_whisper` | Aditi + shaping | -2 st, 0.88x, heavy reverb, 5.2 kHz ceiling |
| `hype` | Arjun + shaping | +1 st, 1.08x, louder |

Timing is per line, not per script:

```json
"audio": {
  "use_puter_tts": true,
  "voice_accent": "deep_dark",
  "tts_lines": [
    { "text": "Andhere mein ek shakti jaag rahi hai", "start_time": "00:00:00.500" },
    { "text": "Aur ab sab badal jayega", "start_time": "00:00:03.250", "voice": "hype" }
  ]
}
```

`start_time` keeps milliseconds, lines are de-overlapped and clamped to the edit,
and the bed ducks under them.

### Editing modes

| Key | Mode | Look | Motion |
|---|---|---|---|
| `anime_edits` | Anime Edits | saturation 1.32, contrast 1.22, bloom 0.42 | directional impact shake, RGB split, motion blur, 0.9-2.2s jump cuts |
| `haunted` | Haunted Mode | saturation 0.55, cold tint, vignette 0.62 | slow handheld drift, 2.6-5.5s crossfades |
| `playful` | Playful Mode | saturation 1.42, warm tint, bloom 0.26 | beat-locked pulse zoom, 1.2-3.0s zoom punches |
| `normal` | Normal Edits | saturation 1.06, vignette 0.18 | none, 2.5-5.0s crossfades |

---

## Notes and assumptions

* **Kimi K3 model id.** `NIM_MODEL` defaults to `moonshotai/kimi-k3`, verified
  against the live `GET /v1/models` catalogue. NIM pins `top_p` at `0.95` for
  this model and rejects any other value; the client reads the required value
  out of the 400 response and retries automatically, so other pinned
  parameters are handled too. If the id is not enabled on your account the
  orchestrator retries with `NIM_FALLBACK_MODEL` and records a job warning.
* **NIM rate limits are per minute.** On HTTP 429 the client waits
  5s / 15s / 30s / 60s (or the `Retry-After` value) and retries the *same*
  model - switching model ids would just burn another request against the same
  account-wide budget.
* **Puter driver ids.** Puter routes every AI capability through
  `POST /drivers/call` with an `interface` / `driver` / `method` triple. The
  defaults (`puter-tts` + `aws-polly`, `puter-image-generation` +
  `openai-image-generation`) are configurable in `config.py` / `.env`, and the
  response decoder accepts raw bytes, `data:` URIs, base64 and URLs so a change
  in Puter's envelope shape does not break the pipeline.
* **Graceful degradation.** A missing NIM key uses the deterministic editor; a
  missing Puter key skips the TTS and inpainting and swaps the AI stickers for
  locally rendered glowing motion-graphic badges (`procedural_sticker`), so the
  timeline still carries the graphics Kimi asked for; a missing Whisper model
  skips captions. Every substitution is reported to the app as a job warning
  instead of failing the render.
* **Puter.js needs a token.** `POST /drivers/call` answers
  `401 {"code": "token_missing"}` without one - there is no anonymous tier, so
  TTS, AI stickers and inpainting all require `X-Puter-Key`.
* Inpainting is sampled (`INPAINT_FPS`, default 2 fps, max 24 frames per
  segment) and blended back with a feathered mask - a full 60 fps round trip
  through an image API would be prohibitively slow and expensive.

## License

MIT - see [LICENSE](LICENSE).
