# How a render runs

## 0. Upload

`POST /api/v1/render` streams the clips to
`backend/data/uploads/<upload_id>/clip_NN.mp4` and immediately returns a
`job_id`. A `ThreadPoolExecutor` (size `MAX_CONCURRENT_JOBS`) runs the pipeline;
status is mirrored to `data/jobs/<job_id>/status.json` so polling survives a
backend restart.

## 1. Analyse (`analyzing`)

`video_renderer.probe_clip` reads duration / geometry / fps / audio presence for
every clip (ffprobe, falling back to OpenCV + an `ffmpeg -i` probe).

If a YouTube reference URL was supplied, `orchestrator.fetch_youtube_reference`
resolves its title, channel, description, tags, chapters and duration - via the
**YouTube Data API v3** when a token is present, otherwise via `yt-dlp`.

## 2. Plan (`planning`) - Kimi K3 on NVIDIA NIM

`orchestrator.KimiOrchestrator` posts to
`{NIM_BASE_URL}/chat/completions` with the blueprint system prompt verbatim,
`response_format: json_object`, and a user turn carrying the prompt, the clip
table, the reference block and the hard constraints (legal enum values, segment
length bounds, sticker/inpaint budgets, the TTS word budget).

The answer is parsed by `extract_json` (tolerant of code fences and trailing
prose), validated into `schemas.EditPlan`, then `sanitise_plan` clamps it:
segment windows are pushed inside the real clip durations, `source_index` is
repaired, segments shorter than 0.8 s or longer than 8 s are fixed, and the
sticker/inpaint budgets are enforced.

If NIM is unreachable, the model id is not enabled on the account, or the answer
cannot be parsed, `build_fallback_plan` produces a deterministic fast-cut
timeline instead and the reason is recorded as a job warning.

### The timeline contract

```json
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
      "source_index": 0,
      "puter_sticker": {
        "generate_prompt": "3D glowing subscribe button",
        "position": "bottom_center",
        "animation": "pop_up"
      },
      "puter_inpaint": {
        "active": true,
        "target_object": "sky",
        "replace_prompt": "dark stormy sky with lightning"
      },
      "text_overlay": { "text": "WATCH THIS", "style": "3d_pop", "position": "center" }
    }
  ],
  "captions": { "enabled": true, "highlight_color": "#FFD400", "position": "bottom_center" }
}
```

`cut_type`: `jump_cut` | `hard_cut` | `crossfade` | `speed_ramp` | `zoom_punch`
`position`: `top_left` … `bottom_right` (9 anchors)
`animation`: `pop_up` | `fade_in` | `slide_up` | `slide_down` | `zoom_out` | `shake` | `none`

## 3. Voiceover (`tts`) - Puter.js

`PuterClient.text_to_speech` calls
`POST {PUTER_BASE_URL}/drivers/call` with
`{"interface": "puter-tts", "driver": "aws-polly", "method": "synthesize", "args": {...}}`.

The accent map picks an `en-IN` voice (`Kajal` neural female, `Arjun` neural
male, `Aditi` for Hinglish). Scripts longer than 2800 characters are split on
sentence boundaries, synthesised chunk by chunk and concatenated with
`ffmpeg -f concat` into a single `.mp3`.

The response decoder handles raw `audio/mpeg` bytes, `{"result": "<url>"}`,
`data:` URIs and base64 blobs alike.

## 4. Stickers (`stickers`) - Puter.js txt2img + rembg

For each `puter_sticker`:

1. the prompt is enriched (*"sticker art, bold clean outline, isolated on a
   plain flat white background, no text"*) and sent to the txt2img driver,
2. **`rembg`** (`isnet-general-use`) cuts the subject out,
3. the alpha channel is trimmed to the subject's bounding box with 8 px padding,
4. the result is saved as a transparent PNG in the job's `assets/` directory.

## 5. Inpainting (`inpainting`) - OpenCV + Puter image-to-image

For each segment with `puter_inpaint.active`:

1. `extract_frames` pulls key frames from the segment window with OpenCV at
   `INPAINT_FPS` (default 2 fps, capped at 24 frames per segment),
2. `build_mask` paints the region to repaint white:
   * `sky` / `cloud` / `sunset` → HSV threshold for blue and bright pixels in
     the upper two thirds, morphologically closed and opened; if too little is
     detected it falls back to the top third,
   * `ground` / `road` / `water` → bottom 40 %, `left`/`right`/`top` → that half,
     `full`/`background` → everything, otherwise a centred box,
   the mask is dilated and feathered,
3. each key frame + mask + `replace_prompt` goes to the Puter inpainting driver,
4. every frame of the window is then streamed through OpenCV and blended with
   the nearest repainted key frame using the feathered mask
   (`out = orig·(1-α) + edited·α`), so motion outside the mask is untouched,
5. frames are piped straight into `ffmpeg` (rawvideo → libx264) and the original
   audio for that window is muxed back in.

Any failure here degrades to the original footage plus a job warning.

## 6. Timeline render (`rendering`)

Per segment: subclip → **scale-to-cover + centre crop** to 1080x1920 → speed
ramp → `zoom_punch` (a 9 % OpenCV push-in over the segment) or `crossfade` →
force 60 fps. The segments are concatenated (`method="compose"`) and written to
`base.mp4`.

## 7. Audio

The original bed is attenuated to `BACKGROUND_AUDIO_GAIN` (0.18) whenever a
voiceover exists and the Puter TTS track is mixed on top. If the voiceover is
longer than the picture, the last frame is held so nothing is cut off.

## 8. Captions (`captions`) - Faster-Whisper

The clean TTS `.mp3` (or the mixed audio when there is no voiceover) is
transcribed with `word_timestamps=True` and VAD filtering. Every word becomes a
Pillow-rendered RGBA image - bold face with an outline and a blurred drop shadow,
Devanagari-aware font selection - shown for its own timestamp with a short
upward pop. No ImageMagick required.

## 9. Overlays and encode (`encoding`)

Stickers and 3D text overlays are placed on one of nine anchors and animated
(`pop_up` uses an ease-out-back scale around the anchor). Everything is
composited and encoded:

```
libx264, preset medium, CRF 20, bitrate 12M, profile high, level 4.2,
pix_fmt yuv420p, +faststart, AAC 192k, 1080x1920 @ 60fps
```

`+faststart` puts the moov atom first, which is what makes the ExoPlayer
preview start instantly over the Range-enabled `/stream` endpoint.

## Failure policy

| Missing | Effect |
|---|---|
| NVIDIA NIM key / model | deterministic fallback editor, job warning |
| Puter key | no TTS, no stickers, no inpainting - the cut still renders |
| Puter call fails mid-render | that asset is skipped, job warning |
| faster-whisper / model | no captions, job warning |
| ffprobe | OpenCV + `ffmpeg -i` fallback probe |
| a segment cannot be built | it is dropped; the render fails only if *all* fail |
