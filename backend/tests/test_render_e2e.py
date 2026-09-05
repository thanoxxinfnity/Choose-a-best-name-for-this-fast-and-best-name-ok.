"""Slow end-to-end render test with a stubbed Puter.js client.

Exercises the paths the offline unit tests cannot reach: sticker overlays with
every animation, 3D text, TTS mixing, inpainting compositing and word level
captions - all without touching the network.

    cd backend && python -m pytest tests/test_render_e2e.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_renderer  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402
from schemas import (  # noqa: E402
    AudioSpec,
    CaptionSpec,
    EditPlan,
    ProjectMeta,
    PuterInpaint,
    PuterSticker,
    TextOverlay,
    TimelineSegment,
)
from video_renderer import CaptionWord, VideoRenderer, probe_clip  # noqa: E402

pytestmark = pytest.mark.slow


class FakePuter:
    """Stands in for PuterClient: writes plausible local assets, no network."""

    is_configured = True

    def text_to_speech(self, text, output_path, accent="indian_accent", speed=1.0):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=320:duration=4",
             "-c:a", "libmp3lame", str(output_path)],
            check=True, capture_output=True,
        )
        return output_path

    def generate_sticker(self, prompt, output_path, remove_background=True, size=1024):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[40:216, 40:216] = (255, 90, 40, 255)  # opaque blob, transparent rim
        Image.fromarray(rgba, "RGBA").save(output_path, "PNG")
        return output_path

    def inpaint_image(self, image_path, mask_path, prompt, output_path, strength=0.85):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            recoloured = np.array(image.convert("RGB"))
        recoloured[:, :, 2] = 255  # obvious "repainted" tint
        Image.fromarray(recoloured).save(output_path, "PNG")
        return output_path


@pytest.fixture(scope="module")
def source_clips(tmp_path_factory) -> list[Path]:
    directory = tmp_path_factory.mktemp("clips")
    clips = []
    for index, pattern in enumerate(("testsrc", "smptebars")):
        path = directory / f"clip_{index}.mp4"
        subprocess.run(
            [ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"{pattern}=size=1280x720:rate=30:duration=8",
             "-f", "lavfi", "-i", f"sine=frequency={440 + index * 220}:duration=8",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True,
        )
        clips.append(path)
    return clips


def _plan() -> EditPlan:
    return EditPlan(
        project_meta=ProjectMeta(resolution="1080x1920", fps=60),
        audio=AudioSpec(
            use_puter_tts=True,
            voice_accent="indian_accent",
            tts_script="Doston, aaj hum dekhenge kuch bilkul alag.",
        ),
        captions=CaptionSpec(enabled=True),
        edit_timeline=[
            TimelineSegment(
                start_time="00:00:00", end_time="00:00:02", cut_type="jump_cut",
                source_index=0,
                puter_sticker=PuterSticker(
                    generate_prompt="3D glowing subscribe button",
                    position="bottom_center", animation="pop_up",
                ),
                text_overlay=TextOverlay(text="WATCH THIS", style="3d_pop", position="center"),
            ),
            TimelineSegment(
                start_time="00:00:01", end_time="00:00:03", cut_type="speed_ramp",
                source_index=1, speed=1.5,
                puter_sticker=PuterSticker(
                    generate_prompt="fire emoji", position="top_right", animation="slide_up",
                ),
            ),
            TimelineSegment(
                start_time="00:00:03", end_time="00:00:05", cut_type="zoom_punch",
                source_index=0,
                puter_inpaint=PuterInpaint(
                    active=True, target_object="sky",
                    replace_prompt="dark stormy sky with lightning",
                ),
            ),
        ],
    )


def test_full_render_with_every_feature(tmp_path, source_clips, monkeypatch):
    monkeypatch.setattr(
        video_renderer, "transcribe_words",
        lambda *_args, **_kwargs: [
            CaptionWord("DOSTON", 0.20, 0.70),
            CaptionWord("AAJ", 0.75, 1.10),
            CaptionWord("DEKHENGE", 1.15, 1.90),
        ],
    )

    stages: list[str] = []
    renderer = VideoRenderer(
        plan=_plan(),
        clip_paths=source_clips,
        workspace=tmp_path / "ws",
        puter=FakePuter(),
        progress=lambda stage, progress, message: stages.append(stage.value),
    )
    output = renderer.render(tmp_path / "final.mp4")

    assert output.exists() and output.stat().st_size > 20_000
    info = probe_clip(output)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert round(info["fps"]) == 60
    assert info["has_audio"] is True
    assert info["duration"] > 3.0

    # Every stage of the blueprint pipeline was visited.
    assert {"tts", "stickers", "inpainting", "rendering", "captions", "encoding"} <= set(stages)

    # The stubs never fail, so nothing should have degraded.
    assert renderer.warnings == [], renderer.warnings


def test_render_degrades_without_puter(tmp_path, source_clips):
    renderer = VideoRenderer(
        plan=_plan(), clip_paths=source_clips, workspace=tmp_path / "ws2", puter=None,
    )
    output = renderer.render(tmp_path / "plain.mp4")
    info = probe_clip(output)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert any("voiceover" in warning for warning in renderer.warnings)
    assert any("stickers" in warning for warning in renderer.warnings)


def test_sound_effects_land_on_the_cuts_and_reach_the_mix(tmp_path, source_clips):
    """SFX are placed off the *rendered* boundaries, not the planner's timecodes.

    A segment's ``start_time`` describes where it was taken from in the source;
    where it lands in the edit is whatever the segments before it added up to.
    Getting that wrong puts every whoosh in the wrong place, so the boundaries
    are recorded as the timeline is cut.
    """
    plan = _plan()
    plan.audio.use_puter_tts = False
    plan.captions.enabled = False

    renderer = VideoRenderer(
        plan=plan, clip_paths=source_clips, workspace=tmp_path / "ws3",
        # This checks the audio track; rendering it at 1080p pays four times
        # the encode for pixels the test never looks at.
        puter=None, theme="anime_edits", enable_sfx=True, export="720p60",
    )
    output = renderer.render(tmp_path / "sfx.mp4")
    assert output.exists()

    # Segment 1 opens the edit so it gets nothing; segment 2 is a speed ramp
    # (a transition) and segment 3 a zoom punch (a hit).
    assert len(renderer._cut_times) == 1
    assert len(renderer._impact_times) == 1
    assert renderer._impact_times[0] > renderer._cut_times[0] > 0.05

    track = tmp_path / "ws3" / "assets" / "sfx_track.wav"
    assert track.exists(), "the SFX track was never written"
    peak = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-nostats", "-i", str(track),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    assert "max_volume:" in peak
    level = float(peak.split("max_volume:")[1].split("dB")[0].strip())
    assert level > -20.0, f"the SFX track is effectively silent ({level}dB)"


def test_sfx_can_be_switched_off(tmp_path, source_clips):
    plan = _plan()
    plan.audio.use_puter_tts = False
    plan.captions.enabled = False
    renderer = VideoRenderer(
        plan=plan, clip_paths=source_clips, workspace=tmp_path / "ws4",
        puter=None, enable_sfx=False, export="720p60",
    )
    renderer.render(tmp_path / "nosfx.mp4")
    assert not (tmp_path / "ws4" / "assets" / "sfx_track.wav").exists()


def test_motion_transitions_reach_the_rendered_video(tmp_path, source_clips):
    """A transition that exists only in a unit test is a transition nobody sees.

    The bridge frames are synthesised from the last frame of one shot and the
    first of the next, so they are in neither source clip - if they survive
    into the output, the splice worked.
    """
    from schemas import CaptionSpec, EditPlan, TimelineSegment

    plan = EditPlan(
        captions=CaptionSpec(enabled=False),
        edit_timeline=[
            TimelineSegment(start_time="0", end_time="1.5", cut_type="crossfade",
                            source_index=0),
            TimelineSegment(start_time="0", end_time="1.5", cut_type="crossfade",
                            source_index=1, transition="whip_pan"),
            TimelineSegment(start_time="2", end_time="3.5", cut_type="crossfade",
                            source_index=0, transition="glitch_slice"),
        ],
    )
    plan.audio.use_puter_tts = False

    renderer = VideoRenderer(
        plan=plan, clip_paths=source_clips, workspace=tmp_path / "ws_mg",
        puter=None, theme="ae_hype", export="720p60", enable_transitions=True,
    )
    output = renderer.render(tmp_path / "mg.mp4")

    assert renderer._transitions_used == 2, renderer.warnings
    assert not any("Transition" in warning for warning in renderer.warnings), renderer.warnings
    info = probe_clip(output)
    # Two 0.22s bridges on top of 4.5s of cut.
    assert info["duration"] > 4.7, info


def test_transitions_can_be_switched_off(tmp_path, source_clips):
    from schemas import CaptionSpec, EditPlan, TimelineSegment

    plan = EditPlan(
        captions=CaptionSpec(enabled=False),
        edit_timeline=[
            TimelineSegment(start_time="0", end_time="1.5", cut_type="crossfade",
                            source_index=0),
            TimelineSegment(start_time="0", end_time="1.5", cut_type="crossfade",
                            source_index=1, transition="whip_pan"),
        ],
    )
    plan.audio.use_puter_tts = False
    renderer = VideoRenderer(
        plan=plan, clip_paths=source_clips, workspace=tmp_path / "ws_nomg",
        puter=None, theme="ae_hype", export="720p60", enable_transitions=False,
    )
    renderer.render(tmp_path / "nomg.mp4")
    assert renderer._transitions_used == 0


def test_a_jump_cut_is_never_softened_by_a_transition(tmp_path, source_clips):
    """That cut type *is* the effect; bridging it undoes what it exists to do."""
    from schemas import CaptionSpec, EditPlan, TimelineSegment

    plan = EditPlan(
        captions=CaptionSpec(enabled=False),
        edit_timeline=[
            TimelineSegment(start_time=str(i), end_time=str(i + 1),
                            cut_type="jump_cut", source_index=0)
            for i in range(5)
        ],
    )
    plan.audio.use_puter_tts = False
    renderer = VideoRenderer(
        plan=plan, clip_paths=source_clips, workspace=tmp_path / "ws_jump",
        puter=None, theme="ae_hype", export="720p60", enable_transitions=True,
    )
    renderer.render(tmp_path / "jump.mp4")
    assert renderer._transitions_used == 0
