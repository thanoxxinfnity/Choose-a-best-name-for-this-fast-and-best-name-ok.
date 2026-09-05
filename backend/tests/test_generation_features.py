"""Intro/outro generation, keyframe animation insertion and timed voiceover.

Puter has no anonymous tier, so the video and speech calls are stubbed with
real ffmpeg-produced media. Everything downstream - conforming, splicing,
timeline offsets, audio placement - is the production code path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_renderer  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402
from schemas import (  # noqa: E402
    AudioSpec,
    CaptionSpec,
    EditPlan,
    GeneratedSequence,
    ProjectMeta,
    PuterAnimate,
    TimelineSegment,
    TtsLine,
)
from video_renderer import VideoRenderer, probe_clip  # noqa: E402
from voices import VOICE_PROFILES, resolve_voice, shape_voice  # noqa: E402

pytestmark = pytest.mark.slow

FF = ffmpeg_binary()


def _make_video(path: Path, pattern: str, seconds: float, size: str = "360x640") -> Path:
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"{pattern}=size={size}:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=420:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    return path


def _make_speech(path: Path, seconds: float) -> Path:
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=180:duration={seconds}",
         "-af", "tremolo=f=6:d=0.5", "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )
    return path


class FakePuter:
    """Stands in for PuterClient: real files, no network."""

    is_configured = True
    api_key = "fake"

    def __init__(self) -> None:
        self.tts_calls: list[tuple[str, str]] = []

    def text_to_speech(self, text, output_path, accent="indian_accent", speed=1.0):
        self.tts_calls.append((text, accent))
        raw = Path(output_path).with_suffix(".raw.mp3")
        _make_speech(raw, max(1.0, len(text.split()) / 2.6))
        # Exercise the real shaping chain for the requested profile.
        shape_voice(raw, Path(output_path), resolve_voice(accent), ffmpeg=FF)
        return Path(output_path)

    def generate_sticker(self, prompt, output_path, remove_background=True, size=1024):
        raise RuntimeError("not used here")


class FakeVideoClient:
    """Stands in for PuterVideoClient."""

    is_configured = True

    def __init__(self, api_key: str = "") -> None:
        self.calls: list[tuple[str, str, float]] = []

    def text_to_video(self, prompt, output_path, seconds=5.0, **kwargs):
        self.calls.append(("t2v", prompt, seconds))
        _make_video(Path(output_path), "smptebars", seconds, size="480x854")
        return type("R", (), {"path": Path(output_path), "model": "t2v", "seconds": seconds})

    def image_to_video(self, image_path, prompt, output_path, seconds=5.0, **kwargs):
        self.calls.append(("i2v", prompt, seconds))
        assert Path(image_path).exists(), "i2v must be seeded with a real still"
        _make_video(Path(output_path), "testsrc", seconds, size="480x854")
        return type("R", (), {"path": Path(output_path), "model": "i2v", "seconds": seconds})


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    return _make_video(tmp_path_factory.mktemp("src") / "clip.mp4", "testsrc", 10.0)


def _plan(**overrides) -> EditPlan:
    plan = EditPlan(
        project_meta=ProjectMeta(resolution="540x960", fps=30),
        audio=AudioSpec(
            use_puter_tts=True,
            voice_accent="deep_dark",
            tts_lines=[
                TtsLine(text="Andhere mein ek shakti jaag rahi hai", start_time="00:00:00.500"),
                TtsLine(text="Aur ab sab badal jayega", start_time=3.25, voice="hype"),
            ],
        ),
        captions=CaptionSpec(enabled=False),
        edit_timeline=[
            TimelineSegment(start_time="00:00:00", end_time="00:00:02", source_index=0),
            TimelineSegment(start_time="00:00:03", end_time="00:00:05", source_index=0),
        ],
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def _render(plan, source, tmp_path, name, monkeypatch, video_client=None):
    client = video_client or FakeVideoClient()
    monkeypatch.setattr(video_renderer, "PuterVideoClient", lambda api_key="": client)
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / f"ws_{name}",
        puter=FakePuter(), theme="anime_edits",
    )
    out = renderer.render(tmp_path / f"{name}.mp4")
    return renderer, client, out


# --------------------------------------------------------------- intro/outro --

def test_intro_and_outro_are_generated_and_spliced(source, tmp_path, monkeypatch):
    plan = _plan(
        intro=GeneratedSequence(active=True, prompt="cursed energy swirling", seconds=2.0,
                                mode="i2v", text="MOJA AI"),
        outro=GeneratedSequence(active=True, prompt="energy fading to black", seconds=1.5,
                                mode="t2v", text="FOLLOW"),
    )
    renderer, client, out = _render(plan, source, tmp_path, "bookends", monkeypatch)

    kinds = [call[0] for call in client.calls]
    assert "i2v" in kinds, "the intro should animate a frame of the edit"
    assert "t2v" in kinds, "the outro asked for text-to-video"

    info = probe_clip(out)
    # 4s of cut + 2s intro + 1.5s outro, minus encoder rounding.
    assert info["duration"] > 6.5, info
    assert (info["width"], info["height"]) == (540, 960)
    assert not any("intro" in w and "skipped" in w for w in renderer.warnings), renderer.warnings


def test_intro_seeds_image_to_video_with_the_first_frame(source, tmp_path, monkeypatch):
    plan = _plan(intro=GeneratedSequence(active=True, prompt="opening", seconds=1.5, mode="i2v"))
    _renderer, client, _out = _render(plan, source, tmp_path, "seed", monkeypatch)
    i2v = [call for call in client.calls if call[0] == "i2v"]
    assert i2v, "no image-to-video call was made"
    # The story context from the analysis/theme is folded into the prompt.
    assert "opening" in i2v[0][1]


def test_intro_falls_back_to_a_title_card_without_puter(source, tmp_path, monkeypatch):
    plan = _plan(intro=GeneratedSequence(active=True, prompt="x", seconds=1.5, text="MOJA AI"))
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws_nokey",
        puter=None, theme="normal",
    )
    out = renderer.render(tmp_path / "nokey.mp4")
    assert probe_clip(out)["duration"] > 4.5, "the title card should still be spliced in"
    assert any("title card" in w for w in renderer.warnings), renderer.warnings


# ------------------------------------------------------- keyframe animation --

def test_keyframe_animation_is_inserted_into_the_timeline(source, tmp_path, monkeypatch):
    plan = _plan()
    plan.edit_timeline[1].puter_animate = PuterAnimate(
        active=True, prompt="the hero powers up", seconds=2.0, mode="insert",
    )
    renderer, client, out = _render(plan, source, tmp_path, "insert", monkeypatch)

    i2v = [call for call in client.calls if call[0] == "i2v"]
    assert len(i2v) == 1
    assert "powers up" in i2v[0][1]
    # 4s of cut plus the 2s animation.
    assert probe_clip(out)["duration"] > 5.5
    assert renderer.warnings == [], renderer.warnings


def test_two_keyframes_split_the_animation_budget(source, tmp_path, monkeypatch):
    plan = _plan()
    plan.edit_timeline[0].puter_animate = PuterAnimate(
        active=True, prompt="clash", seconds=4.0,
        keyframe_time=0.5, second_keyframe_time=1.5,
    )
    _renderer, client, _out = _render(plan, source, tmp_path, "two", monkeypatch)
    i2v = [call for call in client.calls if call[0] == "i2v"]
    assert len(i2v) == 2, "both key frames should be animated"
    assert all(call[2] == pytest.approx(2.0) for call in i2v), "budget split across frames"


def test_replace_mode_swaps_the_segment(source, tmp_path, monkeypatch):
    # No voiceover here: a line running past the last cut legitimately extends
    # the edit (the renderer holds the final frame rather than clipping speech),
    # which would mask the length this test is actually measuring.
    plan = _plan(audio=AudioSpec(use_puter_tts=False))
    plan.edit_timeline[1].puter_animate = PuterAnimate(
        active=True, prompt="stylised", seconds=2.0, mode="replace",
    )
    _renderer, _client, out = _render(plan, source, tmp_path, "replace", monkeypatch)
    # 2s segment + 2s animation replacing the other 2s segment.
    assert probe_clip(out)["duration"] == pytest.approx(4.0, abs=0.6)


def test_a_late_voice_line_extends_the_edit_instead_of_being_cut(source, tmp_path, monkeypatch):
    """The picture holds on its last frame so narration is never truncated."""
    plan = _plan(audio=AudioSpec(
        use_puter_tts=True, voice_accent="deep_dark",
        tts_lines=[TtsLine(text="Aur yahin se sab kuch badal jayega", start_time=3.25)],
    ))
    _renderer, _client, out = _render(plan, source, tmp_path, "tail", monkeypatch)
    # The cut is 4s; the line starts at 3.25s and runs past it.
    assert probe_clip(out)["duration"] > 4.3


def test_animation_without_puter_warns_and_continues(source, tmp_path):
    plan = _plan()
    plan.edit_timeline[0].puter_animate = PuterAnimate(active=True, prompt="x", seconds=2.0)
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws_anim_nokey", puter=None,
    )
    out = renderer.render(tmp_path / "anim_nokey.mp4")
    assert out.exists()
    assert any("keyframe animation was skipped" in w for w in renderer.warnings)


# ------------------------------------------------------------ timed voice ----

def test_each_line_is_synthesised_with_its_own_voice(source, tmp_path, monkeypatch):
    plan = _plan()
    puter = FakePuter()
    monkeypatch.setattr(video_renderer, "PuterVideoClient", lambda api_key="": FakeVideoClient())
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws_voice",
        puter=puter, theme="anime_edits",
    )
    out = renderer.render(tmp_path / "voice.mp4")

    assert len(puter.tts_calls) == 2, puter.tts_calls
    assert puter.tts_calls[0][1] == "deep_dark", "line 1 uses the plan's voice"
    assert puter.tts_calls[1][1] == "hype", "line 2 overrides it"
    assert probe_clip(out)["has_audio"] is True


def test_lines_shift_when_an_intro_is_spliced_in(source, tmp_path, monkeypatch):
    """A generated intro pushes the whole edit back; the lines must follow."""
    plan = _plan(intro=GeneratedSequence(active=True, prompt="opening", seconds=2.0, mode="i2v"))
    renderer, _client, _out = _render(plan, source, tmp_path, "shift", monkeypatch)
    assert renderer._intro_offset == pytest.approx(2.0, abs=0.4)


def test_no_intro_means_no_offset(source, tmp_path, monkeypatch):
    renderer, _client, _out = _render(_plan(), source, tmp_path, "nooffset", monkeypatch)
    assert renderer._intro_offset == 0.0


# ------------------------------------------------------------------ voices ---

def test_deep_dark_profile_actually_lowers_the_pitch(tmp_path):
    import numpy as np

    src = _make_speech(tmp_path / "src.mp3", 2.0)

    def peak_hz(path: Path) -> float:
        raw = subprocess.run(
            [FF, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", "22050", "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
        return float(np.fft.rfftfreq(samples.size, 1 / 22050)[int(np.argmax(spectrum))])

    plain = peak_hz(src)
    deep = peak_hz(shape_voice(src, tmp_path / "deep.mp3", VOICE_PROFILES["deep_dark"], ffmpeg=FF))
    expected = plain * 2 ** (VOICE_PROFILES["deep_dark"].pitch_semitones / 12)
    assert deep == pytest.approx(expected, rel=0.06), (plain, deep, expected)
    assert deep < plain * 0.85, "the deep profile must be audibly lower"


def test_shaping_preserves_duration_within_the_tempo_factor(tmp_path):
    src = _make_speech(tmp_path / "src2.mp3", 3.0)
    profile = VOICE_PROFILES["deep_dark"]
    out = shape_voice(src, tmp_path / "deep2.mp3", profile, ffmpeg=FF)

    def duration(path: Path) -> float:
        raw = subprocess.run(
            [FF, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", "22050", "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
        return len(raw) / 2 / 22050

    # Only the profile's tempo should change the length; the pitch shift is
    # compensated. Reverb adds a short tail.
    assert duration(out) == pytest.approx(duration(src) / profile.tempo, abs=0.5)


# --------------------------------------------------------- voiceover switch --

def _apply_voiceover_switch(plan, enable_voiceover):
    """Mirror of the override JobManager._run applies to the plan."""
    if enable_voiceover is False:
        plan.audio.use_puter_tts = False
        plan.audio.tts_lines = []
        plan.audio.tts_script = ""
        plan.audio.background_music_gain = 1.0
    return plan


def _voiced_plan():
    return _plan(audio=AudioSpec(
        use_puter_tts=True, voice_accent="deep_dark",
        tts_lines=[TtsLine(text="Andhere mein ek shakti", start_time=0.5)],
    ))


@pytest.mark.parametrize("switch,expected", [(None, True), (True, True), (False, False)])
def test_voiceover_switch_states(switch, expected):
    plan = _apply_voiceover_switch(_voiced_plan(), switch)
    assert plan.audio.has_voiceover is expected


def test_voiceover_off_silences_tts_and_unducks_the_bed(source, tmp_path, monkeypatch):
    """The button must really stop synthesis, not just mute it."""
    plan = _apply_voiceover_switch(_voiced_plan(), False)
    puter = FakePuter()
    monkeypatch.setattr(video_renderer, "PuterVideoClient", lambda api_key="": FakeVideoClient())
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws_off", puter=puter,
    )
    out = renderer.render(tmp_path / "off.mp4")

    assert puter.tts_calls == [], "no speech should have been synthesised"
    assert plan.audio.background_music_gain == 1.0, "the bed should not be ducked"
    info = probe_clip(out)
    assert info["has_audio"] is True, "the original audio must survive"
    # Without narration the edit is exactly the two 2s segments.
    assert info["duration"] == pytest.approx(4.0, abs=0.6)


def test_voiceover_on_still_synthesises(source, tmp_path, monkeypatch):
    plan = _apply_voiceover_switch(_voiced_plan(), True)
    puter = FakePuter()
    monkeypatch.setattr(video_renderer, "PuterVideoClient", lambda api_key="": FakeVideoClient())
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws_on", puter=puter,
    )
    renderer.render(tmp_path / "on.mp4")
    assert len(puter.tts_calls) == 1
    assert puter.tts_calls[0][1] == "deep_dark"
