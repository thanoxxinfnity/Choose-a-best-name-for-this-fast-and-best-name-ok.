"""Audio enhancement, colour pop, chroma key and background replacement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enhancers import (  # noqa: E402
    AUDIO_PROFILES,
    _audio_codec_args,
    build_audio_chain,
    chroma_key_video,
    enhance_audio,
    measure_loudness,
    resolve_audio_profile,
)
from puter_integration import ffmpeg_binary  # noqa: E402
from vfx import chroma_key, color_pop, composite_over  # noqa: E402

FF = ffmpeg_binary()


def _saturation(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 1].mean())


# --------------------------------------------------------------- colour pop --

def test_color_pop_lifts_dull_colours_more_than_vivid_ones():
    """Vibrance, not saturation: the lift is proportional to the headroom.

    Both patches share a hue well away from skin, so only their saturation
    differs, and the comparison is on *relative* lift - a small multiplier on
    an already-high value can still be a large absolute step.
    """
    dull = np.full((8, 8, 3), (105, 115, 130), np.uint8)     # desaturated blue
    vivid = np.full((8, 8, 3), (20, 60, 250), np.uint8)      # saturated blue

    dull_base, vivid_base = _saturation(dull), _saturation(vivid)
    dull_lift = (_saturation(color_pop(dull, 0.6)) - dull_base) / dull_base
    vivid_lift = (_saturation(color_pop(vivid, 0.6)) - vivid_base) / vivid_base

    assert dull_lift > vivid_lift * 3, (dull_lift, vivid_lift)
    assert vivid_lift < 0.12, "an already vivid colour must barely move"


def test_color_pop_is_a_no_op_at_zero():
    frame = np.full((8, 8, 3), (100, 150, 200), np.uint8)
    assert np.array_equal(color_pop(frame, 0.0), frame)


def test_color_pop_protects_skin_tones():
    skin = np.full((8, 8, 3), (222, 170, 140), np.uint8)
    protected = _saturation(color_pop(skin, 0.8, protect_skin=True))
    unprotected = _saturation(color_pop(skin, 0.8, protect_skin=False))
    assert protected < unprotected


def test_color_pop_never_overflows():
    frame = np.full((8, 8, 3), (255, 0, 0), np.uint8)
    out = color_pop(frame, 1.0)
    assert out.dtype == np.uint8 and out.max() <= 255


# ---------------------------------------------------------------- chroma key --

def test_chroma_key_removes_the_screen_and_keeps_the_subject():
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:, :8] = (0, 177, 64)
    frame[:, 8:] = (200, 30, 30)
    rgba = chroma_key(frame)
    assert rgba.shape == (16, 16, 4)
    assert rgba[8, 2, 3] == 0, "the green screen should be fully transparent"
    assert rgba[8, 13, 3] == 255, "the subject should be fully opaque"


def test_chroma_key_tolerates_uneven_lighting():
    """A shadowed patch of the same green must still key out."""
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:8, :] = (0, 177, 64)
    frame[8:, :] = (0, 105, 38)      # same hue, much darker
    rgba = chroma_key(frame, tolerance=0.32)
    assert rgba[2, 8, 3] == 0
    assert rgba[13, 8, 3] == 0, "keying in chroma space should ignore brightness"


def test_chroma_key_suppresses_green_spill():
    spilled = np.full((8, 8, 3), (150, 210, 150), np.uint8)  # subject with a cast
    rgba = chroma_key(spilled, tolerance=0.1, spill=1.0)
    assert rgba[4, 4, 1] <= spilled[4, 4, 1], "green should not increase"


def test_composite_over_replaces_the_keyed_area():
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:, :8] = (0, 177, 64)
    frame[:, 8:] = (200, 30, 30)
    background = np.full((16, 16, 3), (10, 10, 200), np.uint8)
    out = composite_over(chroma_key(frame), background)
    assert tuple(out[8, 2]) == (10, 10, 200), "background shows through"
    assert tuple(out[8, 13]) == (200, 30, 30), "subject is untouched"


def test_composite_over_resizes_a_mismatched_background():
    rgba = np.dstack([np.zeros((20, 10, 3), np.uint8),
                      np.full((20, 10), 128, np.uint8)])
    background = np.full((5, 5, 3), 255, np.uint8)
    assert composite_over(rgba, background).shape == (20, 10, 3)


# ------------------------------------------------------------------- audio ---

def test_profiles_and_aliases():
    assert resolve_audio_profile("voice").key == "voice"
    assert resolve_audio_profile("").key == "balanced"
    assert resolve_audio_profile("nonsense").key == "balanced"
    assert resolve_audio_profile("off").key == "off"


def test_off_profile_has_an_empty_chain():
    assert build_audio_chain(AUDIO_PROFILES["off"]) == ""


def test_loudness_normalisation_comes_last_in_the_chain():
    chain = build_audio_chain(AUDIO_PROFILES["voice"]).split(",")
    assert chain[-1].startswith("alimiter")
    assert chain[-2].startswith("loudnorm"), chain


def test_codec_follows_the_container():
    """Forcing AAC into a .wav writes a file that decodes to silence."""
    assert _audio_codec_args(Path("a.wav")) == ["-c:a", "pcm_s16le"]
    assert _audio_codec_args(Path("a.mp3"))[:2] == ["-c:a", "libmp3lame"]
    assert _audio_codec_args(Path("a.m4a"))[:2] == ["-c:a", "aac"]
    assert _audio_codec_args(Path("a.unknown"))[:2] == ["-c:a", "aac"]


@pytest.fixture(scope="module")
def quiet_audio(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "quiet.wav"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=200:duration=5",
         "-f", "lavfi", "-i", "anoisesrc=d=5:c=pink:a=0.02",
         "-filter_complex",
         "[0][1]amix=inputs=2:weights=1 0.3,volume=-22dB,tremolo=f=4:d=0.5",
         "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.mark.slow
@pytest.mark.parametrize("key", ["voice", "music", "podcast", "balanced"])
def test_enhancement_hits_its_loudness_target(quiet_audio, tmp_path, key):
    profile = AUDIO_PROFILES[key]
    out = enhance_audio(quiet_audio, tmp_path / f"{key}.wav", profile)
    measured = measure_loudness(out)
    assert measured is not None, "the enhanced file must be measurable"
    assert measured == pytest.approx(profile.target_lufs, abs=2.5), (key, measured)


@pytest.mark.slow
def test_off_profile_leaves_the_audio_alone(quiet_audio, tmp_path):
    before = measure_loudness(quiet_audio)
    out = enhance_audio(quiet_audio, tmp_path / "off.wav", AUDIO_PROFILES["off"])
    assert measure_loudness(out) == pytest.approx(before, abs=0.1)


@pytest.mark.slow
def test_enhancement_raises_a_quiet_source(quiet_audio, tmp_path):
    before = measure_loudness(quiet_audio)
    out = enhance_audio(quiet_audio, tmp_path / "loud.wav", AUDIO_PROFILES["voice"])
    assert measure_loudness(out) > before + 20, "a -48 LUFS source must come up a lot"


def test_measure_loudness_reports_silence_as_unmeasurable(tmp_path):
    silent = tmp_path / "silent.wav"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
         "-c:a", "pcm_s16le", str(silent)],
        check=True, capture_output=True,
    )
    assert measure_loudness(silent) is None, "ebur128's -70 floor is not a measurement"


# ------------------------------------------------------- background removal --

@pytest.mark.slow
def test_chroma_key_video_replaces_the_background(tmp_path):
    source = tmp_path / "green.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=0x00B140:size=160x120:rate=12:duration=1",
         "-f", "lavfi", "-i", "color=c=red:size=40x40:rate=12:duration=1",
         "-filter_complex", "[0][1]overlay=x=60:y=40",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )
    background = tmp_path / "bg.png"
    cv2.imwrite(str(background), np.full((120, 160, 3), (200, 20, 20), np.uint8))  # BGR blue

    out = chroma_key_video(source, tmp_path / "keyed.mp4", background=background)
    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None

    corner = frame[10, 10]        # was green screen -> background
    middle = frame[60, 80]        # was the red square -> subject
    assert corner[0] > 120 and corner[2] < 90, f"background not applied: {corner}"
    assert middle[2] > 120 and middle[0] < 90, f"subject not preserved: {middle}"
