"""Export presets up to 4K 60fps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_presets import (  # noqa: E402
    DEFAULT_PRESET,
    PRESETS,
    bitrate_for,
    check_source_headroom,
    describe_cost,
    resolve_preset,
)
from puter_integration import ffmpeg_binary  # noqa: E402
from schemas import EditPlan, ProjectMeta, TimelineSegment  # noqa: E402
from video_renderer import VideoRenderer, probe_clip  # noqa: E402

FF = ffmpeg_binary()


# ------------------------------------------------------------------ presets --

def test_every_preset_is_coherent():
    for key, preset in PRESETS.items():
        assert preset.key == key
        assert preset.width % 2 == 0 and preset.height % 2 == 0, "encoders need even dimensions"
        assert 24 <= preset.fps <= 60
        assert preset.encoder() in ("libx264", "libx265")
        assert preset.video_bitrate().endswith("M")


def test_aliases_and_fallback():
    assert resolve_preset("4k").key == "4k60"
    assert resolve_preset("UHD").key == "4k60"
    assert resolve_preset("2K").key == "1440p60"
    assert resolve_preset("draft").key == "720p60"
    assert resolve_preset(None).key == DEFAULT_PRESET
    assert resolve_preset("something silly").key == DEFAULT_PRESET


def test_4k_is_actually_4k():
    preset = resolve_preset("4k")
    assert (preset.width, preset.height) == (2160, 3840)
    assert preset.fps == 60
    assert preset.pixels == 2160 * 3840
    assert preset.is_vertical


def test_landscape_4k_is_3840x2160():
    preset = resolve_preset("4k_landscape")
    assert (preset.width, preset.height) == (3840, 2160)
    assert not preset.is_vertical


def test_bitrate_scales_with_pixel_rate():
    small = bitrate_for(1080, 1920, 60)
    large = bitrate_for(2160, 3840, 60)
    assert int(large.rstrip("M")) > int(small.rstrip("M")) * 3


def test_hevc_asks_for_fewer_bits_than_h264():
    assert int(bitrate_for(2160, 3840, 60, "hevc").rstrip("M")) < \
        int(bitrate_for(2160, 3840, 60, "h264").rstrip("M"))


def test_bitrate_is_bounded():
    assert bitrate_for(64, 64, 24) == "4M"          # floor
    assert int(bitrate_for(7680, 4320, 120).rstrip("M")) <= 120  # ceiling


def test_relative_cost():
    assert describe_cost(PRESETS["1080p60"]) == pytest.approx(1.0)
    assert describe_cost(PRESETS["4k60"]) == pytest.approx(4.0)
    assert describe_cost(PRESETS["4k30"]) == pytest.approx(2.0)
    assert describe_cost(PRESETS["720p60"]) < 0.5


# ---------------------------------------------------------------- headroom --

def test_headroom_warns_about_an_upscale():
    warning = check_source_headroom(PRESETS["4k60"], [(1280, 720)])
    assert warning and "3.0x upscale" in warning


def test_headroom_is_quiet_when_the_source_can_fill_the_frame():
    assert check_source_headroom(PRESETS["1080p60"], [(1080, 1920)]) is None
    assert check_source_headroom(PRESETS["4k60"], [(3840, 2160)]) is None


def test_headroom_uses_the_weakest_source():
    warning = check_source_headroom(PRESETS["4k60"], [(3840, 2160), (640, 360)])
    assert warning and "360px" in warning


def test_headroom_ignores_unmeasured_sources():
    assert check_source_headroom(PRESETS["4k60"], [(0, 0)]) is None


# ------------------------------------------------------------- real render --

@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("export") / "src.mp4"
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=2160x3840:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    return path


def _one_second_plan() -> EditPlan:
    return EditPlan(
        project_meta=ProjectMeta(resolution="1080x1920", fps=60),
        edit_timeline=[TimelineSegment(start_time="00:00:00", end_time="00:00:01",
                                       source_index=0)],
    )


@pytest.mark.slow
def test_render_at_4k_produces_a_4k_file(source, tmp_path):
    renderer = VideoRenderer(
        plan=_one_second_plan(), clip_paths=[source],
        workspace=tmp_path / "ws4k", puter=None, export="4k60",
    )
    out = renderer.render(tmp_path / "out4k.mp4")
    info = probe_clip(out)
    assert (info["width"], info["height"]) == (2160, 3840), info
    assert round(info["fps"]) == 60, info


@pytest.mark.slow
def test_the_preset_overrides_the_plan_resolution(source, tmp_path):
    """A plan written for 1080p must still export at the requested preset."""
    plan = _one_second_plan()
    renderer = VideoRenderer(
        plan=plan, clip_paths=[source], workspace=tmp_path / "ws720",
        puter=None, export="720p60",
    )
    out = renderer.render(tmp_path / "out720.mp4")
    assert (probe_clip(out)["width"], probe_clip(out)["height"]) == (720, 1280)
    # The plan is updated so downstream consumers see the truth.
    assert plan.project_meta.resolution == "720x1280"


@pytest.mark.slow
def test_landscape_preset_renders_landscape(source, tmp_path):
    renderer = VideoRenderer(
        plan=_one_second_plan(), clip_paths=[source],
        workspace=tmp_path / "wswide", puter=None, export="1080p60_wide",
    )
    info = probe_clip(renderer.render(tmp_path / "wide.mp4"))
    assert (info["width"], info["height"]) == (1920, 1080)
