"""The synthesised sound-effects library and its placement."""

from __future__ import annotations

import array
import math
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sfx  # noqa: E402
from puter_integration import ffmpeg_binary  # noqa: E402

FF = ffmpeg_binary()


def _peak(path: Path) -> float:
    """Sample peak, 0..1, read straight out of the wav rather than via ffmpeg.

    Measuring the file independently of the code under test is the point: the
    normaliser and the check must not share a measurement, or a bug in the
    measurement passes itself.
    """
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2
        samples = array.array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    if not samples:
        return 0.0
    return max(abs(value) for value in samples) / 32768.0


def _duration(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


# ------------------------------------------------------------- the library --

def test_every_effect_renders_and_hits_its_target_level(tmp_path):
    """A whoosh that is 10dB quieter than the impact next to it is a bug.

    Each effect declares a peak; a band-pass or a steep decay eats most of the
    source's amplitude, so the level has to be measured and corrected rather
    than guessed at with a fixed gain.
    """
    off_target = []
    for key, spec in sfx.LIBRARY.items():
        path = sfx.render_sfx(key, tmp_path / f"{key}.wav", ffmpeg=FF)
        assert path is not None and path.exists(), f"{key} did not render"
        peak = _peak(path)
        if abs(peak - spec.target_peak) > 0.06:
            off_target.append(f"{key}: wanted {spec.target_peak}, got {peak:.3f}")
    assert not off_target, "; ".join(off_target)


def test_effects_are_the_length_they_claim(tmp_path):
    for key, spec in sfx.LIBRARY.items():
        path = sfx.render_sfx(key, tmp_path / f"{key}.wav", ffmpeg=FF)
        assert abs(_duration(path) - spec.duration) < 0.05, key


def test_rendering_twice_gives_the_same_bytes(tmp_path):
    """Noise sources are seeded, so a probe and the final pass agree.

    Without a seed the normaliser measures one noise burst and gains a
    different one, which is how an effect ends up several dB off target.
    """
    first = sfx.render_sfx("whoosh", tmp_path / "a" / "whoosh.wav", ffmpeg=FF)
    second = sfx.render_sfx("whoosh", tmp_path / "b" / "whoosh.wav", ffmpeg=FF)
    assert first.read_bytes() == second.read_bytes()


def test_the_cache_name_changes_when_the_recipe_does(monkeypatch):
    """A retuned effect must not keep serving the wav from before the fix."""
    before = sfx.cache_name("impact_bright")
    tweaked = dict(sfx.LIBRARY)
    tweaked["impact_bright"] = sfx.SfxSpec(
        **{**sfx.LIBRARY["impact_bright"].__dict__, "target_peak": 0.9}
    )
    monkeypatch.setattr(sfx, "LIBRARY", tweaked)
    assert sfx.cache_name("impact_bright") != before


def test_an_unknown_effect_renders_nothing(tmp_path):
    assert sfx.render_sfx("does_not_exist", tmp_path / "x.wav", ffmpeg=FF) is None


# ------------------------------------------------------------- placement ----

def test_impacts_win_over_cuts_at_the_same_moment():
    """Two effects stacked on one frame read as a mistake, not as emphasis."""
    placements = sfx.plan_placements(
        cut_times=[2.0, 5.0], impact_times=[2.0], theme="anime_edits", duration=10.0
    )
    at_two = {p.key for p in placements if abs(p.at - 2.0) < 0.01}
    # A riser may share the moment - it builds *into* the hit - but the cut's
    # own whoosh must give way rather than double up on it.
    assert "impact" in at_two
    assert "whoosh" not in at_two


def test_a_cut_far_from_any_impact_still_gets_its_whoosh():
    placements = sfx.plan_placements(
        cut_times=[5.0], impact_times=[2.0], theme="anime_edits", duration=10.0
    )
    assert any(p.key == "whoosh" and abs(p.at - 5.0) < 0.01 for p in placements)


def test_a_riser_is_laid_in_ahead_of_the_last_impact():
    placements = sfx.plan_placements(
        cut_times=[1.0], impact_times=[3.0, 8.0], theme="haunted", duration=12.0
    )
    risers = [p for p in placements if p.key == "riser"]
    assert len(risers) == 1
    assert risers[0].at == pytest.approx(8.0)


def test_the_opening_frame_never_gets_an_effect():
    """An effect on frame zero is clipped by its own lead-in and just clicks."""
    placements = sfx.plan_placements(
        cut_times=[0.0, 0.03, 4.0], impact_times=[0.0], duration=10.0
    )
    assert all(p.at > 0.05 for p in placements)


def test_placements_past_the_end_are_dropped():
    placements = sfx.plan_placements(
        cut_times=[2.0, 30.0], impact_times=[], duration=10.0
    )
    assert all(p.at < 10.0 for p in placements)


def test_a_theme_with_no_build_effect_gets_no_riser():
    placements = sfx.plan_placements(
        cut_times=[1.0], impact_times=[4.0], theme="normal", duration=8.0
    )
    assert {p.key for p in placements} == {"whoosh_soft", "impact"}


def test_the_effect_count_is_capped():
    placements = sfx.plan_placements(
        cut_times=[float(i) for i in range(1, 60)], theme="anime_edits",
        duration=90.0, max_effects=8,
    )
    assert len(placements) == 8


def test_every_theme_names_effects_that_exist():
    for theme, palette in sfx.THEME_SFX.items():
        for role, key in palette.items():
            assert key == "" or key in sfx.LIBRARY, f"{theme}/{role} -> {key}"


# ------------------------------------------------------------- the track ----

def test_the_mixed_track_is_the_right_length_and_does_not_clip(tmp_path):
    """Overlapping effects sum well past full scale if nothing holds them back."""
    placements = sfx.plan_placements(
        cut_times=[1.0, 2.0, 3.0, 4.0], impact_times=[2.5, 5.0],
        theme="anime_edits", duration=8.0,
    )
    track = sfx.build_sfx_track(placements, tmp_path / "track.wav", duration=8.0,
                                workspace=tmp_path / "cache", ffmpeg=FF)
    assert track is not None and track.exists()
    assert _duration(track) == pytest.approx(8.0, abs=0.05)
    peak = _peak(track)
    assert 0.2 < peak < 0.95, peak


def test_a_whoosh_starts_before_the_cut_it_motivates(tmp_path):
    """The whoosh has to be audible *into* the cut, so it leads it."""
    placements = [sfx.SfxPlacement("whoosh", 3.0)]
    track = sfx.build_sfx_track(placements, tmp_path / "t.wav", duration=6.0,
                                workspace=tmp_path / "cache", ffmpeg=FF)
    with wave.open(str(track)) as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        samples = array.array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    # Something must be sounding a tenth of a second before the cut lands.
    index = int(2.9 * rate) * channels
    window = samples[index:index + channels * int(0.05 * rate)]
    assert max(abs(v) for v in window) > 200


def test_no_placements_means_no_track(tmp_path):
    assert sfx.build_sfx_track([], tmp_path / "t.wav", duration=5.0, ffmpeg=FF) is None


def test_the_catalogue_describes_every_effect():
    described = sfx.describe_library()
    assert {item["key"] for item in described} == set(sfx.LIBRARY)
    assert all(item["label"] and item["duration"] > 0 for item in described)
