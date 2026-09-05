"""Procedural transitions, 3D type and impact particles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motion_graphics as mg  # noqa: E402

H, W = 120, 200


def _shot(colour):
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = colour
    return frame


A = _shot((30, 30, 200))
B = _shot((200, 30, 30))


def _closeness(frame, colour):
    """Mean absolute distance from a flat colour, 0 = identical."""
    return float(np.abs(frame.astype(int) - np.array(colour)).mean())


# ------------------------------------------------------------ every kind ----

@pytest.mark.parametrize("kind", mg.transition_names())
def test_a_transition_starts_on_the_outgoing_shot_and_ends_on_the_incoming(kind):
    """Whatever happens in the middle, the ends have to be the two shots.

    A transition that has not finished arriving by progress 1.0 leaves a
    ghost of the previous shot on the first frame of the new one.
    """
    start = mg.render_transition(A, B, 0.0, kind)
    end = mg.render_transition(A, B, 1.0, kind)
    assert _closeness(start, (30, 30, 200)) < 40, kind
    assert _closeness(end, (200, 30, 30)) < 40, kind


@pytest.mark.parametrize("kind", mg.transition_names())
def test_the_middle_is_neither_shot(kind):
    """If the midpoint is just one of the two shots, nothing is happening."""
    middle = mg.render_transition(A, B, 0.5, kind)
    assert _closeness(middle, (30, 30, 200)) > 15, kind
    assert _closeness(middle, (200, 30, 30)) > 15, kind


@pytest.mark.parametrize("kind", mg.transition_names())
def test_geometry_and_type_survive(kind):
    frame = mg.render_transition(A, B, 0.5, kind)
    assert frame.shape == A.shape and frame.dtype == np.uint8, kind


@pytest.mark.parametrize("kind", mg.transition_names())
def test_progress_outside_the_range_is_clamped_not_crashed(kind):
    assert mg.render_transition(A, B, -3.0, kind).shape == A.shape
    assert mg.render_transition(A, B, 9.0, kind).shape == A.shape


@pytest.mark.parametrize("kind", mg.transition_names())
def test_the_same_cut_looks_the_same_every_render(kind):
    """A transition that differs between two runs of one edit is a bug that
    only ever shows up as 'it looked better last time'."""
    first = mg.render_transition(A, B, 0.42, kind)
    second = mg.render_transition(A, B, 0.42, kind)
    assert np.array_equal(first, second), kind


def test_mismatched_frame_sizes_are_conformed():
    small = np.zeros((60, 100, 3), np.uint8)
    frame = mg.render_transition(A, small, 0.5, "radial_wipe")
    assert frame.shape == A.shape


def test_an_unknown_transition_dissolves_rather_than_failing():
    frame = mg.render_transition(A, B, 0.5, "does_not_exist")
    assert frame.shape == A.shape
    # A plain dissolve sits between the two.
    assert _closeness(frame, (30, 30, 200)) > 15
    assert _closeness(frame, (200, 30, 30)) > 15


# ------------------------------------------------------- specific behaviour --

def test_the_whip_pan_actually_smears():
    """Sliding one frame off and another on without blur is a slideshow."""
    sharp = np.zeros((H, W, 3), np.uint8)
    sharp[:, W // 2:] = 255  # one hard vertical edge
    whipped = mg.whip_pan(sharp, sharp, 0.5)
    # A vertical edge blurred horizontally produces intermediate columns.
    columns = whipped.mean(axis=(0, 2))
    assert ((columns > 20) & (columns < 235)).sum() > 3


def test_the_glitch_tears_the_frame_into_bands():
    frame = mg.glitch_slice(A, B, 0.5)
    # Measured on one channel, not the mean: the two shots here have the same
    # mean brightness by coincidence, so a mean cannot tell them apart.
    rows = frame[..., 0].mean(axis=1)
    assert rows.max() - rows.min() > 20


def test_the_glitch_is_seeded_so_two_cuts_differ_but_one_cut_repeats():
    one = mg.render_transition(A, B, 0.5, "glitch_slice", seed=1)
    two = mg.render_transition(A, B, 0.5, "glitch_slice", seed=2)
    again = mg.render_transition(A, B, 0.5, "glitch_slice", seed=1)
    assert not np.array_equal(one, two)
    assert np.array_equal(one, again)


def test_the_light_sweep_burns_brighter_than_either_shot():
    frame = mg.light_sweep(A, B, 0.5)
    assert frame.max() > max(A.max(), B.max())


def test_the_radial_wipe_opens_from_the_centre():
    frame = mg.radial_wipe(A, B, 0.35)
    centre = frame[H // 2, W // 2]
    corner = frame[2, 2]
    assert _closeness(centre, (200, 30, 30)) < _closeness(corner, (200, 30, 30))


# ------------------------------------------------------------- the frames ---

def test_transition_frames_excludes_both_endpoints():
    """Emitting them would repeat frames the timeline already has - a stutter."""
    frames = mg.transition_frames(A, B, "radial_wipe", 5)
    assert len(frames) == 5
    # The property is that no emitted frame *is* an endpoint, not that the
    # first one already looks different - a slow-opening wipe barely has.
    assert not np.array_equal(frames[0], A)
    assert not np.array_equal(frames[-1], B)


def test_no_frames_asked_for_means_none_produced():
    assert mg.transition_frames(A, B, "whip_pan", 0) == []
    assert mg.transition_frames(A, B, "whip_pan", -4) == []


# ------------------------------------------------------------- the palette --

def test_every_theme_names_transitions_that_exist():
    for theme, palette in mg.THEME_TRANSITIONS.items():
        for kind in palette:
            assert kind in mg.TRANSITIONS, f"{theme} -> {kind}"


def test_a_long_edit_rotates_through_a_themes_transitions():
    """Using one transition for every cut is the same as using none."""
    picked = [mg.pick_transition("ae_hype", index) for index in range(6)]
    assert len(set(picked)) > 1
    assert picked[0] == picked[3]  # three in the ae_hype palette


def test_an_unknown_theme_still_gets_a_transition():
    assert mg.pick_transition("no_such_theme", 0) in mg.TRANSITIONS


# ---------------------------------------------------------------- 3D type ---

def test_the_title_has_a_visible_extrusion_not_just_a_shadow():
    """The side wall has to be its own shade, or it is a drop shadow."""
    frame = mg.extruded_title("A", (400, 240), 1.0)
    opaque = frame[..., 3] > 200
    assert opaque.sum() > 200

    rgb = frame[..., :3][opaque]
    # Face, outline and extrusion are three distinct tones.
    tones = np.unique((rgb // 40).sum(axis=1))
    assert len(tones) >= 3, tones


def test_the_title_punches_in_and_settles():
    small = mg.extruded_title("HI", (400, 240), 0.05)
    peak = mg.extruded_title("HI", (400, 240), 0.45)
    settled = mg.extruded_title("HI", (400, 240), 1.0)

    def ink(frame):
        return int((frame[..., 3] > 128).sum())

    assert ink(small) < ink(settled), "it should arrive small"
    assert ink(peak) > ink(settled), "it should overshoot before settling"


def test_the_title_fades_up_from_nothing():
    assert mg.extruded_title("HI", (400, 240), 0.0)[..., 3].max() < 40


def test_empty_text_makes_an_empty_frame():
    frame = mg.extruded_title("   ", (200, 120), 0.7)
    assert frame.shape == (120, 200, 4)
    assert frame[..., 3].max() == 0


def test_devanagari_titles_render():
    frame = mg.extruded_title("नमस्ते", (500, 240), 1.0)
    assert int((frame[..., 3] > 128).sum()) > 100


# -------------------------------------------------------------- particles ---

def test_particles_add_light_and_burn_out():
    base = _shot((20, 20, 20))
    early = mg.particle_burst(base.copy(), 0.05)
    late = mg.particle_burst(base.copy(), 0.40, life=0.45)

    assert early.mean() > base.mean(), "no sparks were drawn"
    assert late.mean() < early.mean(), "they should be fading"


def test_particles_outside_their_life_leave_the_frame_alone():
    base = _shot((20, 20, 20))
    assert np.array_equal(mg.particle_burst(base.copy(), 0.9, life=0.45), base)
    assert np.array_equal(mg.particle_burst(base.copy(), -0.1), base)


def test_particles_spread_out_from_their_centre():
    base = _shot((10, 10, 10))
    burst = mg.particle_burst(base.copy(), 0.35, centre=(0.25, 0.5), seed=3)
    left = burst[:, : W // 2].mean()
    right = burst[:, W // 2 :].mean()
    assert left > right


def test_the_same_hit_sparks_the_same_way():
    base = _shot((10, 10, 10))
    assert np.array_equal(
        mg.particle_burst(base.copy(), 0.2, seed=5),
        mg.particle_burst(base.copy(), 0.2, seed=5),
    )
