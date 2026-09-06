"""Every switch is connected to something.

The one promise this app makes that a feature test cannot check is that
nothing in the UI is decorative. A toggle can compile, render, animate and
post its value perfectly while the backend ignores the field entirely - and
that failure looks exactly like success from every side except the output.

So this walks the actual chain: the Android form field, the FastAPI form
parameter, the JobRequest attribute, and the argument the renderer receives.
A switch that stops somewhere along it fails here rather than silently doing
nothing on a real render.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jobs  # noqa: E402
import main  # noqa: E402
from video_renderer import VideoRenderer  # noqa: E402

ANDROID = (
    Path(__file__).resolve().parents[2]
    / "android/app/src/main/java/com/aivideo/editor"
)

# Every knob the render form accepts, and where each one has to arrive.
# "renderer" names the VideoRenderer argument it must end up as; None means it
# is consumed earlier in the pipeline (planning, micro-features, research).
SWITCHES = {
    "enable_captions": None,
    "voice_accent": None,
    "max_stickers": None,
    "max_inpaints": None,
    "theme": "theme",
    "enable_voiceover": None,
    "enable_animation": None,
    "max_animations": None,
    "enable_intro": None,
    "enable_outro": None,
    "auto_silence_cut": None,
    "auto_beat_sync": None,
    "auto_reframe": "auto_reframe",
    "enable_sfx": "enable_sfx",
    "enable_transitions": "enable_transitions",
    "auto_highlight": None,
    "review_plan": None,
    "export_preset": "export",
}


def _render_endpoint_source() -> str:
    return inspect.getsource(main.create_render_job)


def _android_source() -> str:
    return (ANDROID / "ApiClient.kt").read_text(encoding="utf-8")


@pytest.mark.parametrize("field", sorted(SWITCHES))
def test_the_api_accepts_the_field(field):
    source = _render_endpoint_source()
    assert re.search(rf"\b{field}\b.*Form\(", source) or f'alias="{field}"' in source, (
        f"/api/v1/render does not accept '{field}'"
    )


@pytest.mark.parametrize("field", sorted(SWITCHES))
def test_the_api_passes_the_field_into_the_job(field):
    source = _render_endpoint_source()
    assert re.search(rf"{field}\s*=", source), (
        f"'{field}' is accepted by the API but never put on the JobRequest"
    )


@pytest.mark.parametrize("field", sorted(SWITCHES))
def test_the_job_request_carries_the_field(field):
    carried = {item.name for item in dataclasses.fields(jobs.JobRequest)}
    assert field in carried, f"JobRequest has no '{field}'"


@pytest.mark.parametrize("field", sorted(SWITCHES))
def test_the_pipeline_actually_reads_the_field(field):
    """A field nothing reads is the same as no field at all."""
    source = inspect.getsource(jobs.JobManager)
    assert f"request.{field}" in source, (
        f"the pipeline never reads request.{field} - the switch does nothing"
    )


@pytest.mark.parametrize(
    "field", sorted(name for name, target in SWITCHES.items() if target)
)
def test_the_renderer_takes_the_argument(field):
    parameters = inspect.signature(VideoRenderer.__init__).parameters
    assert SWITCHES[field] in parameters, (
        f"VideoRenderer has no '{SWITCHES[field]}' parameter for '{field}'"
    )


@pytest.mark.parametrize("field", sorted(SWITCHES))
def test_the_android_client_sends_the_field(field):
    source = _android_source()
    assert f'addFormDataPart("{field}"' in source, (
        f"the Android client never sends '{field}'"
    )


# ------------------------------------------------- the toggles in the UI ----

# Booleans the user can actually see and press, and the state they bind to.
UI_TOGGLES = {
    "autoSilenceCut": "setAutoSilenceCut",
    "autoBeatSync": "setAutoBeatSync",
    "autoReframe": "setAutoReframe",
    "enableSfx": "setEnableSfx",
    "enableTransitions": "setEnableTransitions",
    "autoHighlight": "setAutoHighlight",
    "reviewPlan": "setReviewPlan",
}


@pytest.mark.parametrize("state,setter", sorted(UI_TOGGLES.items()))
def test_each_visible_toggle_is_bound_to_state_and_a_setter(state, setter):
    screen = (ANDROID / "MainActivity.kt").read_text(encoding="utf-8")
    view_model = (ANDROID / "EditorViewModel.kt").read_text(encoding="utf-8")

    assert f"state.{state}" in screen, f"nothing in the UI reads state.{state}"
    assert f"viewModel::{setter}" in screen, f"no control calls {setter}"
    assert f"fun {setter}(" in view_model, f"the view model has no {setter}"
    assert f"{state} = current.{state}" in view_model, (
        f"'{state}' never reaches the upload call - the toggle is decorative"
    )


def _routes_the_app_calls(source: str) -> set:
    """Every /api/v1 path in the Kotlin source, as a route shape.

    The app builds its URLs as templates - "${baseUrl(context)}/api/v1/jobs/$job"
    - so a path never starts at a quote, and the old pattern that required one
    matched nothing at all. Kotlin interpolations become the path parameter
    they stand for, so /api/v1/jobs/$job lines up with /api/v1/jobs/{job_id}.
    """
    found = set()
    for raw in re.findall(r"(/api/v1/[A-Za-z0-9/\-_$.{}]+)", source):
        # ${item.id} and $job are both one path segment with a value in it.
        shaped = re.sub(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_.]*", "{}", raw)
        found.add(shaped.rstrip("/"))
    return found


def test_the_route_scan_can_actually_fail():
    """The check above passed for months while matching nothing.

    A guard that cannot fail is worse than no guard, so this pins the scanner
    itself: it must find a real call, and must not find one that is absent.
    """
    calls = _routes_the_app_calls(_android_source())
    assert "/api/v1/render" in calls
    assert "/api/v1/voices" in calls
    assert "/api/v1/jobs/{}" in calls, "an interpolated path must still be seen"
    assert "/api/v1/nonsense" not in calls


def test_every_endpoint_the_app_calls_exists():
    """A client calling a route the server does not serve fails only at runtime."""
    served = set()
    for route in main.app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/api/v1"):
            served.add(re.sub(r"\{[^}]*\}", "{}", path).rstrip("/"))

    source = _android_source() + (ANDROID / "GalleryViewModel.kt").read_text(encoding="utf-8")
    missing = sorted(_routes_the_app_calls(source) - served)
    assert not missing, f"the app calls routes the API does not serve: {missing}"
