"""The free image route: sizing rules, model choice, and the fallback order.

These are offline. The horde itself is exercised by hand against the live
service - what is worth pinning here is the logic that decides *what* to ask
it for, because that is what breaks silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import horde  # noqa: E402
import image_models  # noqa: E402


# --- sizing ---------------------------------------------------------------

def test_dimensions_snap_to_the_grid_the_service_accepts():
    # The horde takes multiples of 64 and rejects anything else outright.
    for value in (100, 513, 767, 1345):
        assert horde.snap(value) % 64 == 0
    assert horde.snap(10) == 64, "must never round down to zero"


def test_oversized_requests_shrink_instead_of_being_refused():
    width, height = horde.fit_budget(2048, 3584)
    assert width * height <= horde.MAX_ANON_PIXELS
    assert width % 64 == 0 and height % 64 == 0
    # The frame still has to fit the video, so the shape must survive.
    assert abs((width / height) - (2048 / 3584)) < 0.05


def test_a_size_already_inside_the_budget_is_left_alone():
    assert horde.fit_budget(768, 1344) == (768, 1344)


# --- model choice ---------------------------------------------------------

def test_a_look_picks_a_model_that_has_workers_right_now():
    online = [{"name": "Rag Illustrious Mix", "count": 8}]
    # Nova is the first choice but is offline here, so the next one wins.
    assert horde.pick_model("anime", online=online) == "Rag Illustrious Mix"


def test_photoreal_and_anime_do_not_return_the_same_model():
    online = [{"name": m.name, "count": 5} for m in horde.CURATED]
    assert horde.pick_model("anime", online) != horde.pick_model("photoreal", online)


def test_an_unreachable_model_list_still_yields_a_model():
    # The picker must not be able to fail a render just because the status
    # route is having a bad minute.
    assert horde.pick_model("anime", online=[]) == horde.DEFAULT_ANIME


# --- the catalogue --------------------------------------------------------

def test_horde_models_are_free_and_need_no_key():
    entries = image_models.no_key_models()
    assert entries, "the catalogue should offer at least one keyless model"
    assert all(m.free and m.provider == "horde" for m in entries)


def test_the_horde_prefix_is_stripped_before_the_service_sees_it():
    model = image_models.resolve("horde:ICBINP XL")
    assert model.provider == "horde"
    # Sending "horde:ICBINP XL" as a model name matches nothing on the horde.
    assert image_models.horde_name(model) == "ICBINP XL"
    assert image_models.horde_name(image_models.resolve("flux")) == "flux"


def test_both_services_appear_under_one_look_filter():
    providers = {m.provider for m in image_models.for_look("photoreal")}
    assert providers == {"pollinations", "horde"}


def test_generate_routes_a_horde_model_to_the_horde(monkeypatch, tmp_path):
    seen = {}

    def fake(prompt, destination, **kwargs):
        seen.update(kwargs, prompt=prompt)
        Path(destination).write_bytes(b"x" * 4096)
        return Path(destination)

    monkeypatch.setattr(horde, "generate", fake)
    out = image_models.generate("a cat", tmp_path / "a.webp", api_key="",
                                model="horde:Juggernaut XL", horde_key="abc")
    assert out.exists()
    assert seen["model"] == "Juggernaut XL"   # prefix gone
    assert seen["api_key"] == "abc"


def test_a_horde_model_needs_no_pollinations_key(monkeypatch, tmp_path):
    # The whole point of this route: it works when the paid one cannot.
    monkeypatch.setattr(horde, "generate",
                        lambda prompt, destination, **kw: Path(destination))
    image_models.generate("a cat", tmp_path / "a.webp", api_key="",
                          model="horde:Nova Anime XL")


def test_a_pollinations_model_without_a_key_still_refuses():
    with pytest.raises(RuntimeError, match="Pollinations key"):
        image_models.generate("a cat", "/tmp/nope.webp", api_key="", model="flux")


# --- batching -------------------------------------------------------------

def test_the_whole_batch_is_queued_before_anything_is_collected():
    """The reason this module exists.

    Queue wait dominates generation time, so jobs must wait concurrently. If
    a future edit made this collect each image before submitting the next,
    a twenty-shot batch would go from minutes to hours - and nothing else
    here would fail. So assert the interleaving directly.
    """
    order = []

    def fake_submit(prompt, destination, **kwargs):
        order.append(("submit", prompt))
        return horde.Ticket(job_id=f"j{len(order)}", prompt=prompt,
                            destination=Path(destination))

    def fake_collect(tickets, **kwargs):
        order.append(("collect", len(tickets)))
        for ticket in tickets:
            ticket.done = True
        return list(tickets)

    import unittest.mock as mock
    with mock.patch.object(horde, "submit", fake_submit), \
         mock.patch.object(horde, "collect_all", fake_collect), \
         mock.patch.object(horde, "pick_model", lambda *a, **k: "M"):
        horde.generate_many(["a", "b", "c"], ["/tmp/1", "/tmp/2", "/tmp/3"])

    assert [step for step, _ in order] == ["submit", "submit", "submit", "collect"]
    assert order[-1] == ("collect", 3), "all three must be collected together"


def test_one_rejected_submission_does_not_lose_the_rest_of_the_batch():
    def fake_submit(prompt, destination, **kwargs):
        if prompt == "b":
            raise horde.HordeError("refused")
        return horde.Ticket(job_id="j", prompt=prompt, destination=Path(destination))

    import unittest.mock as mock
    with mock.patch.object(horde, "submit", fake_submit), \
         mock.patch.object(horde, "collect_all",
                           lambda tickets, **k: [setattr(t, "done", True) for t in tickets]), \
         mock.patch.object(horde, "pick_model", lambda *a, **k: "M"):
        tickets = horde.generate_many(["a", "b", "c"], ["/tmp/1", "/tmp/2", "/tmp/3"])

    assert len(tickets) == 3
    assert tickets[1].failed == "refused"
    assert [t.prompt for t in tickets] == ["a", "b", "c"], "order must be preserved"


# --- the censored-placeholder trap ----------------------------------------

class _Response:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _status_with(generation: dict):
    """A finished status response carrying one generation."""
    return _Response({"generations": [generation], "kudos": 18.0})


def _png(width: int, height: int) -> bytes:
    """A PNG that is realistically sized.

    A flat fill compresses to under the 2KB floor and would trip that guard
    before the one under test, so give it noise the way a real image has it.
    """
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _ticket(tmp_path, width=768, height=1344):
    return horde.Ticket(job_id="j1", prompt="a rooftop",
                        destination=tmp_path / "shot.webp",
                        extra={"width": width, "height": height})


def _fake_status(monkeypatch, generation: dict, image: bytes):
    monkeypatch.setattr(horde, "_session",
                        lambda: type("S", (), {"get": staticmethod(
                            lambda *a, **k: _status_with(generation))})())
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(content=image))


def test_a_censored_image_is_a_failure_even_though_the_job_says_ok(monkeypatch, tmp_path):
    """The trap that made six 'successful' images black CENSORED cards.

    The horde marks such a job done, with state "ok" and a real image
    attached - the image is just a placeholder. Only the ``censored`` field
    tells the truth, so a check on ``state`` alone passes every one of them
    straight into the render.
    """
    _fake_status(monkeypatch,
                 {"img": "https://x/i.webp", "state": "ok", "censored": True,
                  "worker_name": "w"},
                 _png(512, 512))

    with pytest.raises(horde.HordeError, match="safety filter"):
        horde._collect(_ticket(tmp_path), api_key="x")
    assert not (tmp_path / "shot.webp").exists(), "a placeholder must not be saved"


def test_the_wrong_size_coming_back_is_refused(monkeypatch, tmp_path):
    # A square where a portrait was asked for cannot be cut into the frame,
    # whatever the flags claim about it.
    _fake_status(monkeypatch,
                 {"img": "https://x/i.webp", "state": "ok", "censored": False,
                  "worker_name": "w"},
                 _png(512, 512))

    with pytest.raises(horde.HordeError, match="768x1344"):
        horde._collect(_ticket(tmp_path), api_key="x")


def test_a_real_image_of_the_right_size_is_saved(monkeypatch, tmp_path):
    _fake_status(monkeypatch,
                 {"img": "https://x/i.webp", "state": "ok", "censored": False,
                  "worker_name": "w", "model": "Nova Anime XL"},
                 _png(768, 1344))

    ticket = _ticket(tmp_path)
    out = horde._collect(ticket, api_key="x")
    assert out.exists() and out.stat().st_size > 2048
    assert ticket.extra["returned_size"] == (768, 1344)
    assert ticket.extra["worker"] == "w"


# --- refusals that look like successes ------------------------------------

def test_an_all_black_frame_is_rejected(monkeypatch, tmp_path):
    """The second way a refusal arrives, and the quietest.

    With the safety filter turned off, a blocked prompt comes back the right
    size with ``censored`` false - and every pixel zero. Two measured runs
    returned exactly this. Nothing in the envelope says so, so if the pixels
    are not checked a black frame goes straight into the cut.
    """
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((1344, 768, 3), dtype=np.uint8)).save(buffer, format="PNG")

    _fake_status(monkeypatch,
                 {"img": "https://x/i.webp", "state": "ok", "censored": False,
                  "worker_name": "w"},
                 buffer.getvalue())

    with pytest.raises(horde.HordeError, match="empty black frame"):
        horde._collect(_ticket(tmp_path), api_key="x")
    assert not (tmp_path / "shot.webp").exists()


def test_a_dark_but_real_frame_is_kept(monkeypatch, tmp_path):
    # A night shot is dark. It must not be mistaken for a refusal, so the
    # test that rejects blank frames has to be about variation, not level.
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    dark = rng.integers(0, 24, (1344, 768, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(dark).save(buffer, format="PNG")

    _fake_status(monkeypatch,
                 {"img": "https://x/i.webp", "state": "ok", "censored": False,
                  "worker_name": "w"},
                 buffer.getvalue())

    assert horde._collect(_ticket(tmp_path), api_key="x").exists()


# --- the submission rate limit --------------------------------------------

def test_a_rate_limited_submission_is_retried_not_dropped(monkeypatch):
    """The horde allows 2 submissions a second and 429s the rest.

    A twenty-shot batch submitted in a tight loop loses most of itself to
    this, and each loss looks like an ordinary refusal, so it must be
    retried rather than recorded as failed.
    """
    calls = []

    class _Post:
        def __init__(self, code):
            self.status_code, self.text = code, "2 per 1 second"

        def json(self):
            return {"id": "abc", "kudos": 6.0}

    def post(url, **kwargs):
        calls.append(url)
        return _Post(429 if len(calls) < 3 else 202)

    monkeypatch.setattr(horde, "_session",
                        lambda: type("S", (), {"post": staticmethod(post)})())
    monkeypatch.setattr(horde.time, "sleep", lambda *a: None)

    ticket = horde.submit("a rooftop", "/tmp/x.webp", model="M")
    assert ticket.job_id == "abc"
    assert len(calls) == 3, "it must keep trying, not give up on the first 429"


def test_submissions_are_spaced_out(monkeypatch):
    # The retry above is the safety net; the spacing is what stops the batch
    # tripping the limit in the first place.
    slept = []
    monkeypatch.setattr(horde.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(horde, "submit",
                        lambda prompt, destination, **k: horde.Ticket(
                            job_id="j", prompt=prompt, destination=Path(destination)))
    monkeypatch.setattr(horde, "collect_all", lambda tickets, **k: list(tickets))

    horde.generate_many(["a", "b", "c"], ["/tmp/1", "/tmp/2", "/tmp/3"], model="M")
    assert len([s for s in slept if s >= horde.SUBMIT_INTERVAL]) >= 2


# --- retrying a refusal on another model ----------------------------------

def test_a_refused_shot_is_retried_on_a_different_model(monkeypatch):
    """One model refused six prompts out of six in testing.

    Without a retry on another model, a whole batch comes back empty for a
    reason that has nothing to do with what was asked for.
    """
    tried = []

    def fake_round(prompts, destinations, model, *args, **kwargs):
        tried.append((model, list(prompts)))
        out = []
        for prompt, destination in zip(prompts, destinations):
            ticket = horde.Ticket(job_id="j", prompt=prompt,
                                  destination=Path(destination))
            if model == horde.CURATED[0].name and prompt == "b":
                ticket.failed = "The worker returned an empty black frame."
            else:
                ticket.done = True
            out.append(ticket)
        return out

    monkeypatch.setattr(horde, "_round", fake_round)
    results = horde.generate_many(["a", "b"], ["/tmp/1", "/tmp/2"], look="anime")

    assert [t.done for t in results] == [True, True]
    assert len(tried) == 2, "the refused shot should have had a second model"
    assert tried[1][1] == ["b"], "only the refused shot should be redone"
    assert tried[1][0] != tried[0][0], "and on a different model"


def test_a_failure_that_another_model_cannot_fix_is_not_retried(monkeypatch):
    # Retrying a job the horde faulted just burns another queue wait.
    tried = []

    def fake_round(prompts, destinations, model, *args, **kwargs):
        tried.append(model)
        ticket = horde.Ticket(job_id="j", prompt=prompts[0],
                              destination=Path(destinations[0]))
        ticket.failed = "The horde faulted this job."
        return [ticket]

    monkeypatch.setattr(horde, "_round", fake_round)
    results = horde.generate_many(["a"], ["/tmp/1"], look="anime")
    assert len(tried) == 1
    assert not results[0].done
