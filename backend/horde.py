"""Free image generation on the AI Horde, a volunteer GPU pool.

Worth being exact about what this is, because it is easy to reach for it
expecting the wrong thing: the horde generates **images**. It has no video
route at all - the published API is 59 routes and none of them make a
video, so a clip still has to be animated from stills by the renderer.

What it does give is images with no bill and no quota, which nothing else
here does. The cost is paid in latency instead of money: work is ranked by
kudos, and a request from the shared anonymous account starts near the back.
A measured 512x512 job sat at queue position 189 and took 281 seconds, of
which only about 30 were spent generating.

That ratio is the whole design of this module: jobs must wait concurrently,
not in series. Measured, six portrait shots queued together finished in 490
seconds against 245 for a single one alone - twice the wall time for six
times the work, because only two workers were serving us. Not free, but the
serial version of the same batch would have been about half an hour. So
``generate_many`` submits the whole batch before collecting any of it, and
single ``generate`` is just the one-item case.

The other thing measured here, and the one that bites: a job blocked by a
worker's safety filter comes back **done, state "ok", with an image
attached** - and the image is a black card reading CENSORED. Six harmless
anime prompts came back that way in a row. ``censored`` is the only field
that says so, which is why ``_collect`` checks it and the pixel dimensions
rather than trusting the job state.

A registered account (free, aihorde.net/register) has its own kudos and
starts further forward; the anonymous key is the fallback so the app works
with no signup at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

BASE = "https://aihorde.net/api/v2"
# The shared account everyone falls back to. It works, it is just slow.
ANON_KEY = "0000000000"
CLIENT_AGENT = "moja-ai:1.0:https://github.com/moja-ai"

# The horde takes multiples of 64 only, and refuses very large jobs from
# low-kudos accounts.
SIZE_STEP = 64
MAX_ANON_PIXELS = 1024 * 1024

# The API answers "2 per 1 second" with HTTP 429 above this. A tight submit
# loop over a twenty-shot batch therefore loses most of the batch, so leave a
# margin rather than sitting on the limit.
SUBMIT_INTERVAL = 0.75
SUBMIT_RETRIES = 4


@dataclass(frozen=True)
class HordeModel:
    name: str            # exactly as the horde spells it
    look: str            # "anime" | "photoreal"
    note: str = ""


# Picked from the 165 models online by worker count, because a model with
# two workers queues behind one with ten no matter how good it is.
CURATED: tuple[HordeModel, ...] = (
    # Order is by what actually drew, not by reputation. AlbedoBase produced
    # the only real image in the first round of testing; Nova refused six
    # prompts out of six - including one with no people in it at all - so it
    # sits behind the models that worked.
    HordeModel("AlbedoBase XL (SDXL)", "anime", "General SDXL. Proven to draw."),
    HordeModel("Rag Illustrious Mix", "anime", "Illustrious base, sharp lineart."),
    HordeModel("526Mix-Animated", "anime", "Softer, more painterly cels."),
    HordeModel("Nova Anime XL", "anime",
               "Clean modern anime, but its filter refuses a lot."),
    HordeModel("ICBINP XL", "photoreal", "Photographic people."),
    HordeModel("Juggernaut XL", "photoreal", "Photoreal with strong lighting."),
    HordeModel("Realistic Vision", "photoreal", "Portraits."),
    HordeModel("AbsoluteReality", "photoreal", "Places and backgrounds."),
)

DEFAULT_ANIME = "AlbedoBase XL (SDXL)"
DEFAULT_PHOTOREAL = "ICBINP XL"


class HordeError(RuntimeError):
    """The horde refused or lost the job, with its own words where it gave any."""


def _session():
    import requests  # noqa: PLC0415

    return requests.Session()


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "apikey": api_key or ANON_KEY,
        "Client-Agent": CLIENT_AGENT,
        "Content-Type": "application/json",
    }


def snap(value: int, step: int = SIZE_STEP) -> int:
    """Round a dimension to what the horde accepts."""
    return max(step, int(round(value / step)) * step)


def fit_budget(width: int, height: int, budget: int = MAX_ANON_PIXELS) -> tuple[int, int]:
    """Shrink to the pixel budget, keeping the aspect ratio and the 64 grid."""
    width, height = snap(width), snap(height)
    if width * height <= budget:
        return width, height
    scale = (budget / (width * height)) ** 0.5
    return snap(width * scale), snap(height * scale)


def online_models(timeout: int = 30) -> List[dict]:
    """Every image model with at least one worker, most workers first."""
    response = _session().get(f"{BASE}/status/models?type=image", timeout=timeout)
    response.raise_for_status()
    models = [m for m in response.json() if m.get("count", 0) > 0]
    models.sort(key=lambda m: -m["count"])
    return models


def pick_model(look: str = "anime", online: Optional[Sequence[dict]] = None) -> str:
    """The curated model for a look that actually has workers right now."""
    wanted = [m.name for m in CURATED if m.look == look] or [DEFAULT_ANIME]
    try:
        live = {m["name"] for m in (online if online is not None else online_models())}
    except Exception as exc:  # the picker must not fail the render
        logger.warning("horde model list unreachable (%s); using %s", exc, wanted[0])
        return wanted[0]
    for name in wanted:
        if name in live:
            return name
    return wanted[0]


@dataclass
class Ticket:
    """One submitted job, before its image exists."""

    job_id: str
    prompt: str
    destination: Path
    kudos: float = 0.0
    done: bool = False
    failed: str = ""
    waited: float = 0.0
    extra: dict = field(default_factory=dict)


def submit(prompt: str, destination, api_key: str = ANON_KEY,
           model: str = "", look: str = "anime",
           width: int = 768, height: int = 1344, steps: int = 24,
           sampler: str = "k_euler_a", cfg_scale: float = 7.0,
           seed: int = 0, nsfw: bool = False, censor: bool = True,
           timeout: int = 60) -> Ticket:
    """Put one image in the queue and return its ticket. Does not wait."""
    width, height = fit_budget(width, height)
    chosen = model or pick_model(look)
    params: Dict[str, object] = {
        "sampler_name": sampler, "width": width, "height": height,
        "steps": steps, "cfg_scale": cfg_scale, "n": 1,
    }
    if seed:
        params["seed"] = str(seed)
    body = {
        "prompt": prompt,
        "params": params,
        "nsfw": nsfw,
        # Kept on deliberately. Turning it off does not get the picture drawn
        # - a blocked job comes back as a blank black frame instead of a card
        # - and leaving it on at least sets the ``censored`` flag, which is
        # the one unambiguous signal that a refusal happened.
        "censor_nsfw": censor and not nsfw,
        "r2": True,
        # Slow and low-VRAM workers roughly double the pool we can land on,
        # and at the back of the queue any worker is better than a fast one
        # we are not eligible for.
        "slow_workers": True,
        "extra_slow_workers": True,
        "models": [chosen],
    }
    session = _session()
    for attempt in range(SUBMIT_RETRIES):
        response = session.post(f"{BASE}/generate/async", headers=_headers(api_key),
                                json=body, timeout=timeout)
        if response.status_code != 429:
            break
        # Rate limited rather than refused: this job is still perfectly good,
        # it was only offered too quickly. Backing off keeps the batch whole.
        time.sleep(SUBMIT_INTERVAL * (2 ** attempt))
    if response.status_code == 401:
        raise HordeError("The horde rejected this API key.")
    if response.status_code == 429:
        raise HordeError("The horde is rate limiting submissions; try a smaller batch.")
    if response.status_code >= 400:
        raise HordeError(f"The horde refused the job: HTTP {response.status_code} "
                         f"{response.text[:200]}")
    payload = response.json()
    return Ticket(job_id=payload["id"], prompt=prompt,
                  destination=Path(destination), kudos=payload.get("kudos", 0.0),
                  extra={"model": chosen, "width": width, "height": height})


def check(job_id: str, api_key: str = ANON_KEY, timeout: int = 30) -> dict:
    """Where the job is in the queue. Cheap - safe to call in a poll loop."""
    response = _session().get(f"{BASE}/generate/check/{job_id}",
                              headers=_headers(api_key), timeout=timeout)
    response.raise_for_status()
    return response.json()


def _is_blank(data: bytes) -> bool:
    """True when the image carries no picture at all.

    The third way a refusal arrives. With the safety filter on, a blocked job
    returns a CENSORED card; with it off, the same job returns a frame that is
    uniformly black - correct size, ``censored`` false, nothing drawn. Both
    measured. Only the pixels distinguish the second kind from a real image,
    so they have to be looked at.
    """
    try:
        import io  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(data)) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=float)
    except Exception:
        return False        # unreadable is a different fault; do not mask it
    # A real frame, even a night shot, varies. A refusal is flat to the bit.
    return bool(pixels.std() < 1.0)


def _dimensions(data: bytes):
    """The real size of the returned image, or None if it will not open."""
    import io  # noqa: PLC0415

    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


def _collect(ticket: Ticket, api_key: str, timeout: int = 120) -> Path:
    """Download a finished job's image to its destination."""
    import requests  # noqa: PLC0415

    response = _session().get(f"{BASE}/generate/status/{ticket.job_id}",
                              headers=_headers(api_key), timeout=timeout)
    response.raise_for_status()
    status = response.json()
    generations = status.get("generations") or []
    if not generations:
        raise HordeError("The horde finished the job but returned no image.")

    first = generations[0]
    # A censored job is not a failed job as far as the queue is concerned: it
    # comes back done, with state "ok", and a real .webp attached - the .webp
    # is just a black card reading CENSORED. Six innocuous anime prompts were
    # answered this way in testing, so this is the common case, not the edge
    # one, and the only honest signal is this dedicated field.
    if first.get("censored"):
        raise HordeError(
            "A worker's safety filter replaced this image with a placeholder. "
            "The filter fires on plenty of harmless anime prompts; rewording "
            "it, or picking another Horde model, usually clears it."
        )
    if first.get("state") == "censored":
        raise HordeError("A worker censored this prompt.")

    image = requests.get(first["img"], timeout=timeout)
    image.raise_for_status()
    if len(image.content) < 2048:
        raise HordeError(f"The horde returned only {len(image.content)} bytes.")

    # Check the pixels rather than trusting the envelope. The renderer crops
    # to a portrait frame, so a square image is a wrong result even when
    # every flag above says the job went fine.
    wanted = (ticket.extra.get("width"), ticket.extra.get("height"))
    got = _dimensions(image.content)
    if got and all(wanted) and got != wanted:
        raise HordeError(
            f"Asked for {wanted[0]}x{wanted[1]} and got {got[0]}x{got[1]}. "
            f"That is usually a filter placeholder rather than a drawing."
        )

    if _is_blank(image.content):
        raise HordeError(
            "The worker returned an empty black frame - its safety filter "
            "blocked the prompt without saying so. Another Horde model "
            "usually draws it."
        )

    ticket.destination.parent.mkdir(parents=True, exist_ok=True)
    ticket.destination.write_bytes(image.content)
    ticket.extra.update(worker=first.get("worker_name"), model=first.get("model"),
                        kudos_spent=status.get("kudos"), returned_size=got)
    return ticket.destination


def collect_all(tickets: Sequence[Ticket], api_key: str = ANON_KEY,
                timeout: float = 1800.0, poll: float = 15.0,
                on_progress: Optional[Callable[[List[Ticket]], None]] = None
                ) -> List[Ticket]:
    """Wait for already-submitted tickets, downloading each as it lands.

    The tickets queue against each other rather than in series, so this costs
    about what the slowest single image costs, not the sum.
    """
    started = time.time()
    pending = {t.job_id: t for t in tickets}
    while pending and time.time() - started < timeout:
        for job_id, ticket in list(pending.items()):
            try:
                state = check(job_id, api_key)
            except Exception as exc:      # a blip in the status route is not
                logger.debug("check %s failed: %s", job_id[:8], exc)
                continue                  # a reason to abandon a queued job
            if state.get("faulted"):
                ticket.failed = "The horde faulted this job."
                pending.pop(job_id)
            elif not state.get("is_possible", True):
                ticket.failed = ("No worker online can run this job - the model "
                                 "or the resolution is unavailable.")
                pending.pop(job_id)
            elif state.get("done"):
                try:
                    _collect(ticket, api_key)
                    ticket.done = True
                except Exception as exc:
                    ticket.failed = str(exc)
                ticket.waited = time.time() - started
                pending.pop(job_id)
        if on_progress:
            on_progress(list(tickets))
        if pending:
            time.sleep(poll)

    for ticket in pending.values():
        ticket.failed = (f"Still queued after {timeout:.0f}s. The anonymous key "
                         f"is last in line; a free account at aihorde.net is faster.")
    return list(tickets)


def _refused(ticket: Ticket) -> bool:
    """Whether this shot failed in a way another model might not."""
    reason = ticket.failed.lower()
    return any(word in reason for word in ("safety filter", "black frame", "censored"))


def _round(prompts: Sequence[str], destinations: Sequence, model: str,
           api_key: str, width: int, height: int, steps: int, seed: int,
           nsfw: bool, timeout: float, poll: float,
           on_progress: Optional[Callable[[List[Ticket]], None]]) -> List[Ticket]:
    """One pass over a batch with one model: submit all, then collect all."""
    tickets: List[Ticket] = []
    for index, (prompt, destination) in enumerate(zip(prompts, destinations)):
        try:
            tickets.append(submit(
                prompt, destination, api_key=api_key, model=model,
                width=width, height=height, steps=steps,
                seed=(seed + index) if seed else 0, nsfw=nsfw))
        except HordeError as exc:
            failed = Ticket(job_id="", prompt=prompt, destination=Path(destination))
            failed.failed = str(exc)
            tickets.append(failed)
        # Stay under the documented submission rate; see SUBMIT_INTERVAL.
        if index + 1 < len(prompts):
            time.sleep(SUBMIT_INTERVAL)

    queued = [t for t in tickets if t.job_id]
    if queued:
        collect_all(queued, api_key=api_key, timeout=timeout, poll=poll,
                    on_progress=on_progress)
    return tickets


def generate_many(prompts: Sequence[str], destinations: Sequence,
                  api_key: str = ANON_KEY, model: str = "", look: str = "anime",
                  width: int = 768, height: int = 1344, steps: int = 24,
                  seed: int = 0, nsfw: bool = False,
                  timeout: float = 1800.0, poll: float = 15.0,
                  retries: int = 2,
                  on_progress: Optional[Callable[[List[Ticket]], None]] = None
                  ) -> List[Ticket]:
    """Draw a whole batch. Every job is queued before any is collected.

    A shot refused by one model's safety filter is retried on the next
    curated model rather than abandoned. That is not a nicety: one model
    refused six prompts out of six, so without this a whole batch can come
    back empty for a reason that has nothing to do with the prompts.
    """
    if len(prompts) != len(destinations):
        raise ValueError("prompts and destinations must be the same length")

    if model:
        ladder = [model]
    else:
        ladder = [m.name for m in CURATED if m.look == look] or [DEFAULT_ANIME]
    ladder = ladder[:max(1, retries + 1)]

    results = _round(prompts, destinations, ladder[0], api_key, width, height,
                     steps, seed, nsfw, timeout, poll, on_progress)

    for fallback in ladder[1:]:
        retry_at = [i for i, t in enumerate(results) if not t.done and _refused(t)]
        if not retry_at:
            break
        logger.info("%d shot(s) refused; retrying on '%s'", len(retry_at), fallback)
        again = _round([prompts[i] for i in retry_at],
                       [destinations[i] for i in retry_at], fallback, api_key,
                       width, height, steps, seed, nsfw, timeout, poll, on_progress)
        for slot, ticket in zip(retry_at, again):
            if ticket.done or not results[slot].failed:
                results[slot] = ticket

    return results


def generate(prompt: str, destination, api_key: str = ANON_KEY, model: str = "",
             look: str = "anime", width: int = 768, height: int = 1344,
             steps: int = 24, seed: int = 0, nsfw: bool = False,
             timeout: float = 900.0, poll: float = 10.0) -> Path:
    """Draw one image and return its path. Raises HordeError on failure."""
    ticket = generate_many([prompt], [destination], api_key=api_key, model=model,
                           look=look, width=width, height=height, steps=steps,
                           seed=seed, nsfw=nsfw, timeout=timeout, poll=poll)[0]
    if not ticket.done:
        raise HordeError(ticket.failed or "The horde returned nothing.")
    return ticket.destination
