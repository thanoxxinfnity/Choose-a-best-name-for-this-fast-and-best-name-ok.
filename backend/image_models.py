"""Which image model to draw with, and what each one is good for.

The app had one hard-wired image path. That is the wrong shape for the thing
it is used for: a sticker, an anime key frame and a photorealistic background
plate are three different jobs, and the model that does one well does the
others badly. So the model becomes a choice, and this is the list to choose
from.

Every entry here was called with a real key and its result looked at. The
service's own catalogue prices are not a reliable guide to what a free
account can run - zimage is listed at a cost and works, nanobanana is listed
at nothing and answers 402 - so ``free`` records what actually came back
rather than what the price list claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE = "https://gen.pollinations.ai"


@dataclass(frozen=True)
class ImageModel:
    key: str
    label: str
    # What it is for, so the picker can be sorted by job rather than by name.
    look: str            # "anime" | "photoreal" | "both"
    free: bool
    seconds: float       # measured, not promised
    note: str = ""


# Measured on 2026-09-09 against a real account, one call each, same prompt.
MODELS: Tuple[ImageModel, ...] = (
    ImageModel("zimage", "Z-Image (anime)", "anime", True, 8.2,
               "Crisp anime and illustration. The default, and the fastest good one."),
    ImageModel("flux", "FLUX", "both", True, 9.6,
               "Sharpest all-rounder here. Real photos and illustration both."),
    ImageModel("dreamshaper", "Dreamshaper", "both", True, 6.8,
               "Fastest. Softer detail, good for backgrounds behind a subject."),
    ImageModel("microsoft/mai-image-2.5-flash", "MAI 2.5 Flash", "photoreal", True, 20.6,
               "Photorealistic people and places."),
    ImageModel("gptimage", "GPT Image", "photoreal", True, 25.7,
               "Photoreal, follows long prompts closely."),
    ImageModel("gpt-image-2", "GPT Image 2", "photoreal", True, 49.8,
               "Highest fidelity of the free ones, and the slowest by far."),
    # Present and better, but they answer 402 without a paid balance. Listed so
    # the picker can show them as locked rather than hiding what exists.
    ImageModel("nanobanana-2", "Nano Banana 2", "both", False, 0.0,
               "Sharper detail and much better text in the image."),
    ImageModel("nanobanana-pro", "Nano Banana Pro", "photoreal", False, 0.0,
               "Studio quality up to 4K."),
    ImageModel("seedream5-pro", "Seedream 5 Pro", "both", False, 0.0,
               "High-end illustration."),
    ImageModel("flux-2-flex", "FLUX 2 Flex", "both", False, 0.0,
               "The newer FLUX."),
)

BY_KEY: Dict[str, ImageModel] = {model.key: model for model in MODELS}
DEFAULT_MODEL = "zimage"


def resolve(key: Optional[str]) -> ImageModel:
    """The named model, or the default when the name is unknown or empty."""
    return BY_KEY.get((key or "").strip(), BY_KEY[DEFAULT_MODEL])


def free_models() -> List[ImageModel]:
    return [model for model in MODELS if model.free]


def for_look(look: str) -> List[ImageModel]:
    """Models suited to a job: 'anime', 'photoreal', or everything."""
    look = (look or "").strip().lower()
    if look not in ("anime", "photoreal"):
        return list(MODELS)
    return [m for m in MODELS if m.look in (look, "both")]


def image_url(prompt: str, model: str = DEFAULT_MODEL, width: int = 768,
              height: int = 1344, seed: int = 0) -> str:
    """The generation URL for a prompt.

    Also the reference URL: this service addresses a generated image by the
    request that made it and caches it immutably, so the same prompt, model
    and seed name the same picture forever.
    """
    from urllib.parse import quote, urlencode  # noqa: PLC0415

    query = urlencode({"model": resolve(model).key, "width": int(width),
                       "height": int(height), "seed": int(seed)})
    return f"{BASE}/image/{quote((prompt or '').strip()[:1200], safe='')}?{query}"


def generate(prompt: str, destination, api_key: str, model: str = DEFAULT_MODEL,
             width: int = 768, height: int = 1344, seed: int = 0,
             timeout: int = 180):
    """Draw one image to ``destination``. Returns the path.

    Raises RuntimeError with the service's own words on failure, because the
    two failures that matter - no balance for a locked model, and a model that
    is simply down - are indistinguishable from a generic message.
    """
    from pathlib import Path  # noqa: PLC0415

    import requests  # noqa: PLC0415

    chosen = resolve(model)
    if not api_key:
        raise RuntimeError("Image generation needs a Pollinations key.")

    url = image_url(prompt, chosen.key, width, height, seed)
    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"},
                            timeout=timeout)
    if response.status_code == 402:
        raise RuntimeError(
            f"'{chosen.label}' needs a paid balance. The free models are: "
            f"{', '.join(m.label for m in free_models())}."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"'{chosen.label}' failed: HTTP {response.status_code} "
            f"{response.text[:160]}"
        )
    # A JSON body with a 200 is an error wearing a success code.
    if "image" not in (response.headers.get("content-type") or ""):
        raise RuntimeError(
            f"'{chosen.label}' returned {response.headers.get('content-type')} "
            f"instead of an image: {response.text[:160]}"
        )
    if len(response.content) < 2048:
        raise RuntimeError(
            f"'{chosen.label}' returned only {len(response.content)} bytes"
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination
