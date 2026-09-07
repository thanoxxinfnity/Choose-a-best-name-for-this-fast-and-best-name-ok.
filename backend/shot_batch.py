"""Generating a batch of shots against a budget, and stopping when it is spent.

The last batch of fourteen shots produced one. The other thirteen failed on
"Insufficient funds", one after another, because nothing was watching the
money - each call was fired, billed or refused, and the loop went on to the
next. On a free tier that is thirteen wasted round trips; on a paid one it
would have been a surprise bill.

So a batch here knows three things it did not know before: what each shot is
expected to cost, how much it is allowed to spend in total, and what it has
already made. It stops on its own when the budget is gone, it skips shots that
are already on disk so an interrupted run resumes where it stopped, and it
gives up early when the provider says the balance is empty rather than
discovering that once per remaining shot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# What a four second clip costs, in dollars, on the models worth using. Quoted
# by the service itself when it refuses a request for lack of funds, which is
# the only place these are stated exactly.
CLIP_PRICES: Dict[str, float] = {
    "minimax/minimax-h3-max-turbo": 0.031,
    "wan-fast": 0.050,
    "p-video": 0.080,
    "seedance-pro": 0.100,
    "minimax-h3": 0.200,
    "wan-3.0": 0.272,
    "seedance-2.0-fast": 0.280,
    "grok-video-pro": 0.280,
    "seedance-2.0-mini": 0.405,
}
DEFAULT_PRICE = 0.10

# Phrases a provider uses when the account is empty. One of these means every
# remaining shot will fail the same way, so the batch stops instead of asking
# the same question thirty more times.
BROKE_MARKERS = ("insufficient balance", "insufficient funds", "no credits remaining",
                 "payment required", "quota exceeded")


def price_of(model: str, seconds: float = 4.0) -> float:
    """Expected cost of one clip, in dollars."""
    per_four = CLIP_PRICES.get(model, DEFAULT_PRICE)
    return per_four * (max(seconds, 0.5) / 4.0)


def is_out_of_money(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in BROKE_MARKERS)


@dataclass
class Shot:
    """One thing to generate."""

    name: str
    prompt: str
    seconds: float = 4.0


@dataclass
class BatchResult:
    made: List[Path] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    not_attempted: List[str] = field(default_factory=list)
    spent: float = 0.0
    stopped_because: str = ""

    def summary(self) -> str:
        parts = [f"{len(self.made)} made"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} already there")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.not_attempted:
            parts.append(f"{len(self.not_attempted)} not attempted")
        line = ", ".join(parts) + f" - about ${self.spent:.2f} spent"
        return line + (f" ({self.stopped_because})" if self.stopped_because else "")


def generate_batch(
    shots: Sequence[Shot],
    provider,
    destination: Path,
    budget: float,
    model: str = "wan-fast",
    resolution: str = "720x1280",
    on_event: Optional[Callable[[str], None]] = None,
    min_bytes: int = 100_000,
) -> BatchResult:
    """Generate ``shots`` into ``destination``, spending at most ``budget``.

    ``budget`` is in dollars and is checked BEFORE each call, against what the
    next shot is expected to cost - so the batch stops one shot short rather
    than one shot over.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = BatchResult()

    def say(line: str) -> None:
        logger.info(line)
        if on_event:
            on_event(line)

    for index, shot in enumerate(shots):
        target = destination / f"{shot.name}.mp4"
        if target.exists() and target.stat().st_size >= min_bytes:
            result.skipped.append(shot.name)
            say(f"skip  {shot.name} (already generated)")
            continue

        cost = price_of(model, shot.seconds)
        if result.spent + cost > budget + 1e-9:
            result.not_attempted.extend(s.name for s in shots[index:]
                                        if not (destination / f"{s.name}.mp4").exists())
            result.stopped_because = (
                f"budget of ${budget:.2f} would be exceeded by the next shot"
            )
            say(f"stop  {result.stopped_because}")
            break

        try:
            started = time.time()
            provider.text_to_video(shot.prompt, target, seconds=shot.seconds,
                                   resolution=resolution, model=model)
            size = target.stat().st_size if target.exists() else 0
            if size < min_bytes:
                raise RuntimeError(f"only {size} bytes came back")
            result.spent += cost
            result.made.append(target)
            say(f"ok    {shot.name}  {time.time() - started:.0f}s  "
                f"{size / 1e6:.1f}MB  (${result.spent:.2f} of ${budget:.2f})")
        except Exception as exc:
            target.unlink(missing_ok=True)
            if is_out_of_money(exc):
                result.not_attempted.extend(s.name for s in shots[index:])
                result.stopped_because = "the account ran out of balance"
                say(f"stop  {result.stopped_because}: {str(exc)[:120]}")
                break
            result.failed.append(shot.name)
            say(f"fail  {shot.name}: {type(exc).__name__}: {str(exc)[:120]}")

    say(result.summary())
    return result


def load_shots(path: Path) -> List[Shot]:
    """Read a shot list from JSON: [{"name":..., "prompt":..., "seconds":...}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Shot(name=str(item["name"]), prompt=str(item["prompt"]),
             seconds=float(item.get("seconds", 4.0)))
        for item in data
    ]
