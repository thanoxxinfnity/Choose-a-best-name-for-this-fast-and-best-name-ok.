"""Building one long video out of many short generated clips.

A generator that makes five seconds at a time can make eight minutes, but only
by making ninety-six clips, and the arithmetic on that is the whole story:

    8 minutes / 5s = 96 clips.  At one clip per ten minutes, sixteen hours.

The rate that matters is how fast the provider *renders*, not how fast it
accepts. A service can take ten requests a minute and still finish one clip
every ten minutes; queueing faster only makes the queue longer. So this asks
the endpoint what its real throughput is and says how long the job will take
before starting it, rather than after.

Everything here is resumable. A job measured in hours will be interrupted -
a restart, a dropped connection, a closed quota window - and re-rendering
clips that already exist would be the most expensive possible way to recover.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from config import settings

logger = logging.getLogger(__name__)

# What a free tier tends to manage when nothing better is known. Deliberately
# pessimistic: a promise that comes in early is a good surprise.
ASSUMED_SECONDS_PER_CLIP = 600.0


@dataclass
class Shot:
    """One generated clip in the sequence."""

    prompt: str
    seconds: float = 5.0
    image: Optional[str] = None      # a still to animate, for image-to-video
    index: int = 0

    @property
    def filename(self) -> str:
        return f"shot_{self.index:03d}.mp4"


@dataclass
class SequenceEstimate:
    clips: int
    seconds_each: float
    total_seconds: float
    render_rate_seconds: float
    wall_clock_seconds: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        hours = self.wall_clock_seconds / 3600.0
        pace = (f"about one clip every {self.render_rate_seconds / 60:.0f} minutes"
                if self.render_rate_seconds >= 90
                else f"about {60 / max(self.render_rate_seconds, 1):.1f} clips a minute")
        length = (f"{hours:.1f} hours" if hours >= 1
                  else f"{self.wall_clock_seconds / 60:.0f} minutes")
        return (
            f"{self.clips} clips of {self.seconds_each:.0f}s makes "
            f"{self.total_seconds / 60:.1f} minutes of video. The endpoint renders "
            f"{pace} ({self.source}), so expect about {length} of waiting."
        )


def estimate(
    clips: int,
    seconds_each: float = 5.0,
    base_url: str = "",
    timeout: int = 30,
) -> SequenceEstimate:
    """How long a sequence will really take, asked of the endpoint itself.

    Acceptance rate is the number a service advertises and the number that
    does not matter. This looks for the render rate and falls back to a
    pessimistic assumption rather than to the flattering one.
    """
    base = (base_url or settings.videoforge_base_url).rstrip("/")
    rate = ASSUMED_SECONDS_PER_CLIP
    source = "assumed, the endpoint did not say"

    try:
        response = requests.get(f"{base}/api/v1/stats", timeout=timeout)
        if response.status_code == 200:
            body = response.json()
            data = body.get("data", body) if isinstance(body, dict) else {}
            capacity = str(data.get("capacity_rpm") or "")
            found = _render_seconds_from(capacity)
            if found:
                rate, source = found, f"reported: {capacity.strip()}"
    except Exception as exc:
        logger.debug("could not read %s stats: %s", base, exc)

    return SequenceEstimate(
        clips=clips, seconds_each=seconds_each,
        total_seconds=clips * seconds_each,
        render_rate_seconds=rate,
        wall_clock_seconds=clips * rate,
        source=source,
    )


def _render_seconds_from(capacity: str) -> Optional[float]:
    """Pull a render rate out of whatever prose the endpoint reports.

    Written against what this service actually says - "renders ~1 per 10 min" -
    and deliberately narrow: a phrase it does not recognise leaves the
    pessimistic default in place instead of inventing a number from it.
    """
    import re

    text = capacity.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*per\s*(\d+(?:\.\d+)?)?\s*(min|minute|hour|sec)", text)
    if not match:
        return None
    count = float(match.group(1)) or 1.0
    per = float(match.group(2) or 1.0)
    unit = match.group(3)
    span = per * {"sec": 1.0, "min": 60.0, "minute": 60.0, "hour": 3600.0}[unit]
    return span / max(count, 0.001)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class SequenceState:
    """What has already been made, so a restart does not redo it."""

    workspace: Path
    done: Dict[int, str] = field(default_factory=dict)
    failed: Dict[int, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.workspace / "sequence_state.json"

    def load(self) -> "SequenceState":
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.done = {int(k): v for k, v in (data.get("done") or {}).items()}
                self.failed = {int(k): v for k, v in (data.get("failed") or {}).items()}
            except Exception as exc:
                logger.warning("sequence state unreadable (%s), starting fresh", exc)
        return self

    def save(self) -> None:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"done": self.done, "failed": self.failed}, indent=1),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("could not persist the sequence state: %s", exc)


def build_sequence(
    shots: Sequence[Shot],
    workspace: Path,
    provider: Any,
    on_progress: Optional[Callable[[str, float], None]] = None,
    stop_after_failures: int = 3,
) -> SequenceState:
    """Render every shot, skipping any that already exist.

    Failures do not abandon the run: a sequence long enough to be worth
    building is long enough that one bad clip is likely, and losing the other
    ninety-five to it would be absurd. It gives up only when failures start
    looking systemic rather than incidental.
    """
    workspace = Path(workspace)
    state = SequenceState(workspace).load()
    clips_dir = workspace / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    def report(message: str, fraction: float) -> None:
        logger.info("sequence: %s (%.0f%%)", message, fraction * 100)
        if on_progress:
            try:
                on_progress(message, fraction)
            except Exception:
                logger.debug("sequence progress hook raised", exc_info=True)

    consecutive_failures = 0
    for position, shot in enumerate(shots):
        destination = clips_dir / shot.filename
        if shot.index in state.done and destination.exists():
            report(f"shot {shot.index + 1}/{len(shots)} already made", position / len(shots))
            continue

        report(f"shot {shot.index + 1}/{len(shots)}", position / len(shots))
        try:
            if shot.image:
                provider.image_to_video(Path(shot.image), shot.prompt, destination,
                                        seconds=shot.seconds)
            else:
                provider.text_to_video(shot.prompt, destination, seconds=shot.seconds)
            state.done[shot.index] = str(destination)
            state.failed.pop(shot.index, None)
            consecutive_failures = 0
        except Exception as exc:
            logger.warning("shot %s failed: %s", shot.index, exc)
            state.failed[shot.index] = str(exc)[:300]
            consecutive_failures += 1
            if consecutive_failures >= stop_after_failures:
                state.save()
                raise RuntimeError(
                    f"{consecutive_failures} shots failed in a row, so this looks like "
                    f"the service rather than the shots - stopping with "
                    f"{len(state.done)} of {len(shots)} made. Re-running keeps them."
                ) from exc
        state.save()

    report("all shots made", 1.0)
    return state


def stitch(
    state: SequenceState,
    shots: Sequence[Shot],
    destination: Path,
    ffmpeg: Optional[str] = None,
) -> Optional[Path]:
    """Join the finished clips in order, skipping any that never arrived."""
    from puter_integration import ffmpeg_binary

    ordered = [
        Path(state.done[shot.index]) for shot in shots
        if shot.index in state.done and Path(state.done[shot.index]).exists()
    ]
    if not ordered:
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = destination.parent / "sequence_list.txt"
    listing.write_text(
        "\n".join(f"file '{path.resolve()}'" for path in ordered), encoding="utf-8"
    )

    import subprocess

    result = subprocess.run(
        [ffmpeg or ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         # Re-encode rather than copy: generated clips can differ in profile or
         # frame rate, and a concat of streams that disagree plays back broken.
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(destination)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        logger.warning("stitching failed: %s", result.stderr[:300])
        return None
    return destination
