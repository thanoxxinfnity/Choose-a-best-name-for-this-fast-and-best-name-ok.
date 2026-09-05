"""The in-app gallery: everything Moja AI has produced, in one place.

Renders live in their job workspace and generated clips live in the cache, so
the gallery is an index over both rather than a third copy of the media. Each
entry carries a poster frame generated on demand and cached next to the video.

Deleting an entry deletes the media. Nothing here silently keeps a copy.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from puter_integration import ffmpeg_binary

logger = logging.getLogger(__name__)

GENERATED_DIRNAME = "generated"
THUMBNAIL_DIRNAME = ".thumbs"
THUMBNAIL_WIDTH = 360


@dataclass
class GalleryItem:
    item_id: str
    kind: str                    # "render" | "generated"
    filename: str
    path: str
    size_bytes: int
    created_at: float
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    title: str = ""
    prompt: str = ""
    theme: str = ""
    export_preset: str = ""
    provider: str = ""
    has_thumbnail: bool = False

    @property
    def created_iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_iso"] = self.created_iso
        data["stream_url"] = f"/api/v1/gallery/{self.item_id}/stream"
        data["thumbnail_url"] = f"/api/v1/gallery/{self.item_id}/thumbnail"
        data["download_url"] = f"/api/v1/gallery/{self.item_id}/download"
        return data


def generated_dir() -> Path:
    directory = settings.data_dir / GENERATED_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _probe(path: Path) -> Dict[str, Any]:
    from video_renderer import probe_clip

    try:
        return probe_clip(path)
    except Exception:
        return {}


def _render_items() -> List[GalleryItem]:
    """Finished renders, read from each job's own workspace and status file."""
    items: List[GalleryItem] = []
    if not settings.jobs_dir.exists():
        return items

    for job_dir in settings.jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        videos = sorted(job_dir.glob("moja_ai_final.mp4")) or sorted(
            job_dir.glob("final_*.mp4")
        )
        if not videos:
            continue
        video = videos[0]

        status: Dict[str, Any] = {}
        status_file = job_dir / "status.json"
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        if status and status.get("stage") not in (None, "completed"):
            continue

        stat = video.stat()
        items.append(GalleryItem(
            item_id=f"render:{job_dir.name}",
            kind="render",
            filename=video.name,
            path=str(video),
            size_bytes=stat.st_size,
            created_at=status.get("finished_at") or stat.st_mtime,
            duration=float(status.get("duration_seconds") or 0.0),
            width=int(status.get("output_width") or 0),
            height=int(status.get("output_height") or 0),
            fps=float(status.get("output_fps") or 0.0),
            title=(status.get("prompt") or "Untitled edit")[:80],
            prompt=status.get("prompt", ""),
            theme=status.get("theme", ""),
            export_preset=status.get("export_preset", ""),
        ))
    return items


def _generated_items() -> List[GalleryItem]:
    """Clips produced by the generation endpoints and kept deliberately."""
    items: List[GalleryItem] = []
    directory = generated_dir()
    for video in sorted(directory.glob("*.mp4")):
        stat = video.stat()
        meta: Dict[str, Any] = {}
        sidecar = video.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        items.append(GalleryItem(
            item_id=f"generated:{video.stem}",
            kind="generated",
            filename=video.name,
            path=str(video),
            size_bytes=stat.st_size,
            created_at=meta.get("created_at") or stat.st_mtime,
            duration=float(meta.get("seconds") or 0.0),
            title=(meta.get("prompt") or video.stem)[:80],
            prompt=meta.get("prompt", ""),
            provider=meta.get("provider", ""),
        ))
    return items


def list_items(kind: str = "", limit: int = 200, fill_missing: bool = True) -> List[GalleryItem]:
    """Everything in the gallery, newest first."""
    items = _render_items() + _generated_items()
    if kind:
        items = [item for item in items if item.kind == kind]
    items.sort(key=lambda item: item.created_at, reverse=True)
    items = items[:max(1, limit)]

    if fill_missing:
        for item in items:
            # Older jobs predate the geometry fields on the status file.
            if not item.width or not item.duration:
                info = _probe(Path(item.path))
                item.width = item.width or int(info.get("width") or 0)
                item.height = item.height or int(info.get("height") or 0)
                item.fps = item.fps or float(info.get("fps") or 0.0)
                item.duration = item.duration or float(info.get("duration") or 0.0)
            item.has_thumbnail = thumbnail_path(item).exists()
    return items


def find_item(item_id: str) -> Optional[GalleryItem]:
    for item in list_items(limit=10_000, fill_missing=False):
        if item.item_id == item_id:
            item.has_thumbnail = thumbnail_path(item).exists()
            return item
    return None


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def thumbnail_path(item: GalleryItem) -> Path:
    video = Path(item.path)
    directory = video.parent / THUMBNAIL_DIRNAME
    return directory / f"{video.stem}.jpg"


def ensure_thumbnail(item: GalleryItem, at_fraction: float = 0.25) -> Optional[Path]:
    """A poster frame, generated once and cached beside the video."""
    destination = thumbnail_path(item)
    if destination.exists():
        return destination
    video = Path(item.path)
    if not video.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)

    duration = item.duration or float(_probe(video).get("duration") or 0.0)
    # A quarter in avoids both the black first frame and the end card.
    seek = max(0.2, duration * at_fraction) if duration else 0.5

    result = subprocess.run(
        [
            ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{seek:.2f}", "-i", str(video), "-frames:v", "1",
            "-vf", f"scale={THUMBNAIL_WIDTH}:-2", "-q:v", "4", str(destination),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        logger.warning("Thumbnail failed for %s: %s", video.name, result.stderr[:160])
        return None
    return destination


# ---------------------------------------------------------------------------
# Writing and deleting
# ---------------------------------------------------------------------------


def keep_generated(
    source: Path,
    prompt: str = "",
    provider: str = "",
    model: str = "",
    seconds: float = 0.0,
) -> Optional[GalleryItem]:
    """Move a generated clip out of the temp cache and into the gallery."""
    source = Path(source)
    if not source.exists() or source.stat().st_size == 0:
        return None

    stem = f"{int(time.time())}_{provider or 'clip'}"
    destination = generated_dir() / f"{stem}.mp4"
    try:
        shutil.copyfile(source, destination)
    except Exception as exc:
        logger.warning("Could not add the clip to the gallery: %s", exc)
        return None

    destination.with_suffix(".json").write_text(
        json.dumps({
            "prompt": prompt, "provider": provider, "model": model,
            "seconds": seconds, "created_at": time.time(),
        }, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    item = GalleryItem(
        item_id=f"generated:{stem}", kind="generated", filename=destination.name,
        path=str(destination), size_bytes=destination.stat().st_size,
        created_at=time.time(), duration=seconds, title=prompt[:80] or destination.stem,
        prompt=prompt, provider=provider,
    )
    ensure_thumbnail(item)
    return item


def delete_item(item_id: str) -> bool:
    """Remove an entry and the media behind it."""
    item = find_item(item_id)
    if item is None:
        return False

    if item.kind == "render":
        # A render owns its whole job workspace.
        workspace = Path(item.path).parent
        if workspace.parent == settings.jobs_dir:
            shutil.rmtree(workspace, ignore_errors=True)
            return True
        Path(item.path).unlink(missing_ok=True)
        return True

    video = Path(item.path)
    video.unlink(missing_ok=True)
    video.with_suffix(".json").unlink(missing_ok=True)
    thumbnail_path(item).unlink(missing_ok=True)
    return True


def stats() -> Dict[str, Any]:
    items = list_items(limit=10_000, fill_missing=False)
    return {
        "count": len(items),
        "renders": sum(1 for item in items if item.kind == "render"),
        "generated": sum(1 for item in items if item.kind == "generated"),
        "total_bytes": sum(item.size_bytes for item in items),
        "newest": items[0].created_iso if items else "",
    }
