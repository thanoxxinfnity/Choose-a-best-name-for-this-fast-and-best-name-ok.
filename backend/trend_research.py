"""Mine YouTube for what actually performs in a niche.

`search.list` costs 100 quota units against a 10,000/day default, so this
module searches once, then batches the cheap (1 unit) `videos.list` lookup for
statistics, and caches everything on disk.  A careless implementation burns the
day's quota in a hundred calls.

What comes back is not "the algorithm" - it is measurable structure: how long
the winners run, how their titles are built, which tags recur, and how engaged
their audiences are.  :func:`derive_style` turns that into editing parameters
the planner can actually use.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

from config import settings

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

CACHE_TTL_SECONDS = 6 * 3600

_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")
_WORD = re.compile(r"[A-Za-z0-9']+")

# Words that say nothing about the edit itself.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "this", "that", "it", "you", "your", "my", "me", "we", "i", "at", "by",
    "from", "be", "are", "was", "new", "best", "top", "video", "shorts", "short",
    "youtube", "subscribe", "like", "watch", "full", "part", "ep", "episode",
}


@dataclass
class TrendVideo:
    video_id: str
    title: str
    channel: str
    published_at: str
    duration: float = 0.0
    views: int = 0
    likes: int = 0
    comments: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def engagement(self) -> float:
        """Likes+comments per thousand views - the part a creator controls."""
        if self.views <= 0:
            return 0.0
        return (self.likes + self.comments) / self.views * 1000.0


@dataclass
class TrendReport:
    query: str
    niche: str = ""
    sampled: int = 0
    median_duration: float = 0.0
    duration_range: tuple = (0.0, 0.0)
    median_views: int = 0
    top_engagement: float = 0.0
    common_tags: List[str] = field(default_factory=list)
    title_keywords: List[str] = field(default_factory=list)
    hook_patterns: List[str] = field(default_factory=list)
    examples: List[TrendVideo] = field(default_factory=list)
    error: str = ""

    def to_prompt_block(self, limit: int = 6) -> str:
        if self.error:
            return f"Trend research unavailable: {self.error}"
        lines = [
            f"Researched {self.sampled} top-performing '{self.query}' shorts on YouTube.",
            f"Winning length: about {self.median_duration:.0f}s "
            f"(range {self.duration_range[0]:.0f}-{self.duration_range[1]:.0f}s).",
        ]
        if self.median_views:
            lines.append(f"Median views in the sample: {self.median_views:,}.")
        if self.title_keywords:
            lines.append("Words that recur in the winning titles: "
                         + ", ".join(self.title_keywords[:12]) + ".")
        if self.common_tags:
            lines.append("Tags they share: " + ", ".join(self.common_tags[:12]) + ".")
        if self.hook_patterns:
            lines.append("Hook shapes that recur: " + "; ".join(self.hook_patterns[:4]) + ".")
        if self.examples:
            lines.append("Examples:")
            for video in self.examples[:limit]:
                lines.append(
                    f"  - \"{video.title[:80]}\" ({video.duration:.0f}s, "
                    f"{video.views:,} views, {video.engagement:.1f} eng/1k)"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration_range"] = list(self.duration_range)
        return data


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


def _cache_path(query: str, limit: int) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", query.lower())[:60]
    directory = settings.cache_dir / "trends"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe}_{limit}.json"


def research_trends(
    query: str,
    api_key: str = "",
    limit: int = 25,
    niche: str = "",
    use_cache: bool = True,
    timeout: Optional[int] = None,
) -> TrendReport:
    """Search YouTube for a niche and summarise what the winners have in common."""
    query = (query or "").strip()
    if not query:
        return TrendReport(query="", error="empty query")

    key = (api_key or settings.youtube_api_key or "").strip()
    if not key:
        return TrendReport(query=query, niche=niche, error="no YouTube Data API key")

    cache = _cache_path(query, limit)
    if use_cache and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                return _report_from_dict(json.loads(cache.read_text(encoding="utf-8")))
            except Exception:
                logger.debug("trend cache unreadable, refetching")

    timeout = timeout or settings.youtube_timeout
    try:
        # 1 search (100 units) ...
        response = requests.get(
            SEARCH_URL,
            params={
                "part": "snippet", "q": query, "type": "video",
                "videoDuration": "short", "order": "viewCount",
                "maxResults": min(max(limit, 1), 50), "key": key,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return TrendReport(query=query, niche=niche,
                               error=f"search HTTP {response.status_code}: {response.text[:160]}")
        ids = [
            item["id"]["videoId"]
            for item in response.json().get("items", [])
            if item.get("id", {}).get("videoId")
        ]
        if not ids:
            return TrendReport(query=query, niche=niche, error="no results")

        # ... then one batched details call (1 unit) for all of them.
        details = requests.get(
            VIDEOS_URL,
            params={"part": "snippet,contentDetails,statistics",
                    "id": ",".join(ids), "key": key},
            timeout=timeout,
        )
        if details.status_code != 200:
            return TrendReport(query=query, niche=niche,
                               error=f"videos HTTP {details.status_code}: {details.text[:160]}")
        videos = [_parse_video(item) for item in details.json().get("items", [])]
    except requests.RequestException as exc:
        return TrendReport(query=query, niche=niche, error=f"network error: {exc}")

    report = summarise(query, videos, niche=niche)
    try:
        cache.write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    except Exception:
        logger.debug("could not write the trend cache")
    return report


def _parse_video(item: Dict[str, Any]) -> TrendVideo:
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    return TrendVideo(
        video_id=item.get("id", ""),
        title=snippet.get("title", ""),
        channel=snippet.get("channelTitle", ""),
        published_at=snippet.get("publishedAt", ""),
        duration=parse_iso_duration(item.get("contentDetails", {}).get("duration", "")),
        views=int(stats.get("viewCount", 0) or 0),
        likes=int(stats.get("likeCount", 0) or 0),
        comments=int(stats.get("commentCount", 0) or 0),
        tags=list(snippet.get("tags") or []),
    )


def parse_iso_duration(value: str) -> float:
    match = _ISO_DURATION.match(value or "")
    if not match:
        return 0.0
    days, hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


# The API's videoDuration=short means "under 4 minutes", not "a Short", so a
# 4-minute AMV lands in a Shorts sample and drags the statistics with it.
SHORT_FORM_CEILING_SECONDS = 90.0


def summarise(
    query: str,
    videos: Sequence[TrendVideo],
    niche: str = "",
    short_form_only: bool = True,
) -> TrendReport:
    """Reduce a sample of videos to the structure they share."""
    usable = [video for video in videos if video.duration > 0]
    if not usable:
        return TrendReport(query=query, niche=niche, error="no usable results")

    if short_form_only:
        short_form = [v for v in usable if v.duration <= SHORT_FORM_CEILING_SECONDS]
        # Only drop the long tail if enough short-form examples remain to be
        # representative; otherwise the niche genuinely is long-form.
        if len(short_form) >= max(4, len(usable) // 3):
            usable = short_form

    usable.sort(key=lambda video: video.views, reverse=True)
    durations = [video.duration for video in usable]

    tag_counts: Dict[str, int] = {}
    for video in usable:
        for tag in {tag.lower().strip() for tag in video.tags if tag.strip()}:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    word_counts: Dict[str, int] = {}
    for video in usable:
        for word in {w.lower() for w in _WORD.findall(video.title) if len(w) > 2}:
            if word in _STOPWORDS:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1

    return TrendReport(
        query=query,
        niche=niche,
        sampled=len(usable),
        median_duration=statistics.median(durations),
        duration_range=(min(durations), max(durations)),
        median_views=int(statistics.median(video.views for video in usable)),
        top_engagement=round(max(video.engagement for video in usable), 2),
        common_tags=[
            tag for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1])
            if count > 1
        ][:20],
        title_keywords=[
            word for word, count in sorted(word_counts.items(), key=lambda kv: -kv[1])
            if count > 1
        ][:20],
        hook_patterns=_hook_patterns(usable),
        examples=usable[:8],
    )


def _hook_patterns(videos: Sequence[TrendVideo]) -> List[str]:
    """Recurring title shapes, which mirror the on-screen hook."""
    patterns: Dict[str, int] = {}
    for video in videos:
        title = video.title.strip()
        if not title:
            continue
        lowered = title.lower()
        if title.endswith("?") or lowered.startswith(("how ", "why ", "what ", "can ")):
            patterns["a question as the hook"] = patterns.get("a question as the hook", 0) + 1
        if "vs" in lowered.split() or " vs " in lowered:
            patterns["an X vs Y matchup"] = patterns.get("an X vs Y matchup", 0) + 1
        if re.search(r"\b(top|best)\s*\d+|\b\d+\s+(things|ways|tips|reasons)", lowered):
            patterns["a numbered list"] = patterns.get("a numbered list", 0) + 1
        if sum(1 for char in title if char.isupper()) > len(title) * 0.45:
            patterns["ALL CAPS shouting"] = patterns.get("ALL CAPS shouting", 0) + 1
        if any(marker in lowered for marker in ("edit", "amv", "shorts", "fyp")):
            patterns["an explicit 'edit' label"] = patterns.get("an explicit 'edit' label", 0) + 1
        if any(char in title for char in "🔥💀😱⚡✨"):
            patterns["emoji in the title"] = patterns.get("emoji in the title", 0) + 1
    return [
        pattern for pattern, count in sorted(patterns.items(), key=lambda kv: -kv[1])
        if count > 1
    ]


def _report_from_dict(data: Dict[str, Any]) -> TrendReport:
    examples = [TrendVideo(**video) for video in data.get("examples", [])]
    data = dict(data)
    data["examples"] = examples
    data["duration_range"] = tuple(data.get("duration_range") or (0.0, 0.0))
    return TrendReport(**data)


# ---------------------------------------------------------------------------
# Turning research into editing parameters
# ---------------------------------------------------------------------------


def derive_style(report: TrendReport) -> Dict[str, Any]:
    """Concrete editing guidance implied by the research."""
    if report.error or not report.sampled:
        return {}

    target = max(8.0, min(report.median_duration, 60.0))
    # Short-form winners cut fast; scale the segment length to the total so a
    # 15s short is not given 5s shots.
    if target <= 20:
        segment = (0.8, 1.8)
    elif target <= 35:
        segment = (1.2, 2.6)
    else:
        segment = (1.8, 3.5)

    return {
        "target_duration": round(target),
        "segment_seconds": segment,
        "suggested_cuts": max(4, int(target / ((segment[0] + segment[1]) / 2))),
        "title_keywords": report.title_keywords[:8],
        "tags": report.common_tags[:10],
        "hooks": report.hook_patterns[:4],
    }
