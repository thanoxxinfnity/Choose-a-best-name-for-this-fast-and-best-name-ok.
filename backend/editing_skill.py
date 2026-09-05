"""Kimi's editing skill: a craft manual it carries into every render.

Everything else feeds the model *evidence* for one job - the footage analysis,
the trend report, a couple of exemplars. This is different: a standing body of
editing craft that goes into the **system** prompt, so the model is an editor
before it is told anything about the current clip.

The skill has two halves:

* **Craft rules** - curated, hand-written, version controlled. What makes a cut
  land, where text goes, how sparingly to use stickers, how to time a voice
  line. These do not change per render.
* **Genre playbooks** - learned. Every trend report the app runs folds measured
  numbers for that niche back into the skill (winning length, shot cadence,
  the hooks that recur), so the manual gets more specific the more the app is
  used.

The whole thing is persisted, versioned and inspectable through
``GET /api/v1/skill``, so it can be read and corrected rather than being an
opaque prompt buried in the code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import settings

logger = logging.getLogger(__name__)

SKILL_VERSION = 1
SKILL_FILENAME = "editing_skill.json"
MAX_PLAYBOOKS = 24


# ---------------------------------------------------------------------------
# Curated craft
# ---------------------------------------------------------------------------

CRAFT: Dict[str, List[str]] = {
    "The first second": [
        "The hook is frame one, not a build-up. Put the strongest image and the "
        "on-screen promise inside the first 1.0s or the viewer is already gone.",
        "Never open on a fade from black, a logo, or a slow establishing shot in "
        "short form. Those are long-form grammar.",
        "The opening text says what the viewer will get, not what the video is "
        "called. 'WAIT FOR IT' outperforms a title.",
    ],
    "Where to cut": [
        "Cut on motion and on impact, not on stillness. A cut that lands on a "
        "hit reads as intentional; the same cut a beat late reads as a mistake.",
        "One idea per shot. If a shot needs two things explained, it is two shots.",
        "Vary shot length deliberately: a run of identical 1.5s cuts goes numb. "
        "Break the rhythm before the payoff, not during it.",
        "Cut away from a shot at its peak, not after it. Leaving the best frame "
        "on screen for an extra half second is the most common way to kill pace.",
    ],
    "Text on screen": [
        "Two to four words. Text competes with the picture; a sentence loses.",
        "Text enters with a cut, never mid-shot. Mid-shot text looks like a bug.",
        "Never stack text and a sticker in the same third of the frame.",
        "Reserve the last card for one call to action, and only one.",
    ],
    "Stickers and graphics": [
        "A sticker is punctuation. Three or four in a short is plenty; one every "
        "cut is noise and the viewer stops seeing any of them.",
        "Stickers land on hits and reveals. A sticker over filler footage draws "
        "attention to the fact that it is filler.",
        "The sticker should describe something actually on screen. A generic "
        "'subscribe' badge over a fight scene is a mismatch the viewer feels.",
    ],
    "Sound and voice": [
        "The track carries a montage. If the music is doing the work, do not add "
        "narration on top of it - drop the voice or drop the music bed.",
        "Voice lines go in the gaps between hits, never over them.",
        "Duck the bed under a line and bring it straight back. A bed that stays "
        "ducked sounds broken.",
        "Two short lines beat one long one. Give each 2-6 seconds of room and "
        "never overlap them.",
    ],
    "Pacing": [
        "Match cadence to energy, not to taste: high-energy footage wants ~1-2s "
        "shots, calm footage wants 3-5s and long crossfades.",
        "Shake, RGB split and zoom punches are accents. Applied to every cut "
        "they stop being accents and become a headache.",
        "End on the strongest remaining moment, not on whatever is left over.",
    ],
    "Honesty about the footage": [
        "Only describe subjects that the analysis reports are actually on "
        "screen. Inventing a subject produces stickers and text that do not "
        "match the video.",
        "Timestamps must come from the clips supplied, inside their real "
        "durations. A timeline that references footage that does not exist "
        "cannot be rendered.",
    ],
}


# ---------------------------------------------------------------------------
# Learned genre playbooks
# ---------------------------------------------------------------------------


@dataclass
class GenrePlaybook:
    """Measured numbers for one niche, learned from trend research."""

    niche: str
    query: str = ""
    sampled: int = 0
    median_duration: float = 0.0
    shot_seconds: List[float] = field(default_factory=list)
    suggested_cuts: int = 0
    hooks: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    updated_at: float = 0.0
    observations: int = 1

    def to_line(self) -> str:
        shots = (
            f"{self.shot_seconds[0]:.1f}-{self.shot_seconds[1]:.1f}s shots"
            if len(self.shot_seconds) == 2 else "unknown shot length"
        )
        parts = [
            f"{self.niche}: winners run ~{self.median_duration:.0f}s, {shots}, "
            f"about {self.suggested_cuts} cuts"
        ]
        if self.hooks:
            parts.append("hooks that recur: " + "; ".join(self.hooks[:3]))
        if self.keywords:
            parts.append("words that recur: " + ", ".join(self.keywords[:6]))
        return " | ".join(parts) + f" (from {self.sampled} videos)"


@dataclass
class EditingSkill:
    version: int = SKILL_VERSION
    craft: Dict[str, List[str]] = field(default_factory=lambda: {
        section: list(rules) for section, rules in CRAFT.items()
    })
    playbooks: Dict[str, GenrePlaybook] = field(default_factory=dict)
    updated_at: float = 0.0

    # ------------------------------------------------------------- prompting
    def to_system_block(self, niche: str = "", max_playbooks: int = 3) -> str:
        """The skill as it appears in Kimi's system prompt."""
        lines = ["[EDITING SKILL - how this editor works, independent of any one job]"]
        for section, rules in self.craft.items():
            lines.append(f"\n{section}:")
            lines.extend(f"  - {rule}" for rule in rules)

        chosen = self._relevant_playbooks(niche, max_playbooks)
        if chosen:
            lines.append("\nMeasured genre playbooks (learned from real top performers):")
            lines.extend(f"  - {playbook.to_line()}" for playbook in chosen)
        return "\n".join(lines)

    def _relevant_playbooks(self, niche: str, limit: int) -> List[GenrePlaybook]:
        if not self.playbooks:
            return []
        wanted = (niche or "").strip().lower()
        ordered = sorted(
            self.playbooks.values(),
            key=lambda p: (p.niche.lower() == wanted, p.observations, p.updated_at),
            reverse=True,
        )
        return ordered[:limit]

    # -------------------------------------------------------------- learning
    def learn_from_research(self, niche: str, report: Any, style: Dict[str, Any]) -> bool:
        """Fold one trend report into the playbook for its niche.

        Repeated observations of the same niche are averaged rather than
        overwritten, so one unusual sample cannot rewrite the playbook.
        """
        niche = (niche or "").strip().lower() or "general"
        if getattr(report, "error", "") or not getattr(report, "sampled", 0):
            return False

        shot = list(style.get("segment_seconds") or [])
        incoming = GenrePlaybook(
            niche=niche,
            query=getattr(report, "query", ""),
            sampled=int(getattr(report, "sampled", 0)),
            median_duration=float(getattr(report, "median_duration", 0.0)),
            shot_seconds=[float(value) for value in shot][:2],
            suggested_cuts=int(style.get("suggested_cuts") or 0),
            hooks=list(getattr(report, "hook_patterns", []) or [])[:4],
            keywords=list(getattr(report, "title_keywords", []) or [])[:8],
            updated_at=time.time(),
        )

        existing = self.playbooks.get(niche)
        if existing is None:
            self.playbooks[niche] = incoming
        else:
            self.playbooks[niche] = _blend(existing, incoming)

        self._prune()
        self.updated_at = time.time()
        return True

    def _prune(self) -> None:
        if len(self.playbooks) <= MAX_PLAYBOOKS:
            return
        # Keep the best observed and most recent; drop one-off niches first.
        ranked = sorted(
            self.playbooks.values(),
            key=lambda p: (p.observations, p.updated_at),
            reverse=True,
        )
        self.playbooks = {playbook.niche: playbook for playbook in ranked[:MAX_PLAYBOOKS]}

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "craft": self.craft,
            "playbooks": {key: asdict(value) for key, value in self.playbooks.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditingSkill":
        playbooks = {
            key: GenrePlaybook(**value)
            for key, value in (data.get("playbooks") or {}).items()
        }
        craft = data.get("craft") or {}
        # A stored skill from an older version keeps any curated section it is
        # missing, so upgrading the code cannot silently drop craft rules.
        merged = {section: list(rules) for section, rules in CRAFT.items()}
        merged.update({section: list(rules) for section, rules in craft.items()})
        return cls(
            version=int(data.get("version", SKILL_VERSION)),
            craft=merged,
            playbooks=playbooks,
            updated_at=float(data.get("updated_at", 0.0)),
        )


def _blend(existing: GenrePlaybook, incoming: GenrePlaybook) -> GenrePlaybook:
    """Weighted average of two observations of the same niche."""
    total = existing.observations + 1
    weight = existing.observations / total

    def mix(old: float, new: float) -> float:
        return old * weight + new * (1 - weight)

    shots = existing.shot_seconds or incoming.shot_seconds
    if len(existing.shot_seconds) == 2 and len(incoming.shot_seconds) == 2:
        shots = [
            round(mix(existing.shot_seconds[0], incoming.shot_seconds[0]), 2),
            round(mix(existing.shot_seconds[1], incoming.shot_seconds[1]), 2),
        ]

    return GenrePlaybook(
        niche=existing.niche,
        query=incoming.query or existing.query,
        sampled=existing.sampled + incoming.sampled,
        median_duration=round(mix(existing.median_duration, incoming.median_duration), 1),
        shot_seconds=shots,
        suggested_cuts=int(round(mix(existing.suggested_cuts, incoming.suggested_cuts))),
        hooks=_merge_unique(existing.hooks, incoming.hooks, 5),
        keywords=_merge_unique(existing.keywords, incoming.keywords, 10),
        updated_at=time.time(),
        observations=total,
    )


def _merge_unique(first: Sequence[str], second: Sequence[str], limit: int) -> List[str]:
    seen: List[str] = []
    for value in list(first) + list(second):
        if value and value not in seen:
            seen.append(value)
    return seen[:limit]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_CACHE: Optional[EditingSkill] = None


def skill_path() -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / SKILL_FILENAME


def load_skill(refresh: bool = False) -> EditingSkill:
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    path = skill_path()
    if path.exists():
        try:
            _CACHE = EditingSkill.from_dict(json.loads(path.read_text(encoding="utf-8")))
            return _CACHE
        except Exception as exc:
            logger.warning("Editing skill unreadable (%s), starting from the curated one", exc)
    _CACHE = EditingSkill()
    return _CACHE


def save_skill(skill: EditingSkill) -> Path:
    global _CACHE
    path = skill_path()
    try:
        path.write_text(json.dumps(skill.to_dict(), indent=1, ensure_ascii=False),
                        encoding="utf-8")
        _CACHE = skill
    except Exception as exc:
        logger.warning("Could not persist the editing skill: %s", exc)
    return path


def learn_from_research(niche: str, report: Any, style: Dict[str, Any]) -> bool:
    """Public hook: fold a trend report into the persistent skill."""
    skill = load_skill()
    if skill.learn_from_research(niche, report, style):
        save_skill(skill)
        return True
    return False


def reset_skill() -> EditingSkill:
    """Back to the curated craft, dropping everything learned."""
    global _CACHE
    _CACHE = EditingSkill()
    save_skill(_CACHE)
    return _CACHE
