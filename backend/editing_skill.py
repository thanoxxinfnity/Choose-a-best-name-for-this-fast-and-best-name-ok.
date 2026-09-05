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

SKILL_VERSION = 3
SKILL_FILENAME = "editing_skill.json"
MAX_PLAYBOOKS = 24
# A mistake has to recur before it is worth spending prompt on.
HABIT_THRESHOLD = 3
MAX_HABITS = 12


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
    "The AE hype grammar (velocity edits)": [
        "This school cuts *on* the beat, not near it. A hard cut, a zoom punch "
        "and an impact flash all land on the same frame as the hit - three "
        "accents on one frame read as one deliberate impact, spread across "
        "three frames they read as three mistakes.",
        "Shots run 0.7-1.8s. The grammar is a run of short shots building to "
        "one held beat, then straight back into the run; a montage of evenly "
        "spaced 1s shots has no shape.",
        "Speed is a punctuation mark: ramp *into* the hit (speed above 1.0 on "
        "the shot before it) and sit at normal speed on the payoff. Ramping the "
        "payoff itself throws away the frame you were building to.",
        "Every third or fourth cut is the biggest one. Reserve the zoom punch "
        "and the flash for those; use plain jump cuts in between or the accents "
        "stop meaning anything.",
        "Text in this style is one or two words, hard-cut in on the hit and hard "
        "out - never faded, never mid-shot. It punctuates the beat like the cut "
        "does.",
        "Do not put a sticker on a flash frame. The flash washes it out and both "
        "accents are wasted.",
        "The last hit is the loudest: leave the strongest single frame of the "
        "footage on the final impact and end within a beat of it.",
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


# What a recurring finding means, said back to the model as a habit to break.
# Keyed by the plan doctor's rule names; anything unmapped is skipped rather
# than paraphrased badly.
HABIT_LESSONS: Dict[str, str] = {
    "metronome": "you keep writing shots that are all the same length - vary "
                 "them deliberately, especially before the payoff",
    "accent_inflation": "you keep marking almost every cut as a zoom punch - "
                        "spend an accent every third or fourth cut, not on all of them",
    "slow_open": "you keep opening on a long establishing shot - the promise "
                 "has to land inside the first second",
    "soft_open": "you keep opening on a crossfade - short form has no room to fade in",
    "sticker_crowding": "you keep putting stickers on back-to-back shots - they "
                        "need a shot between them to read",
    "overlap": "you keep putting the text and the sticker in the same third of "
               "the frame - they fight each other there",
    "wordy_overlay": "you keep writing overlays longer than four words - only "
                     "the first few get read",
    "voice_overlap": "you keep starting a voice line before the previous one "
                     "has finished - give each 2-6 seconds of its own",
    "voice_on_the_hit": "you keep landing voice lines on cuts - they belong in "
                        "the gaps between hits",
    "voice_past_the_end": "you keep writing narration longer than the edit it "
                          "sits in - count the words against the runtime",
    "stub_ending": "you keep ending on a fragment of a shot - end on the "
                   "strongest remaining moment, with room to land",
    "past_the_end": "you keep referencing footage past the end of the clip - "
                    "every timecode must exist in the uploads",
    "missing_source": "you keep pointing at clips that were not uploaded",
}


# ---------------------------------------------------------------------------
# Learned genre playbooks
# ---------------------------------------------------------------------------


@dataclass
class Habit:
    """A mistake this editor has been caught making, and how often."""

    rule: str
    count: int = 0
    last_seen: float = 0.0

    def to_line(self) -> str:
        lesson = HABIT_LESSONS.get(self.rule, "")
        return f"{lesson} (caught {self.count} times)" if lesson else ""


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
    # True when the references were handed over deliberately rather than found
    # by a search. Someone choosing ten edits is a much stronger statement
    # about what they want than the top ten results for a keyword.
    chosen: bool = False

    def to_line(self) -> str:
        shots = (
            f"{self.shot_seconds[0]:.1f}-{self.shot_seconds[1]:.1f}s shots"
            if len(self.shot_seconds) == 2 else "unknown shot length"
        )
        source = "references you chose" if self.chosen else "winners"
        parts = [
            f"{self.niche}: {source} run ~{self.median_duration:.0f}s, {shots}, "
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
    habits: Dict[str, Habit] = field(default_factory=dict)
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

        habits = self.recurring_habits()
        if habits:
            # The curated rules say what good looks like; these say which of
            # them this editor has actually been failing, which is a sharper
            # instruction than the rule on its own.
            lines.append("\nHabits the review keeps catching in your timelines - break these:")
            lines.extend(f"  - {habit.to_line()}" for habit in habits)
        return "\n".join(lines)

    def recurring_habits(self, limit: int = 5) -> List[Habit]:
        """The mistakes frequent enough to be worth naming."""
        ranked = [
            habit for habit in self.habits.values()
            if habit.count >= HABIT_THRESHOLD and habit.to_line()
        ]
        ranked.sort(key=lambda habit: (habit.count, habit.last_seen), reverse=True)
        return ranked[:limit]

    def learn_from_review(self, rules: Sequence[str]) -> bool:
        """Record what the plan doctor caught this time.

        One bad timeline is noise - a model has an off draft like anyone. The
        same finding three times is a habit, and only then does it earn a line
        in the prompt, because a prompt that lists every stumble teaches
        nothing and costs tokens on every render.
        """
        counted = 0
        now = time.time()
        for rule in rules:
            if rule not in HABIT_LESSONS:
                continue
            habit = self.habits.get(rule) or Habit(rule=rule)
            habit.count += 1
            habit.last_seen = now
            self.habits[rule] = habit
            counted += 1
        if not counted:
            return False
        if len(self.habits) > MAX_HABITS:
            ranked = sorted(self.habits.values(),
                            key=lambda habit: (habit.count, habit.last_seen), reverse=True)
            self.habits = {habit.rule: habit for habit in ranked[:MAX_HABITS]}
        self.updated_at = now
        return True

    def _relevant_playbooks(self, niche: str, limit: int) -> List[GenrePlaybook]:
        if not self.playbooks:
            return []
        wanted = (niche or "").strip().lower()
        ordered = sorted(
            self.playbooks.values(),
            key=lambda p: (p.niche.lower() == wanted, p.chosen, p.observations, p.updated_at),
            reverse=True,
        )
        return ordered[:limit]

    # -------------------------------------------------------------- learning
    def learn_from_research(self, niche: str, report: Any, style: Dict[str, Any],
                            chosen: bool = False) -> bool:
        """Fold one trend report into the playbook for its niche.

        Repeated observations of the same niche are averaged rather than
        overwritten, so one unusual sample cannot rewrite the playbook - with
        one exception. References someone handed over deliberately are not an
        observation to average in; they are an instruction. Averaging ten
        chosen edits into a keyword search dilutes exactly the signal that was
        being given, so a chosen set replaces a searched one outright.
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
            chosen=chosen,
        )

        existing = self.playbooks.get(niche)
        if existing is None or (chosen and not existing.chosen):
            self.playbooks[niche] = incoming
        elif existing.chosen and not chosen:
            # A search must not water down what the user picked.
            return False
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
            "habits": {key: asdict(value) for key, value in self.habits.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditingSkill":
        playbooks = {
            key: GenrePlaybook(**value)
            for key, value in (data.get("playbooks") or {}).items()
        }
        habits = {
            key: Habit(**value) for key, value in (data.get("habits") or {}).items()
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
            habits=habits,
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
        chosen=existing.chosen or incoming.chosen,
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


def learn_from_research(niche: str, report: Any, style: Dict[str, Any],
                        chosen: bool = False) -> bool:
    """Public hook: fold a trend report into the persistent skill."""
    skill = load_skill()
    if skill.learn_from_research(niche, report, style, chosen=chosen):
        save_skill(skill)
        return True
    return False


def learn_from_review(rules: Sequence[str]) -> bool:
    """Public hook: record what the plan doctor caught, so it stops recurring."""
    skill = load_skill()
    if skill.learn_from_review(rules):
        save_skill(skill)
        return True
    return False


def reset_skill() -> EditingSkill:
    """Back to the curated craft, dropping everything learned."""
    global _CACHE
    _CACHE = EditingSkill()
    save_skill(_CACHE)
    return _CACHE
