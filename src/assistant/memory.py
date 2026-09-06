"""Long-term memory: a small local JSON store behind remember/forget tools.

Kinds of items:
- "preference": the owner's own standing instructions ("from now on when I say
  movie time…") — applied without being asked.
- "house": the same, but how the HOME runs for anyone in it ("make 2700 the
  house default") — shared, not personal.
- "fact": things to recall on demand ("remember the wifi password is…").
- lesson / observation / episode: written by the session-end reflection pass.

Every item carries its evidence: a `subject` (the short scope it belongs to —
lights, music, calendar, house, will), where it came from (`source`), when it
was last confirmed (`last_verified`), how much we trust it (`confidence`) and,
when it corrects an earlier item, the id it `supersedes`.

Two things follow from that, and they are the point of the file:

- Replacement is atomic. "Actually make that 2700" writes ONE new item that
  supersedes the old id, in a single file write. There is no window in which
  the store holds neither value — the old one is retired only because the new
  one exists, and it stays on disk as history.
- Retrieval, not recital. Past a small store, the instructions get the
  preferences whose subject is in play (from the tools and entities of the
  last few minutes) plus the newest handful of everything else — never the
  whole store. Lessons need real confidence to be injected at all, so one
  unlucky evening does not become a house truth.

Rows written before any of this load with defaults; nothing is lost.

Privacy posture (docs/FEATURES.md): owner-directed, deliberately stored items
only — never ambient conversation. The file is local and gitignored.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

# preference/house/fact: stored via the voice tools (deliberate, user-directed).
# lesson/observation/episode: written by the session-end reflection pass —
# lessons are operational recipes injected into future instructions;
# observations await the user's consent before becoming preferences;
# episodes are the searchable conversation journal.
KINDS = ("preference", "house", "fact", "lesson", "observation", "episode")

# Where an item came from: the owner said it, reflection distilled it, or a
# tool outcome proved it. Confidence is read against this.
SOURCES = ("voice", "reflection", "tool")

# The scopes worth naming out loud. Free-form (a new one costs nothing), but
# these are what the instructions suggest and what inference falls back on.
SUBJECTS = ("lights", "music", "climate", "tv", "calendar", "house", "will")

INJECT_WHOLE_BELOW = 15  # a small store is still cheap to inject entire
RECENT_OTHERS = 5  # past that: what is in play, plus this many newest others
LESSON_MIN_CONFIDENCE = 0.6  # below this a lesson is a hunch, not a house truth
INFERENCE_CONFIDENCE = 0.4  # one reflection with nothing to check it against
VERIFIED_CONFIDENCE = 0.7  # a tool outcome in the transcript backed it up
CONFIDENCE_CEILING = 0.95  # seen again and again is still not certainty

_SUBJECT_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lights", ("light", "lamp", "bulb", "brightness", "dim", "kelvin",
                "colour", "color", "lumen")),
    ("music", ("music", "song", "playlist", "album", "artist", "track", "volume",
               "speaker", "play", "radio", "airplay", "spotify")),
    ("tv", ("tv", "television", "netflix", "youtube", "app")),
    ("calendar", ("calendar", "appointment", "meeting", "event", "schedule")),
    ("climate", ("thermostat", "temperature", "degrees", "heat", "cool", "furnace")),
    ("will", ("call me", "my name", "i am")),
    ("house", ("house", "home", "apartment", "guest", "everyone")),
)

# What a tool or an entity says about the subject of the moment.
_TOOL_SUBJECTS = {
    "set_lights": "lights",
    "get_lights": "lights",
    "undo_last": "lights",
    "play_music": "music",
    "browse_music": "music",
    "media_control": "music",
    "launch_app": "tv",
    "show_me": "tv",
    "list_calendar_events": "calendar",
    "create_calendar_event": "calendar",
    "delete_calendar_event": "calendar",
}
_DOMAIN_SUBJECTS = {
    "light": "lights",
    "scene": "lights",
    "media_player": "music",
    "climate": "climate",
    "fan": "climate",
    "switch": "house",
    "sensor": "house",
    "binary_sensor": "house",
    "cover": "house",
    "lock": "house",
}


def normalize_subject(subject: str) -> str:
    return " ".join(str(subject or "").split()).lower()[:32]


def infer_subject(text: str) -> str:
    """The scope an item is about, when nobody said. A fallback for rows
    stored before subjects existed — an explicit subject always wins."""
    lowered = str(text or "").lower()
    for subject, words in _SUBJECT_WORDS:
        if any(re.search(rf"\b{re.escape(word)}", lowered) for word in words):
            return subject
    return ""


def subjects_in_play(tools: Iterable[str] = (), entities: Iterable[str] = ()) -> set[str]:
    """What this stretch of conversation is about, from the tools just called
    and the entities they touched. "will" is always in play — the person
    talking is the one constant — so what he asked to be called never falls
    off the end of a long store."""
    found = {"will"}
    for tool in tools:
        subject = _TOOL_SUBJECTS.get(str(tool))
        if subject:
            found.add(subject)
    for entity in entities:
        subject = _DOMAIN_SUBJECTS.get(str(entity).split(".", 1)[0])
        if subject:
            found.add(subject)
    return found


@dataclass(frozen=True)
class MemoryItem:
    id: int
    kind: str
    text: str
    created: str
    subject: str = ""
    source: str = "voice"
    last_verified: str = ""
    confidence: float = 1.0
    supersedes: int | None = None

    @property
    def scope(self) -> str:
        """The subject to file it under — stated, or read off the text."""
        return self.subject or infer_subject(self.text)


_FIELD_NAMES = frozenset(f.name for f in fields(MemoryItem))


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._next_id = 1
        self._items: list[MemoryItem] = []
        self._load()

    # ── the file ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._next_id = data.get("next_id", 1)
        items: list[MemoryItem] = []
        for row in data.get("items", []):
            if not isinstance(row, dict):
                continue
            # Older rows have fewer fields; a newer app may have written more.
            # Both load: unknown keys are dropped, missing ones take defaults.
            known = {k: v for k, v in row.items() if k in _FIELD_NAMES}
            with contextlib.suppress(TypeError, ValueError):
                items.append(MemoryItem(**known))
        self._items = items
        # A hand-edited or half-written file must never hand out a live id.
        self._next_id = max([self._next_id, *(item.id + 1 for item in items)])

    def _save(self) -> None:
        """One write, atomically replaced — an interrupted save leaves the
        previous store intact rather than a truncated one."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "items": [asdict(item) for item in self._items]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def _commit(self, items: list[MemoryItem], next_id: int) -> None:
        """Swap in a whole new store, or keep the old one. Nothing partial
        reaches disk or the caller — this is what makes replacement atomic."""
        was_items, was_next = self._items, self._next_id
        self._items, self._next_id = items, next_id
        try:
            self._save()
        except OSError:
            self._items, self._next_id = was_items, was_next
            raise

    # ── writing ────────────────────────────────────────────────────────────

    def add(
        self,
        kind: str,
        text: str,
        *,
        subject: str = "",
        source: str = "voice",
        confidence: float = 1.0,
        supersedes: int | None = None,
        last_verified: str | None = None,
    ) -> MemoryItem:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        text = text.strip()
        if not text:
            raise ValueError("empty memory text")
        if supersedes is not None:
            old = self.get(int(supersedes))
            if old is None:
                raise ValueError(f"no memory with id {supersedes} to supersede")
            supersedes = old.id
        today = _today()
        if last_verified is None:
            # The owner saying it IS the verification; reflection must earn it.
            last_verified = today if source == "voice" else ""
        item = MemoryItem(
            id=self._next_id,
            kind=kind,
            text=text,
            created=today,
            subject=normalize_subject(subject),
            source=source,
            last_verified=last_verified,
            confidence=max(0.0, min(1.0, float(confidence))),
            supersedes=supersedes,
        )
        self._commit([*self._items, item], self._next_id + 1)
        return item

    def replace(
        self,
        item_id: int,
        text: str,
        *,
        kind: str | None = None,
        subject: str | None = None,
        source: str = "voice",
    ) -> MemoryItem:
        """Correct a memory in ONE write: the new version supersedes the old,
        which stays on disk as history. There is no moment in between where
        the store holds neither value."""
        old = self.get(int(item_id))
        if old is None:
            raise ValueError(f"no memory with id {item_id}")
        return self.add(
            kind or old.kind,
            text,
            subject=old.subject if subject is None else subject,
            source=source,
            supersedes=old.id,
        )

    def reinforce(self, item_id: int, *, confidence: float | None = None) -> MemoryItem | None:
        """Seen again: raise what we trust it at and stamp today. A lesson
        that keeps turning out true earns its way into the instructions."""
        for index, item in enumerate(self._items):
            if item.id != item_id:
                continue
            raised = (
                max(item.confidence, float(confidence))
                if confidence is not None
                else item.confidence + 0.2
            )
            fresher = replace(
                item,
                confidence=max(0.0, min(CONFIDENCE_CEILING, raised)),
                last_verified=_today(),
            )
            items = list(self._items)
            items[index] = fresher
            self._commit(items, self._next_id)
            return fresher
        return None

    def forget(self, item_id: int) -> bool:
        """Delete outright — for something the owner wants gone. Correcting a
        memory is `replace`, which keeps the old version as history.

        The versions behind it go too: forgetting "2700 kelvin" must not
        resurrect the "3000 kelvin" it corrected."""
        by_id = {item.id: item for item in self._items}
        doomed: set[int] = set()
        cursor: int | None = item_id
        while cursor is not None and cursor in by_id and cursor not in doomed:
            doomed.add(cursor)
            cursor = by_id[cursor].supersedes
        if not doomed:
            return False
        self._commit([item for item in self._items if item.id not in doomed], self._next_id)
        return True

    # ── reading ────────────────────────────────────────────────────────────

    def superseded_ids(self) -> set[int]:
        """Ids some later item corrects. Derived from the links themselves,
        so a replacement is one row on disk and can never half-apply."""
        return {item.supersedes for item in self._items if item.supersedes is not None}

    def get(self, item_id: int, *, include_superseded: bool = False) -> MemoryItem | None:
        retired = set() if include_superseded else self.superseded_ids()
        for item in self._items:
            if item.id == item_id and item.id not in retired:
                return item
        return None

    def items(
        self,
        kind: str | None = None,
        *,
        subject: str = "",
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        retired = set() if include_superseded else self.superseded_ids()
        wanted = normalize_subject(subject)
        return [
            item
            for item in self._items
            if (kind is None or item.kind == kind)
            and item.id not in retired
            and (not wanted or item.scope == wanted)
        ]

    def scoped(self, kind: str, subjects: Iterable[str] = ()) -> list[MemoryItem]:
        """What to inject for this stretch: everything while the store is
        small, then only what is in play plus the newest few others."""
        live = self.items(kind)
        if len(live) <= INJECT_WHOLE_BELOW:
            return live
        wanted = {normalize_subject(s) for s in subjects if normalize_subject(s)}
        matched = [item for item in live if item.scope in wanted]
        matched_ids = {item.id for item in matched}
        others = [item for item in live if item.id not in matched_ids][-RECENT_OTHERS:]
        chosen = matched_ids | {item.id for item in others}
        return [item for item in live if item.id in chosen]

    # ── rendering for the instructions ─────────────────────────────────────

    def _render(self, items: list[MemoryItem], empty: str) -> str:
        lines = [f"- [id {i.id}{(' · ' + i.scope) if i.scope else ''}] {i.text}" for i in items]
        return "\n".join(lines) or empty

    def preferences_text(self, subjects: Iterable[str] = ()) -> str:
        return self._render(self.scoped("preference", subjects), "(none stored yet)")

    def house_text(self, subjects: Iterable[str] = ()) -> str:
        return self._render(self.scoped("house", subjects), "(none set)")

    def lessons_text(
        self, limit: int = 15, *, min_confidence: float = LESSON_MIN_CONFIDENCE
    ) -> str:
        lessons = [x for x in self.items("lesson") if x.confidence >= min_confidence][-limit:]
        return self._render(lessons, "(none yet)")

    def observations_text(self, limit: int = 5) -> str:
        return self._render(self.items("observation")[-limit:], "(none)")
