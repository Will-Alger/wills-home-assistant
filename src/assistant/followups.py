"""Follow-ups: things she promised to bring up later, by trigger.

"When I get home, remind me to take the chicken out" (arrival), "when I
leave, remind me to lock the door" (departure), "next time we talk, ask me
how the demo went" (conversation), or a plain time. Entity-based "tell me
when X" stays a watch (events.py). One store, so "what are you waiting on?"
is one question.

Time / arrival / departure follow-ups fire as urgent `followup`
notifications (owner-set: they speak through quiet hours, and reach his
phone when he is away — the point of "remind me to lock up when I leave").
Conversation ones are rendered into her instructions and retired only when
she actually brought one up — she called raise_follow_up, or her own words
overlap the promise. A session where he only asked for the hallway light
leaves it waiting. Persisted in data/followups.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from assistant.scheduler import _next_occurrence, spoken_time

TRIGGERS = ("time", "arrival", "departure", "conversation")

# Function words long enough to clear the four-letter bar. Without them
# "you wanted me to tell you about that" would match every promise she has.
# Kept deliberately short: too many, and a promise whose only content words
# are ordinary verbs ("ask how the demo went") could never match at all.
STOP_WORDS = frozenset(
    {
        "about", "after", "again", "also", "always", "another", "anything", "around",
        "away", "back", "because", "been", "before", "being", "both", "could", "does",
        "doing", "done", "down", "each", "else", "even", "ever", "every", "from", "going",
        "gone", "have", "having", "here", "just", "like", "more", "most", "much", "never",
        "next", "okay", "once", "only", "other", "over", "please", "really", "right",
        "same", "should", "since", "some", "sorry", "still", "such", "sure", "than",
        "thank", "thanks", "that", "thats", "their", "them", "then", "there", "these",
        "they", "thing", "things", "this", "those", "through", "very", "well", "were",
        "what", "when", "where", "which", "while", "will", "with", "would", "yeah",
        "your", "yours",
    }
)


def significant_words(text: str) -> set[str]:
    """What a sentence is actually about: four letters or more, no filler."""
    # The stem before any apostrophe, so "the hallway's off" still says hallway.
    words = [w.split("'")[0] for w in re.findall(r"[a-z']+", str(text).lower().replace("’", "'"))]
    return {w for w in words if len(w) >= 4} - STOP_WORDS


def mentions(said: str, what: str) -> bool:
    """Did she bring this up? Two words in common is the whole test — one
    ("the demo") is the kind of coincidence an ordinary sentence produces."""
    return len(significant_words(said) & significant_words(what)) >= 2


@dataclass
class FollowUp:
    id: int
    what: str
    trigger: str  # time | arrival | departure | conversation
    fire_at: float | None = None
    context: str = ""
    active: bool = True
    created: float = 0.0
    fired: float | None = None


class FollowUpStore:
    def __init__(
        self,
        path: Path,
        *,
        announcer: Any | None = None,
        journal: Any | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._announcer = announcer
        self._journal = journal
        self._now = now
        self._next_id = 1
        self._items: list[FollowUp] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = int(data.get("next_id", 1))
        known = set(FollowUp.__dataclass_fields__)
        for row in data.get("followups", []):
            if isinstance(row, dict) and "id" in row and "what" in row:
                self._items.append(FollowUp(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "followups": [asdict(f) for f in self._items]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── creating / managing ────────────────────────────────────────────────

    def add(
        self,
        what: str,
        *,
        trigger: str,
        at: str = "",
        in_seconds: float | None = None,
        context: str = "",
    ) -> FollowUp:
        what = " ".join(str(what).split())
        if not what:
            raise ValueError("a follow-up needs what to bring up")
        trigger = str(trigger or "").strip().lower()
        if trigger == "next_conversation":
            trigger = "conversation"
        if trigger not in TRIGGERS:
            raise ValueError("the trigger must be time, arrival, departure, or conversation")
        now = self._now()
        fire_at = None
        if trigger == "time":
            if in_seconds is not None:
                if in_seconds <= 0:
                    raise ValueError("the delay must be positive")
                fire_at = now + float(in_seconds)
            else:
                fire_at = _next_occurrence(at, [], now)
                if fire_at is None:
                    raise ValueError(f"a time follow-up needs at like 18:00 or in_seconds (got {at!r})")
        item = FollowUp(
            id=self._next_id,
            what=what,
            trigger=trigger,
            fire_at=fire_at,
            context=" ".join(str(context or "").split()),
            created=now,
        )
        self._next_id += 1
        self._items.append(item)
        self._save()
        return item

    def active(self) -> list[FollowUp]:
        return [f for f in self._items if f.active]

    def get(self, item_id: int) -> FollowUp | None:
        return next((f for f in self._items if f.id == int(item_id)), None)

    def cancel(self, item_id: int) -> FollowUp | None:
        item = self.get(item_id)
        if item is None or not item.active:
            return None
        item.active = False
        self._save()
        return item

    def for_conversation(self) -> list[FollowUp]:
        return [f for f in self.active() if f.trigger == "conversation"]

    def mark_raised(self, ids: Iterable[int]) -> int:
        """She brought these up in a conversation: retire them."""
        wanted = {int(i) for i in ids}
        now = self._now()
        count = 0
        for f in self._items:
            if f.id in wanted and f.active:
                f.active = False
                f.fired = now
                count += 1
        if count:
            self._save()
        return count

    def settle_conversation(
        self, ids: Iterable[int], said: str, *, raised: Iterable[int] = ()
    ) -> tuple[list[int], list[int]]:
        """A conversation just ended. Retire only the follow-ups she really
        brought up — she called raise_follow_up on them, or her own words
        overlap the promise. The rest keep waiting, and the journal says so:
        a session happening is not the same as a promise being kept.

        Returns (retired ids, still pending ids)."""
        explicit = {int(i) for i in raised}
        kept: list[int] = []
        pending: list[int] = []
        for item_id in (int(i) for i in ids):
            item = self.get(item_id)
            if item is None or not item.active:
                continue
            (kept if item_id in explicit or mentions(said, item.what) else pending).append(item_id)
        if kept:
            self.mark_raised(kept)
        if self._journal is not None:
            for item_id in kept:
                with contextlib.suppress(Exception):
                    self._journal.write(
                        "followup",
                        f"follow-up {item_id} raised in conversation",
                        source="conversation",
                        data={"id": item_id},
                    )
            for item_id in pending:
                with contextlib.suppress(Exception):
                    self._journal.write(
                        "followup",
                        f"follow-up {item_id} not raised: not mentioned",
                        source="conversation",
                        data={"id": item_id},
                    )
        return kept, pending

    # ── firing ─────────────────────────────────────────────────────────────

    def _raise(self, item: FollowUp, now: float, *, why: str) -> None:
        item.active = False
        item.fired = now
        text = f"Follow-up: {item.what}" + (f" ({item.context})" if item.context else "")
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.write("followup", f"{why}: {item.what}", source=item.trigger, data={"id": item.id})
        if self._announcer is not None:
            with contextlib.suppress(Exception):
                self._announcer.enqueue(
                    text,
                    kind="followup",
                    ref=f"followup:{item.id}",
                    priority="urgent",  # he asked for it; it beats quiet hours and reaches his phone
                    expires_in_s=12 * 3600,
                    context={"followup": item.id},
                )

    def tick(self, now: float | None = None) -> list[FollowUp]:
        now = self._now() if now is None else now
        raised = [f for f in self.active() if f.trigger == "time" and f.fire_at is not None and f.fire_at <= now]
        for f in raised:
            self._raise(f, now, why="time")
        if raised:
            self._save()
        return raised

    def on_presence(self, transition: Any) -> list[FollowUp]:
        kind = str(getattr(transition, "kind", ""))
        trigger = {"arrived": "arrival", "left": "departure"}.get(kind)
        if trigger is None:
            return []
        now = self._now()
        raised = [f for f in self.active() if f.trigger == trigger]
        for f in raised:
            self._raise(f, now, why=kind)
        if raised:
            self._save()
        return raised

    # ── reading back ───────────────────────────────────────────────────────

    @staticmethod
    def when_text(item: FollowUp) -> str:
        if item.trigger == "time" and item.fire_at:
            return spoken_time(item.fire_at)
        return {
            "arrival": "when he gets home",
            "departure": "when he leaves",
            "conversation": "next time you talk",
        }.get(item.trigger, item.trigger)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": f.id,
                "what": f.what,
                "when": self.when_text(f),
                **({"context": f.context} if f.context else {}),
            }
            for f in self.active()
        ]

    def text(self) -> str:
        rows = self.for_conversation()
        if not rows:
            return "none"
        return "; ".join(f"[id {f.id}] {f.what}" + (f" ({f.context})" if f.context else "") for f in rows)
