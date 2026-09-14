"""What felt wrong, and what it adds up to.

An item is one piece of the owner's feedback about how a conversation went
("she cut me off", "I wish she waited until I was done before acknowledging"),
tagged, tied to the sessions it is about, with a status that moves as it is
dealt with. A need is what several items add up to — "shorter replies", "wait
until he is done" — the thing that actually gets built, and the thing a task
on the board is opened for. Persisted in data/feedback.json; read and written
from the event loop and from the dashboard's HTTP thread, so every access
holds one lock. Writing never raises into a conversation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("new", "triaged", "planned", "fixed", "wontfix")
NEED_STATUSES = ("open", "planned", "building", "done")
TAGS = (
    "cut-off", "no-reply", "too-eager", "too-slow", "wrong-action", "misheard",
    "music", "wake", "style", "praise", "other",
)


def _clean(text: str, limit: int = 1000) -> str:
    return " ".join(str(text).split())[:limit]


def _tags(tags: Iterable[str] | None) -> list[str]:
    seen: list[str] = []
    for tag in tags or ():
        slug = "-".join(str(tag).strip().lower().replace("_", "-").split())[:24]
        if slug and slug not in seen:
            seen.append(slug)
    return seen[:8]


@dataclass
class Item:
    id: int
    created: float
    text: str
    tags: list[str] = field(default_factory=list)
    sessions: list[int] = field(default_factory=list)
    status: str = "new"
    need: int | None = None
    source: str = "dashboard"  # dashboard | voice
    unit: str = ""
    updated: float = 0.0
    # The part of the conversation he meant: the lines he selected (first to
    # last, everything between), and the timeline stamps that bound them.
    excerpt: str = ""
    range: dict[str, Any] = field(default_factory=dict)


@dataclass
class Need:
    id: int
    created: float
    title: str
    tags: list[str] = field(default_factory=list)
    status: str = "open"
    task: int | None = None  # the board task it was promoted to
    notes: str = ""
    updated: float = 0.0


class FeedbackStore:
    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self._path = Path(path)
        self._now = now
        self._lock = threading.Lock()
        self._items: list[Item] = []
        self._needs: list[Need] = []
        self._next_item = 1
        self._next_need = 1
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        known_item, known_need = set(Item.__dataclass_fields__), set(Need.__dataclass_fields__)
        self._items = [Item(**{k: v for k, v in row.items() if k in known_item})
                       for row in data.get("items", []) if isinstance(row, dict) and "id" in row]
        self._needs = [Need(**{k: v for k, v in row.items() if k in known_need})
                       for row in data.get("needs", []) if isinstance(row, dict) and "id" in row]
        self._next_item = max([int(data.get("next_item", 1))] + [i.id + 1 for i in self._items])
        self._next_need = max([int(data.get("next_need", 1))] + [n.id + 1 for n in self._needs])

    def _save(self) -> None:
        payload = {
            "next_item": self._next_item, "next_need": self._next_need,
            "items": [asdict(i) for i in self._items], "needs": [asdict(n) for n in self._needs],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass  # a store that will not write never breaks a conversation

    # ── items ──────────────────────────────────────────────────────────────

    def add(self, text: str, *, tags: Iterable[str] | None = None, sessions: Iterable[int] | None = None,
            source: str = "dashboard", unit: str = "", excerpt: str = "",
            range: dict[str, Any] | None = None) -> Item | None:
        text = _clean(text)
        if not text:
            return None
        with self._lock:
            item = Item(
                id=self._next_item, created=self._now(), text=text, tags=_tags(tags),
                sessions=sorted({int(s) for s in (sessions or ()) if str(s).lstrip("-").isdigit()}),
                source=source, unit=unit, updated=self._now(),
                excerpt=str(excerpt or "")[:2000],
                range={k: v for k, v in (range or {}).items() if k in ("from_ts", "to_ts", "lines")},
            )
            self._next_item += 1
            self._items.append(item)
            self._save()
            return item

    def update(self, item_id: int, **changes: Any) -> Item | None:
        with self._lock:
            item = next((i for i in self._items if i.id == int(item_id)), None)
            if item is None:
                return None
            if "status" in changes and changes["status"] in STATUSES:
                item.status = changes["status"]
            if "tags" in changes:
                item.tags = _tags(changes["tags"])
            if "text" in changes and _clean(changes["text"]):
                item.text = _clean(changes["text"])
            if "sessions" in changes:
                item.sessions = sorted({int(s) for s in changes["sessions"] if str(s).lstrip("-").isdigit()})
            if "need" in changes:
                need = changes["need"]
                item.need = int(need) if need is not None and str(need).isdigit() else None
            item.updated = self._now()
            self._save()
            return item

    def items(self, *, status: str | None = None, need: int | None = None,
              session: int | None = None) -> list[Item]:
        with self._lock:
            rows = list(self._items)
        if status:
            rows = [i for i in rows if i.status == status]
        if need is not None:
            rows = [i for i in rows if i.need == int(need)]
        if session is not None:
            rows = [i for i in rows if int(session) in i.sessions]
        return sorted(rows, key=lambda i: (i.created, i.id), reverse=True)

    def tag_counts(self, *, open_only: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items():
            if open_only and item.status in ("fixed", "wontfix"):
                continue
            for tag in item.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # ── needs ──────────────────────────────────────────────────────────────

    def add_need(self, title: str, *, tags: Iterable[str] | None = None, items: Iterable[int] | None = None,
                 notes: str = "") -> Need | None:
        title = _clean(title, 200)
        if not title:
            return None
        with self._lock:
            need = Need(id=self._next_need, created=self._now(), title=title, tags=_tags(tags),
                        notes=_clean(notes), updated=self._now())
            self._next_need += 1
            self._needs.append(need)
            for item in self._items:
                if item.id in {int(i) for i in (items or ())}:
                    item.need = need.id
                    if item.status == "new":
                        item.status = "triaged"
            self._save()
            return need

    def update_need(self, need_id: int, **changes: Any) -> Need | None:
        with self._lock:
            need = next((n for n in self._needs if n.id == int(need_id)), None)
            if need is None:
                return None
            if "status" in changes and changes["status"] in NEED_STATUSES:
                need.status = changes["status"]
            if "title" in changes and _clean(changes["title"], 200):
                need.title = _clean(changes["title"], 200)
            if "tags" in changes:
                need.tags = _tags(changes["tags"])
            if "notes" in changes:
                need.notes = _clean(changes["notes"])
            if "task" in changes:
                task = changes["task"]
                need.task = int(task) if task is not None and str(task).isdigit() else None
            need.updated = self._now()
            self._save()
            return need

    def needs(self, *, status: str | None = None) -> list[Need]:
        with self._lock:
            rows = list(self._needs)
        if status:
            rows = [n for n in rows if n.status == status]
        return sorted(rows, key=lambda n: (n.created, n.id), reverse=True)

    # ── for the dashboard and the prompt ───────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": [asdict(i) for i in sorted(self._items, key=lambda i: (i.created, i.id), reverse=True)],
                "needs": [asdict(n) for n in sorted(self._needs, key=lambda n: (n.created, n.id), reverse=True)],
                "statuses": list(STATUSES), "need_statuses": list(NEED_STATUSES), "tags": list(TAGS),
            }

    def open_needs_text(self, limit: int = 5) -> str:
        """One line for her instructions: what he has asked for, in his words."""
        rows = [n for n in self.needs() if n.status in ("open", "planned", "building")][:limit]
        return "; ".join(f"{n.title} ({n.status})" for n in rows)
