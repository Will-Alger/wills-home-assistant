"""The journal: what she did and saw, day by day.

Notifications are what she wants the owner's attention for; the journal is
everything that happened — tool actions she ran, scheduled things that
fired, watches that hit, her own notifications' lifecycle, comings and
goings, each conversation. "Did the porch light come on last night?", "what
did you do while I was gone?", "when did I leave today?" are answered here.

Storage: one small append-only JSONL file per local day under data/journal/
(a few dozen rows a day), pruned after `keep_days`. Reads touch only the
days asked for. Writing never raises — a journal must never break a command.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

KINDS = (
    "tool", "action", "watch", "schedule", "notification", "session", "presence",
    "task", "push", "followup", "focus", "system",
)
_MAX_DAYS_PER_QUERY = 14


@dataclass(frozen=True)
class Entry:
    ts: float
    kind: str
    source: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)


def _day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d")


def spoken_stamp(ts: float) -> str:
    when = datetime.fromtimestamp(ts).astimezone()
    return f"{when.strftime('%a')} {when.strftime('%I:%M %p').lstrip('0')}"


class Journal:
    def __init__(
        self,
        directory: Path,
        *,
        now: Callable[[], float] = time.time,
        keep_days: int = 90,
    ) -> None:
        self._dir = directory
        self._now = now
        self._keep_days = max(1, int(keep_days))
        self._last_prune_day = ""

    @property
    def directory(self) -> Path:
        return self._dir

    # ── writing ────────────────────────────────────────────────────────────

    def write(
        self, kind: str, text: str, *, source: str = "app", data: dict[str, Any] | None = None
    ) -> Entry | None:
        text = " ".join(str(text).split())[:500]
        if not text:
            return None
        entry = Entry(
            ts=self._now(), kind=str(kind), source=str(source)[:80], text=text, data=dict(data or {})
        )
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with (self._dir / f"{_day_of(entry.ts)}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), default=str) + "\n")
        except OSError:
            return entry  # the row is lost, the command is not
        return entry

    def prune(self) -> int:
        """Delete day files older than keep_days; returns how many."""
        today = _day_of(self._now())
        if today == self._last_prune_day:
            return 0
        self._last_prune_day = today
        cutoff = _day_of(self._now() - self._keep_days * 86400)
        gone = 0
        if not self._dir.exists():
            return 0
        for path in self._dir.glob("*.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink()
                    gone += 1
                except OSError:
                    continue
        return gone

    # ── reading ────────────────────────────────────────────────────────────

    def _days(self, since: float, until: float) -> list[Path]:
        start = datetime.fromtimestamp(since).astimezone().date()
        end = datetime.fromtimestamp(until).astimezone().date()
        paths = []
        day = start
        while day <= end:
            paths.append(self._dir / f"{day.isoformat()}.jsonl")
            day += timedelta(days=1)
        return paths

    def _read(self, path: Path) -> list[Entry]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        out: list[Entry] = []
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a crash mid-write
            if isinstance(row, dict) and "ts" in row and "text" in row:
                out.append(
                    Entry(
                        ts=float(row["ts"]),
                        kind=str(row.get("kind", "")),
                        source=str(row.get("source", "")),
                        text=str(row["text"]),
                        data=dict(row.get("data") or {}),
                    )
                )
        return out

    def query(
        self,
        text: str = "",
        *,
        since: float | None = None,
        until: float | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[Entry]:
        """Entries in the window, best matches first, then newest first.
        Default window: the last two days; never more than 14."""
        now = self._now()
        until = now + 60 if until is None else until
        since = until - 2 * 86400 if since is None else since
        since = max(since, until - _MAX_DAYS_PER_QUERY * 86400)
        wanted = {k for k in kinds} if kinds else None
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]
        scored: list[tuple[int, float, Entry]] = []
        for path in self._days(since, until):
            for entry in self._read(path):
                if entry.ts < since or entry.ts >= until:
                    continue
                if wanted is not None and entry.kind not in wanted:
                    continue
                if words:
                    # the data (tool args, entity ids) counts too: "hallway" lives
                    # in set_lights' arguments, not in its terse result text
                    hay = f"{entry.text} {entry.source} {entry.kind} {json.dumps(entry.data)}".lower()
                    hits = sum(1 for w in words if w in hay)
                    if not hits:
                        continue
                    score = 3 if hits == len(words) else 1
                else:
                    score = 0
                scored.append((score, entry.ts, entry))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [entry for _, _, entry in scored[: max(1, int(limit))]]

    @staticmethod
    def spoken(entry: Entry) -> dict[str, Any]:
        row = {"when": spoken_stamp(entry.ts), "kind": entry.kind, "text": entry.text}
        if entry.source and entry.source != "app":
            row["source"] = entry.source
        return row
