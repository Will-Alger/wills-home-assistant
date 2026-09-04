"""Conversation continuity: a short log of recent sessions, so she can say
"about that thermostat thing from this morning" and answer "what did we talk
about earlier?".

One row per conversation: when it started and ended, why it ended, the first
thing the owner said, and — once reflection has run — a one-line summary.
The last few (within a day) are rendered into her instructions. Persisted in
data/sessions.json, last `keep` rows.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class SessionRow:
    id: int
    started: float
    ended: float | None = None
    kind: str = "wake"  # wake | ptt (the hotkey) | announce
    ended_by: str = ""
    first_user_line: str = ""
    summary: str = ""
    tools: list[str] = field(default_factory=list)
    announced: list[int] = field(default_factory=list)
    responses: int = 0
    cost_usd: float = 0.0

    @property
    def about(self) -> str:
        return self.summary or self.first_user_line


def when_label(ts: float, now: float) -> str:
    """'this morning at 8:12 AM', 'last night at 11:02 PM', 'Monday at 3:10 PM'."""
    when = datetime.fromtimestamp(ts).astimezone()
    today = datetime.fromtimestamp(now).astimezone().date()
    clock = when.strftime("%I:%M %p").lstrip("0")
    if when.date() == today:
        part = "this morning" if when.hour < 12 else "this afternoon" if when.hour < 18 else "this evening"
        return f"{part} at {clock}"
    if when.date() == today - timedelta(days=1):
        return f"{'last night' if when.hour >= 18 else 'yesterday'} at {clock}"
    return f"{when.strftime('%A')} at {clock}"


class SessionLog:
    def __init__(self, path: Path, *, keep: int = 50, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._keep = max(1, keep)
        self._now = now
        self._next_id = 1
        self._rows: list[SessionRow] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = int(data.get("next_id", 1))
        known = set(SessionRow.__dataclass_fields__)
        for row in data.get("sessions", []):
            if isinstance(row, dict) and "id" in row and "started" in row:
                self._rows.append(SessionRow(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = self._rows[-self._keep :]
        payload = {"next_id": self._next_id, "sessions": [asdict(r) for r in self._rows]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self, row_id: int) -> SessionRow | None:
        return next((r for r in self._rows if r.id == int(row_id)), None)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self, kind: str = "wake") -> SessionRow:
        row = SessionRow(id=self._next_id, started=self._now(), kind=kind)
        self._next_id += 1
        self._rows.append(row)
        self._save()
        return row

    def finish(
        self,
        row_id: int,
        *,
        ended_by: str = "",
        first_user_line: str = "",
        tools: Iterable[str] = (),
        announced: Iterable[int] = (),
        responses: int = 0,
        cost_usd: float = 0.0,
    ) -> SessionRow | None:
        row = self.get(row_id)
        if row is None:
            return None
        row.ended = self._now()
        row.ended_by = str(ended_by)
        row.first_user_line = " ".join(str(first_user_line).split())[:120]
        row.tools = [str(t) for t in tools][:30]
        row.announced = [int(i) for i in announced]
        row.responses = int(responses)
        row.cost_usd = float(cost_usd)
        self._save()
        return row

    def set_summary(self, row_id: int, summary: str) -> None:
        row = self.get(row_id)
        summary = " ".join(str(summary).split())[:300]
        if row is not None and summary:
            row.summary = summary
            self._save()

    # ── reading back ───────────────────────────────────────────────────────

    def recent(
        self, *, within_s: float = 86400.0, limit: int = 3, with_user_only: bool = True
    ) -> list[SessionRow]:
        now = self._now()
        rows = [
            r
            for r in self._rows
            if r.ended is not None
            and now - r.started <= within_s
            and (not with_user_only or r.first_user_line)
        ]
        rows.sort(key=lambda r: r.started, reverse=True)
        return rows[: max(1, limit)]

    def recent_text(self, limit: int = 3) -> str:
        """For her instructions: 'this morning at 8:12 AM: <summary>; ...'."""
        now = self._now()
        rows = self.recent(limit=limit)
        if not rows:
            return "none in the last day"
        return "; ".join(f"{when_label(r.started, now)}: {r.about}" for r in rows)

    def rows(self, since: float | None = None, *, limit: int = 10) -> list[dict[str, Any]]:
        now = self._now()
        rows = [r for r in self._rows if r.ended is not None and (since is None or r.started >= since)]
        rows.sort(key=lambda r: r.started, reverse=True)
        return [
            {
                "when": when_label(r.started, now),
                "about": r.about or "(nothing said)",
                "kind": r.kind,
                "ended_by": r.ended_by,
                **({"tools": r.tools} if r.tools else {}),
            }
            for r in rows[:limit]
        ]
