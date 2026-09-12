"""Every decision, every line said, every tool — one row each, always on.

The engine already narrates itself through `engine.tap(kind, **fields)`; until
today those rows only existed while a manual recording ran. The timeline is
the same rows, always, without audio: one JSONL file with the wall-clock
stamp, the session id and the unit on every row, so a question about any
conversation ("why did that one close?", "how long from his last word to her
first?") is a filter, not a memory. Append-only, rotated by size, and a write
that fails is never the conversation's problem.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_KEEP_BYTES = 25 * 1024 * 1024  # rotate past this; one older file is kept


class Timeline:
    def __init__(
        self,
        path: Path,
        *,
        unit: str = "desktop",
        now: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
        keep_bytes: int = _KEEP_BYTES,
    ) -> None:
        self._path = Path(path)
        self._unit = unit
        self._now = now
        self._clock = clock
        self._keep = keep_bytes
        self._lock = threading.Lock()
        self._session: int | None = None
        self._t0 = 0.0
        self._written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session(self) -> int | None:
        return self._session

    # ── writing ────────────────────────────────────────────────────────────

    def session_started(self, session: int | None, kind: str = "wake") -> None:
        self._session = int(session) if session is not None else None
        self._t0 = self._clock()
        self.event("session", what="opened", session_kind=kind)

    def session_ended(self, ended_by: str = "", *, cost_usd: float = 0.0, replied: bool = False) -> None:
        self.event("session", what="closed", by=ended_by, cost_usd=round(float(cost_usd), 5), replied=replied)
        self._session = None

    def event(self, kind: str, **fields: Any) -> None:
        row = {
            "ts": round(self._now(), 3),
            "t": round(self._clock() - self._t0, 3) if self._session is not None else None,
            "session": self._session,
            "unit": self._unit,
            "kind": str(kind),
            **fields,
        }
        line = json.dumps(row, default=str) + "\n"
        with self._lock, contextlib.suppress(Exception):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            self._written += len(line)
            if self._written > 1_000_000:
                self._written = 0
                self._rotate_if_large()

    def _rotate_if_large(self) -> None:
        with contextlib.suppress(OSError):
            if self._path.stat().st_size > self._keep:
                older = self._path.with_suffix(".1.jsonl")
                if older.exists():
                    older.unlink()
                os.replace(self._path, older)

    def tap(self, *sinks: Callable[..., None] | None) -> Callable[..., None]:
        """A tap that writes here and to every other sink (a recording, say)."""
        others = [s for s in sinks if s is not None]

        def fanout(kind: str, **fields: Any) -> None:
            self.event(kind, **fields)
            for sink in others:
                with contextlib.suppress(Exception):
                    sink(kind, **fields)

        return fanout

    # ── reading ────────────────────────────────────────────────────────────

    def rows(self, *, session: int | None = None, since: float | None = None,
             limit: int = 2000, kinds: set[str] | None = None) -> list[dict[str, Any]]:
        """Newest last. Reads the file, so any thread may ask."""
        out: list[dict[str, Any]] = []
        for row in self._iter():
            if session is not None and row.get("session") != int(session):
                continue
            if since is not None and float(row.get("ts") or 0) < since:
                continue
            if kinds and row.get("kind") not in kinds:
                continue
            out.append(row)
        return out[-max(1, int(limit)):]

    def _iter(self) -> Iterator[dict[str, Any]]:
        for path in (self._path.with_suffix(".1.jsonl"), self._path):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a crash mid-write
                if isinstance(row, dict) and "kind" in row:
                    yield row
