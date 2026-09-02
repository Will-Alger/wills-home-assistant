"""Routines: structured "when I ask for X under condition Y, do it like Z."

Distinct from free-text preferences (which the model interprets), a routine is
applied DETERMINISTICALLY in the tool executor: when a matching tool call
arrives inside the routine's window, its `defaults` fill in any fields the
model left out and its `overrides` are applied regardless. "TV volume defaults
to 65%" or "when I ask for lights after 5pm, use warm orange" never depend on
the model remembering. Routines are also described in her instructions so she
can explain and honor them in conversation. Persisted in data/routines.json.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class Routine:
    id: int
    description: str  # the owner's words, read back
    tool: str  # which tool it shapes: set_lights, media_control, play_music, ... ("" = any)
    defaults: dict[str, Any] = field(default_factory=dict)  # used when the model omitted the field
    overrides: dict[str, Any] = field(default_factory=dict)  # always applied
    after: str = ""  # "HH:MM" local window start
    before: str = ""  # window end (wraps overnight)
    days: list[str] = field(default_factory=list)
    match: dict[str, Any] = field(default_factory=dict)  # e.g. {"action": "volume_set"}
    active: bool = True
    created: float = field(default_factory=time.time)
    applied: int = 0


def _minutes(hhmm: str) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", hhmm or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _in_window(after: str, before: str, when: datetime) -> bool:
    start, end = _minutes(after), _minutes(before)
    minute = when.hour * 60 + when.minute
    if start is None and end is None:
        return True
    if start is None:
        return minute < end  # type: ignore[operator]
    if end is None:
        return minute >= start
    return start <= minute < end if start <= end else (minute >= start or minute < end)


def _matches(routine: Routine, tool: str, tool_input: dict[str, Any], when: datetime) -> bool:
    if not routine.active:
        return False
    if routine.tool and routine.tool != tool:
        return False
    if routine.days and _DAYS[when.weekday()] not in [d.lower()[:3] for d in routine.days]:
        return False
    if not _in_window(routine.after, routine.before, when):
        return False
    return all(str(tool_input.get(k, "")).lower() == str(v).lower() for k, v in routine.match.items())


_LIGHT_FIELDS = ("brightness_pct", "rgb_color", "color_temp_kelvin", "transition_seconds")


def _apply(routine: Routine, tool: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(tool_input))  # deep copy; tool inputs are plain JSON
    if tool == "set_lights":
        # per-change defaults for lights being turned ON, e.g. a warm evening color
        for change in out.get("changes", []):
            if not isinstance(change, dict) or change.get("turn", "on") != "on":
                continue
            colored = any(k in change for k in ("rgb_color", "color_temp_kelvin"))
            for key, value in routine.defaults.items():
                if key in ("rgb_color", "color_temp_kelvin") and colored:
                    continue  # the speaker asked for a specific color
                if key in _LIGHT_FIELDS and key not in change:
                    change[key] = value
            for key, value in routine.overrides.items():
                if key in _LIGHT_FIELDS:
                    change[key] = value
        for key, value in routine.defaults.items():
            if key not in _LIGHT_FIELDS and key not in out:
                out[key] = value
    else:
        for key, value in routine.defaults.items():
            if key not in out or out[key] in (None, ""):
                out[key] = value
    for key, value in routine.overrides.items():
        if tool != "set_lights" or key not in _LIGHT_FIELDS:
            out[key] = value
    return out


class RoutineStore:
    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._now = now
        self._next_id = 1
        self._routines: list[Routine] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = int(data.get("next_id", 1))
        known = set(Routine.__dataclass_fields__)
        for row in data.get("routines", []):
            if isinstance(row, dict) and "id" in row:
                self._routines.append(Routine(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "routines": [asdict(r) for r in self._routines]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(
        self,
        description: str,
        *,
        tool: str = "",
        defaults: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
        after: str = "",
        before: str = "",
        days: list[str] | None = None,
        match: dict[str, Any] | None = None,
    ) -> Routine:
        description = " ".join(str(description).split())
        if not description:
            raise ValueError("a routine needs a description in the owner's words")
        if not (defaults or overrides):
            raise ValueError("a routine needs defaults or overrides to apply")
        for key, value in (("after", after), ("before", before)):
            if value and _minutes(value) is None:
                raise ValueError(f"{key} must look like 17:00 (got {value!r})")
        routine = Routine(
            id=self._next_id,
            description=description,
            tool=str(tool or ""),
            defaults=dict(defaults or {}),
            overrides=dict(overrides or {}),
            after=str(after or ""),
            before=str(before or ""),
            days=[str(d) for d in (days or [])],
            match=dict(match or {}),
            created=self._now(),
        )
        self._next_id += 1
        self._routines.append(routine)
        self._save()
        return routine

    def remove(self, routine_id: int) -> Routine | None:
        for r in self._routines:
            if r.id == int(routine_id) and r.active:
                r.active = False
                self._save()
                return r
        return None

    def active(self) -> list[Routine]:
        return [r for r in self._routines if r.active]

    def apply(self, tool: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], list[Routine]]:
        """The tool input with every matching routine applied, and which ones."""
        when = datetime.fromtimestamp(self._now()).astimezone()
        applied: list[Routine] = []
        out = tool_input
        for routine in self._routines:
            if _matches(routine, tool, out, when):
                out = _apply(routine, tool, out)
                routine.applied += 1
                applied.append(routine)
        if applied:
            self._save()
        return out, applied

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "rule": r.description,
                "tool": r.tool or "any",
                "window": " ".join(
                    p for p in (f"after {r.after}" if r.after else "", f"before {r.before}" if r.before else "",
                                f"on {', '.join(r.days)}" if r.days else "") if p
                ) or "always",
                "defaults": r.defaults,
                "overrides": r.overrides,
                "applied": r.applied,
            }
            for r in self.active()
        ]

    def text(self) -> str:
        rows = self.active()
        if not rows:
            return "(none)"
        return "\n".join(f"- [id {r.id}] {r.description}" for r in rows)
