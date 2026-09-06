"""Working context: the last few minutes, kept across the close.

A session ends and "a little dimmer" or "let's do the second option" has
nothing to bind to — the session log (sessions.py) keeps what was SAID, not
what it was about. This is the small record in between: the topic, the
entities her last home actions touched, the last successful action, a
question she asked and never got answered, temporary overrides ("just for
tonight…"), and the jobs still running.

Every field carries its own timestamp and dies on its own clock — half an
hour, ten minutes for an unanswered question — because a stale reference is
worse than none: it makes her act confidently on the wrong lamps. When a
field is gone she asks what they meant. Durable things (preferences, facts)
belong in memory.py and are not this; nothing here outlives the hour.
Persisted in data/context.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

DEFAULT_TTL_S = 1800.0  # half an hour: past that, "it" is a guess
QUESTION_TTL_S = 600.0  # an unanswered question goes cold faster than a lamp

_TTL: dict[str, float] = {
    "topic": DEFAULT_TTL_S,
    "entities": DEFAULT_TTL_S,
    "action": DEFAULT_TTL_S,
    "question": QUESTION_TTL_S,
    "overrides": DEFAULT_TTL_S,
    "jobs": DEFAULT_TTL_S,
}
_FIELDS = tuple(_TTL)
_MAX_ENTITIES = 6  # more than this and nothing is "most likely" any more
_MAX_OVERRIDES = 3

_EMPTY = (
    "nothing recent — 'it' or 'that' has nothing to bind to yet, so ask what they mean"
)

# The lead-in she would strip to hear the actual choice on offer.
_QUESTION_LEAD = re.compile(
    r"^(?:so\s+|and\s+)?(?:would you (?:like|prefer)|do you want|did you want|"
    r"should i|shall i|which(?: one)?(?: would you like)?|what about|"
    r"do you mean)\b[:,]?\s*",
    re.IGNORECASE,
)
_OR_SPLIT = re.compile(r",?\s+\bor\b\s+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# "Just for tonight", "only this once" — a rule that applies now and is NOT a
# standing routine. Deliberately narrow: bare "for now" is how he says
# goodbye ("that's all for now"), so the marker must carry just/only.
_TEMPORARY = re.compile(
    r"\b(?:just|only)\s+(?:for\s+)?(?:now|tonight|today|this once|this time)\b"
    r"|\bthis time only\b",
    re.IGNORECASE,
)


def _ago(seconds: float) -> str:
    """How long ago, the way she would say it in one breath."""
    if seconds < 90:
        return "a moment ago"
    return f"{round(seconds / 60)} minutes ago"


def question_options(question: str) -> list[str]:
    """'Warm white or amber?' -> ['warm white', 'amber'] — so "the second
    one" still resolves in the session after the one that offered them."""
    text = _QUESTION_LEAD.sub("", str(question).strip().rstrip("?").strip())
    parts = [" ".join(p.split()).strip(" ,.;:") for p in _OR_SPLIT.split(text)]
    parts = [p for p in parts if 0 < len(p) <= 40]
    return parts if len(parts) >= 2 else []


def pending_question(transcript: Iterable[tuple[str, str]]) -> tuple[str, list[str]] | None:
    """Her last words were a question nobody answered — what "yes", "the
    second one" or "let's do that" would attach to next time. Anything he
    says afterwards counts as the answer, whatever it was."""
    said = ""
    for role, text in transcript:
        if role == "you":
            said = ""
        elif role == "alexa":
            said = str(text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(said.strip()) if s.strip()]
    if not sentences or not sentences[-1].endswith("?"):
        return None
    last = " ".join(sentences[-1].split())[:200]
    return last, question_options(last)


def temporary_override(said: str) -> str:
    """"Just for tonight, keep the volume down" — a rule for this stretch
    only, which is exactly why it must not become a routine."""
    text = " ".join(str(said).split())
    return text[:160] if text and len(text) <= 160 and _TEMPORARY.search(text) else ""


class WorkingContext:
    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._now = now
        # field -> {"value": ..., "at": timestamp}; absent means never set
        self._fields: dict[str, dict[str, Any]] = {}
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name in _FIELDS:
            row = data.get(name)
            if isinstance(row, dict) and "value" in row:
                self._fields[name] = {"value": row["value"], "at": float(row.get("at", 0.0))}

    def _save(self) -> None:
        with contextlib.suppress(OSError):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._fields, indent=1), encoding="utf-8")
            os.replace(tmp, self._path)

    # ── the clock ──────────────────────────────────────────────────────────

    def _set(self, name: str, value: Any) -> None:
        self._fields[name] = {"value": value, "at": self._now()}
        self._save()

    def _drop(self, name: str) -> None:
        if self._fields.pop(name, None) is not None:
            self._save()

    def _live(self, name: str) -> dict[str, Any] | None:
        """The field, or None once its own clock has run out."""
        row = self._fields.get(name)
        if row is None:
            return None
        if self._now() - float(row.get("at", 0.0)) > _TTL[name]:
            return None
        return row

    def _value(self, name: str, default: Any = None) -> Any:
        row = self._live(name)
        return default if row is None else row["value"]

    def _age(self, name: str) -> float:
        row = self._live(name)
        return 0.0 if row is None else max(0.0, self._now() - float(row["at"]))

    # ── writing ────────────────────────────────────────────────────────────

    def note_topic(self, text: str) -> None:
        """One line: what this conversation was about."""
        topic = " ".join(str(text).split())[:160]
        if topic:
            self._set("topic", topic)

    def note_entities(self, items: Iterable[Any]) -> None:
        """What the last home action touched, newest first. Merged rather
        than replaced: lights and then a speaker is two candidates, and two
        candidates is precisely when she should ask which. Each entity keeps
        its OWN moment, so a lamp from half an hour ago cannot ride along on
        a speaker she touched just now."""
        now = self._now()
        fresh: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                entity_id, name = str(item.get("id", "")), str(item.get("name", ""))
            else:
                entity_id, name = str(item[0]), str(item[1])
            if entity_id:
                fresh.append({"id": entity_id, "name": name or entity_id, "at": now})
        if not fresh:
            return
        known = {row["id"] for row in fresh}
        fresh.extend(row for row in self.entities() if row["id"] not in known)
        self._set("entities", fresh[:_MAX_ENTITIES])

    def note_action(self, tool: str, args: dict[str, Any] | None, outcome: str) -> None:
        """The last thing that actually worked — what "do that again" or "a
        little more" is a modification of."""
        self._set(
            "action",
            {
                "tool": str(tool),
                "args": _small(args),
                "outcome": " ".join(str(outcome).split())[:160],
            },
        )

    def ask(self, question: str, options: Iterable[str] = ()) -> None:
        text = " ".join(str(question).split())[:200]
        if text:
            self._set(
                "question",
                {"text": text, "options": [" ".join(str(o).split())[:60] for o in options][:6]},
            )

    def answered(self) -> None:
        self._drop("question")

    def note_override(self, text: str) -> None:
        rule = " ".join(str(text).split())[:160]
        if not rule:
            return
        rules = [r for r in (self._value("overrides", []) or []) if r != rule]
        self._set("overrides", [rule, *rules][:_MAX_OVERRIDES])

    def note_job(self, key: str, text: str) -> None:
        """Something still running she may be asked about ("is it done?")."""
        label = " ".join(str(text).split())[:120]
        if not label:
            return
        jobs = dict(self._value("jobs", {}) or {})
        jobs[str(key)] = label
        self._set("jobs", jobs)

    def clear_job(self, key: str) -> None:
        jobs = dict(self._value("jobs", {}) or {})
        if jobs.pop(str(key), None) is not None:
            self._set("jobs", jobs)

    def clear(self) -> None:
        self._fields = {}
        self._save()

    # ── reading back ───────────────────────────────────────────────────────

    def entities(self) -> list[dict[str, Any]]:
        """The entities still fresh enough to be what "it" means, newest
        first — each on its own clock, not the field's last write."""
        now = self._now()
        rows = self._value("entities", []) or []
        return [
            {"id": str(row["id"]), "name": str(row.get("name") or row["id"]), "at": float(row["at"])}
            for row in rows
            if isinstance(row, dict) and row.get("id") and now - float(row.get("at", 0.0)) <= _TTL["entities"]
        ]

    def snapshot(self) -> dict[str, Any]:
        """Every field that is still live — expired ones are simply absent."""
        live = {name: self._value(name) for name in _FIELDS if self._live(name) is not None}
        if "entities" in live:
            fresh = self.entities()
            live["entities"] = fresh
            if not fresh:
                del live["entities"]
        return live

    def text(self) -> str:
        """One short paragraph for the `{context}` placeholder."""
        parts: list[str] = []
        action = self._value("action")
        topic = self._value("topic")
        if isinstance(action, dict):
            outcome = action.get("outcome") or "done"
            # The arguments matter as much as the outcome: "a little brighter"
            # is only answerable if she can see it was 30% a minute ago.
            said = json.dumps(action.get("args") or {}, separators=(",", ":"))[:200]
            parts.append(
                f"Just now ({_ago(self._age('action'))}): {action.get('tool')}"
                f"{' ' + said if said != '{}' else ''} — {outcome}"
            )
            if topic:
                parts.append(f"the topic was {topic}")
        elif topic:
            parts.append(f"Just now ({_ago(self._age('topic'))}): {topic}")
        entities = self.entities()
        if entities:
            named = ", ".join(
                row["id"] if row["name"] == row["id"] else f"{row['name']} [{row['id']}]"
                for row in entities
            )
            parts.append(f"'it', 'that' or 'them' most likely means {named}")
        question = self._value("question")
        if isinstance(question, dict) and question.get("text"):
            asked = f'you asked "{question["text"]}" and got no answer'
            options = question.get("options") or []
            if options:
                asked += " (the options were: " + ", ".join(options) + ")"
            parts.append(asked)
        overrides = self._value("overrides") or []
        if overrides:
            parts.append("just for this stretch: " + "; ".join(overrides))
        jobs = self._value("jobs") or {}
        if jobs:
            parts.append("still running: " + "; ".join(jobs.values()))
        return ("; ".join(p.rstrip(". ") for p in parts) + ".") if parts else _EMPTY


def _small(args: dict[str, Any] | None) -> dict[str, Any]:
    """Tool arguments, shrunk to what is worth carrying — the record is read
    aloud-adjacent, not an audit log (that is the journal's job)."""
    try:
        text = json.dumps(args or {}, default=str)
    except (TypeError, ValueError):
        return {}
    return json.loads(text) if len(text) <= 400 else {}
