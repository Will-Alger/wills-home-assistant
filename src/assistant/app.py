"""The app's idle loop, extracted so it can be tested without audio.

While idle, the app watches every 80 ms mic frame for the wake word AND
checks whether an announcement is due — the two things that can start a
conversation. `wait_for_trigger` returns which one happened (or "restart"
when something else — a phone approve — asked for one, or "reconfigure" when
the microphone or speaker was changed and the idle mic must be reopened).
`record_session` is the bookkeeping after a conversation: the session log,
the journal, and reflection.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from assistant.context import pending_question, temporary_override


class RescanClock:
    """While she is on a fallback microphone (the saved one unplugged, or the
    default a virtual input), the idle loop re-scans the devices every
    `every_s` so the real microphone is picked up the moment it is back —
    no restart, no voice command she could not hear anyway. `due()` is polled
    from the idle loop and fires once per period; inactive clocks never fire."""

    def __init__(self, active: bool, every_s: float = 30.0, now: Callable[[], float] = time.monotonic) -> None:
        self._active = active
        self._every = every_s
        self._now = now
        self._next = now() + every_s

    def due(self) -> bool:
        if not self._active or self._now() < self._next:
            return False
        self._next = self._now() + self._every
        return True


async def wait_for_trigger(
    source: Any,
    wake: Any,
    announcer: Any = None,
    restart: Callable[[], bool] | None = None,
    reconfigure: Callable[[], bool] | None = None,
    convert: Callable[[bytes], bytes] | None = None,
    stall_s: float | None = None,
    on_score: Callable[[float, bool], None] | None = None,
) -> str:
    """Block until the wake phrase is heard ("wake"), an announcement is
    ready to be spoken ("announce"), a restart was requested ("restart"),
    the audio devices changed ("reconfigure"), or no frame arrived for
    `stall_s` seconds ("stalled" — an unplugged microphone gives no error,
    only silence). `convert` turns a source frame into what the detector
    expects (the session mic runs at 24 kHz, the wake model wants 16 kHz)
    so one microphone serves idle and talk — and whatever he says right
    after the wake phrase queues on that same stream and reaches the
    session first. `on_score` sees every idle frame's wake score and
    whether it fired — that is how the latency log times the wake and
    keeps the near misses."""
    while True:
        if stall_s is not None:
            try:
                frame = await asyncio.wait_for(source.get_frame(), timeout=stall_s)
            except TimeoutError:
                return "stalled"
        else:
            frame = await source.get_frame()
        fired = wake.detect(convert(frame) if convert is not None else frame)
        if on_score is not None:
            with contextlib.suppress(Exception):  # measuring never blocks a wake
                on_score(float(getattr(wake, "last_score", 0.0)), fired)
        if fired:
            return "wake"
        if announcer is not None and announcer.due():
            return "announce"
        if restart is not None and restart():
            return "restart"
        if reconfigure is not None and reconfigure():
            return "reconfigure"


async def record_session(
    sessions: Any,
    journal: Any,
    stats: Any,
    row: Any,
    reflector: Any,
    timings: dict[str, Any] | None = None,
    context: Any = None,
) -> Any | None:
    """Close the session's row, journal it, then let reflection add the
    one-line summary (its fallback is the first thing the owner said).
    Sequential on purpose: the row exists before the summary updates it.
    `timings` is the last turn's latency row, kept on the session so "how
    fast were you?" never has to re-read the log. `context` is the expiring
    working context (context.py): the topic, an unanswered question and any
    "just for tonight" rule outlive the close by half an hour."""
    first = next((text for role, text in stats.transcript if role == "you"), "")
    if sessions is not None and row is not None:
        with contextlib.suppress(Exception):
            sessions.finish(
                row.id,
                ended_by=stats.ended_by,
                first_user_line=first,
                tools=stats.tool_calls,
                announced=getattr(stats, "announced", []),
                responses=stats.responses,
                cost_usd=stats.cost_usd,
                timings=timings or {},
            )
    if journal is not None:
        with contextlib.suppress(Exception):
            journal.write(
                "session",
                first or "(announcement only)",
                source=getattr(row, "kind", "wake") if row is not None else "wake",
                data={
                    "id": getattr(row, "id", 0),
                    "ended_by": stats.ended_by,
                    "tools": list(stats.tool_calls)[:20],
                },
            )
    if context is not None:
        with contextlib.suppress(Exception):
            note_context(context, stats, first)
    if reflector is None:
        return None
    return await reflect_session(reflector, sessions, row, stats, context)


def note_context(context: Any, stats: Any, first: str) -> None:
    """What this conversation leaves behind for the next one: the topic (the
    first thing he said, until reflection replaces it with the summary), a
    question of hers he never answered, and any "just for tonight" rule.
    A session where she never spoke — an announcement nobody heard — leaves
    an earlier pending question exactly where it was."""
    context.note_topic(first)
    if any(role == "alexa" for role, _ in stats.transcript):
        asked = pending_question(stats.transcript)
        if asked is None:
            context.answered()
        else:
            context.ask(asked[0], asked[1])
    for role, said in stats.transcript:
        if role == "you" and (rule := temporary_override(said)):
            context.note_override(rule)


async def reflect_session(
    reflector: Any, sessions: Any, row: Any, stats: Any, context: Any = None
) -> Any | None:
    """Reflection is a Claude CLI call that takes seconds: the runner awaits
    it in the BACKGROUND, because while it ran in line the wake-word mic
    stayed closed and "hey alexa" right after a conversation went unheard."""
    reflection = None
    if stats.transcript:
        with contextlib.suppress(Exception):  # learning must never break the loop
            reflection = await reflector.reflect(stats.transcript)
    summary = getattr(reflection, "summary", "") if reflection is not None else ""
    if summary and sessions is not None and row is not None:
        with contextlib.suppress(Exception):
            sessions.set_summary(row.id, summary)
    if summary and context is not None:
        with contextlib.suppress(Exception):
            context.note_topic(summary)  # a better topic than his opening line
    return reflection
