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

import contextlib
from collections.abc import Callable
from typing import Any


async def wait_for_trigger(
    source: Any,
    wake: Any,
    announcer: Any = None,
    restart: Callable[[], bool] | None = None,
    reconfigure: Callable[[], bool] | None = None,
) -> str:
    """Block until the wake phrase is heard ("wake"), an announcement is
    ready to be spoken ("announce"), a restart was requested ("restart"),
    or the audio devices changed ("reconfigure")."""
    while True:
        frame = await source.get_frame()
        if wake.detect(frame):
            return "wake"
        if announcer is not None and announcer.due():
            return "announce"
        if restart is not None and restart():
            return "restart"
        if reconfigure is not None and reconfigure():
            return "reconfigure"


async def record_session(
    sessions: Any, journal: Any, stats: Any, row: Any, reflector: Any
) -> Any | None:
    """Close the session's row, journal it, then let reflection add the
    one-line summary (its fallback is the first thing the owner said).
    Sequential on purpose: the row exists before the summary updates it."""
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
    reflection = None
    if reflector is not None and stats.transcript:
        with contextlib.suppress(Exception):  # learning must never break the loop
            reflection = await reflector.reflect(stats.transcript)
    summary = getattr(reflection, "summary", "") if reflection is not None else ""
    if summary and sessions is not None and row is not None:
        with contextlib.suppress(Exception):
            sessions.set_summary(row.id, summary)
    return reflection
