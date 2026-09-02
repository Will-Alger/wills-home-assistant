"""The morning briefing: calendar, tasks awaiting approval, today's schedule,
and what she told him that he hasn't acknowledged. A scheduled `briefing`
item calls the coroutine this returns; the announcer speaks the result."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any


def compose_briefing(
    calendar: Any, board: Any, scheduler: Any, announcer: Any, owner: str
) -> Callable[[], Awaitable[str]]:
    async def compose() -> str:
        from assistant.calendar.base import local_tz, spoken_when

        now = datetime.now(tz=local_tz())
        parts = [f"Good morning, {owner}. It's {now.strftime('%A, %B %d').replace(' 0', ' ')}."]
        if calendar is not None:
            try:
                end = now.replace(hour=23, minute=59, second=59)
                events = await calendar.list_events(now, end)
                if events:
                    items = "; ".join(f"{e.summary} {spoken_when(e)}" for e in events[:5])
                    parts.append(f"Today: {items}.")
                else:
                    parts.append("Nothing on the calendar today.")
            except Exception as err:  # noqa: BLE001
                parts.append(f"I couldn't read the calendar: {str(err)[:80]}.")
        if board is not None:
            waiting = [t for t in board.tasks() if t.state in ("built", "staged")]
            if waiting:
                names = ", ".join(f"task {t.id} {t.title}" for t in waiting[:3])
                parts.append(f"Awaiting your approval: {names}.")
            asking = [t for t in board.tasks() if t.state == "needs_input"]
            if asking:
                parts.append(f"Task {asking[0].id} needs your answer: {asking[0].question[:120]}")
        if scheduler is not None:
            rows = [r for r in scheduler.describe() if r["kind"] != "briefing"]
            soon = [r for r in rows if r["next"].startswith("today")]
            if soon:
                parts.append(
                    "Scheduled today: " + "; ".join(f"{r['label']} {r['next']}" for r in soon[:4]) + "."
                )
        if announcer is not None:
            unread = [a for a in announcer.unread() if a.kind not in ("presence", "briefing")]
            if unread:
                texts = "; ".join(a.text[:100] for a in unread[:3])
                more = f"; and {len(unread) - 3} more" if len(unread) > 3 else ""
                parts.append(f"Unread from me: {texts}{more}.")
        return " ".join(parts)

    return compose
