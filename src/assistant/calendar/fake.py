"""In-memory calendar implementing CalendarApi — tests and `--fake` mode.

Same role as FakeHome: lets the calendar tools be exercised end to end
without Will's iCloud credentials or a network round trip.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime

from assistant.calendar.base import CalendarError, CalendarEvent, as_datetime


@dataclass
class FakeCalendar:
    events: list[CalendarEvent] = field(default_factory=list)
    names: list[str] = field(default_factory=lambda: ["Home", "Work"])
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    async def calendar_names(self) -> list[str]:
        return list(self.names)

    async def list_events(
        self, start: datetime, end: datetime, *, calendar: str | None = None
    ) -> list[CalendarEvent]:
        name = self._pick(calendar)
        hits = [
            event
            for event in self.events
            if event.calendar == name
            and as_datetime(event.end or event.start, end_of_day=True) >= start
            and as_datetime(event.start) <= end
        ]
        return sorted(hits, key=lambda event: as_datetime(event.start))

    async def create_event(
        self,
        *,
        summary: str,
        start: datetime | date,
        end: datetime | date | None = None,
        calendar: str | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> CalendarEvent:
        event = CalendarEvent(
            uid=f"fake-{next(self._ids)}",
            summary=summary,
            start=start,
            end=end,
            calendar=self._pick(calendar),
            location=location,
            description=description,
        )
        self.events.append(event)
        return event

    async def delete_event(self, uid: str, *, calendar: str | None = None) -> None:
        remaining = [event for event in self.events if event.uid != uid]
        if len(remaining) == len(self.events):
            raise CalendarError(f"no event with id {uid}")
        self.events = remaining

    def _pick(self, calendar: str | None) -> str:
        if not calendar:
            return self.names[0]
        for name in self.names:
            if name.lower() == calendar.strip().lower():
                return name
        raise CalendarError(
            f"I don't see a calendar called {calendar!r} — there's {', '.join(self.names)}"
        )
