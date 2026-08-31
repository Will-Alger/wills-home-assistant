"""Calendar types and the CalendarApi protocol.

The brain talks to `CalendarApi`; the real iCloud CalDAV client and the
in-memory fake both implement it, so the calendar tools can be developed and
tested without Apple credentials or a network round trip.

Times crossing this boundary are timezone-aware datetimes in the machine's
local zone. An all-day event carries a bare `date` instead — iCloud stores
those as DATE values, and collapsing them to midnight would make the
assistant say "at twelve in the morning" about a birthday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Protocol


class CalendarError(RuntimeError):
    """A failure phrased so the assistant can read it aloud verbatim."""


def local_tz() -> tzinfo:
    tz = datetime.now().astimezone().tzinfo
    assert tz is not None  # astimezone() always attaches one
    return tz


@dataclass(frozen=True)
class CalendarEvent:
    uid: str
    summary: str
    start: datetime | date
    end: datetime | date | None = None
    calendar: str = ""
    location: str | None = None
    description: str | None = None

    @property
    def all_day(self) -> bool:
        return not isinstance(self.start, datetime)


def parse_when(value: str) -> datetime | date:
    """Parse an ISO date or datetime out of a tool call.

    "2026-09-02" -> a date (all-day). "2026-09-02T15:00" (space instead of
    T is accepted too) -> a local-zone datetime; an explicit offset or Z is
    honored and converted to local.
    """
    text = value.strip().replace(" ", "T")
    if not text:
        raise CalendarError("I need a date and time for that.")
    try:
        parsed = date.fromisoformat(text) if len(text) == 10 else datetime.fromisoformat(text)
    except ValueError as err:
        raise CalendarError(
            f"I couldn't read {value!r} as a date — use 2026-09-02 or 2026-09-02T15:00."
        ) from err
    if isinstance(parsed, datetime):
        tz = local_tz()
        return parsed.replace(tzinfo=tz) if parsed.tzinfo is None else parsed.astimezone(tz)
    return parsed


def as_datetime(value: datetime | date, *, end_of_day: bool = False) -> datetime:
    """Coerce a date or datetime to a local-zone datetime, for range math."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=local_tz())
    moment = time(23, 59, 59) if end_of_day else time(0, 0)
    return datetime.combine(value, moment, tzinfo=local_tz())


def _clock(moment: datetime) -> str:
    """3:05 PM — hand-built because %-I/%#I differ across Windows and Linux."""
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def _day(value: datetime | date) -> str:
    return f"{value:%a %b} {value.day}"


def spoken_when(event: CalendarEvent) -> str:
    """A short, speakable "when" for one event."""
    if event.all_day:
        return f"{_day(event.start)}, all day"
    start = as_datetime(event.start)
    when = f"{_day(start)}, {_clock(start)}"
    if isinstance(event.end, datetime):
        end = as_datetime(event.end)
        return f"{when} to {_clock(end) if end.date() == start.date() else _day(end)}"
    return when


def spoken_now(moment: datetime | None = None) -> str:
    """Current local date and time, phrased for instructions and tool results.

    Without it the model has no idea what "tomorrow at three" means — a
    Realtime session carries no clock of its own.
    """
    moment = moment or datetime.now(tz=local_tz())
    zone = moment.tzname() or ""
    stamp = f"{moment:%A, %B} {moment.day}, {moment.year} at {_clock(moment)}"
    return f"{stamp} {zone}".strip()


def default_end(start: datetime | date, minutes: int) -> datetime | date | None:
    """All-day events end when the day does; timed ones get a default length."""
    if isinstance(start, datetime):
        return start + timedelta(minutes=minutes)
    return None


class CalendarApi(Protocol):
    async def calendar_names(self) -> list[str]:
        """Display names of the account's event calendars."""
        ...

    async def list_events(
        self, start: datetime, end: datetime, *, calendar: str | None = None
    ) -> list[CalendarEvent]:
        """Events overlapping [start, end), recurrences expanded, sorted."""
        ...

    async def create_event(
        self,
        *,
        summary: str,
        start: datetime | date,
        end: datetime | date | None = None,
        calendar: str | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> CalendarEvent: ...

    async def delete_event(self, uid: str, *, calendar: str | None = None) -> None:
        """Remove an event by uid (used by the setup check, not by voice)."""
        ...
