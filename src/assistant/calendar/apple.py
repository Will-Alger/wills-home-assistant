"""Apple Calendar over iCloud CalDAV.

iCloud speaks plain CalDAV at https://caldav.icloud.com with Basic auth: the
Apple ID and an APP-SPECIFIC password (appleid.apple.com -> Sign-In and
Security -> App-Specific Passwords). The account password is rejected, and
two-factor prompts never reach a headless assistant.

Shapes verified against the installed caldav 3.2 source (2026-08-31), not
memory: `DAVClient(url=..., username=..., password=...)` does RFC 6764
discovery, `principal().get_calendars()` lists the collections,
`Calendar.search(start=, end=, event=True, expand=True)` returns each
occurrence as its own `Event` (recurrences expanded client-side), and
`Calendar.add_event(summary=..., dtstart=..., dtend=...)` builds the VEVENT
through `caldav.lib.vcal.create_ical` — any icalendar property may be passed
as a keyword — and PUTs it. So create-event support (the open question in
docs/FEATURES.md) is real and generic, not iCloud-specific.

The library is synchronous, so every call hops to a worker thread; one lock
serializes them, because a DAVClient session is not thread-safe.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

import caldav

from assistant.calendar.base import (
    CalendarError,
    CalendarEvent,
    as_datetime,
    local_tz,
)

ICLOUD_CALDAV_URL = "https://caldav.icloud.com"


class AppleCalendar:
    """CalendarApi backed by a real iCloud (or any CalDAV) account."""

    def __init__(
        self,
        username: str,
        app_password: str,
        *,
        url: str = ICLOUD_CALDAV_URL,
        default_calendar: str = "",
        timeout: int = 20,
    ) -> None:
        if not username or not app_password:
            raise CalendarError(
                "the calendar isn't set up yet — it needs an iCloud Apple ID and "
                "an app-specific password"
            )
        self._url = url
        self._username = username
        self._password = app_password
        self._default_calendar = default_calendar
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._calendars: list[Any] | None = None

    async def calendar_names(self) -> list[str]:
        return await self._call(lambda: [_display_name(c) for c in self._event_calendars()])

    async def list_events(
        self, start: datetime, end: datetime, *, calendar: str | None = None
    ) -> list[CalendarEvent]:
        def read() -> list[CalendarEvent]:
            target = self._pick(calendar)
            found = target.search(start=start, end=end, event=True, expand=True)
            events = [_to_event(obj, _display_name(target)) for obj in found]
            return sorted(events, key=lambda e: as_datetime(e.start))

        return await self._call(read)

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
        def write() -> CalendarEvent:
            target = self._pick(calendar)
            created = target.add_event(
                summary=summary,
                dtstart=start,
                dtend=end,
                location=location,
                description=description,
            )
            return _to_event(created, _display_name(target))

        return await self._call(write)

    async def delete_event(self, uid: str, *, calendar: str | None = None) -> None:
        def remove() -> None:
            target = self._pick(calendar)
            found = target.event_by_uid(uid)
            found.delete()

        await self._call(remove)

    async def _call(self, work: Any) -> Any:
        """Run one blocking caldav call off the loop, with speakable errors."""
        async with self._lock:
            try:
                return await asyncio.to_thread(work)
            except CalendarError:
                raise
            except Exception as err:
                raise CalendarError(_explain(err)) from err

    def _event_calendars(self) -> list[Any]:
        if self._calendars is None:
            client = caldav.DAVClient(
                url=self._url,
                username=self._username,
                password=self._password,
                timeout=self._timeout,
            )
            found = list(client.principal().get_calendars())
            self._calendars = [c for c in found if _holds_events(c)] or found
        if not self._calendars:
            raise CalendarError("that iCloud account has no calendars in it")
        return self._calendars

    def _pick(self, name: str | None) -> Any:
        wanted = (name or self._default_calendar or "").strip().lower()
        calendars = self._event_calendars()
        if not wanted:
            return calendars[0]
        names = [_display_name(c) for c in calendars]
        for calendar, display in zip(calendars, names, strict=True):
            if display.lower() == wanted:
                return calendar
        partial = [c for c, d in zip(calendars, names, strict=True) if wanted in d.lower()]
        if len(partial) == 1:
            return partial[0]
        raise CalendarError(
            f"I don't see a calendar called {name or self._default_calendar!r} — "
            f"there's {', '.join(names)}"
        )


def _holds_events(calendar: Any) -> bool:
    """Skip iCloud's reminder lists, which live in the same calendar home."""
    try:
        return "VEVENT" in calendar.get_supported_components()
    except Exception:  # noqa: BLE001 — the property is optional (RFC 4791 5.2.3)
        return True


def _display_name(calendar: Any) -> str:
    try:
        return str(calendar.get_display_name() or "").strip() or "Calendar"
    except Exception:  # noqa: BLE001 — a nameless collection is still usable
        return "Calendar"


def _to_local(value: Any) -> datetime | date | None:
    if isinstance(value, datetime):
        tz = local_tz()
        return value.astimezone(tz) if value.tzinfo else value.replace(tzinfo=tz)
    return value if isinstance(value, date) else None


def _to_event(obj: Any, calendar_name: str) -> CalendarEvent:
    component = obj.icalendar_component
    start = _to_local(getattr(component.get("dtstart"), "dt", None))
    if start is None:
        raise CalendarError("the calendar returned an event with no start time")
    end = _to_local(getattr(component.get("dtend"), "dt", None))
    if end is None:
        duration = getattr(component.get("duration"), "dt", None)
        if duration is not None and isinstance(start, datetime):
            end = start + duration
    return CalendarEvent(
        uid=str(component.get("uid") or ""),
        summary=str(component.get("summary") or "(untitled)"),
        start=start,
        end=end,
        calendar=calendar_name,
        location=_text(component.get("location")),
        description=_text(component.get("description")),
    )


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _explain(err: Exception) -> str:
    message = f"{type(err).__name__}: {err}"
    lowered = message.lower()
    if "401" in lowered or "unauthor" in lowered or "authoriz" in lowered:
        return (
            "iCloud rejected those calendar credentials — it needs an "
            "app-specific password from appleid.apple.com, not the Apple ID password"
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "iCloud didn't answer in time — the calendar is unreachable right now"
    return f"the calendar request failed ({message})"
