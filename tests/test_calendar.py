"""Calendar tools and the iCloud CalDAV mapping — no network, no credentials.

The AppleCalendar tests stub only the DAVClient transport: the ICS that goes
to iCloud is built by the real caldav/icalendar code and read back through
our real mapping, so a keyword the library would reject fails here first.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import caldav
import pytest
from caldav.lib import vcal

from assistant.brain.tools import CALENDAR_TOOLS, ToolExecutor
from assistant.calendar.base import (
    CalendarError,
    CalendarEvent,
    local_tz,
    parse_when,
    spoken_when,
)
from assistant.calendar.fake import FakeCalendar
from assistant.home.fake import FakeHome

TZ = local_tz()


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


# ── parsing and phrasing ────────────────────────────────────────────────────


def test_parse_when_reads_dates_datetimes_and_offsets() -> None:
    assert parse_when("2026-09-02") == date(2026, 9, 2)
    assert parse_when("2026-09-02T15:00") == at("2026-09-02T15:00")
    assert parse_when("2026-09-02 15:00") == at("2026-09-02T15:00")
    # An explicit offset is honored, then converted into the local zone.
    assert parse_when("2026-09-02T15:00:00Z") == datetime(2026, 9, 2, 15, tzinfo=UTC).astimezone(TZ)


def test_parse_when_rejects_free_text_with_a_speakable_error() -> None:
    with pytest.raises(CalendarError) as err:
        parse_when("next tuesday")
    assert "2026-09-02" in str(err.value)  # tells the model the shape it needs


def test_spoken_when_is_readable_aloud() -> None:
    timed = CalendarEvent("1", "Dentist", at("2026-09-03T15:00"), at("2026-09-03T16:00"))
    assert spoken_when(timed) == "Thu Sep 3, 3:00 PM to 4:00 PM"
    all_day = CalendarEvent("2", "Birthday", date(2026, 9, 5))
    assert spoken_when(all_day) == "Sat Sep 5, all day"


# ── the voice tools, against the fake calendar ──────────────────────────────


@pytest.fixture
def executor() -> tuple[ToolExecutor, FakeCalendar]:
    calendar = FakeCalendar()
    return ToolExecutor(FakeHome(), calendar), calendar


async def test_create_then_list_round_trips(executor) -> None:
    tools, calendar = executor
    start = (datetime.now(tz=TZ) + timedelta(days=1)).replace(microsecond=0)
    result, is_error = await tools.execute(
        "create_calendar_event",
        {"summary": "Dinner with Sam", "start": start.isoformat(), "location": "Via Carota"},
    )
    assert not is_error
    assert "Dinner with Sam" in result
    stored = calendar.events[0]
    assert stored.end == start + timedelta(hours=1)  # default length
    assert stored.location == "Via Carota"

    listed, is_error = await tools.execute("list_calendar_events", {})
    assert not is_error
    payload = json.loads(listed)
    assert [event["summary"] for event in payload["events"]] == ["Dinner with Sam"]
    assert payload["events"][0]["location"] == "Via Carota"
    assert payload["now"]  # the model needs a clock to resolve "tomorrow"


async def test_list_window_defaults_to_a_week_and_honors_an_explicit_range(executor) -> None:
    tools, calendar = executor
    now = datetime.now(tz=TZ)
    await calendar.create_event(summary="Soon", start=now + timedelta(days=2))
    await calendar.create_event(summary="Later", start=now + timedelta(days=30))

    default_window = json.loads((await tools.execute("list_calendar_events", {}))[0])
    assert [e["summary"] for e in default_window["events"]] == ["Soon"]

    wide = json.loads(
        (
            await tools.execute(
                "list_calendar_events",
                {"start": now.date().isoformat(), "end": (now + timedelta(days=40)).date().isoformat()},
            )
        )[0]
    )
    assert [e["summary"] for e in wide["events"]] == ["Soon", "Later"]


async def test_date_only_start_creates_an_all_day_event(executor) -> None:
    tools, calendar = executor
    result, is_error = await tools.execute(
        "create_calendar_event", {"summary": "Flight day", "start": "2026-09-05"}
    )
    assert not is_error and "all day" in result
    assert calendar.events[0].start == date(2026, 9, 5)
    assert calendar.events[0].end is None


async def test_duration_minutes_sets_the_end(executor) -> None:
    tools, calendar = executor
    await tools.execute(
        "create_calendar_event",
        {"summary": "Standup", "start": "2026-09-02T09:00", "duration_minutes": 15},
    )
    assert calendar.events[0].end == at("2026-09-02T09:15")


async def test_delete_requires_confirmation_and_uses_the_uid(executor) -> None:
    tools, calendar = executor
    await calendar.create_event(
        summary="Mistake", start=datetime.now(tz=TZ) + timedelta(days=1)
    )
    uid = calendar.events[0].uid
    # listings expose the uid so the model can name the exact event
    listed = json.loads((await tools.execute("list_calendar_events", {}))[0])
    assert listed["events"][0]["uid"] == uid

    text, is_error = await tools.execute("delete_calendar_event", {"uid": uid})
    assert is_error and "explicit yes" in text
    assert calendar.events  # still there — no confirmation, no deletion

    text, is_error = await tools.execute(
        "delete_calendar_event", {"uid": uid, "confirmed": True}
    )
    assert not is_error and "deleted" in text
    assert calendar.events == []

    text, is_error = await tools.execute(
        "delete_calendar_event", {"uid": "nope", "confirmed": True}
    )
    assert is_error  # honest error for an unknown event


async def test_backwards_event_and_unknown_calendar_come_back_as_errors(executor) -> None:
    tools, _ = executor
    result, is_error = await tools.execute(
        "create_calendar_event",
        {"summary": "Time travel", "start": "2026-09-02T15:00", "end": "2026-09-02T14:00"},
    )
    assert is_error and "before it starts" in result

    result, is_error = await tools.execute(
        "list_calendar_events", {"calendar": "Bowling League"}
    )
    assert is_error and "Bowling League" in result


async def test_tools_are_inert_without_a_configured_calendar() -> None:
    tools = ToolExecutor(FakeHome())
    result, is_error = await tools.execute("list_calendar_events", {})
    assert is_error and "isn't connected" in result


def test_calendar_tools_declare_realtime_ready_schemas() -> None:
    for tool in CALENDAR_TOOLS:
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]
    create = next(t for t in CALENDAR_TOOLS if t["name"] == "create_calendar_event")
    assert create["input_schema"]["required"] == ["summary", "start"]


# ── the iCloud client, with a stubbed transport ─────────────────────────────


class StubDavCalendar:
    """Stands in for a caldav Calendar; builds and parses REAL icalendar data."""

    def __init__(self, name: str, components: list[str], events: list[str]) -> None:
        self._name = name
        self._components = components
        self.stored = [caldav.Event(client=None, data=ics) for ics in events]
        self.searches: list[dict[str, Any]] = []

    def get_display_name(self) -> str:
        return self._name

    def get_supported_components(self) -> list[str]:
        return self._components

    def search(self, **kwargs: Any) -> list[caldav.Event]:
        self.searches.append(kwargs)
        return list(self.stored)

    def add_event(self, **ical_data: Any) -> caldav.Event:
        # The real builder the library would use — rejects bad keywords here.
        ics = vcal.create_ical(objtype="VEVENT", **ical_data)
        event = caldav.Event(client=None, data=ics)
        self.stored.append(event)
        return event


class StubPrincipal:
    def __init__(self, calendars: list[StubDavCalendar]) -> None:
        self._calendars = calendars

    def get_calendars(self) -> list[StubDavCalendar]:
        return list(self._calendars)


class StubDavClient:
    last: StubDavClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        StubDavClient.last = self

    def principal(self) -> StubPrincipal:
        return StubPrincipal(StubDavClient.calendars)


_TIMED_ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:evt-1
SUMMARY:Dentist
DTSTART:20260903T190000Z
DTEND:20260903T200000Z
LOCATION:12 Main St
END:VEVENT
END:VCALENDAR
"""

_ALL_DAY_ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:evt-2
SUMMARY:Sam's birthday
DTSTART;VALUE=DATE:20260905
END:VEVENT
END:VCALENDAR
"""


@pytest.fixture
def apple(monkeypatch):
    from assistant.calendar import apple as apple_module

    home = StubDavCalendar("Home", ["VEVENT"], [_TIMED_ICS, _ALL_DAY_ICS])
    reminders = StubDavCalendar("Groceries", ["VTODO"], [])
    StubDavClient.calendars = [reminders, home]
    monkeypatch.setattr(apple_module.caldav, "DAVClient", StubDavClient)
    return apple_module.AppleCalendar("will@icloud.com", "abcd-efgh-ijkl-mnop"), home


async def test_reminder_lists_are_skipped_when_choosing_a_calendar(apple) -> None:
    calendar, _ = apple
    assert await calendar.calendar_names() == ["Home"]


async def test_list_events_maps_icloud_data_into_local_time(apple) -> None:
    calendar, dav = apple
    start, end = at("2026-09-01T00:00"), at("2026-09-30T00:00")
    events = await calendar.list_events(start, end)

    assert dav.searches == [{"start": start, "end": end, "event": True, "expand": True}]
    dentist, birthday = events
    assert dentist.summary == "Dentist"
    assert dentist.location == "12 Main St"
    assert dentist.start == datetime(2026, 9, 3, 19, tzinfo=UTC).astimezone(TZ)
    assert dentist.calendar == "Home"
    assert birthday.all_day and birthday.start == date(2026, 9, 5)


async def test_create_event_builds_a_vevent_icloud_accepts(apple) -> None:
    calendar, dav = apple
    created = await calendar.create_event(
        summary="Dinner with Sam",
        start=at("2026-09-04T19:30"),
        end=at("2026-09-04T21:00"),
        location="Via Carota",
        description="booked by voice",
    )
    ics = dav.stored[-1].data
    assert "SUMMARY:Dinner with Sam" in ics
    assert "LOCATION:Via Carota" in ics
    assert "UID:" in ics and "DTSTAMP:" in ics  # iCloud rejects a VEVENT without them
    assert created.summary == "Dinner with Sam"
    assert created.start == at("2026-09-04T19:30")
    assert created.end == at("2026-09-04T21:00")


async def test_all_day_creation_keeps_a_date_value(apple) -> None:
    calendar, dav = apple
    created = await calendar.create_event(summary="Flight day", start=date(2026, 9, 5))
    assert "DTSTART;VALUE=DATE:20260905" in dav.stored[-1].data
    assert created.all_day


async def test_credentials_reach_the_client_and_401s_are_explained(apple, monkeypatch) -> None:
    calendar, _ = apple
    await calendar.calendar_names()
    assert StubDavClient.last.kwargs["username"] == "will@icloud.com"
    assert StubDavClient.last.kwargs["password"] == "abcd-efgh-ijkl-mnop"
    assert StubDavClient.last.kwargs["url"] == "https://caldav.icloud.com"

    from assistant.calendar import apple as apple_module

    class Unauthorized(StubDavClient):
        def principal(self):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(apple_module.caldav, "DAVClient", Unauthorized)
    fresh = apple_module.AppleCalendar("will@icloud.com", "wrong")
    with pytest.raises(CalendarError) as err:
        await fresh.calendar_names()
    assert "app-specific password" in str(err.value)


def test_missing_credentials_fail_before_any_request() -> None:
    from assistant.calendar.apple import AppleCalendar

    with pytest.raises(CalendarError):
        AppleCalendar("", "")
