"""Delivery: the policy table (presence x priority x quiet x settle), the
Announcer honoring it, and the Courier (arrival marker, pushes with retry)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from assistant.announce import Announcement, Announcer
from assistant.calendar.base import CalendarEvent
from assistant.delivery import Courier, DeliveryPolicy, DeliverySettings
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.journal import Journal
from assistant.presence import Presence

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def at(hour: int, minute: int = 0) -> float:
    return datetime(2026, 9, 2, hour, minute, tzinfo=LOCAL).timestamp()


def item(kind: str = "task", priority: str = "normal", mode: str = "speak") -> Announcement:
    return Announcement(id=1, text="x", kind=kind, priority=priority, mode=mode)


def away_presence(tmp_path: Path, clock: Clock) -> Presence:
    presence = Presence(tmp_path / "p.json", "person.will", owner="Will", now=clock)
    presence.state, presence.since = "away", clock.at
    return presence


def test_policy_by_presence_and_priority(tmp_path: Path) -> None:
    clock = Clock(at(12))
    presence = Presence(tmp_path / "p.json", "person.will", now=clock)
    policy = DeliveryPolicy(quiet=lambda _now: False, presence=presence)
    now = clock.at
    assert policy.decide(item(), now) == "speak"  # unknown whereabouts = treat as home
    presence.state = "away"
    assert policy.decide(item("task"), now) == "push"
    assert policy.decide(item("action"), now) == "hold"  # waits for the arrival welcome
    assert policy.decide(item("action", "urgent"), now) == "push"
    assert policy.decide(item("nudge", mode="inbox"), now) == "inbox"
    assert policy.decide(item("nudge", "urgent", mode="inbox"), now) == "push"
    presence.state, presence.since = "home", now
    assert policy.decide(item(), now) == "hold"  # just walked in: let him get in the door
    assert policy.decide(item(), now + 91) == "speak"
    night = DeliveryPolicy(quiet=lambda _now: True, presence=presence)
    assert night.decide(item(), now + 91) == "hold"
    assert night.decide(item(priority="urgent"), now + 91) == "speak"
    assert DeliveryPolicy(quiet=lambda _now: False).decide(item(), now) == "speak"  # no presence at all


def test_announcer_holds_while_away_and_queues_pushes(tmp_path: Path) -> None:
    clock = Clock(at(12))
    presence = away_presence(tmp_path, clock)
    a = Announcer(
        tmp_path / "a.json", now=clock,
        policy=DeliveryPolicy(quiet=lambda _now: False, presence=presence).decide,
    )
    built = a.enqueue("Task 7 is built.", kind="task")
    porch = a.enqueue("Done: porch light on.", kind="action")
    assert not a.due()  # nobody home: nothing is spoken
    assert [x.id for x in a.pending_push()] == [built.id]  # ...but the task goes to his phone
    a.mark_pushed(built.id)
    assert a.pending_push() == [] and built.pushed is not None and built.open  # still spoken on arrival
    assert a.unread_summary().startswith("2 unread")  # held items count as unread

    presence.state, presence.since = "home", clock.at
    clock.at += 91  # settled
    assert a.due() and [x.id for x in a.take_due()] == [built.id, porch.id]


async def test_courier_arrival_marker_only_when_something_is_held(tmp_path: Path) -> None:
    clock = Clock(at(18))
    presence = away_presence(tmp_path, clock)
    a = Announcer(
        tmp_path / "a.json", now=clock,
        policy=DeliveryPolicy(quiet=lambda _now: False, presence=presence).decide,
    )
    courier = Courier(a, presence=presence, owner="Will", now=clock)
    presence.observe("home")
    clock.at += 61
    await courier.tick()
    assert [t.kind for t in courier.transitions] == ["arrived"]
    assert a.pending() == []  # nothing was waiting: no welcome for nothing

    presence.state, presence.since = "away", clock.at
    a.enqueue("Task 7 is built.", kind="task", ref="task:7:1:built")
    presence.observe("home")
    clock.at += 61
    await courier.tick()
    assert [x.kind for x in a.pending()] == ["task", "presence"]
    assert not a.due()  # parking: the welcome waits for the settle grace
    clock.at += 91
    assert a.due()
    marker = next(x for x in a.pending() if x.kind == "presence")
    assert marker.text == "Will just got home." and marker.expires == clock.at - 91 + 600


def test_focus_holds_normal_pushes_urgent_and_expires(tmp_path: Path) -> None:
    clock = Clock(at(14))
    settings = DeliverySettings(tmp_path / "d.json", now=clock)
    policy = DeliveryPolicy(quiet=lambda _now: False, settings=settings)
    assert policy.decide(item(), clock.at) == "speak"
    focus = settings.set_focus("a call", 60)
    assert focus["source"] == "voice" and settings.describe()["focus"].startswith("a call for 60")
    assert policy.decide(item(), clock.at) == "hold"
    assert policy.decide(item(priority="urgent"), clock.at) == "push"
    assert not settings.auto_focus("meeting", clock.at + 7200)  # the calendar never overrides his word
    clock.at += 3601
    assert settings.focus_active() is None and policy.decide(item(), clock.at) == "speak"
    assert settings.auto_focus("meeting", clock.at + 600) and settings.focus_active()["source"] == "calendar"
    assert not settings.clear_focus(source="voice") and settings.clear_focus(source="calendar")
    assert DeliverySettings(tmp_path / "d.json", now=clock).focus_active() is None


def test_kind_preferences_override_defaults_and_validate(tmp_path: Path) -> None:
    clock = Clock(at(14))
    settings = DeliverySettings(tmp_path / "d.json", now=clock)
    presence = away_presence(tmp_path, clock)
    policy = DeliveryPolicy(quiet=lambda _now: False, presence=presence, settings=settings)
    assert policy.decide(item("watch"), clock.at) == "push"
    settings.set_preference("watch", "away", "hold")
    assert policy.decide(item("watch"), clock.at) == "hold"
    assert policy.decide(item("watch", "urgent"), clock.at) == "push"  # urgent always wins
    settings.set_preference("*", "away", "inbox")
    assert policy.decide(item("action"), clock.at) == "inbox" and policy.decide(item("watch"), clock.at) == "hold"
    presence.state, presence.since = "home", clock.at - 3600
    settings.set_preference("action", "home", "journal")
    assert policy.decide(item("action"), clock.at) == "journal" and policy.decide(item("task"), clock.at) == "speak"
    night = DeliveryPolicy(quiet=lambda _now: True, presence=presence, settings=settings)
    settings.set_preference("followup", "quiet", "speak")
    assert night.decide(item("followup"), clock.at) == "speak" and night.decide(item("task"), clock.at) == "hold"
    for bad in (("watch", "away", "speak"), ("watch", "sometimes", "hold"), ("", "home", "speak")):
        try:
            settings.set_preference(*bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert settings.clear_preference("watch", "away") and not settings.clear_preference("watch", "away")
    assert "action: home → journal" in DeliverySettings(tmp_path / "d.json", now=clock).text()


async def test_apply_decisions_parks_and_journals(tmp_path: Path) -> None:
    clock = Clock(at(14))
    settings = DeliverySettings(tmp_path / "d.json", now=clock)
    settings.set_preference("action", "home", "journal")
    settings.set_preference("milestone", "home", "inbox")
    a = Announcer(tmp_path / "a.json", now=clock, policy=DeliveryPolicy(quiet=lambda _n: False, settings=settings).decide)
    seen: list[tuple[int, str]] = []
    a.subscribe(lambda it, ev: seen.append((it.id, ev)))
    done = a.enqueue("Done: porch light on.", kind="action")
    progress = a.enqueue("Progress on task 7: tests passing", kind="milestone")
    built = a.enqueue("Task 7 is built.", kind="task")
    assert a.due()
    courier = Courier(a, settings=settings, now=clock)
    await courier.tick()  # the minute work runs on the first tick
    assert done.state == "resolved" and (done.id, "journaled") in seen  # recorded, never raised
    assert progress.mode == "inbox" and progress.unread and (progress.id, "parked") in seen
    assert [x.id for x in a.take_due()] == [built.id]


async def test_calendar_meeting_sets_an_auto_focus_that_never_overrides_voice(tmp_path: Path) -> None:
    clock = Clock(at(14))
    now = datetime.fromtimestamp(clock.at).astimezone()
    from datetime import timedelta

    class BusyCalendar:
        def __init__(self) -> None:
            self.events = [
                CalendarEvent(uid="1", summary="Team sync", start=now - timedelta(minutes=10), end=now + timedelta(minutes=20)),
                CalendarEvent(uid="2", summary="Birthday", start=now.date(), end=now.date()),
                CalendarEvent(uid="3", summary="Run club", start=now - timedelta(minutes=5), end=now + timedelta(hours=1)),
            ]

        async def list_events(self, start, end, *, calendar=None):
            return list(self.events)

    settings = DeliverySettings(tmp_path / "d.json", now=clock)
    a = Announcer(tmp_path / "a.json", now=clock)
    calendar = BusyCalendar()
    courier = Courier(a, settings=settings, calendar=calendar, focus_from_calendar=True, now=clock)
    await courier.tick()
    focus = settings.focus_active()
    assert focus is not None and focus["name"] == "meeting" and focus["source"] == "calendar"
    assert focus["until"] == (now + timedelta(minutes=20)).timestamp()  # the run club is not a meeting
    settings.set_focus("dinner", 30)
    clock.at += 301
    await courier.tick()
    assert settings.focus_active()["name"] == "dinner"  # his word stands
    settings.clear_focus()
    calendar.events = []
    clock.at += 301
    await courier.tick()
    assert settings.focus_active() is None
    off = Courier(Announcer(tmp_path / "b.json", now=clock), settings=settings, calendar=calendar, now=clock)
    calendar.events = [CalendarEvent(uid="4", summary="Interview", start=now, end=now + timedelta(hours=9))]
    clock.at += 301
    await off.tick()
    assert settings.focus_active() is None  # focus_from_calendar is off


async def test_escalation_pushes_stale_unread_once_and_expires_old(tmp_path: Path) -> None:
    class FakePusher:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str | None]] = []

        async def push(self, item: Announcement, *, level: str | None = None) -> None:
            self.calls.append((item.id, level))

    clock = Clock(at(14))
    journal = Journal(tmp_path / "journal", now=clock)
    a = Announcer(tmp_path / "a.json", now=clock)
    stale = a.enqueue("Task 7 is built.", kind="task")
    a.take_due()
    a.mark_delivered([stale.id])
    heard = a.enqueue("Revision 2 is built.", kind="task")
    a.take_due()
    a.mark_delivered([heard.id])
    a.mark_read([heard.id])
    ancient = a.enqueue("The plant light turned on.", kind="watch")
    a.take_due()
    a.mark_delivered([ancient.id])
    ancient.delivered = clock.at - 4 * 86400  # spoken four days ago, never acknowledged
    pusher = FakePusher()
    courier = Courier(a, pusher=pusher, journal=journal, escalate_after_s=4 * 3600, unread_expire_s=3 * 86400, now=clock)
    clock.at += 5 * 3600
    await courier.tick()
    assert pusher.calls == [(stale.id, "passive")] and stale.pushed is not None  # silent nudge to the phone
    assert ancient.state == "resolved" and heard.pushed is None
    assert any("expired unread" in e.text for e in journal.query(kinds=["notification"]))
    clock.at += 61
    await courier.tick()
    assert len(pusher.calls) == 1  # once


async def test_nudges_for_builds_left_waiting(tmp_path: Path) -> None:
    from tests.test_dispatch import fake_runner, make_repo
    from tests.test_tasks import make_board

    repo = make_repo(tmp_path)
    clock = Clock(at(14))
    board, announcer = make_board(repo)
    board._now = clock
    announcer._now = clock
    task = board.draft("Greeting", "x")
    task.state = "built"
    task.updated = clock.at - 3 * 86400
    board._save()
    courier = Courier(
        announcer, board=board, nudge_after_s=2 * 86400, unread_expire_s=30 * 86400, now=clock
    )
    await courier.tick()
    (nudge,) = announcer.unread()
    assert nudge.kind == "nudge" and nudge.mode == "inbox" and "waiting 3 days" in nudge.text
    assert nudge.actions == ["approve", "later"] and not announcer.due()  # never spoken on its own
    clock.at += 86400
    await courier.tick()
    assert len(announcer.unread()) == 1  # not again for three days
    clock.at += 3 * 86400
    await courier.tick()
    assert len(announcer.unread()) == 2 and "waiting 7 days" in announcer.unread()[-1].text
    assert fake_runner(repo) is not None


def test_delivery_tools_by_voice(tmp_path: Path) -> None:
    clock = Clock(at(14))
    settings = DeliverySettings(tmp_path / "d.json", now=clock)
    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", delivery=settings)
    text, is_error = engine._execute_delivery_tool("set_focus", {"name": "a call", "minutes": 45})
    assert not is_error and text.startswith("focus 'a call' for 45 minutes") and engine._instructions_stale
    text, is_error = engine._execute_delivery_tool(
        "set_notification_preference", {"kind": "watch", "when": "away", "action": "hold"}
    )
    assert not is_error and text == "watch notifications while away: hold"
    text, is_error = engine._execute_delivery_tool(
        "set_notification_preference", {"kind": "watch", "when": "away", "action": "speak"}
    )
    assert is_error and "push, hold, inbox, journal" in text
    info = json.loads(engine._execute_delivery_tool("list_notification_settings", {})[0])
    assert info["preferences"] == {"watch": {"away": "hold"}} and info["focus"].startswith("a call")
    assert engine._execute_delivery_tool("clear_focus", {})[0] == "focus cleared"
    assert engine._execute_delivery_tool("clear_focus", {})[0] == "no focus was set"
    assert engine._execute_delivery_tool("clear_notification_preference", {"kind": "watch"})[0] == "back to the default"


async def test_courier_pushes_once_and_retries_a_failure(tmp_path: Path) -> None:
    class FakePusher:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str | None]] = []
            self.fail_first = True

        async def push(self, item: Announcement, *, level: str | None = None) -> None:
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("HA down")
            self.calls.append((item.id, level))

    clock = Clock(at(12))
    presence = away_presence(tmp_path, clock)
    a = Announcer(
        tmp_path / "a.json", now=clock,
        policy=DeliveryPolicy(quiet=lambda _now: False, presence=presence).decide,
    )
    pusher = FakePusher()
    courier = Courier(a, presence=presence, pusher=pusher, now=clock)
    built = a.enqueue("Task 7 is built.", kind="task")
    await courier.tick()
    assert pusher.calls == [] and built.pushed is None  # the first try failed
    clock.at += 30
    await courier.tick()
    assert pusher.calls == []  # not before the retry delay
    clock.at += 31
    await courier.tick()
    assert pusher.calls == [(built.id, None)] and built.pushed is not None
    await courier.tick()
    assert len(pusher.calls) == 1  # never twice
