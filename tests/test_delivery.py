"""Delivery: the policy table (presence x priority x quiet x settle), the
Announcer honoring it, and the Courier (arrival marker, pushes with retry)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from assistant.announce import Announcement, Announcer
from assistant.delivery import Courier, DeliveryPolicy
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
