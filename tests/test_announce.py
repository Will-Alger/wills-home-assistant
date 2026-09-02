"""The announcement queue — persistence, quiet hours, backoff, inbox. No audio."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from assistant.announce import Announcer, in_quiet_hours, parse_quiet_hours, write_inbox

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def noon() -> float:
    return datetime(2026, 9, 1, 12, 0, tzinfo=LOCAL).timestamp()


def test_enqueue_persists_and_reloads(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "announcements.json", now=clock)
    item = a.enqueue("The calendar build is done.", kind="task", ref="job:1:done")
    assert item.id == 1 and item.open

    again = Announcer(tmp_path / "announcements.json", now=clock)
    assert [x.text for x in again.pending()] == ["The calendar build is done."]
    assert again.due()


def test_same_ref_is_not_queued_twice_while_open(tmp_path: Path) -> None:
    a = Announcer(tmp_path / "a.json", now=Clock(noon()))
    first = a.enqueue("built", ref="job:1:done")
    dup = a.enqueue("built again", ref="job:1:done")
    assert dup.id == first.id and len(a.pending()) == 1
    a.mark_delivered([first.id])
    assert a.enqueue("built again", ref="job:1:done").id != first.id  # delivered → new one ok


def test_take_due_schedules_backoff_then_gives_up(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock, max_attempts=2)
    a.enqueue("hello")
    (first,) = a.take_due()
    assert first.attempts == 1 and first.next_attempt == clock.at + 60.0
    assert a.take_due() == []  # not due again until the backoff passes
    clock.at += 61
    (second,) = a.take_due()
    assert second.attempts == 2
    clock.at += 301
    assert a.take_due() == []  # third attempt exceeds max → cancelled, never spoken
    assert a.pending() == []


def test_mark_delivered_clears_it_and_records_history(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    item = a.enqueue("done")
    a.take_due()
    a.mark_delivered([item.id])
    assert not a.due() and a.pending() == []
    assert [h["text"] for h in a.history(since=noon() - 1)] == ["done"]


def test_quiet_hours_hold_normal_but_not_urgent(tmp_path: Path) -> None:
    window = parse_quiet_hours("23:00-08:00")
    assert in_quiet_hours(window, datetime(2026, 9, 1, 23, 30, tzinfo=LOCAL))
    assert in_quiet_hours(window, datetime(2026, 9, 1, 2, 0, tzinfo=LOCAL))
    assert not in_quiet_hours(window, datetime(2026, 9, 1, 8, 0, tzinfo=LOCAL))
    assert parse_quiet_hours("") is None

    night = Clock(datetime(2026, 9, 1, 23, 30, tzinfo=LOCAL).timestamp())
    a = Announcer(tmp_path / "a.json", quiet_hours="23:00-08:00", now=night)
    a.enqueue("the build finished")
    assert not a.due()  # waits for morning
    a.enqueue("staged build crashed", priority="urgent", ref="sys:rollback")
    assert a.due()
    (urgent,) = a.take_due()
    assert urgent.text == "staged build crashed"
    a.mark_delivered([urgent.id])
    night.at = datetime(2026, 9, 2, 8, 5, tzinfo=LOCAL).timestamp()
    assert [x.text for x in a.take_due()] == ["the build finished"]


def test_inbox_drops_from_other_processes_are_swept(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    write_inbox(tmp_path, "rolled back to main", kind="system", ref="sys:rb", priority="urgent")
    inbox = tmp_path / "announcements" / "inbox"
    (inbox / "partial.json").write_text("{not json", encoding="utf-8")  # half-written
    assert a.sweep_inbox() == 1
    assert [x.text for x in a.pending()] == ["rolled back to main"]
    assert a.pending()[0].priority == "urgent"
    assert (inbox / "partial.json").exists()  # left for a later sweep, never crashes
    assert not list(inbox.glob("*.json.tmp"))


def test_expiry_and_cancel(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    a.enqueue("stale soon", expires_in_s=10)
    a.enqueue("job one", ref="job:1:milestone:1")
    a.enqueue("job one done", ref="job:1:done")
    assert a.cancel("job:1:") == 2
    clock.at += 11
    assert a.take_due() == [] and a.pending() == []


def test_due_is_cheap_and_cached(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    assert not a.due()
    a.enqueue("x")
    assert a.due()  # enqueue invalidates the cache
    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert data["next_id"] == 2 and data["items"][0]["text"] == "x"


# ── milestone 11: notifications ────────────────────────────────────────────


def test_read_state_lifecycle(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    seen: list[tuple[int, str]] = []
    a.subscribe(lambda item, event: seen.append((item.id, event)))
    item = a.enqueue("Task 7 is built.", kind="task", ref="task:7:1:built")
    a.take_due()
    a.mark_delivered([item.id])
    assert item.state == "spoken" and item.unread and not item.open  # said, not proven heard
    clock.at += 5
    held = a.enqueue("the porch light came on", kind="watch")  # never spoken in this test
    assert [x.id for x in a.unread()] == [item.id, held.id]
    summary = a.unread_summary()
    assert summary.startswith("1 unread since") and "[1] task: Task 7 is built." in summary
    # (the held one is about to be spoken anyway, so it is left out of the teaser)

    assert a.mark_read([item.id]) == 1 and item.state == "read" and not item.unread
    assert a.mark_read([item.id]) == 0
    assert a.mark_read([held.id]) == 1 and not held.open and a.pending() == []  # listed = heard
    assert a.mark_unread([held.id]) == 1 and held.open and a.due()
    again = Announcer(tmp_path / "a.json", now=clock)
    assert again.get(item.id).read is not None and again.get(held.id).state == "pending"
    assert (item.id, "spoken") in seen and (item.id, "read") in seen and (held.id, "unread") in seen


def test_explicit_group_bumps_count_and_ephemeral_kinds_read_when_spoken(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock)
    first = a.enqueue("the plant light turned on", kind="watch", ref="watch:3:1", group="watch:3")
    clock.at += 30
    same = a.enqueue("the plant light turned off", kind="watch", ref="watch:3:2", group="watch:3")
    assert same.id == first.id and first.count == 2 and first.text == "the plant light turned off"
    assert len(a.pending()) == 1
    clock.at += 601
    assert a.enqueue("again", kind="watch", ref="watch:3:3", group="watch:3").id != first.id  # window passed
    m1 = a.enqueue("Progress on task 7: plan settled", ref="task:7:1:milestone:1")
    m2 = a.enqueue("Progress on task 7: tests passing", ref="task:7:1:milestone:2")
    assert m1.id != m2.id  # no group key → never merged, shared prefix or not
    timer = a.enqueue("Your timer is up.", kind="timer")
    a.take_due()
    a.mark_delivered([timer.id])
    assert timer.state == "read"  # ephemeral: nobody wants it in tomorrow's recap
    assert a.get(m1.id).state == "pending"  # handed out, not yet spoken


def test_resolve_inbox_and_parking(tmp_path: Path) -> None:
    clock = Clock(noon())
    a = Announcer(tmp_path / "a.json", now=clock, max_attempts=1)
    nudge = a.enqueue("Task 7 has been waiting two days.", kind="nudge", mode="inbox", ref="task:7:nudge")
    assert not a.due() and nudge.unread and nudge.state == "inbox"
    built = a.enqueue(
        "Task 7 is built.", ref="task:7:1:built", context={"task_id": 7}, actions=["approve", "later"]
    )
    assert Announcer(tmp_path / "a.json", now=clock).get(built.id).actions == ["approve", "later"]
    assert a.resolve("task:7:") == 2 and built.state == "resolved" and a.unread() == []
    assert a.resolve("task:7:") == 0 and a.resolve("") == 0

    stubborn = a.enqueue("hello")
    a.take_due()  # attempt 1 = the cap
    clock.at += 61
    assert a.take_due() == [] and stubborn.mode == "inbox" and stubborn.unread  # parked, never lost
    assert a.pending() == [] and [x.id for x in a.unread()] == [stubborn.id]
    assert a.items()[-1]["state"] == "inbox"


def test_legacy_rows_load_and_unread_survive_trimming(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    legacy = {
        "id": 1, "text": "old row", "kind": "task", "ref": "", "priority": "normal",
        "created": noon(), "expires": None, "attempts": 1, "next_attempt": 0,
        "delivered": noon(), "cancelled": False,
    }
    path.write_text(json.dumps({"next_id": 2, "items": [legacy]}), encoding="utf-8")
    clock = Clock(noon() + 100)
    a = Announcer(path, now=clock, keep=3)
    old = a.get(1)
    assert old.state == "spoken" and old.mode == "speak" and old.unread and old.context == {}
    for i in range(3):
        item = a.enqueue(f"spoken {i}")
        a.take_due()
        a.mark_delivered([item.id])
    done = a.enqueue("read one")
    a.take_due()
    a.mark_delivered([done.id])
    a.mark_read([done.id])
    a.enqueue("one more")
    remaining = {x.id for x in a._items}
    assert done.id not in remaining  # the only finished row was trimmed
    assert 1 in remaining and len(a.unread()) == 5  # every unread row kept its id
