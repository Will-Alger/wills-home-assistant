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
