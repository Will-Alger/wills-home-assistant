"""The journal: day files, ranked queries, torn lines, pruning — and the
writers that feed it (engine tools, scheduler, watcher, board, announcer)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine, SessionStats
from assistant.events import EventWatcher, WatchStore
from assistant.home.fake import FakeHome
from assistant.journal import Journal
from assistant.scheduler import Scheduler
from assistant.tasks import TaskBoard
from tests.fake_realtime import FakeClient
from tests.test_dispatch import fake_runner, make_repo
from tests.test_events import event

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 9, day, hour, minute, tzinfo=LOCAL).timestamp()


def test_write_rotates_by_local_day_and_query_ranks(tmp_path: Path) -> None:
    clock = Clock(at(1, 23, 50))
    journal = Journal(tmp_path / "journal", now=clock)
    journal.write("tool", "set_lights: porch light on", source="voice", data={"ok": True})
    clock.at = at(2, 0, 10)  # past midnight → a new day file
    journal.write("schedule", "Done: porch light off", source="action")
    clock.at = at(2, 8, 0)
    journal.write("watch", "the front door opened", source="binary_sensor.front_door")
    clock.at = at(2, 8, 1)
    journal.write("tool", "media_control: paused", source="voice")
    assert sorted(p.name for p in (tmp_path / "journal").glob("*.jsonl")) == [
        "2026-09-01.jsonl", "2026-09-02.jsonl",
    ]

    clock.at = at(2, 9, 0)
    hits = journal.query("porch light")
    assert [h.text for h in hits] == ["Done: porch light off", "set_lights: porch light on"]  # all words, newest first
    assert journal.query("porch")[0].kind == "schedule"
    assert journal.query("door porch")[0].text == "the front door opened"  # partial hits: newest first
    assert journal.query("thermostat") == []
    assert [h.kind for h in journal.query(kinds=["tool"])] == ["tool", "tool"]
    assert len(journal.query(limit=2)) == 2 and journal.query(limit=2)[0].text == "media_control: paused"
    assert [h.text for h in journal.query(since=at(2, 0, 0))] == [
        "media_control: paused", "the front door opened", "Done: porch light off",
    ]
    assert journal.query(since=at(2, 0, 0), until=at(2, 1, 0))[0].text == "Done: porch light off"
    spoken = Journal.spoken(hits[0])
    assert spoken["kind"] == "schedule" and spoken["when"].startswith("Wed") and spoken["source"] == "action"


def test_torn_line_is_skipped_and_prune_deletes_old_days(tmp_path: Path) -> None:
    clock = Clock(at(2, 12))
    journal = Journal(tmp_path / "journal", now=clock, keep_days=3)
    journal.write("tool", "first")
    with (tmp_path / "journal" / "2026-09-02.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"ts": 1, "kind": "tool", "te')  # crashed mid-write
    assert [e.text for e in journal.query()] == ["first"]
    (tmp_path / "journal" / "2026-08-20.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "journal" / "2026-08-31.jsonl").write_text("{}\n", encoding="utf-8")
    assert journal.prune() == 1  # only the one older than 3 days
    assert not (tmp_path / "journal" / "2026-08-20.jsonl").exists()
    assert (tmp_path / "journal" / "2026-08-31.jsonl").exists()
    assert journal.prune() == 0  # once a day is enough
    assert journal.write("tool", "   ") is None


async def test_engine_journals_acting_tools_not_lookups(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", journal=journal
    )
    client = FakeClient()

    def call(name: str, args: dict) -> SimpleNamespace:
        item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id="c1")
        return SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))

    stats = SessionStats()
    await engine._handle_response_done(
        client.connection, call("set_lights", {"changes": [{"target": "Hallway", "turn": "on"}]}), stats
    )
    await engine._handle_response_done(client.connection, call("get_lights", {}), stats)
    rows = journal.query(kinds=["tool"])
    assert len(rows) == 1 and rows[0].text.startswith("set_lights:") and rows[0].data["ok"]
    assert rows[0].data["args"]["changes"][0]["target"] == "Hallway"

    text, is_error = engine._execute_journal_tool("journal_search", {"query": "hallway"})
    assert not is_error and json.loads(text)[0]["kind"] == "tool"
    text, is_error = engine._execute_journal_tool("journal_search", {"query": "thermostat"})
    assert not is_error and "nothing in the journal" in text
    text, is_error = engine._execute_journal_tool("journal_search", {"since": "whenever"})
    assert is_error
    config = await engine._session_config(None)
    assert "journal_search" in {t["name"] for t in config["tools"]}


async def test_scheduler_watcher_board_and_announcer_write_rows(tmp_path: Path) -> None:
    clock = Clock(at(2, 17, 55))
    journal = Journal(tmp_path / "journal", now=clock)
    announcer = Announcer(tmp_path / "a.json", now=clock)
    announcer.subscribe(
        lambda item, ev: journal.write("notification", f"{ev}: {item.text}", source=item.kind)
    )
    sched = Scheduler(tmp_path / "schedule.json", announcer=announcer, now=clock, journal=journal)
    sched.schedule(kind="reminder", label="call mom", in_seconds=60, message="call your mom")
    clock.at += 61
    await sched.tick()

    store = WatchStore(tmp_path / "watches.json", now=clock)
    store.add(entity_id="binary_sensor.front_door", message="the {entity} opened", to_state="on", once=False)
    watcher = EventWatcher("http://ha", "tok", store, announcer, journal=journal)
    watcher.handle(event("binary_sensor.front_door", "off", "on"))
    watcher.handle(event("binary_sensor.front_door", "off", "on"))  # flapping: one row, count 2
    (hit,) = [a for a in announcer.pending() if a.kind == "watch"]
    assert hit.count == 2 and hit.group == "watch:1"

    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=fake_runner(repo), journal=journal)
    board.draft("Porch light", "make it warm")

    kinds = {e.kind for e in journal.query(limit=40)}
    assert {"schedule", "watch", "task", "notification"} <= kinds
    assert any(e.text.startswith("task 1 'Porch light' drafted") for e in journal.query(kinds=["task"]))
    assert len(journal.query(kinds=["watch"])) == 2  # every hit is journaled even when grouped
