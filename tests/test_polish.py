"""Phase 5 polish: announcement history by voice, ranked task search, and the
approval count in her wake-time status line."""

from __future__ import annotations

import json
from pathlib import Path

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.tasks import TaskBoard
from tests.test_dispatch import fake_runner, make_repo


def test_announcement_history_by_window(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", announcer=announcer
    )
    text, is_error = engine._execute_system_tool("announcement_history", {"since": "today"})
    assert not is_error and "haven't announced" in text

    item = announcer.enqueue("Task 7 is built.", kind="task")
    announcer.take_due()
    announcer.mark_delivered([item.id])
    text, is_error = engine._execute_system_tool("announcement_history", {"since": "today"})
    assert not is_error
    rows = json.loads(text)
    assert rows[0]["said"] == "Task 7 is built." and rows[0]["kind"] == "task" and rows[0]["when"]
    text, _ = engine._execute_system_tool("announcement_history", {"since": "2"})
    assert json.loads(text)[0]["said"] == "Task 7 is built."
    text, is_error = engine._execute_system_tool("announcement_history", {"since": "whenever"})
    assert is_error


def test_search_ranks_title_matches_first(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=fake_runner(repo))
    body_hit = board.draft("Startup greeting", "mention the calendar in the greeting")
    title_hit = board.draft("Calendar reminders", "remind me of events")
    rows = json.loads(board.search("calendar"))
    assert [r["id"] for r in rows] == [title_hit.id, body_hit.id]
    assert board.search("calendar reminders greeting") == "no past tasks match that"


def test_status_line_counts_tasks_awaiting_approval(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=fake_runner(repo))
    a = board.draft("One", "x")
    b = board.draft("Two", "y")
    a.state = "built"
    b.state = "built"
    board._save()
    line = board.status_line()
    assert line.startswith("2 awaiting your approval; ")
    assert "task 1 'One'" in line and "task 2 'Two'" in line
