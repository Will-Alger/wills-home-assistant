"""The morning briefing includes what she told him that he hasn't heard."""

from __future__ import annotations

from pathlib import Path

from assistant.announce import Announcer
from assistant.briefing import compose_briefing
from assistant.calendar.fake import FakeCalendar
from assistant.scheduler import Scheduler
from assistant.tasks import Iteration, TaskBoard
from tests.test_dispatch import fake_runner, make_repo


async def test_briefing_includes_the_unread_digest(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=fake_runner(repo))
    built = board.draft("Porch light", "x")
    built.state = "built"
    stuck = board.draft("Greeting", "y")
    stuck.state = "needs_input"
    stuck.iterations.append(Iteration(n=1, status="done", question="first or full name?"))
    board._save()
    announcer = Announcer(tmp_path / "a.json")
    for n in range(4):
        item = announcer.enqueue(f"Notification number {n}.", kind="watch", ref=f"watch:{n}:1")
        announcer.take_due()
        announcer.mark_delivered([item.id])
    scheduler = Scheduler(tmp_path / "s.json", announcer=announcer)

    text = await compose_briefing(FakeCalendar(), board, scheduler, announcer, "Will")()
    assert text.startswith("Good morning, Will.")
    assert "Awaiting your approval: task 1 Porch light." in text
    assert "Task 2 needs your answer: first or full name?" in text
    assert "Unread from me: Notification number 0.; Notification number 1.; Notification number 2.; and 1 more." in text
    assert len(announcer.unread()) == 4  # the briefing itself marks nothing read

    quiet = await compose_briefing(None, None, None, Announcer(tmp_path / "b.json"), "Will")()
    assert quiet.count(".") >= 2 and "Unread" not in quiet
