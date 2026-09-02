"""Two-way async: a coding agent stops on a QUESTION:, the task parks in
needs_input with a question notification, and the owner's answer resumes
the same agent. Throwaway repo + fake claude, no Max usage."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.dispatch import DispatchError
from tests.test_dispatch import make_repo
from tests.test_phase4 import env_mode, settle
from tests.test_tasks import make_board


async def test_agent_question_parks_the_task_and_the_answer_resumes_it(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "ask-once")
    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)
    task = board.draft("Startup greeting", "greet the owner by name")
    await board.start(task.id)
    await settle(board, task.id)

    task = board.get(task.id)
    assert task.state == "needs_input"
    assert task.question.startswith("should the greeting use his first name")
    assert task.current.status == "done" and task.current.kind == "build"
    (asked,) = [a for a in announcer.pending() if a.kind == "question"]
    assert "needs your call: should the greeting" in asked.text and "answer task 1" in asked.text
    assert asked.context == {"task_id": task.id} and asked.actions == ["answer"]
    assert board.status_line().startswith("1 waiting on your answer")
    assert "needs your answer" in board.status_line()
    with pytest.raises(DispatchError, match="needs one of"):
        await board.approve(task.id)  # nothing is built yet
    with pytest.raises(DispatchError, match="needs one of"):
        await board.switch_build(task.id)
    with pytest.raises(DispatchError, match="needs the owner"):
        await board.answer(task.id, "   ")

    await board.answer(task.id, "first name")
    assert board.get(task.id).state == "revising"
    assert asked.state == "resolved"  # the question is off the unread list the moment he answers
    await settle(board, task.id)
    task = board.get(task.id)
    assert task.state == "built"
    assert [it.kind for it in task.iterations] == ["build", "answer"]
    assert "resumed the earlier session" in task.iterations[1].summary  # --resume, same agent
    assert task.iterations[1].question.startswith("should the greeting") and task.iterations[1].feedback == "first name"
    doc = (Path(task.worktree) / "docs" / "tasks" / f"{task.slug}.md").read_text(encoding="utf-8")
    assert "## Answer 2" in doc and "A: first name" in doc and "Q: should the greeting" in doc
    assert any("Task 1, 'Startup greeting', is built" in a.text for a in announcer.pending())
    assert board.status_line().startswith("1 awaiting your approval")
    with pytest.raises(DispatchError, match="needs one of"):
        await board.answer(task.id, "again")  # nothing is being asked now


async def test_feedback_can_stand_in_for_an_answer(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "ask-once")
    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)
    task = board.draft("Greeting", "greet")
    await board.start(task.id)
    await settle(board, task.id)
    assert board.get(task.id).state == "needs_input"
    await board.revise(task.id, "use the first name, and keep it to three words")
    await settle(board, task.id)
    task = board.get(task.id)
    assert task.state == "built" and task.iterations[1].kind == "revise"
    assert not [a for a in announcer.unread() if a.kind == "question"]  # superseded by the revision
