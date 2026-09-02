"""Phase 4: revisions resume the same agent, builds survive restarts,
quiet deaths are resumed once, hung starts fail honestly."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from assistant.announce import Announcer
from assistant.dispatch import Dispatcher, pid_alive
from assistant.tasks import TaskBoard
from tests.test_dispatch import FAKE_CLAUDE, make_repo


def runner(repo: Path, mode: str = "", **kw) -> Dispatcher:
    return Dispatcher(repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=30.0, **kw)


def env_mode(monkeypatch, tmp_path: Path, mode: str) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(tmp_path / "fake-state.marker"))


async def settle(board: TaskBoard, task_id: int, timeout: float = 25.0) -> None:
    async with asyncio.timeout(timeout):
        while board.get(task_id).state in ("building", "revising"):
            await asyncio.sleep(0.1)


def test_pid_alive_knows_live_and_dead_processes() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(4_000_000)
    assert not pid_alive(0)


async def test_revision_resumes_the_same_agent_and_records_feedback(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "")
    repo = make_repo(tmp_path)
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=runner(repo), announcer=announcer, resume_delay_s=0)
    task = board.draft("Volume ducking", "duck the music when spoken to")
    await board.start(task.id)
    await settle(board, task.id)
    assert board.get(task.id).state == "built"
    first = board.get(task.id).iterations[0]
    assert first.session_id == "sess-fake" and "(session " in first.summary  # pre-assigned id was passed

    await board.revise(task.id, "it ducks too much, keep the music at half volume")
    assert board.get(task.id).state == "revising"
    await settle(board, task.id)
    task = board.get(task.id)
    assert task.state == "built"
    second = task.iterations[1]
    assert second.kind == "revise" and second.status == "done"
    assert "resumed the earlier session" in second.summary  # --resume was used
    assert second.feedback.startswith("it ducks too much")
    doc = (Path(task.worktree) / "docs" / "tasks" / f"{task.slug}.md").read_text(encoding="utf-8")
    assert "## Revision 2" in doc and "half volume" in doc
    texts = [a.text for a in announcer.pending()]
    assert any(t.startswith("Revision 2 of task 1, 'Volume ducking', is built") for t in texts)


async def test_a_build_survives_an_app_restart(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "slow")
    repo = make_repo(tmp_path)
    first_app = TaskBoard(repo, runner=runner(repo), resume_delay_s=0)
    task = first_app.draft("Long build", "takes a while")
    await first_app.start(task.id)
    await asyncio.sleep(0.8)  # the agent is now running detached; "restart" the app
    for bg in first_app._bg:
        bg.cancel()
    await asyncio.sleep(0.1)
    assert first_app.get(task.id).state == "building"

    second_app = TaskBoard(repo, runner=runner(repo), resume_delay_s=0)
    assert second_app.get(task.id).state == "building"  # no longer marked interrupted on load
    await second_app.startup_maintenance()
    await settle(second_app, task.id)
    task = second_app.get(task.id)
    assert task.state == "built" and "did the task" in task.current.summary
    assert any(h["event"] == "restart" for h in task.history)


async def test_a_quiet_death_is_resumed_once(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "no-result-once")
    repo = make_repo(tmp_path)
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=runner(repo), announcer=announcer, resume_delay_s=0)
    task = board.draft("Flaky build", "dies once")
    await board.start(task.id)
    await settle(board, task.id)
    task = board.get(task.id)
    assert task.state == "built"
    kinds = [it.kind for it in task.iterations]
    assert kinds == ["build", "retry"]
    assert task.iterations[0].status == "failed" and "without a result" in task.iterations[0].summary
    assert "resumed the earlier session" in task.iterations[1].summary
    assert any(h["event"] == "resuming" for h in task.history)
    assert not any("stopped without finishing" in a.text for a in announcer.pending())


async def test_a_missing_session_falls_back_to_a_fresh_build(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "no-session-once")
    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=runner(repo), resume_delay_s=0)
    task = board.draft("Lost session", "x")
    await board.start(task.id)
    await settle(board, task.id)
    await board.revise(task.id, "make it blue")
    await settle(board, task.id)
    task = board.get(task.id)
    assert task.state == "built"
    assert [it.kind for it in task.iterations] == ["build", "revise", "build"]
    assert task.iterations[1].status == "failed" and "never started a session" in task.iterations[1].summary
    assert any(h["event"] == "session missing" for h in task.history)


async def test_a_cli_that_never_comes_up_fails_honestly(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "hang-no-init")
    repo = make_repo(tmp_path)
    agent = runner(repo)
    run = await agent.run_agent(
        cwd=repo, prompt="x", log_path=repo / "logs" / "hang.log", init_timeout_s=1.0
    )
    assert run.status == "failed" and "didn't start" in run.summary
    await asyncio.sleep(0.5)
    assert not pid_alive(run.pid)  # killed, not left running
