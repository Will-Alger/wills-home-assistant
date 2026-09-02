"""Phase 3: switching builds, approving from a staged process, rollback
absorption, startup housekeeping, and the ALEXA_HOME split."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from assistant.announce import Announcer
from assistant.dispatch import DispatchError
from assistant.tasks import TaskBoard, pointer_path, read_pointer
from tests.test_dispatch import fake_runner, make_repo


class SyncingRunner(type(fake_runner(Path(".")))):  # type: ignore[misc]
    """The fake runner with uv_sync recorded instead of executed."""

    synced: list[Path]

    async def uv_sync(self, cwd: Path, uv_exe: str = ""):
        if not hasattr(self, "synced"):
            self.synced = []
        self.synced.append(Path(cwd))
        return True, ""


def board_with_built_task(repo: Path, **kw) -> tuple[TaskBoard, Announcer, SyncingRunner]:
    import sys

    from tests.test_dispatch import FAKE_CLAUDE

    runner = SyncingRunner(repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=20.0)
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=runner, announcer=announcer, **kw)
    return board, announcer, runner


async def build(board: TaskBoard, title: str = "Volume ducking") -> int:
    task = board.draft(title, "duck the music when spoken to")
    await board.start(task.id)
    async with asyncio.timeout(15):
        while board.get(task.id).state == "building":
            await asyncio.sleep(0.1)
    assert board.get(task.id).state == "built"
    return task.id


async def test_switch_build_writes_the_pointer_and_asks_for_a_restart(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _, runner = board_with_built_task(repo)
    tid = await build(board)
    assert not board.restart_requested

    text = await board.switch_build(str(tid))
    assert "switching to task 1's build" in text and board.restart_requested
    pointer = read_pointer(repo)
    assert pointer and pointer["task_id"] == tid and pointer["worktree"] == board.get(tid).worktree
    assert board.get(tid).state == "staged"
    assert runner.synced == [Path(board.get(tid).worktree)]  # branch deps synced first
    assert "task 1" in board.status_line() and "staged" in board.status_line()

    board.restart_requested = False
    assert "already running main" not in await board.switch_build("main")
    assert read_pointer(repo) is None and board.get(tid).state == "built"  # reversible, nothing lost
    assert board.restart_requested
    board.restart_requested = False
    assert "already running main" in await board.switch_build("main")
    assert not board.restart_requested


async def test_switching_to_a_second_task_unstages_the_first(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _, _ = board_with_built_task(repo)
    first = await build(board, "First")
    second = await build(board, "Second")
    await board.switch_build(first)
    await board.switch_build(second)
    assert board.get(first).state == "built" and board.get(second).state == "staged"
    assert read_pointer(repo)["task_id"] == second
    assert any(h["event"] == "unstaged" for h in board.get(first).history)

    with pytest.raises(DispatchError, match="needs one of"):
        await board.switch_build(999) if False else await board.switch_build(board.draft("Not built", "x").id)


async def test_approve_from_the_staged_process_syncs_main_and_restarts(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _, runner = board_with_built_task(repo)
    tid = await build(board)
    await board.switch_build(tid)
    # simulate the agent's committed work
    (Path(board.get(tid).worktree) / "DUCK.md").write_text("duck", encoding="utf-8")
    await runner.commit_all(Path(board.get(tid).worktree), "duck")

    # now pretend we ARE the staged process
    staged_board = TaskBoard(repo, runner=runner, staged_task_id=tid)
    result = await staged_board.approve(tid)
    assert "merged" in result and "restarting onto main" in result
    assert staged_board.restart_requested
    assert read_pointer(repo) is None  # back to main on the next launch
    assert runner.synced[-1] == repo  # main's deps synced BEFORE the restart
    assert (repo / "DUCK.md").exists()
    assert staged_board.get(tid).state == "merged" and staged_board.get(tid).cleanup_pending


async def test_abandon_while_staged_returns_to_main(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _, runner = board_with_built_task(repo)
    tid = await build(board)
    await board.switch_build(tid)
    staged_board = TaskBoard(repo, runner=runner, staged_task_id=tid)
    note = staged_board.abandon(tid)
    assert "restarting onto main" in note and staged_board.restart_requested
    assert read_pointer(repo) is None


async def test_startup_absorbs_a_rollback_and_confirms_a_staged_start(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, announcer, runner = board_with_built_task(repo)
    tid = await build(board)
    await board.switch_build(tid)

    # the watchdog rolled back: pointer renamed, task still says staged
    pointer_path(repo).replace(pointer_path(repo).with_name("active_checkout.failed.json"))
    fresh = TaskBoard(repo, runner=runner, announcer=announcer)
    await fresh.startup_maintenance()
    task = fresh.get(tid)
    assert task.state == "built" and "rolled back" in task.last_error
    assert not pointer_path(repo).with_name("active_checkout.failed.json").exists()

    # a process that actually started from the staged build says so, once
    await fresh.switch_build(tid)
    staged = TaskBoard(repo, runner=runner, announcer=announcer, staged_task_id=tid)
    await staged.startup_maintenance()
    texts = [a.text for a in announcer.pending()]
    assert any("restarted into the staged build of task 1" in t for t in texts)
    assert staged.get(tid).state == "staged"
    assert "STAGED build of task 1" in staged.staged_paragraph()
    assert TaskBoard(repo, runner=runner).staged_paragraph() == ""  # main process: nothing


async def test_startup_cleans_up_merged_worktrees(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _, runner = board_with_built_task(repo)
    tid = await build(board)
    (Path(board.get(tid).worktree) / "X.md").write_text("x", encoding="utf-8")
    await runner.commit_all(Path(board.get(tid).worktree), "x")
    await board.approve(tid)
    worktree = Path(board.get(tid).worktree)
    assert worktree.exists() and board.get(tid).cleanup_pending
    await TaskBoard(repo, runner=runner).startup_maintenance()
    assert not worktree.exists()
    assert not TaskBoard(repo, runner=runner).get(tid).cleanup_pending


def test_home_dir_follows_alexa_home(monkeypatch, tmp_path: Path) -> None:
    from assistant.config import code_root, home_dir

    monkeypatch.delenv("ALEXA_HOME", raising=False)
    assert home_dir() == code_root()
    monkeypatch.setenv("ALEXA_HOME", str(tmp_path))
    assert home_dir() == tmp_path.resolve()
    assert code_root() != tmp_path.resolve()  # code stays where it was imported from
    (tmp_path / "data").mkdir()
    assert json.loads('{"ok": true}')["ok"]
