"""The runner: a throwaway git repo + a fake claude CLI. No Max usage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from assistant.dispatch import Dispatcher, DispatchError, load_extra_routines

FAKE_CLAUDE = Path(__file__).with_name("fake_claude.py")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    run("init", "-b", "main")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "README.md").write_text("hi", encoding="utf-8")
    run("add", "-A")
    run("commit", "-m", "init")
    return repo


def push_cloud_branch_to_origin(repo: Path, tmp_path: Path, branch: str = "alexa/cloud-feature") -> None:
    """Simulate a cloud session: a bare origin holding a pushed feature branch."""
    origin = tmp_path / "origin.git"
    run = lambda *a, **kw: subprocess.run(["git", *a], check=True, capture_output=True, **kw)
    run("init", "--bare", str(origin))
    run("remote", "add", "origin", str(origin), cwd=repo)
    run("push", "origin", "main", cwd=repo)
    run("checkout", "-b", branch, cwd=repo)
    (repo / "CLOUD.md").write_text("from the cloud", encoding="utf-8")
    run("add", "-A", cwd=repo)
    run("commit", "-m", "cloud work", cwd=repo)
    run("push", "origin", branch, cwd=repo)
    run("checkout", "main", cwd=repo)
    run("branch", "-D", branch, cwd=repo)  # only the remote has it now


def fake_runner(repo: Path, **kw) -> Dispatcher:
    return Dispatcher(repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=20.0, **kw)


async def test_run_agent_streams_the_session_into_an_agent_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    runner = fake_runner(repo, model="opus")
    assert "--model opus" in runner._claude_cmd  # Opus rides the command line
    worktree, branch = await runner.create_worktree("1-greeting")
    assert worktree.exists() and branch == "alexa/1-greeting"

    kinds: list[str] = []
    run = await runner.run_agent(
        cwd=worktree,
        prompt="add a greeting file",
        log_path=repo / "logs" / "tasks" / "1-greeting-1.log",
        on_update=lambda r, kind: kinds.append(kind),
    )
    assert run.status == "done"
    assert "did the task" in run.summary
    assert run.session_id == "sess-fake" and run.model == "claude-fake-1"
    assert run.milestones == ["tests are passing."]
    assert "Moving on to the commit" in run.progress[-1]
    assert kinds == ["init", "progress", "milestone", "result"]
    assert (repo / "logs" / "tasks" / "1-greeting-1.log").exists()


async def test_agent_without_a_result_is_reported_honestly(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    runner = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" -c "import sys; sys.exit(3)"', timeout_s=20.0
    )
    run = await runner.run_agent(cwd=repo, prompt="doomed", log_path=repo / "logs" / "x.log")
    assert run.status == "failed" and "exit 3" in run.summary


async def test_merge_gates_and_success(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    runner = fake_runner(repo)
    worktree, branch = await runner.create_worktree("2-greeting")
    (worktree / "GREETING.md").write_text("hello", encoding="utf-8")
    assert await runner.commit_all(worktree, "add greeting")

    (repo / "README.md").write_text("dirty", encoding="utf-8")  # tracked file modified
    ok, message = await runner.merge_branch(merge_ref=branch, gates_dir=worktree, title="greeting")
    assert not ok and "uncommitted changes" in message
    (repo / "README.md").write_text("hi", encoding="utf-8")

    ok, message = await runner.merge_branch(merge_ref=branch, gates_dir=worktree, title="greeting")
    assert ok, message
    assert (repo / "GREETING.md").exists()  # landed on main
    assert await runner.remove_worktree(worktree)
    assert not worktree.exists()


async def test_cloud_branch_is_fetched_gated_and_merged(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    push_cloud_branch_to_origin(repo, tmp_path)
    runner = fake_runner(repo)
    merge_ref, checkout = await runner.fetch_remote_branch("alexa/cloud-feature", "3-cloud")
    assert merge_ref == "origin/alexa/cloud-feature" and checkout.exists()
    ok, message = await runner.merge_branch(merge_ref=merge_ref, gates_dir=checkout, title="cloud")
    assert ok, message
    assert (repo / "CLOUD.md").exists()
    await runner.remove_worktree(checkout)
    with pytest.raises(DispatchError, match="could not fetch"):
        await runner.fetch_remote_branch("alexa/nope", "4-nope")


def test_repo_routing_and_registry(tmp_path: Path) -> None:
    runner = Dispatcher(
        tmp_path,
        routine_id="trig_self",
        routine_token="tok_self",
        extra_routines={"my-side-project": {"routine_id": "trig_side", "token": "tok_side"}},
    )
    assert runner.cloud_enabled and runner.own_routine == ("trig_self", "tok_self")
    assert runner.extra_repo_names() == ["my-side-project"]
    assert runner.resolve_repo("my side project")["routine_id"] == "trig_side"  # spoken form
    with pytest.raises(DispatchError, match="my-side-project"):
        runner.resolve_repo("nonexistent-repo")

    assert load_extra_routines(tmp_path) == {}
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "routines.json").write_text(
        '{"good-repo": {"routine_id": "trig_a", "token": "tok_a"}, "missing-token": {"routine_id": "trig_b"}}',
        encoding="utf-8",
    )
    assert list(load_extra_routines(tmp_path)) == ["good-repo"]


async def test_cloud_refresh_returns_the_status_line(tmp_path: Path) -> None:
    runner = Dispatcher(
        tmp_path,
        cloud_status_cmd=f'"{sys.executable}" -c "print(\'chatter\'); print(\'DONE: opened PR 7\')"',
    )
    assert await runner.refresh_cloud("session_x") == "DONE: opened PR 7"


def test_stop_command_matching() -> None:
    from assistant.engines.realtime_engine import is_stop_command

    for phrase in ("Alexa stop", "stop", "STOP.", "alexa, stop listening", "be quiet", "shut up"):
        assert is_stop_command(phrase), phrase
    for phrase in ("stop the music", "don't stop believing", "stop at the store tomorrow"):
        assert not is_stop_command(phrase), phrase
