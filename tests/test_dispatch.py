"""Dispatcher tests: a throwaway git repo + a fake claude CLI. No Max usage."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from assistant.dispatch import Dispatcher

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


async def wait_done(dispatcher: Dispatcher, job_id: str, timeout: float = 15.0) -> None:
    async with asyncio.timeout(timeout):
        while True:
            job = next(j for j in dispatcher.jobs() if j.id == job_id)
            if job.status != "running":
                return
            await asyncio.sleep(0.1)


async def test_full_job_lifecycle(tmp_path) -> None:
    repo = make_repo(tmp_path)
    dispatcher = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=20.0
    )
    job = await dispatcher.start("add a greeting file", "greeting file")

    assert job.status == "running"
    assert job.branch.startswith("alexa/greeting-file")
    assert Path(job.worktree).exists()  # sandboxed worktree created

    await wait_done(dispatcher, job.id)
    job = dispatcher.jobs()[0]
    assert job.status == "done"
    assert "did the task" in job.summary
    assert job.session_id == "sess-fake"
    assert job.model == "claude-fake-1"
    assert job.last_activity == "working on the task now"
    assert (repo / "logs" / "jobs" / f"{job.id}.log").exists()

    report = json.loads(dispatcher.report())
    assert report[0]["status"] == "done"
    assert "claude --resume sess-fake" in report[0]["full_transcript"]

    # persistence: a fresh dispatcher reads the record back
    again = Dispatcher(repo, claude_cmd="unused")
    assert again.jobs()[0].summary == job.summary


async def test_interrupted_jobs_marked_on_restart(tmp_path) -> None:
    repo = make_repo(tmp_path)
    jobs_path = repo / "data" / "jobs.json"
    jobs_path.parent.mkdir()
    jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "title": "t",
                    "request": "r",
                    "status": "running",
                    "branch": "alexa/x",
                    "worktree": "gone",
                    "started": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    dispatcher = Dispatcher(repo, claude_cmd="unused")
    assert dispatcher.jobs()[0].status == "interrupted"


async def test_failed_agent_is_reported(tmp_path) -> None:
    repo = make_repo(tmp_path)
    dispatcher = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" -c "import sys; sys.exit(3)"', timeout_s=20.0
    )
    job = await dispatcher.start("doomed", "doomed")
    await wait_done(dispatcher, job.id)
    job = dispatcher.jobs()[0]
    assert job.status == "failed"
    assert "exit 3" in job.summary
