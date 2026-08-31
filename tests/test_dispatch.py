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


class FakeCloudDispatcher(Dispatcher):
    """Emulates the routine /fire response handling without HTTP; records
    which (routine_id, token) each dispatch would have fired."""

    fired: list[tuple[str, str]]

    async def _start_cloud(self, request, title, routine_id, token, repo=""):
        import time as _t
        import uuid as _u

        from assistant.dispatch import Job

        if not hasattr(self, "fired"):
            self.fired = []
        self.fired.append((routine_id, token))
        job = Job(
            id=f"cloud-{_u.uuid4().hex[:6]}", title=title, request=request,
            status="running", branch="(cloud session â€” lands as a GitHub branch/PR)",
            worktree="", started=_t.time(), mode="cloud",
            session_id="session_01TEST", session_url="https://claude.ai/code/session_01TEST",
            summary="running in the cloud", repo=repo,
        )
        self._jobs[job.id] = job
        self._save()
        return job


async def test_cloud_dispatch_records_session_url(tmp_path) -> None:
    repo = make_repo(tmp_path)

    dispatcher = FakeCloudDispatcher(repo, routine_id="trig_x", routine_token="tok_x")
    assert dispatcher.cloud_enabled
    job = await dispatcher.start("build the thing", "the thing", mode="cloud")
    assert job.mode == "cloud" and "claude.ai/code" in job.session_url
    assert dispatcher.fired == [("trig_x", "tok_x")]  # own routine, own creds

    report = json.loads(dispatcher.report())
    assert report[0]["open_live"] == "https://claude.ai/code/session_01TEST"

    # restart must NOT mark cloud jobs interrupted â€” they run server-side
    again = Dispatcher(repo, claude_cmd="unused")
    assert again.jobs()[0].status == "running"


async def test_multi_repo_dispatch_routes_to_that_repos_routine(tmp_path) -> None:
    repo = make_repo(tmp_path)
    dispatcher = FakeCloudDispatcher(
        repo,
        routine_id="trig_self",
        routine_token="tok_self",
        extra_routines={"my-side-project": {"routine_id": "trig_side", "token": "tok_side"}},
    )
    assert dispatcher.extra_repo_names() == ["my-side-project"]

    job = await dispatcher.start("fix the readme", "readme fix", repo="my side project")
    assert job.repo == "my side project" and job.mode == "cloud"
    assert dispatcher.fired == [("trig_side", "tok_side")]  # that repo's creds, not hers

    # own-repo aliases still use her own routine
    await dispatcher.start("tweak", "tweak", repo="self", mode="cloud")
    assert dispatcher.fired[-1] == ("trig_self", "tok_self")

    report = json.loads(dispatcher.report())
    assert {e["repo"] for e in report} == {"my side project", "own repo"}


async def test_unknown_repo_fails_with_the_configured_list(tmp_path) -> None:
    import pytest

    from assistant.dispatch import DispatchError, load_extra_routines

    repo = make_repo(tmp_path)
    dispatcher = FakeCloudDispatcher(
        repo,
        routine_id="trig_self",
        routine_token="tok_self",
        extra_routines={"my-side-project": {"routine_id": "trig_side", "token": "tok_side"}},
    )
    with pytest.raises(DispatchError, match="my-side-project"):
        await dispatcher.start("do stuff", "stuff", repo="nonexistent-repo")

    # registry loader: valid entries only, absent/broken file -> empty
    assert load_extra_routines(repo) == {}
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "routines.json").write_text(
        json.dumps(
            {
                "good-repo": {"routine_id": "trig_a", "token": "tok_a"},
                "missing-token": {"routine_id": "trig_b"},
            }
        ),
        encoding="utf-8",
    )
    assert list(load_extra_routines(repo)) == ["good-repo"]


async def test_voice_merge_with_gates(tmp_path) -> None:
    repo = make_repo(tmp_path)
    dispatcher = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=20.0
    )
    job = await dispatcher.start("greeting", "greeting")
    await wait_done(dispatcher, job.id)
    # simulate the agent's committed work on the branch
    wt = Path(job.worktree)
    (wt / "GREETING.md").write_text("hello", encoding="utf-8")
    await dispatcher._cmd(["git", "add", "-A"], wt)
    await dispatcher._cmd(["git", "commit", "-m", "add greeting"], wt)

    result = await dispatcher.merge(job.id)
    assert "merged" in result and "restart" in result
    assert (repo / "GREETING.md").exists()  # landed on main
    assert dispatcher.jobs()[0].status == "merged"
    assert dispatcher.jobs()[0].closed  # merging IS the close
    # double-merge refused
    assert "already merged" in await dispatcher.merge(job.id)
    # unknown job refused
    assert "no job" in await dispatcher.merge("nope")


async def test_merge_blocked_by_dirty_main(tmp_path) -> None:
    repo = make_repo(tmp_path)
    dispatcher = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"', timeout_s=20.0
    )
    job = await dispatcher.start("x", "x")
    await wait_done(dispatcher, job.id)
    (repo / "README.md").write_text("dirty", encoding="utf-8")  # tracked file modified
    assert "uncommitted changes" in await dispatcher.merge(job.id)


async def test_open_vs_closed_lifecycle(tmp_path) -> None:
    """Will's away-scenario: jobs persist, finished ones stay visible until
    he considers them dealt with, then close_work archives them."""
    import time as _t

    repo = make_repo(tmp_path)
    dispatcher = FakeCloudDispatcher(repo, routine_id="t", routine_token="k")
    job = await dispatcher.start("build the thing", "the thing", mode="cloud")

    # while running it shows up in the wake-time status line
    assert "the thing" in dispatcher.status_line() and "running" in dispatcher.status_line()

    # finished (say, while Will was at work) â€” still open, flagged with age
    job.status = "done"
    job.finished = _t.time() - 2 * 3600
    job.summary = "opened PR #9"
    dispatcher._save()
    line = dispatcher.status_line()
    assert "done" in line and "not yet closed" in line and "2.0h" in line
    assert json.loads(dispatcher.report())[0]["hours_ago_finished"] == 2.0

    # "yeah, we're done with that one"
    assert "closed" in dispatcher.close(job.id)
    assert dispatcher.status_line() == "none open"
    assert "no open jobs" in dispatcher.report()
    assert json.loads(dispatcher.report(include_closed=True))[0]["closed"] is True
    assert json.loads(dispatcher.report(job.id))[0]["id"] == job.id  # by id still works

    # survives a restart
    again = Dispatcher(repo, claude_cmd="unused")
    assert again.jobs()[0].closed is True
    assert "already closed" in dispatcher.close(job.id)
    assert "no job" in dispatcher.close("nope")


async def test_cloud_refresh_updates_the_job_from_the_live_session(tmp_path) -> None:
    repo = make_repo(tmp_path)
    fake_status = f'"{sys.executable}" -c "print(\'DONE: opened PR 7 on the side project\')"'
    dispatcher = FakeCloudDispatcher(
        repo, routine_id="t", routine_token="k", cloud_status_cmd=fake_status
    )
    job = await dispatcher.start("build", "build it", mode="cloud")
    assert json.loads(dispatcher.report())[0]["note"].startswith("cloud progress")

    pinged = dispatcher.refresh_running_cloud()
    assert pinged == [job.id]
    await asyncio.gather(*dispatcher._tasks)

    job = dispatcher.jobs()[0]
    assert job.status == "done" and job.finished
    assert "PR 7" in job.summary and "live check at" in job.last_activity
    assert "no job" in await dispatcher.refresh("missing")


def test_stop_command_matching() -> None:
    from assistant.engines.realtime_engine import is_stop_command

    for phrase in ("Alexa stop", "stop", "STOP.", "alexa, stop listening", "be quiet", "shut up"):
        assert is_stop_command(phrase), phrase
    for phrase in ("stop the music", "don't stop believing", "stop at the store tomorrow"):
        assert not is_stop_command(phrase), phrase


async def test_own_repo_defaults_local_even_with_cloud_configured(tmp_path) -> None:
    """The full loop (voice merge + restart) needs a local worktree — cloud
    must be opt-in for her own repo."""
    repo = make_repo(tmp_path)
    dispatcher = Dispatcher(
        repo,
        claude_cmd=f'"{sys.executable}" "{FAKE_CLAUDE}"',
        timeout_s=20.0,
        routine_id="trig_x",
        routine_token="tok_x",
    )
    job = await dispatcher.start("greeting", "greeting")
    assert job.mode == "local" and job.branch.startswith("alexa/")
    await wait_done(dispatcher, job.id)


def _push_cloud_branch_to_origin(repo: Path, tmp_path: Path) -> None:
    """Simulate a cloud session: a bare origin holding a pushed feature branch."""
    origin = tmp_path / "origin.git"
    run = lambda *a, **kw: subprocess.run(["git", *a], check=True, capture_output=True, **kw)
    run("init", "--bare", str(origin))
    run("remote", "add", "origin", str(origin), cwd=repo)
    run("push", "origin", "main", cwd=repo)
    run("checkout", "-b", "alexa/cloud-feature", cwd=repo)
    (repo / "CLOUD.md").write_text("from the cloud", encoding="utf-8")
    run("add", "-A", cwd=repo)
    run("commit", "-m", "cloud work", cwd=repo)
    run("push", "origin", "alexa/cloud-feature", cwd=repo)
    run("checkout", "main", cwd=repo)
    run("branch", "-D", "alexa/cloud-feature", cwd=repo)  # only the remote has it now


async def test_cloud_job_merges_by_remote_branch_with_gates(tmp_path) -> None:
    repo = make_repo(tmp_path)
    _push_cloud_branch_to_origin(repo, tmp_path)

    dispatcher = FakeCloudDispatcher(repo, routine_id="t", routine_token="k")
    job = await dispatcher.start("cloud feature", "cloud feature", mode="cloud")

    # gates: not done yet, and done-but-no-branch both refuse with guidance
    assert "only finished jobs" in await dispatcher.merge(job.id, branch="alexa/cloud-feature")
    job.status = "done"
    dispatcher._save()
    assert "remote branch" in await dispatcher.merge(job.id)

    result = await dispatcher.merge(job.id, branch="alexa/cloud-feature")
    assert "merged origin/alexa/cloud-feature" in result, result
    assert (repo / "CLOUD.md").exists()  # landed on main
    assert dispatcher.jobs()[0].status == "merged" and dispatcher.jobs()[0].closed
    assert not (repo / ".worktrees" / f"merge-{job.id}").exists()  # cleaned up


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
