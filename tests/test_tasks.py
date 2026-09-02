"""The task board: draft → build → announce → approve, plus queries and the
import of the older jobs.json. Runs against a throwaway repo + fake claude."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from assistant.announce import Announcer
from assistant.dispatch import DispatchError
from assistant.tasks import TaskBoard
from tests.test_dispatch import fake_runner, make_repo, push_cloud_branch_to_origin


def make_board(repo: Path, **kw) -> tuple[TaskBoard, Announcer]:
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=fake_runner(repo, **kw), announcer=announcer)
    return board, announcer


async def wait_state(board: TaskBoard, task_id: int, timeout: float = 15.0) -> None:
    async with asyncio.timeout(timeout):
        while board.get(task_id).state == "building":
            await asyncio.sleep(0.1)


SPEC = "## Goal\nGreet on startup.\n\n## Voice test\nAsk 'say hi' and hear a greeting."


async def test_draft_build_announce_and_approve(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)

    task = board.draft("Startup greeting", SPEC)
    assert task.id == 1 and task.state == "drafting" and task.slug == "1-startup-greeting"
    assert (repo / "data" / "specs" / "1-startup-greeting.md").read_text(encoding="utf-8").startswith(
        "# Task 1: Startup greeting"
    )
    assert "task 1 'Startup greeting' — drafting" in board.status_line()

    with pytest.raises(DispatchError, match="needs one of"):
        await board.approve(1)  # not built yet

    task = await board.start(1)
    assert task.state == "building" and task.branch == "alexa/1-startup-greeting"
    spec_in_branch = Path(task.worktree) / "docs" / "tasks" / "1-startup-greeting.md"
    assert spec_in_branch.exists()  # the agent's prompt target, committed on the branch
    with pytest.raises(DispatchError, match="is building"):
        await board.start(1)  # already running

    await wait_state(board, 1)
    task = board.get("task 1")
    assert task.state == "built"
    assert task.current.status == "done" and task.current.milestones == 1
    assert task.session_id == "sess-fake"
    texts = [a.text for a in announcer.pending()]
    assert any("Task 1, 'Startup greeting', is built and ready for your test" in t for t in texts)
    milestone = next(r for r in announcer.items(limit=50) if r["kind"] == "milestone")
    assert milestone["text"].startswith("Progress on task 1, 'Startup greeting': tests are passing")
    assert milestone["state"] == "resolved"  # never spoken: "built" superseded it

    # simulate the agent's committed work, then approve → merge gates → main
    (Path(task.worktree) / "GREETING.md").write_text("hello", encoding="utf-8")
    await board._runner.commit_all(Path(task.worktree), "add greeting")
    result = await board.approve("1")
    assert "merged alexa/1-startup-greeting" in result and "restart" in result
    assert (repo / "GREETING.md").exists()
    task = board.get(1)
    assert task.state == "merged" and task.closed and task.cleanup_pending
    assert board.status_line() == "none open"

    # persisted and reloaded
    again = TaskBoard(repo, runner=board._runner)
    assert again.get(1).state == "merged" and again.get(1).iterations[0].summary


async def test_queries_windows_and_search(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _ = make_board(repo)
    one = board.draft("Calendar reminders", "remind me of events")
    two = board.draft("Web search tool", "search the web")
    board.abandon(two.id)

    open_rows = json.loads(board.list())
    assert [r["id"] for r in open_rows] == [one.id]  # abandoned hidden by default
    everything = json.loads(board.list(include_closed=True))
    assert {r["id"] for r in everything} == {1, 2}
    today = json.loads(board.list(since="today"))
    assert {r["id"] for r in today} == {1, 2}
    assert board.list(since="yesterday", until="yesterday") == "no tasks match that"
    with pytest.raises(ValueError, match="ISO date"):
        board.list(since="whenever")

    assert json.loads(board.search("calendar"))[0]["id"] == one.id
    assert json.loads(board.search("web search"))[0]["id"] == two.id
    assert board.search("thermostat") == "no past tasks match that"
    detail = json.loads(board.detail(one.id))
    assert detail["spec"] == "remind me of events" and detail["history"][0]["event"] == "drafted"


async def test_failed_build_can_be_retried_and_abandoned(tmp_path: Path) -> None:
    import sys

    repo = make_repo(tmp_path)
    announcer = Announcer(repo / "data" / "announcements.json")
    from assistant.dispatch import Dispatcher

    doomed = Dispatcher(
        repo, claude_cmd=f'"{sys.executable}" -c "import sys; sys.exit(3)"', timeout_s=20.0
    )
    board = TaskBoard(repo, runner=doomed, announcer=announcer)
    task = board.draft("Doomed feature", "will not build")
    await board.start(task.id)
    await wait_state(board, task.id)
    task = board.get(task.id)
    assert task.state == "failed" and "without a result" in task.last_error
    assert any("stopped without finishing" in a.text for a in announcer.pending())

    # a retry reuses the worktree (and would resume the session if one existed)
    await board.start(task.id)
    assert board.get(task.id).current.kind == "build"  # no session id → fresh build
    await wait_state(board, task.id)
    assert board.get(task.id).state == "failed"
    assert "abandoned" in board.abandon(task.id)
    assert board.get(task.id).closed


async def test_cloud_task_builds_via_routine_and_merges_by_remote_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    push_cloud_branch_to_origin(repo, tmp_path)

    fired: list[tuple[str, str]] = []

    class FakeCloudRunner(type(fake_runner(repo))):
        async def fire_cloud(self, *, request, title, routine_id, token):
            fired.append((routine_id, token))
            return {"session_id": "session_01TEST", "session_url": "https://claude.ai/code/session_01TEST"}

        async def refresh_cloud(self, session_id):
            return "DONE: pushed branch alexa/cloud-feature"

    runner = FakeCloudRunner(repo, routine_id="trig_self", routine_token="tok_self")
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=runner, announcer=announcer)
    task = board.draft("Cloud feature", "do it in the cloud")
    task = await board.start(task.id, mode="cloud")
    assert task.mode == "cloud" and task.state == "building"
    assert fired == [("trig_self", "tok_self")]
    assert json.loads(board.detail(task.id))["open_live"].startswith("https://claude.ai/code/")

    assert "DONE" in await board.refresh(task.id)
    assert board.get(task.id).state == "built"
    assert any("built in the cloud" in a.text for a in announcer.pending())

    assert "remote branch" in await board.approve(task.id)  # needs the branch name
    result = await board.approve(task.id, branch="alexa/cloud-feature")
    assert "merged origin/alexa/cloud-feature" in result
    assert (repo / "CLOUD.md").exists() and board.get(task.id).state == "merged"


def test_jobs_json_is_imported_once(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "data").mkdir()
    (repo / "data" / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "calendar-3edbeb", "title": "Apple Calendar integration", "request": "calendar",
                    "status": "merged", "branch": "alexa/calendar-3edbeb", "worktree": "gone",
                    "started": time.time() - 7200, "finished": time.time() - 3600,
                    "summary": "added calendar tools", "session_id": "sess-1", "closed": True,
                },
                {
                    "id": "cloud-16de16", "title": "Weather feature", "request": "weather",
                    "status": "running", "branch": "(cloud)", "worktree": "", "mode": "cloud",
                    "started": time.time() - 600, "session_url": "https://claude.ai/code/s",
                },
            ]
        ),
        encoding="utf-8",
    )
    board = TaskBoard(repo, runner=fake_runner(repo))
    cal, weather = board.get(1), board.get(2)
    assert cal.state == "merged" and cal.closed and cal.iterations[0].summary == "added calendar tools"
    assert weather.state == "building" and weather.mode == "cloud"
    assert (repo / "data" / "jobs.json.imported").exists()
    assert not (repo / "data" / "jobs.json").exists()
    assert "task 2 'Weather feature' — building" in board.status_line()
    # a second load reads tasks.json, not the (now renamed) jobs file
    assert TaskBoard(repo, runner=fake_runner(repo)).get(2).title == "Weather feature"


async def test_a_build_left_running_is_reconciled_at_startup(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "data").mkdir()
    (repo / "data" / "tasks.json").write_text(
        json.dumps(
            {
                "version": 1,
                "next_id": 2,
                "tasks": [
                    {
                        "id": 1, "slug": "1-x", "title": "X", "spec": "x", "state": "building",
                        "created": 1.0, "updated": 1.0, "mode": "local",
                        "iterations": [{"n": 1, "status": "running", "started": 1.0}],
                        "unknown_future_field": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    board = TaskBoard(repo, runner=fake_runner(repo), resume_delay_s=0)
    assert board.get(1).state == "building"  # loading alone never gives up on a build
    await board.startup_maintenance()  # no live pid, no log: it is over
    await wait_state(board, 1)
    task = board.get(1)
    assert task.state == "failed" and "without a result" in task.last_error
    assert any(h["event"] == "restart" for h in task.history)


async def test_one_yes_merges_several_and_reports_each(tmp_path: Path) -> None:
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.tasks import Iteration

    repo = make_repo(tmp_path)
    board, _ = make_board(repo)
    built: list[int] = []
    for name in ("One", "Two"):
        task = board.draft(name, "x")
        await board.start(task.id)
        await wait_state(board, task.id)
        (Path(task.worktree) / f"{name}.md").write_text(name, encoding="utf-8")
        await board._runner.commit_all(Path(task.worktree), name)
        built.append(task.id)
    stuck = board.draft("Stuck", "y")
    stuck.state = "needs_input"
    stuck.iterations.append(Iteration(n=1, status="done", question="which room?"))
    board._save()

    report = await board.approve_many([*built, stuck.id])
    lines = report.splitlines()
    assert lines[0].startswith("2 of 3 merged.")
    assert lines[1].startswith("task 1 ('One'): merged alexa/1-one")
    assert lines[2].startswith("task 2 ('Two'): merged alexa/2-two")
    assert lines[3].startswith("task 3: not merged: task 3 is needs_input")
    assert (repo / "One.md").exists() and (repo / "Two.md").exists()
    assert board.get(1).state == "merged" and board.get(2).state == "merged"

    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", task_board=board)
    text, is_error = await engine._execute_task_tool("approve_task", {"all": True, "confirmed": True})
    assert is_error and "nothing is built" in text
    text, is_error = await engine._execute_task_tool("approve_task", {"ids": [1], "confirmed": False})
    assert is_error and "explicit" in text
    text, is_error = await engine._execute_task_tool("approve_task", {"confirmed": True})
    assert is_error and "needs an id" in text


async def test_voice_approve_runs_in_the_background_and_announces(tmp_path: Path) -> None:
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.tasks import Iteration

    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)
    task = board.draft("Ping response", "say pong")
    await board.start(task.id)
    await wait_state(board, task.id)
    (Path(task.worktree) / "PONG.md").write_text("pong", encoding="utf-8")
    await board._runner.commit_all(Path(task.worktree), "pong")
    stuck = board.draft("Stuck", "y")
    stuck.state = "needs_input"
    stuck.iterations.append(Iteration(n=1, status="done", question="which room?"))
    board._save()
    restarts: list[int] = []
    board.on_restart = lambda: restarts.append(1)

    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", task_board=board)
    text, is_error = await engine._execute_task_tool(
        "approve_task", {"ids": [task.id, stuck.id], "confirmed": True}
    )
    assert not is_error and text.startswith("merging task 1 ('Ping response'), task 2 ('Stuck') in the background")
    assert board.get(task.id).state == "built"  # returned at once; the gates are still running
    async with asyncio.timeout(30):
        await asyncio.gather(*board._bg)
    assert board.get(task.id).state == "merged" and (repo / "PONG.md").exists()
    texts = [a.text for a in announcer.pending()]
    assert any(t.startswith("Task 1, 'Ping response' is merged into main.") for t in texts)
    assert any(t.startswith("Task 2, 'Stuck' could not be merged: task 2 is needs_input") for t in texts)
    assert all(a.priority == "urgent" for a in announcer.pending())
    assert restarts == [1] and not board.restart_requested


async def test_a_second_approve_while_one_is_merging_is_refused(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, _ = make_board(repo)
    task = board.draft("Greeting", "x")
    await board.start(task.id)
    await wait_state(board, task.id)
    (Path(task.worktree) / "G.md").write_text("g", encoding="utf-8")
    await board._runner.commit_all(Path(task.worktree), "g")
    first = asyncio.create_task(board.approve(task.id))
    await asyncio.sleep(0.05)  # the merge is underway
    with pytest.raises(DispatchError, match="already being merged"):
        await board.approve(task.id)
    assert "merged" in await first


def test_stale_running_records_on_closed_tasks_do_not_count(tmp_path: Path) -> None:
    """A migrated or abandoned task may still carry a 'running' iteration;
    it must not block switching or starting other work."""
    repo = make_repo(tmp_path)
    board, _ = make_board(repo)
    from assistant.tasks import Iteration

    ghost = board.draft("Ghost", "old cloud probe")
    ghost.state = "building"
    ghost.iterations.append(Iteration(n=1, status="running", started=1.0))
    board._save()
    assert ghost.running
    board.abandon(ghost.id)
    assert not board.get(ghost.id).running
    assert board.get(ghost.id).current.status == "interrupted"
    assert not any(t.running for t in board.tasks())
