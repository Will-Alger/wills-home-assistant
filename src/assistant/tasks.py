"""The task board: the assistant's own Jira, kept in her head.

Every piece of development work she is asked to do becomes a Task with a
spec, a state, and a history. She writes the spec after talking it through,
starts a build (the runner in dispatch.py does the git + `claude -p` work),
announces milestones and completion on her own (announce.py), and — with the
owner's spoken approval — merges the result. Later phases add staging
(running from the branch) and revisions (feedback back to the same agent).

States: drafting → building → built ⇄ staged → merged; side exits
revising (back to built), failed, abandoned. merged/abandoned auto-close.

Storage: data/tasks.json {"version", "next_id", "tasks"} — atomic writes,
tolerant loads (unknown keys dropped, missing keys defaulted), and a one-time
import of the older data/jobs.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from assistant.dispatch import (
    _REVISE_PROMPT,
    _TASK_PROMPT,
    DispatchError,
    _is_own_repo,
)

STATES = ("drafting", "building", "built", "staged", "revising", "merged", "failed", "abandoned")
OPEN_STATES = ("drafting", "building", "built", "staged", "revising", "failed")

_TRANSITIONS: dict[str, set[str]] = {
    "start": {"drafting", "failed"},
    "approve": {"built", "staged"},
    "abandon": {"drafting", "building", "built", "staged", "revising", "failed"},
    "revise": {"built", "staged", "failed"},
    "stage": {"built", "staged"},
}

_MAX_MILESTONES = 3
_STATUS_LINE_LIMIT = 5
POINTER_NAME = "active_checkout.json"


def pointer_path(root: Path) -> Path:
    return root / "data" / POINTER_NAME


def read_pointer(root: Path) -> dict[str, Any] | None:
    path = pointer_path(root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("task_id") else None
    except (json.JSONDecodeError, OSError):
        return None


def write_pointer(root: Path, data: dict[str, Any]) -> None:
    path = pointer_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def clear_pointer(root: Path) -> None:
    pointer_path(root).unlink(missing_ok=True)


@dataclass
class Iteration:
    n: int
    kind: str = "build"  # build | revise | retry
    feedback: str = ""
    started: float = 0.0
    finished: float | None = None
    status: str = "running"  # running | done | failed | interrupted
    summary: str = ""
    cost_usd: float = 0.0
    session_id: str = ""
    model: str = ""
    progress: list[str] = field(default_factory=list)
    milestones: int = 0
    log: str = ""


@dataclass
class Task:
    id: int
    slug: str
    title: str
    spec: str
    state: str = "drafting"
    created: float = 0.0
    updated: float = 0.0
    branch: str = ""
    worktree: str = ""
    mode: str = "local"  # local | cloud
    repo: str = ""  # "" = the assistant's own repository
    session_id: str = ""  # current agent session (--resume target)
    session_url: str = ""  # cloud only
    iterations: list[Iteration] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    staged_at: float | None = None
    merged_at: float | None = None
    closed: bool = False
    last_error: str = ""
    cleanup_pending: bool = False

    @property
    def current(self) -> Iteration | None:
        return self.iterations[-1] if self.iterations else None

    @property
    def running(self) -> bool:
        # an agent is live only while the task is in a working state — stale
        # "running" iterations on closed/failed records never count
        return (
            self.state in ("building", "revising")
            and self.current is not None
            and self.current.status == "running"
        )


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32] or "task"


def _load_task(row: dict[str, Any]) -> Task:
    known = set(Task.__dataclass_fields__)
    data = {k: v for k, v in row.items() if k in known}
    it_known = set(Iteration.__dataclass_fields__)
    data["iterations"] = [
        Iteration(**{k: v for k, v in it.items() if k in it_known})
        for it in row.get("iterations", [])
        if isinstance(it, dict) and "n" in it
    ]
    return Task(**data)


def _window(spec: str | None, now: float) -> tuple[float | None, float | None]:
    """'today' | 'yesterday' | 'week' | ISO date -> (since, until) timestamps."""
    if not spec:
        return None, None
    text = spec.strip().lower()
    local_now = datetime.fromtimestamp(now).astimezone()
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "today":
        return midnight.timestamp(), None
    if text == "yesterday":
        start = midnight - timedelta(days=1)
        return start.timestamp(), midnight.timestamp()
    if text in ("week", "this week", "last 7 days"):
        return (midnight - timedelta(days=7)).timestamp(), None
    try:
        day = datetime.fromisoformat(text).replace(tzinfo=local_now.tzinfo)
    except ValueError as err:
        raise ValueError(
            f"use today, yesterday, week, or an ISO date like 2026-09-01 (got {spec!r})"
        ) from err
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _hours_ago(then: float | None, now: float) -> str:
    if not then:
        return "?"
    hours = (now - then) / 3600
    if hours < 1:
        return f"{round(hours * 60)} min ago"
    return f"{hours:.1f}h ago"


class TaskBoard:
    def __init__(
        self,
        root: Path,
        *,
        runner: Any,
        announcer: Any | None = None,
        now: Callable[[], float] = time.time,
        staged_task_id: int | None = None,
        uv_exe: str = "",
    ) -> None:
        self._root = root
        self._runner = runner
        self._announcer = announcer
        self._now = now
        self._staged_task_id = staged_task_id  # this PROCESS runs that task's build
        self._uv_exe = uv_exe
        self.restart_requested = False  # a switch/approve/abandon wants a restart
        self._path = root / "data" / "tasks.json"
        self._next_id = 1
        self._tasks: dict[int, Task] = {}
        self._bg: list[asyncio.Task] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            self._next_id = int(data.get("next_id", 1))
            for row in data.get("tasks", []):
                if isinstance(row, dict) and "id" in row:
                    task = _load_task(row)
                    # a local build dies with the app; nobody will finish it
                    if task.running and task.mode == "local":
                        task.current.status = "interrupted"
                        task.current.finished = self._now()
                        task.state = "failed"
                        task.last_error = "the build was interrupted by a restart"
                        self._log(task, "interrupted", "restart while building")
                    self._tasks[task.id] = task
        else:
            self._import_jobs()

    def _import_jobs(self) -> None:
        """One-time import of the older data/jobs.json records."""
        old = self._root / "data" / "jobs.json"
        if not old.exists():
            return
        try:
            rows = json.loads(old.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        state_map = {"done": "built", "merged": "merged", "failed": "failed", "interrupted": "failed"}
        for row in sorted(rows, key=lambda r: r.get("started", 0)):
            status = row.get("status", "failed")
            mode = row.get("mode", "local")
            state = state_map.get(status, "building" if mode == "cloud" else "failed")
            task = Task(
                id=self._next_id,
                slug=str(row.get("id", f"job-{self._next_id}")),
                title=str(row.get("title", "untitled")),
                spec=str(row.get("request", "")),
                state=state,
                created=float(row.get("started", 0.0) or 0.0),
                updated=float(row.get("finished") or row.get("started") or 0.0),
                branch=str(row.get("branch", "")),
                worktree=str(row.get("worktree", "")),
                mode=mode,
                repo=str(row.get("repo", "")),
                session_id=str(row.get("session_id", "")),
                session_url=str(row.get("session_url", "")),
                closed=bool(row.get("closed", False)) or state in ("merged", "abandoned"),
                merged_at=float(row.get("finished") or 0.0) if state == "merged" else None,
            )
            task.iterations.append(
                Iteration(
                    n=1,
                    kind="build",
                    started=task.created,
                    finished=row.get("finished"),
                    status={"built": "done", "merged": "done", "building": "running"}.get(
                        state, "failed"
                    ),
                    summary=str(row.get("summary", "")),
                    cost_usd=float(row.get("cost_usd") or 0.0),
                    session_id=task.session_id,
                    model=str(row.get("model", "")),
                )
            )
            task.history.append(
                {"ts": task.created, "event": "imported", "detail": f"from jobs.json ({status})"}
            )
            self._tasks[task.id] = task
            self._next_id += 1
        self._save()
        old.replace(old.with_suffix(".json.imported"))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "next_id": self._next_id,
            "tasks": [asdict(t) for t in self._tasks.values()],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def _log(self, task: Task, event: str, detail: str = "") -> None:
        task.updated = self._now()
        task.history.append({"ts": task.updated, "event": event, "detail": detail})

    def _announce(self, text: str, *, kind: str, ref: str, priority: str = "normal") -> None:
        if self._announcer is None:
            return
        try:
            self._announcer.enqueue(text, kind=kind, ref=ref, priority=priority)
        except Exception:  # noqa: BLE001 — never let an announcement break the board
            return

    # ── lookup ─────────────────────────────────────────────────────────────

    def tasks(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.created, reverse=True)

    def get(self, ref: Any) -> Task:
        """Accepts 7, "7", "task 7", or a slug."""
        text = str(ref).strip().lower()
        digits = re.sub(r"[^0-9]", "", text)
        if digits and int(digits) in self._tasks:
            return self._tasks[int(digits)]
        for task in self._tasks.values():
            if task.slug == text:
                return task
        raise DispatchError(f"no task {ref!r} — list_tasks shows the ids")

    def _check(self, task: Task, action: str) -> None:
        allowed = _TRANSITIONS[action]
        if task.state not in allowed:
            raise DispatchError(
                f"task {task.id} is {task.state}; {action} needs one of: {', '.join(sorted(allowed))}"
            )
        if task.running and action != "abandon":
            raise DispatchError(f"task {task.id} still has an agent running — wait for it to finish")

    def extra_repo_names(self) -> list[str]:
        return list(self._runner.extra_repo_names())

    def running_iterations(self) -> list[tuple[Task, Iteration]]:
        return [(t, t.current) for t in self._tasks.values() if t.running and t.current]

    # ── drafting ───────────────────────────────────────────────────────────

    def draft(self, title: str, spec: str) -> Task:
        title = " ".join(str(title).split())
        spec = str(spec).strip()
        if not title or not spec:
            raise DispatchError("a task needs a title and a spec")
        task = Task(
            id=self._next_id,
            slug=f"{self._next_id}-{_slugify(title)}",
            title=title,
            spec=spec,
            created=self._now(),
        )
        self._next_id += 1
        self._log(task, "drafted", "spec written")
        self._tasks[task.id] = task
        spec_path = self._root / "data" / "specs" / f"{task.slug}.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(self._spec_document(task), encoding="utf-8")
        self._save()
        return task

    @staticmethod
    def _spec_document(task: Task) -> str:
        lines = [f"# Task {task.id}: {task.title}", "", task.spec.strip(), ""]
        for it in task.iterations:
            if it.kind == "revise" and it.feedback:
                stamp = datetime.fromtimestamp(it.started).astimezone().strftime("%Y-%m-%d")
                lines += [f"## Revision {it.n} — {stamp}", "", it.feedback.strip(), ""]
        return "\n".join(lines)

    # ── building ───────────────────────────────────────────────────────────

    async def start(self, ref: Any, *, mode: str = "", repo: str = "") -> Task:
        task = self.get(ref)
        self._check(task, "start")
        if repo and not _is_own_repo(repo):
            entry = self._runner.resolve_repo(repo)
            task.repo = repo
            await self._start_cloud(task, entry["routine_id"], entry["token"])
        elif mode == "cloud":
            if not self._runner.cloud_enabled:
                raise DispatchError("cloud dispatch is not configured (routine id/token missing)")
            rid, token = self._runner.own_routine
            await self._start_cloud(task, rid, token)
        else:
            await self._start_local(task)
        return task

    async def _start_local(self, task: Task) -> None:
        resume = ""
        if task.state == "failed" and task.worktree and Path(task.worktree).exists():
            resume = task.session_id  # pick the agent's own context back up
        else:
            worktree, branch = await self._runner.create_worktree(task.slug)
            task.worktree, task.branch = str(worktree), branch
        worktree = Path(task.worktree)
        spec_path = worktree / "docs" / "tasks" / f"{task.slug}.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(self._spec_document(task), encoding="utf-8")
        await self._runner.commit_all(worktree, f"Task {task.id}: spec — {task.title}")
        n = len(task.iterations) + 1
        iteration = Iteration(
            n=n,
            kind="retry" if resume else "build",
            started=self._now(),
            log=str(self._root / "logs" / "tasks" / f"{task.slug}-{n}.log"),
        )
        task.iterations.append(iteration)
        task.mode = "local"
        task.state = "building"
        task.last_error = ""
        self._log(task, "building", f"iteration {n} ({iteration.kind}) on {task.branch}")
        self._save()
        prompt = (
            _REVISE_PROMPT.format(
                feedback="Continue where you left off; the previous run ended without a result.",
                spec_path=f"docs/tasks/{task.slug}.md",
                n=n,
            )
            if resume
            else _TASK_PROMPT.format(
                spec_path=f"docs/tasks/{task.slug}.md",
                task_id=task.id,
                title=task.title,
                spec=task.spec,
            )
        )
        self._bg.append(asyncio.create_task(self._run_iteration(task, iteration, prompt, resume)))

    async def _run_iteration(self, task: Task, iteration: Iteration, prompt: str, resume: str) -> None:
        def on_update(run: Any, kind: str) -> None:
            iteration.session_id = run.session_id or iteration.session_id
            iteration.model = run.model or iteration.model
            iteration.progress = list(run.progress)
            task.session_id = iteration.session_id or task.session_id
            if kind == "milestone" and iteration.milestones < _MAX_MILESTONES:
                iteration.milestones += 1
                self._announce(
                    f"Progress on task {task.id}, '{task.title}': {run.milestones[-1]}",
                    kind="milestone",
                    ref=f"task:{task.id}:{iteration.n}:milestone:{iteration.milestones}",
                )
            self._save()

        try:
            run = await self._runner.run_agent(
                cwd=Path(task.worktree),
                prompt=prompt,
                log_path=Path(iteration.log),
                resume_session_id=resume,
                on_update=on_update,
            )
            on_update(run, "result")
            iteration.summary = run.summary
            iteration.cost_usd = run.cost_usd
            iteration.status = run.status
        except Exception as err:  # noqa: BLE001 — a build may never crash the app
            iteration.status = "failed"
            iteration.summary = f"{type(err).__name__}: {err}"
        iteration.finished = self._now()
        if iteration.status == "done":
            task.state = "built"
            self._log(task, "built", iteration.summary[:200])
            gist = iteration.summary.strip().split(". ")[0][:160]
            self._announce(
                f"Task {task.id}, '{task.title}', is built and ready for your test."
                + (f" The agent says: {gist}." if gist else "")
                + f" Say 'switch to task {task.id}' to try it, or 'approve task {task.id}' to merge it.",
                kind="task",
                ref=f"task:{task.id}:{iteration.n}:built",
            )
        else:
            task.state = "failed"
            task.last_error = iteration.summary[:300]
            self._log(task, "failed", iteration.summary[:200])
            self._announce(
                f"Task {task.id}, '{task.title}', stopped without finishing: "
                f"{iteration.summary[:140]}. Say 'retry task {task.id}' to pick it back up.",
                kind="task",
                ref=f"task:{task.id}:{iteration.n}:failed",
            )
        self._save()

    async def _start_cloud(self, task: Task, routine_id: str, token: str) -> None:
        info = await self._runner.fire_cloud(
            request=task.spec, title=task.title, routine_id=routine_id, token=token
        )
        n = len(task.iterations) + 1
        task.iterations.append(Iteration(n=n, kind="build", started=self._now()))
        task.mode = "cloud"
        task.session_id = info.get("session_id", "")
        task.session_url = info.get("session_url", "")
        task.branch = "(cloud session — lands as a GitHub branch/PR)"
        task.state = "building"
        self._log(task, "building", "cloud session fired")
        self._save()

    async def refresh(self, ref: Any) -> str:
        """Ask a cloud task's live session how it's going (DONE flips to built)."""
        task = self.get(ref)
        if task.mode != "cloud" or not task.session_id:
            return f"task {task.id} runs locally — its status is already live"
        line = await self._runner.refresh_cloud(task.session_id)
        iteration = task.current
        if iteration is not None:
            iteration.progress = (iteration.progress + [line[:200]])[-10:]
            if line.upper().startswith("DONE") and iteration.status == "running":
                iteration.status = "done"
                iteration.finished = self._now()
                iteration.summary = line[:600]
                task.state = "built"
                self._log(task, "built", line[:200])
                self._announce(
                    f"Task {task.id}, '{task.title}', is built in the cloud: {line[:160]}. "
                    "It merges by its remote branch name once you approve it.",
                    kind="task",
                    ref=f"task:{task.id}:{iteration.n}:built",
                )
        self._save()
        return f"task {task.id}: {line[:400]}"

    # ── finishing ──────────────────────────────────────────────────────────

    async def approve(self, ref: Any, branch: str = "") -> str:
        task = self.get(ref)
        self._check(task, "approve")
        cleanup: Path | None = None
        if task.mode == "cloud":
            if not branch:
                return (
                    f"task {task.id} came from a cloud session — its status names the "
                    "remote branch; call approve_task again with branch=<exact name>"
                )
            merge_ref, cleanup = await self._runner.fetch_remote_branch(branch, task.slug)
            gates_dir = cleanup
        else:
            merge_ref, gates_dir = task.branch, Path(task.worktree)
        try:
            ok, message = await self._runner.merge_branch(
                merge_ref=merge_ref, gates_dir=gates_dir, title=task.title
            )
        finally:
            if cleanup is not None:
                await self._runner.remove_worktree(cleanup)
        if not ok:
            task.last_error = message[:300]
            self._log(task, "merge blocked", message[:200])
            self._save()
            return message
        task.state = "merged"
        task.closed = True
        task.merged_at = self._now()
        task.cleanup_pending = task.mode == "local"
        self._log(task, "merged", merge_ref)
        # main's venv must match the merged lockfile BEFORE anything restarts
        # into it: a crash-looping main is the one thing the watchdog can't
        # roll back from.
        synced, note = await self._runner.uv_sync(self._root, self._uv_exe)
        pointer = read_pointer(self._root)
        if not synced and "not found" not in note:
            self._save()
            return (
                f"{message} Task {task.id} is merged, but syncing main's dependencies failed "
                f"({note}) — NOT restarting; the owner should run uv sync and restart by hand."
            )
        if pointer is not None:
            clear_pointer(self._root)
            for other in self._tasks.values():
                if other.state == "staged":
                    other.state = "built"
                    self._log(other, "unstaged", f"task {task.id} merged")
        self._save()
        if self._staged_task_id is not None:
            self.restart_requested = True
            return (
                f"{message} Task {task.id} is merged into main — restarting onto main now: "
                "say a brief goodbye and end the conversation."
            )
        return f"{message} Task {task.id} is merged — offer to restart yourself so it takes effect."

    def abandon(self, ref: Any) -> str:
        task = self.get(ref)
        self._check(task, "abandon")
        was_staged = task.state == "staged"
        if task.running and task.current is not None:
            task.current.status = "interrupted"
            task.current.finished = self._now()
        task.state = "abandoned"
        task.closed = True
        task.cleanup_pending = task.mode == "local"
        self._log(task, "abandoned")
        pointer = read_pointer(self._root)
        if pointer is not None and int(pointer.get("task_id", 0)) == task.id:
            clear_pointer(self._root)
        self._save()
        note = f"task {task.id} ('{task.title}') abandoned — its branch is kept if you change your mind"
        if was_staged and self._staged_task_id == task.id:
            self.restart_requested = True
            note += "; restarting onto main — say a brief goodbye and end the conversation"
        return note

    # ── queries ────────────────────────────────────────────────────────────

    def list(
        self,
        states: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        include_closed: bool = False,
    ) -> str:
        now = self._now()
        lo, hi = _window(since, now)
        if until:
            _lo2, hi2 = _window(until, now)
            hi = hi2 if hi2 is not None else hi
            if _lo2 is not None and hi is None:
                hi = _lo2
        rows = self.tasks()
        if states:
            wanted = {s.strip().lower() for s in states}
            rows = [t for t in rows if t.state in wanted]
        elif not include_closed and not (since or until):
            rows = [t for t in rows if not t.closed]
        if lo is not None:
            rows = [t for t in rows if t.updated >= lo]
        if hi is not None:
            rows = [t for t in rows if t.updated < hi]
        if not rows:
            if since or until or states:
                return "no tasks match that"
            return "no open tasks" if self._tasks else "no tasks yet"
        return json.dumps([self._row(t, now) for t in rows[:12]])

    def _row(self, task: Task, now: float) -> dict[str, Any]:
        it = task.current
        row: dict[str, Any] = {
            "id": task.id,
            "title": task.title,
            "state": task.state,
            "updated": _hours_ago(task.updated, now),
            "mode": task.mode,
        }
        if task.repo:
            row["repo"] = task.repo
        if task.closed:
            row["closed"] = True
        if it is not None:
            row["iterations"] = len(task.iterations)
            if it.summary:
                row["summary"] = it.summary[:240]
            elif it.progress:
                row["currently"] = it.progress[-1]
        if task.last_error and task.state == "failed":
            row["error"] = task.last_error[:200]
        return row

    def detail(self, ref: Any, log_tail: bool = False) -> str:
        task = self.get(ref)
        now = self._now()
        data: dict[str, Any] = {
            "id": task.id,
            "title": task.title,
            "state": task.state,
            "created": _hours_ago(task.created, now),
            "updated": _hours_ago(task.updated, now),
            "mode": task.mode,
            "repo": task.repo or "own repo",
            "branch": task.branch,
            "spec": task.spec[:4000],
            "iterations": [
                {
                    "n": it.n,
                    "kind": it.kind,
                    "status": it.status,
                    "feedback": it.feedback,
                    "summary": it.summary[:800],
                    "cost_usd": round(it.cost_usd, 4),
                    "model": it.model or "cli default",
                    "latest": it.progress[-1] if it.progress else "",
                }
                for it in task.iterations
            ],
            "history": [
                {
                    "when": _hours_ago(h.get("ts"), now),
                    "event": h.get("event"),
                    "detail": h.get("detail", ""),
                }
                for h in task.history[-12:]
            ],
        }
        if task.last_error:
            data["last_error"] = task.last_error
        if task.mode == "cloud" and task.session_url:
            data["open_live"] = task.session_url
        elif task.session_id:
            data["full_transcript"] = (
                f"claude --resume {task.session_id} (run in a terminal inside {task.worktree})"
            )
        if log_tail and task.current and task.current.log:
            try:
                data["log_tail"] = Path(task.current.log).read_text(
                    encoding="utf-8", errors="replace"
                )[-1500:]
            except OSError:
                data["log_tail"] = "(no log yet)"
        return json.dumps(data)

    def search(self, query: str) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
        if not words:
            return "give me a word or two to search for"
        now = self._now()
        hits = []
        for task in self.tasks():
            haystack = " ".join(
                [task.title, task.spec, task.state]
                + [it.summary + " " + it.feedback for it in task.iterations]
                + [str(h.get("detail", "")) for h in task.history]
            ).lower()
            if all(w in haystack for w in words):
                hits.append(self._row(task, now))
        return json.dumps(hits[:10]) if hits else "no past tasks match that"

    def status_line(self) -> str:
        open_tasks = [t for t in self.tasks() if not t.closed][:_STATUS_LINE_LIMIT]
        if not open_tasks:
            return "none open"
        now = self._now()
        parts = []
        for t in open_tasks:
            where = f" in {t.repo}" if t.repo else ""
            if t.state == "building":
                state = f"building since {_hours_ago(t.current.started if t.current else t.updated, now)}"
            elif t.state == "built":
                state = f"built {_hours_ago(t.updated, now)}, awaiting your test/approval"
            else:
                state = f"{t.state} {_hours_ago(t.updated, now)}"
            parts.append(f"task {t.id} '{t.title}'{where} — {state}")
        return "; ".join(parts)

    def staged_paragraph(self) -> str:
        if self._staged_task_id is None or self._staged_task_id not in self._tasks:
            return ""
        task = self._tasks[self._staged_task_id]
        it = task.current
        gist = (it.summary[:300] if it and it.summary else "(no summary)")
        return (
            f"IMPORTANT: you are running the STAGED build of task {task.id} '{task.title}' "
            f"(iteration {it.n if it else '?'}, not merged). What the agent built: {gist} "
            "When it fits, ask how the test is going. 'ship it' → approve_task after a yes; "
            "'go back to main' → switch_build main; problems → note them precisely for a "
            "revision. "
        )

    async def switch_build(self, target: Any) -> str:
        """Run a built task's branch (restart into it) or go back to main.
        Free and reversible: nothing is merged, abandoned, or lost."""
        text = str(target).strip().lower()
        pointer = read_pointer(self._root)
        if text in ("", "main", "master", "stable", "normal"):
            if pointer is None and self._staged_task_id is None:
                return "already running main — nothing to switch"
            clear_pointer(self._root)
            for other in self._tasks.values():
                if other.state == "staged":
                    other.state = "built"
                    self._log(other, "unstaged", "switched back to main")
            self._save()
            self.restart_requested = True
            return "switching back to main now — say a brief goodbye and end the conversation; back in about twenty seconds"
        task = self.get(target)
        self._check(task, "stage")
        if task.mode != "local" or not task.worktree or not Path(task.worktree).exists():
            raise DispatchError(f"task {task.id} has no local build to run (cloud tasks merge by branch)")
        if any(t.running for t in self._tasks.values()):
            raise DispatchError("a build is still running — switching would kill it; wait for it to finish")
        synced, note = await self._runner.uv_sync(Path(task.worktree), self._uv_exe)
        warning = "" if synced else f" (dependency sync skipped: {note})"
        it = task.current
        write_pointer(
            self._root,
            {
                "task_id": task.id,
                "slug": task.slug,
                "worktree": task.worktree,
                "branch": task.branch,
                "iteration": it.n if it else 0,
                "set_at": self._now(),
            },
        )
        for other in self._tasks.values():
            if other.state == "staged" and other.id != task.id:
                other.state = "built"
                self._log(other, "unstaged", f"switched to task {task.id}")
        task.state = "staged"
        task.staged_at = self._now()
        self._log(task, "staged", "switching into this build")
        self._save()
        self.restart_requested = True
        return (
            f"switching to task {task.id}'s build now{warning} — say a brief goodbye and end the "
            "conversation; back in about twenty seconds running it"
        )

    async def startup_maintenance(self) -> None:
        """Run once at app start: absorb a rollback, confirm a staged start,
        and remove worktrees that are no longer needed."""
        failed = pointer_path(self._root).with_name("active_checkout.failed.json")
        if failed.exists():
            try:
                data = json.loads(failed.read_text(encoding="utf-8"))
                task = self._tasks.get(int(data.get("task_id", 0)))
            except (json.JSONDecodeError, OSError, ValueError):
                task = None
            if task is not None:
                task.state = "built"
                task.last_error = (
                    "the staged build crashed twice at startup and was rolled back to main — "
                    "see logs/alexa.log"
                )
                self._log(task, "rolled back", "crashed at startup twice")
            failed.unlink(missing_ok=True)
        if self._staged_task_id is not None and self._staged_task_id in self._tasks:
            task = self._tasks[self._staged_task_id]
            if task.state != "staged":
                task.state = "staged"
                self._log(task, "staged", "process started from this build")
            it = task.current
            gist = it.summary.strip().split(". ")[0][:160] if it and it.summary else ""
            self._announce(
                f"I've restarted into the staged build of task {task.id}, '{task.title}'."
                + (f" The agent says: {gist}." if gist else "")
                + " Try it out — say 'go back to main' any time.",
                kind="system",
                ref=f"task:{task.id}:staged:{it.n if it else 0}",
            )
        pointer = read_pointer(self._root)
        live = int(pointer.get("task_id", 0)) if pointer else None
        for task in self._tasks.values():
            if not (task.cleanup_pending and task.id != live and task.worktree):
                continue
            if not Path(task.worktree).exists() or await self._runner.remove_worktree(task.worktree):
                task.cleanup_pending = False
                self._log(task, "cleaned up", "worktree removed")
        self._save()
