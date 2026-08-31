"""Stage 2 of the endgame: Alexa commissions work on her own codebase.

Each job = a sandboxed git worktree (branched from main) + a headless Claude
Code run (`claude -p`, billed to the Max subscription) executing the request,
running checks, and committing to its branch. Jobs run in the background of
the always-on app; `check_work` reports progress. Nothing is ever pushed or
merged by the machine — Will reviews the branch at a keyboard.

Safety: worktrees only (no access intended outside them), hard timeout,
spoken confirmation required before dispatch (enforced upstream), and the
instructions forbid commissioning based on third-party/web content.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

_TASK_PROMPT = """\
You are working on the codebase of "Alexa", a voice home assistant, in a git
worktree on its own branch. The assistant itself commissioned this work at
its owner's spoken request.

TASK: {request}

Rules:
- Work only inside this directory. Never read or modify .env files, data/,
  or anything outside this worktree.
- Keep scope tight: do the task, nothing speculative.
- If you changed code, run `uv run ruff check src scripts tests` and
  `uv run pytest -q` and fix what they surface.
- Do not push. Do not merge. When finished: `git add -A` and commit with a
  clear message.
- End your reply with a short plain-language summary of what you did, the
  state of tests, and anything the reviewer should look at.
"""


@dataclass
class Job:
    id: str
    title: str
    request: str
    status: str  # running | done | failed | interrupted
    branch: str
    worktree: str
    started: float
    finished: float | None = None
    summary: str = ""


class DispatchError(RuntimeError):
    pass


class Dispatcher:
    def __init__(
        self,
        root: Path,
        *,
        claude_cmd: str = "claude -p --output-format json --dangerously-skip-permissions",
        timeout_s: float = 1500.0,
    ) -> None:
        self._root = root
        self._claude_cmd = claude_cmd
        self._timeout_s = timeout_s
        self._jobs_path = root / "data" / "jobs.json"
        self._jobs: dict[str, Job] = {}
        self._tasks: list[asyncio.Task] = []
        self._load()

    def _load(self) -> None:
        if not self._jobs_path.exists():
            return
        for row in json.loads(self._jobs_path.read_text(encoding="utf-8")):
            job = Job(**row)
            if job.status == "running":  # app restarted mid-job
                job.status = "interrupted"
            self._jobs[job.id] = job

    def _save(self) -> None:
        self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs_path.write_text(
            json.dumps([asdict(j) for j in self._jobs.values()], indent=2), encoding="utf-8"
        )

    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)

    async def start(self, request: str, title: str) -> Job:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32] or "task"
        slug = f"{slug}-{uuid.uuid4().hex[:6]}"
        worktree = self._root / ".worktrees" / slug
        branch = f"alexa/{slug}"
        proc = await asyncio.create_subprocess_exec(
            "git",
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree),
            "main",
            cwd=self._root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise DispatchError(f"could not create worktree: {err.decode(errors='replace')[:200]}")
        job = Job(
            id=slug,
            title=title,
            request=request,
            status="running",
            branch=branch,
            worktree=str(worktree),
            started=time.time(),
        )
        self._jobs[job.id] = job
        self._save()
        self._tasks.append(asyncio.create_task(self._run(job)))
        return job

    async def _run(self, job: Job) -> None:
        try:
            prompt = _TASK_PROMPT.format(request=job.request)
            proc = await asyncio.create_subprocess_shell(
                self._claude_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=job.worktree,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")), timeout=self._timeout_s
                )
            except TimeoutError:
                proc.kill()
                job.status = "failed"
                job.summary = f"timed out after {self._timeout_s:.0f}s"
                return
            if proc.returncode != 0:
                job.status = "failed"
                job.summary = f"agent exited {proc.returncode}: {stderr.decode(errors='replace')[:300]}"
                return
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
            job.summary = str(envelope.get("result", ""))[:2000]
            job.status = "failed" if envelope.get("is_error") else "done"
        except Exception as err:  # noqa: BLE001 — a job may never crash the app
            job.status = "failed"
            job.summary = f"{type(err).__name__}: {err}"
        finally:
            job.finished = time.time()
            self._save()

    def report(self, job_id: str | None = None) -> str:
        jobs = [self._jobs[job_id]] if job_id and job_id in self._jobs else self.jobs()[:5]
        if not jobs:
            return "no development jobs yet"
        out = []
        for j in jobs:
            elapsed = (j.finished or time.time()) - j.started
            out.append(
                {
                    "id": j.id,
                    "title": j.title,
                    "status": j.status,
                    "branch": j.branch,
                    "minutes": round(elapsed / 60, 1),
                    "summary": j.summary[:600],
                }
            )
        return json.dumps(out)
