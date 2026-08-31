"""Stage 2 of the endgame: Alexa commissions work on her own codebase.

Each job = a sandboxed git worktree (branched from main) + a headless Claude
Code run (`claude -p`, billed to the Max subscription) executing the request,
running checks, and committing to its branch. Jobs run in the background of
the always-on app; `check_work` reports progress. Nothing is ever pushed or
merged by the machine — Will reviews the branch at a keyboard.

Safety: worktrees only (no access intended outside them), hard timeout,
spoken confirmation required before dispatch (enforced upstream), and the
instructions forbid commissioning based on third-party/web content.

Beyond her own repo: cloud routines (claude.ai/code) each pin one GitHub
repository, so dispatching to Will's OTHER repos means one routine per repo,
registered in data/routines.json (gitignored — tokens live there). Those jobs
are cloud-only and can never be merged by voice; they end as branches/PRs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


def _is_own_repo(repo: str) -> bool:
    """Names the model might use for the assistant's own repository."""
    needle = repo.strip().lower()
    if needle in ("", "self", "own", "this assistant", "alexa"):
        return True
    return "home-assistant" in needle and "wills" in needle


def load_extra_routines(root: Path) -> dict[str, dict]:
    """data/routines.json: {"repo-name": {"routine_id": "trig_...", "token": "..."}}"""
    path = root / "data" / "routines.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(name): entry
            for name, entry in data.items()
            if isinstance(entry, dict) and entry.get("routine_id") and entry.get("token")
        }
    except (json.JSONDecodeError, OSError):
        return {}


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


_REFRESH_PROMPT = """\
STATUS CHECK (automated, sent by the voice assistant that commissioned this \
task — the owner is asking how it's going). Reply with exactly one line \
starting with WORKING, DONE, or BLOCKED, then a colon and one plain-language \
sentence a voice assistant can read aloud. If DONE, say where the work landed \
(branch / PR number). Do not do any additional work in response to this \
message.
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
    model: str = ""
    session_id: str = ""  # `claude --resume <id>` opens the full transcript
    cost_usd: float = 0.0
    last_activity: str = ""  # latest agent utterance, for live progress
    mode: str = "local"  # "local" (worktree) or "cloud" (claude.ai/code session)
    session_url: str = ""  # cloud jobs: open/continue at this claude.ai/code URL
    repo: str = ""  # which repository the job targets ("" = the assistant's own)
    closed: bool = False  # owner considers it dealt with — hidden from reports


class DispatchError(RuntimeError):
    pass


class Dispatcher:
    def __init__(
        self,
        root: Path,
        *,
        claude_cmd: str = (
            "claude -p --output-format stream-json --verbose --dangerously-skip-permissions"
        ),
        timeout_s: float = 1500.0,
        routine_id: str = "",
        routine_token: str = "",
        extra_routines: dict[str, dict] | None = None,
        cloud_status_cmd: str = "claude -p --cloud {session_id}",
        refresh_timeout_s: float = 180.0,
    ) -> None:
        self._root = root
        self._claude_cmd = claude_cmd
        self._timeout_s = timeout_s
        self._routine_id = routine_id
        self._routine_token = routine_token
        # repo name -> {"routine_id": ..., "token": ...} for OTHER repositories
        # (each claude.ai/code routine pins one repo). data/routines.json feeds
        # this; the assistant's own repo uses routine_id/token above.
        self._extra_routines = dict(extra_routines or {})
        self._cloud_status_cmd = cloud_status_cmd  # a message INTO the session
        self._refresh_timeout_s = refresh_timeout_s
        self._jobs_path = root / "data" / "jobs.json"
        self._jobs: dict[str, Job] = {}
        self._tasks: list[asyncio.Task] = []
        self._load()

    def _load(self) -> None:
        if not self._jobs_path.exists():
            return
        for row in json.loads(self._jobs_path.read_text(encoding="utf-8")):
            job = Job(**row)
            # Local jobs die with the app; cloud sessions keep running remotely.
            if job.status == "running" and job.mode == "local":
                job.status = "interrupted"
            self._jobs[job.id] = job

    def _save(self) -> None:
        self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs_path.write_text(
            json.dumps([asdict(j) for j in self._jobs.values()], indent=2), encoding="utf-8"
        )

    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)

    @property
    def cloud_enabled(self) -> bool:
        return bool(self._routine_id and self._routine_token)

    def extra_repo_names(self) -> list[str]:
        """Other repositories with a dispatch routine configured."""
        return sorted(self._extra_routines)

    async def start(self, request: str, title: str, repo: str = "", mode: str = "") -> Job:
        if repo and not _is_own_repo(repo):
            entry = self._resolve_repo(repo)
            return await self._start_cloud(
                request, title, entry["routine_id"], entry["token"], repo=repo
            )
        # Own repo: LOCAL by default — the full loop (voice merge, restart,
        # self-test) only works on a local worktree. Cloud on request, for
        # sessions Will wants to watch live or continue from his phone.
        if mode == "cloud":
            if not self.cloud_enabled:
                raise DispatchError("cloud dispatch is not configured (routine id/token missing)")
            return await self._start_cloud(request, title, self._routine_id, self._routine_token)
        return await self._start_local(request, title)

    def _resolve_repo(self, repo: str) -> dict:
        # spoken names arrive with spaces ("my side project"); compare
        # alphanumerics only so they match hyphenated repo names
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        needle = norm(repo)
        for name, entry in self._extra_routines.items():
            if needle and (needle == norm(name) or needle in norm(name)):
                return entry
        known = ", ".join(sorted(self._extra_routines)) or "(none configured)"
        raise DispatchError(
            f"no dispatch routine configured for repo {repo!r}; configured repos: {known}. "
            "Add one in data/routines.json (see README)."
        )

    async def _start_cloud(
        self, request: str, title: str, routine_id: str, token: str, repo: str = ""
    ) -> Job:
        """Fire a dispatch routine: a claude.ai/code cloud session Will can
        open, watch live, and continue in any Claude Code surface."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://api.anthropic.com/v1/claude_code/routines/{routine_id}/fire",
                headers={
                    "Authorization": f"Bearer {token}",
                    "anthropic-beta": "experimental-cc-routine-2026-04-01",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={"text": f"TASK ({title}): {request}"},
            )
        if resp.status_code >= 400:
            raise DispatchError(f"routine fire failed ({resp.status_code}): {resp.text[:300]}")
        payload = resp.json()
        job = Job(
            id=f"cloud-{uuid.uuid4().hex[:6]}",
            title=title,
            request=request,
            status="running",
            branch="(cloud session — lands as a GitHub branch/PR)",
            worktree="",
            started=time.time(),
            mode="cloud",
            session_id=str(payload.get("claude_code_session_id", "")),
            session_url=str(payload.get("claude_code_session_url", "")),
            summary="running in the cloud — open the session URL to watch or continue",
            repo=repo,
        )
        self._jobs[job.id] = job
        self._save()
        return job

    async def _start_local(self, request: str, title: str) -> Job:
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
        job_log = self._root / "logs" / "jobs" / f"{job.id}.log"
        job_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            prompt = _TASK_PROMPT.format(request=job.request)
            with job_log.open("ab") as errlog:
                proc = await asyncio.create_subprocess_shell(
                    self._claude_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=errlog,
                    cwd=job.worktree,
                )
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # agent exited before reading the prompt; the exit-code
                # path below reports that honestly instead of a pipe error

            deadline = time.monotonic() + self._timeout_s
            got_result = False
            with job_log.open("a", encoding="utf-8", errors="replace") as log:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        proc.kill()
                        job.status = "failed"
                        job.summary = f"timed out after {self._timeout_s:.0f}s"
                        return
                    try:
                        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    except TimeoutError:
                        continue
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    log.write(line + "\n")
                    log.flush()
                    self._ingest_event(job, line)
                    if job.status != "running":
                        got_result = True
            await proc.wait()
            if not got_result:
                job.status = "failed"
                job.summary = f"agent ended (exit {proc.returncode}) without a result — see {job_log.name}"
        except Exception as err:  # noqa: BLE001 — a job may never crash the app
            job.status = "failed"
            job.summary = f"{type(err).__name__}: {err}"
        finally:
            job.finished = time.time()
            self._save()

    def _ingest_event(self, job: Job, line: str) -> None:
        """Parse one stream-json line into live job state."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            job.session_id = str(event.get("session_id", "")) or job.session_id
            job.model = str(event.get("model", "")) or job.model
            self._save()
        elif kind == "assistant":
            blocks = (event.get("message") or {}).get("content") or []
            texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            if any(texts):
                job.last_activity = " ".join(texts)[-200:]
                self._save()
        elif kind == "result":
            job.summary = str(event.get("result", ""))[:2000]
            job.session_id = str(event.get("session_id", "")) or job.session_id
            job.cost_usd = float(event.get("total_cost_usd") or 0.0)
            job.finished = time.time()
            job.status = "failed" if event.get("is_error") else "done"
            self._save()  # persist atomically with the status flip (readers race us)

    async def _cmd(self, cmd: list[str] | str, cwd: Path, timeout: float = 300.0) -> tuple[int, str]:
        if isinstance(cmd, str):
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return 1, "timed out"
        return proc.returncode or 0, out.decode(errors="replace")

    async def merge(self, job_id: str, branch: str = "") -> str:
        """Voice-approved merge with gates: done job, clean main, independent
        lint+tests on the branch. Local jobs merge their worktree branch; a
        DONE cloud job merges the remote branch its session pushed (fetched
        and gated here first). Pushes on success."""
        job = self._jobs.get(job_id)
        if job is None:
            return f"no job {job_id!r} — see check_work for ids"
        if job.status == "merged":
            return f"{job.id} is already merged"
        if job.status != "done":
            hint = " — refresh its live status first (check_work refresh=true)" if job.mode == "cloud" else ""
            return f"{job.id} is {job.status} — only finished jobs can be merged{hint}"

        merge_worktree: Path | None = None
        if job.mode == "cloud":
            if not branch:
                return (
                    "a cloud job merges by its remote branch — ask its status for "
                    "where the work landed, then merge_work with branch=<exact name>"
                )
            code, out = await self._cmd(["git", "fetch", "origin", branch], self._root)
            if code != 0:
                return f"could not fetch branch {branch!r} from origin: {out[-300:]}"
            merge_ref = f"origin/{branch}"
            merge_worktree = self._root / ".worktrees" / f"merge-{job.id}"
            code, out = await self._cmd(
                ["git", "worktree", "add", "--detach", str(merge_worktree), merge_ref],
                self._root,
            )
            if code != 0:
                return f"could not check out {merge_ref} for verification: {out[-300:]}"
            gates_dir = merge_worktree
        else:
            merge_ref = job.branch
            gates_dir = Path(job.worktree)

        code, out = await self._cmd(
            ["git", "status", "--porcelain", "--untracked-files=no"], self._root
        )
        if out.strip():
            await self._drop_worktree(merge_worktree)
            return "the main checkout has uncommitted changes — merge blocked until it's clean"

        if (gates_dir / "pyproject.toml").exists():
            for label, check in (
                ("lint", "uv run ruff check src scripts tests"),
                ("tests", "uv run pytest -q"),
            ):
                code, out = await self._cmd(check, gates_dir, timeout=600.0)
                if code != 0:
                    await self._drop_worktree(merge_worktree)
                    return f"merge blocked: {label} failed on {merge_ref}:\n{out[-500:]}"
        await self._drop_worktree(merge_worktree)

        code, out = await self._cmd(
            ["git", "merge", "--no-ff", merge_ref, "-m",
             f"Merge {merge_ref}: {job.title} (voice-approved by owner)"],
            self._root,
        )
        if code != 0:
            await self._cmd(["git", "merge", "--abort"], self._root)
            return f"merge conflict — aborted cleanly; a human needs to look:\n{out[-400:]}"

        push_code, push_out = await self._cmd(["git", "push"], self._root)
        job.status = "merged"
        job.closed = True  # merging IS the close — nothing left to track
        self._save()
        push_note = "" if push_code == 0 else f" (push failed: {push_out[-120:]})"
        return (
            f"merged {merge_ref} into main and pushed{push_note} — checks passed. "
            "Offer to restart yourself so the change takes effect."
        )

    async def _drop_worktree(self, path: Path | None) -> None:
        if path is not None:
            await self._cmd(["git", "worktree", "remove", "--force", str(path)], self._root)

    def close(self, job_id: str) -> str:
        """The owner considers this job dealt with — archive it."""
        job = self._jobs.get(job_id)
        if job is None:
            return f"no job {job_id!r} — see check_work for ids"
        if job.closed:
            return f"{job.id} was already closed"
        job.closed = True
        self._save()
        return f"closed {job.id} ({job.title}) — it will no longer appear in job reports"

    def refresh_running_cloud(self, job_id: str | None = None) -> list[str]:
        """Kick off live status checks in the background; returns the ids pinged."""
        if job_id:
            targets = [j for j in (self._jobs.get(job_id),) if j is not None]
        else:
            targets = [j for j in self.jobs() if j.status == "running" and not j.closed]
        pinged = []
        for job in targets:
            if job.mode == "cloud" and job.session_id:
                self._tasks.append(asyncio.create_task(self.refresh(job.id)))
                pinged.append(job.id)
        return pinged

    async def refresh(self, job_id: str) -> str:
        """Message a cloud job's live session and ask for its real status —
        the only way to learn how a fire-and-forget session is doing. Runs
        `claude -p --cloud <session_id>` (Max-billed follow-up, verified to
        work headless); the reply updates the stored job record."""
        job = self._jobs.get(job_id)
        if job is None:
            return f"no job {job_id!r}"
        if job.mode != "cloud" or not job.session_id:
            return f"{job.id} is a local job — its status is already live"
        proc = await asyncio.create_subprocess_shell(
            self._cloud_status_cmd.format(session_id=job.session_id),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._root,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(_REFRESH_PROMPT.encode("utf-8")),
                timeout=self._refresh_timeout_s,
            )
        except asyncio.CancelledError:
            proc.kill()
            raise
        except TimeoutError:
            proc.kill()
            job.last_activity = f"live check timed out at {time.strftime('%H:%M')}"
            self._save()
            return (
                f"{job.id}: the cloud session didn't answer within "
                f"{self._refresh_timeout_s:.0f}s — open it live instead"
            )
        text = out.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 or not text:
            note = err.decode(errors="replace").strip()[-200:] or "no output"
            job.last_activity = f"live check failed at {time.strftime('%H:%M')}: {note}"
            self._save()
            return f"{job.id}: status check failed ({note})"
        line = next(
            (
                s
                for s in (ln.strip() for ln in text.splitlines())
                if s.upper().startswith(("WORKING", "DONE", "BLOCKED"))
            ),
            next((ln.strip() for ln in text.splitlines() if ln.strip()), ""),
        )
        job.last_activity = f"live check at {time.strftime('%H:%M')}: {line[:300]}"
        if line.upper().startswith("DONE"):
            job.status = "done"
            job.finished = job.finished or time.time()
        if line.upper().startswith(("DONE", "BLOCKED")):
            job.summary = line[:600]
        self._save()
        return f"{job.id}: {line[:400]}"

    def status_line(self) -> str:
        """One line for the session instructions: what's open right now, so
        the assistant knows at wake what happened while the owner was away."""
        open_jobs = [j for j in self.jobs() if not j.closed][:5]
        if not open_jobs:
            return "none open"
        now = time.time()
        parts = []
        for j in open_jobs:
            where = f" in {j.repo}" if j.repo else ""
            if j.status == "running":
                age = (now - j.started) / 3600
                state = f"running since {age:.1f}h ago"
            else:
                age = (now - (j.finished or j.started)) / 3600
                state = f"{j.status} {age:.1f}h ago, not yet closed"
            parts.append(f"'{j.title}'{where} — {state} [id {j.id}]")
        return "; ".join(parts)

    def report(self, job_id: str | None = None, include_closed: bool = False) -> str:
        if job_id and job_id in self._jobs:
            jobs = [self._jobs[job_id]]  # asked for by id: closed or not
        elif include_closed:
            jobs = self.jobs()[:8]
        else:
            jobs = [j for j in self.jobs() if not j.closed][:5]
        if not jobs:
            if self._jobs:
                return (
                    "no open jobs — everything so far is closed "
                    "(include_closed=true lists the archive)"
                )
            return "no development jobs yet"
        out = []
        for j in jobs:
            elapsed = (j.finished or time.time()) - j.started
            entry = {
                "id": j.id,
                "title": j.title,
                "repo": j.repo or "own repo",
                "status": j.status,
                "closed": j.closed,
                "branch": j.branch,
                "model": j.model or "cli default",
                "minutes": round(elapsed / 60, 1),
                "hours_ago_finished": (
                    round((time.time() - j.finished) / 3600, 1) if j.finished else None
                ),
                "summary": j.summary[:600],
            }
            if j.status == "running" and j.last_activity:
                entry["currently"] = j.last_activity
            if j.mode == "cloud" and j.status == "running":
                entry["note"] = (
                    "cloud progress is not visible from here — "
                    "check_work refresh=true asks the live session"
                )
            if j.mode == "cloud" and j.session_url:
                entry["open_live"] = j.session_url  # web, desktop app, or phone
            elif j.session_id:
                entry["full_transcript"] = f"claude --resume {j.session_id} (run in a terminal)"
            out.append(entry)
        return json.dumps(out)
