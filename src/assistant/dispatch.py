"""The runner beneath the task board (see tasks.py): git worktrees, headless
Claude Code runs billed to the Max subscription, merge gates, cloud routines.

This module knows nothing about tasks or announcements — it does one unit of
mechanical work at a time and reports back:

- `create_worktree` / `remove_worktree` / `commit_all` — git plumbing.
- `run_agent` — one `claude -p` run streamed as stream-json into an `AgentRun`
  (session id, model, progress lines, MILESTONE: lines, result, cost). Pass
  `resume_session_id` to continue an earlier run with new instructions.
- `merge_branch` — the voice-merge gates: clean main, ruff + pytest on the
  branch (`uv run`, falling back to `python -m` when uv is missing — see
  `resolve_gate_runner`), `git merge --no-ff`, push. `fetch_remote_branch`
  prepares a cloud session's pushed branch for the same gates.
- `fire_cloud` / `refresh_cloud` — claude.ai/code routines (one per repo).

Safety unchanged: worktrees only, hard timeout, the agent is told never to
touch .env/data or push/merge. Spoken confirmation is enforced upstream.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


def _is_own_repo(repo: str) -> bool:
    """Names the model might use for the assistant's own repository."""
    needle = repo.strip().lower()
    if needle in ("", "self", "own", "this assistant", "alexa"):
        return True
    return "home-assistant" in needle and "wills" in needle


# Asked of a fallback interpreter before the merge gates trust it with ruff and
# pytest: importing them would be slower and could fail for unrelated reasons.
GATE_TOOLS_PROBE = (
    "import importlib.util as u, sys; "
    "sys.exit(0 if u.find_spec('ruff') and u.find_spec('pytest') else 1)"
)

CLOUD_ROUTINES_FILE = "cloud_routines.json"


def load_extra_routines(root: Path) -> dict[str, dict]:
    """data/cloud_routines.json: {"repo-name": {"routine_id": "trig_...", "token": "..."}}
    (renamed from routines.json, which the behavior RoutineStore now owns)."""
    path = root / "data" / CLOUD_ROUTINES_FILE
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


def migrate_cloud_routines(root: Path) -> str:
    """One-time move: an old data/routines.json holding cloud-routine
    credentials becomes data/cloud_routines.json. A behavior-routine file
    (next_id/routines keys) is left alone. Returns a note, or ''."""
    old = root / "data" / "routines.json"
    new = root / "data" / CLOUD_ROUTINES_FILE
    if not old.exists() or new.exists():
        return ""
    try:
        data = json.loads(old.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    if not isinstance(data, dict) or not data or "routines" in data or "next_id" in data:
        return ""
    if not all(isinstance(v, dict) and v.get("routine_id") for v in data.values()):
        return ""
    os.replace(old, new)
    return f"moved cloud routine credentials to data/{CLOUD_ROUTINES_FILE}"


_RULES = """\
Rules:
- Work only inside this directory. Never read or modify .env files, data/,
  logs/, or anything outside this worktree.
- Keep scope tight: do the task, nothing speculative. One feature per task.
- If you changed code, run `uv run ruff check src scripts tests` and
  `uv run pytest -q` and fix what they surface. Tests use fakes — never
  require live services or spend API money.
- Do not push. Do not merge. When finished: `git add -A` and commit with a
  clear message.
- Progress: when you reach a meaningful milestone (plan settled, core code
  in place, tests passing, committed), write one line on its own starting
  with `MILESTONE:` followed by one short plain sentence — at most three per
  task. The owner hears these read aloud.
- Blocked on a decision only the owner can make? Write ONE line on its own
  starting with `QUESTION:` — one plain sentence naming the options — and
  then STOP: end your reply immediately, doing nothing else. You will be
  resumed with his answer. Never guess on such a decision; never ask more
  than one question at a time; never ask what the spec already answers.
- End your reply with a short plain-language summary of what you did, the
  state of tests, and how the owner should test it by voice.
"""

_TASK_PROMPT = (
    """\
You are working on the codebase of "Alexa", a voice home assistant, in a git
worktree on its own branch. The assistant herself commissioned this work at
her owner's spoken request. The spec is committed at {spec_path} — it says
how the owner will test the result by voice; build to that.

TASK {task_id}: {title}

{spec}

"""
    + _RULES
)

_REVISE_PROMPT = (
    """\
You are continuing your own earlier work on this branch of "Alexa", a voice
home assistant. The owner tested it by voice and reports:

{feedback}

The spec at {spec_path} now carries this as "Revision {n}". Fix it on this
branch, keeping what already works.

"""
    + _RULES
)

_ANSWER_PROMPT = (
    """\
You are continuing your own earlier work on this branch of "Alexa", a voice
home assistant. You stopped to ask the owner:

{question}

The owner answers: {answer}

The spec at {spec_path} records this as "Answer {n}". Continue the task from
where you stopped, keeping what already works.

"""
    + _RULES
)

_REFRESH_PROMPT = """\
STATUS CHECK (automated, sent by the voice assistant that commissioned this \
task — the owner is asking how it's going). Reply with exactly one line \
starting with WORKING, DONE, or BLOCKED, then a colon and one plain-language \
sentence a voice assistant can read aloud. If DONE, say where the work landed \
(branch / PR number). Do not do any additional work in response to this \
message.
"""


@dataclass
class GateRunner:
    """How the merge gates run ruff and pytest on a checkout."""

    prefix: list[str]  # [uv, "run"] or [python, "-m"]
    env: dict[str, str] | None = None  # set on the fallback path only
    note: str = ""  # "" behind uv; names the fallback otherwise, to be spoken

    def command(self, *args: str) -> list[str]:
        return [*self.prefix, *args]


@dataclass
class AgentRun:
    """Live state of one headless agent process."""

    status: str = "running"  # running | done | failed
    session_id: str = ""
    model: str = ""
    summary: str = ""
    cost_usd: float = 0.0
    progress: list[str] = field(default_factory=list)  # recent agent utterances
    milestones: list[str] = field(default_factory=list)  # MILESTONE: lines seen
    question: str = ""  # a trailing QUESTION: line — the agent stopped for the owner
    returncode: int | None = None
    pid: int = 0  # the detached process, so a restarted app can re-attach
    saw_init: bool = False  # the CLI came up and reported a session


class DispatchError(RuntimeError):
    pass


def pid_alive(pid: int) -> bool:
    """Is that process still running? (Windows has no kill(pid, 0).)"""
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    """Stop a detached agent and its children (the shell wrapper + node)."""
    if not pid:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            creationflags=NO_WINDOW,
        )
    else:
        import signal

        with contextlib_suppress(OSError):
            os.kill(int(pid), signal.SIGTERM)


def contextlib_suppress(*exc):
    import contextlib

    return contextlib.suppress(*exc)


# She runs with no console (the watchdog hides it); a console child spawned
# without this flag pops up a visible terminal window on Windows.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_DETACH_FLAGS = (
    # NOT DETACHED_PROCESS: a console-less parent makes Windows open a VISIBLE
    # console for every child it starts. CREATE_NO_WINDOW gives the agent its own
    # hidden console, and a child outlives its parent on Windows anyway.
    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    if sys.platform == "win32"
    else 0
)


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
        model: str = "",
        effort: str = "",
    ) -> None:
        self.root = root
        # the coding agent's model/effort ride on the command line (Opus by
        # default from settings; the CLI default otherwise)
        self._claude_cmd = (
            claude_cmd
            + (f" --model {model}" if model else "")
            + (f" --effort {effort}" if effort else "")
        )
        self._timeout_s = timeout_s
        self._routine_id = routine_id
        self._routine_token = routine_token
        # repo name -> {"routine_id": ..., "token": ...} for OTHER repositories
        # (each claude.ai/code routine pins one repo). data/cloud_routines.json
        # feeds this; the assistant's own repo uses routine_id/token above.
        self._extra_routines = dict(extra_routines or {})
        self._cloud_status_cmd = cloud_status_cmd  # a message INTO the session
        self._refresh_timeout_s = refresh_timeout_s

    # ── git plumbing ────────────────────────────────────────────────────────

    async def create_worktree(self, slug: str) -> tuple[Path, str]:
        """A fresh branch alexa/<slug> checked out under .worktrees/<slug>."""
        worktree = self.root / ".worktrees" / slug
        branch = f"alexa/{slug}"
        code, out = await self._cmd(
            ["git", "worktree", "add", "-b", branch, str(worktree), "main"], self.root
        )
        if code != 0:
            raise DispatchError(f"could not create worktree: {out[-300:]}")
        return worktree, branch

    async def remove_worktree(self, path: Path | str) -> bool:
        code, _ = await self._cmd(["git", "worktree", "remove", "--force", str(path)], self.root)
        await self._cmd(["git", "worktree", "prune"], self.root)
        return code == 0

    async def commit_all(self, cwd: Path, message: str) -> bool:
        await self._cmd(["git", "add", "-A"], cwd)
        code, _ = await self._cmd(["git", "commit", "-q", "-m", message], cwd)
        return code == 0

    async def _cmd(
        self,
        cmd: list[str] | str,
        cwd: Path,
        timeout: float = 300.0,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        if isinstance(cmd, str):
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                creationflags=NO_WINDOW, env=env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                creationflags=NO_WINDOW, env=env,
            )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return 1, "timed out"
        return proc.returncode or 0, out.decode(errors="replace")

    # ── the coding agent ────────────────────────────────────────────────────

    async def run_agent(
        self,
        *,
        cwd: Path,
        prompt: str,
        log_path: Path,
        resume_session_id: str = "",
        session_id: str = "",
        on_update: Callable[[AgentRun, str], None] | None = None,
        init_timeout_s: float = 90.0,
    ) -> AgentRun:
        """Launch `claude -p` DETACHED in `cwd` (it outlives this app, so a
        staging restart doesn't kill a build) with its output going to
        `log_path`, then follow that log. `resume_session_id` continues an
        earlier run; `session_id` pre-assigns the id of a fresh one. Calls
        on_update(run, kind) per event (spawn|init|progress|milestone|result)."""
        run = AgentRun(session_id=resume_session_id or session_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._claude_cmd
        if resume_session_id:
            cmd += f" --resume {resume_session_id}"
        elif session_id:
            cmd += f" --session-id {session_id}"
        offset = log_path.stat().st_size if log_path.exists() else 0
        try:
            with log_path.open("ab") as out:
                out.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {cmd}\n".encode())
                out.flush()
                offset = out.tell()
                proc = subprocess.Popen(  # noqa: ASYNC220 — a detached launch; returns at once
                    cmd,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=out,
                    cwd=cwd,
                    creationflags=_DETACH_FLAGS,
                )
            run.pid = proc.pid
            try:
                assert proc.stdin is not None
                proc.stdin.write(prompt.encode("utf-8"))
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # agent exited before reading the prompt; reported below
            if on_update is not None:
                on_update(run, "spawn")
        except Exception as err:  # noqa: BLE001 — a launch failure is a failed run
            run.status = "failed"
            run.summary = f"could not start the coding agent: {type(err).__name__}: {err}"
            return run
        return await self.follow(
            run, log_path, offset=offset, on_update=on_update, init_timeout_s=init_timeout_s
        )

    async def follow(
        self,
        run: AgentRun,
        log_path: Path,
        *,
        offset: int = 0,
        on_update: Callable[[AgentRun, str], None] | None = None,
        init_timeout_s: float = 90.0,
    ) -> AgentRun:
        """Tail a running (or finished) agent's log until it reports a result
        or its process is gone. Also how a restarted app re-attaches."""
        deadline = time.monotonic() + self._timeout_s
        init_deadline = time.monotonic() + init_timeout_s
        buffer = b""
        try:
            while True:
                try:
                    with log_path.open("rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                except OSError:
                    chunk = b""
                if chunk:
                    offset += len(chunk)
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for raw in lines:
                        kind = self._ingest(run, raw.decode("utf-8", errors="replace").rstrip())
                        if kind == "init":
                            run.saw_init = True
                        if kind and on_update is not None:
                            on_update(run, kind)
                        if run.status != "running":
                            return run
                    continue  # drain everything available before checking liveness
                if not pid_alive(run.pid):
                    run.status = "failed"
                    run.summary = (
                        "the coding agent ended without a result"
                        + ("" if run.saw_init else " (it never started a session)")
                        + f" — see {log_path.name}"
                    )
                    return run
                now = time.monotonic()
                if not run.saw_init and now > init_deadline:
                    kill_tree(run.pid)
                    run.status = "failed"
                    run.summary = (
                        f"the coding agent didn't start within {init_timeout_s:.0f}s — "
                        "is the Claude CLI installed and logged in?"
                    )
                    return run
                if now > deadline:
                    kill_tree(run.pid)
                    run.status = "failed"
                    run.summary = f"timed out after {self._timeout_s:.0f}s"
                    return run
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            run.status = "running"  # the agent keeps going detached; re-attach later
            raise

    @staticmethod
    def _ingest(run: AgentRun, line: str) -> str:
        """Parse one stream-json line into the run; returns the event kind."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return ""
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            run.session_id = str(event.get("session_id", "")) or run.session_id
            run.model = str(event.get("model", "")) or run.model
            return "init"
        if kind == "assistant":
            blocks = (event.get("message") or {}).get("content") or []
            texts = [
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            ]
            if not any(texts):
                return ""
            joined = " ".join(texts)
            run.progress = (run.progress + [joined[-200:]])[-10:]
            result = "progress"
            # only a TRAILING question parks the task: an agent that asks and
            # then keeps working has answered itself
            run.question = ""
            for raw in joined.splitlines():
                said = raw.strip()
                if said.upper().startswith("MILESTONE:"):
                    run.milestones.append(said[10:].strip())
                    result = "milestone"
                elif said.upper().startswith("QUESTION:"):
                    run.question = said[9:].strip()
            if run.question:
                result = "question"
            return result
        if kind == "result":
            run.summary = str(event.get("result", ""))[:2000]
            run.session_id = str(event.get("session_id", "")) or run.session_id
            run.cost_usd = float(event.get("total_cost_usd") or 0.0)
            run.status = "failed" if event.get("is_error") else "done"
            return "result"
        return ""

    # ── dependencies ────────────────────────────────────────────────────────

    @staticmethod
    def resolve_uv(configured: str = "") -> str | None:
        """uv.exe: the configured path, PATH, then the WinGet install dir."""
        if configured and Path(configured).exists():
            return configured
        found = shutil.which("uv")
        if found:
            return found
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            hits = glob.glob(str(Path(local) / "Microsoft" / "WinGet" / "Packages" / "astral-sh.uv_*" / "uv.exe"))
            if hits:
                return hits[0]
        return None

    async def uv_sync(self, cwd: Path, uv_exe: str = "") -> tuple[bool, str]:
        """Make a checkout's .venv match its lockfile (a branch may add deps)."""
        uv = self.resolve_uv(uv_exe)
        if uv is None:
            return False, "uv was not found (set UV_EXE in .env)"
        code, out = await self._cmd([uv, "sync", "--quiet"], cwd, timeout=600.0)
        return code == 0, out[-300:].strip()

    @staticmethod
    def _venv_python(cwd: Path) -> str | None:
        """A checkout's own interpreter, if it has one."""
        for rel in (Path("Scripts") / "python.exe", Path("bin") / "python"):
            candidate = cwd / ".venv" / rel
            if candidate.exists():
                return str(candidate)
        return None

    async def _runs_the_gate_tools(self, python: str, cwd: Path) -> bool:
        """Can this interpreter run ruff and pytest as modules?"""
        code, _ = await self._cmd([python, "-c", GATE_TOOLS_PROBE], cwd, timeout=60.0)
        return code == 0

    async def resolve_gate_runner(self, gates_dir: Path, uv_exe: str = "") -> GateRunner:
        """How to run ruff and pytest on a checkout.

        `uv run` first, found the way uv_sync finds it (UV_EXE, PATH, then the
        WinGet folder) — a bare "uv" in a shell fails whenever her service
        starts without uv on PATH, and that blocked every merge with a lint
        error that had nothing to do with the branch. When uv is nowhere, fall
        back to an interpreter that already has both tools — the branch's own
        .venv, else the one she is running in — as `python -m ruff` /
        `python -m pytest`, with the branch's own `src` pinned on PYTHONPATH:
        her interpreter's editable install points at MAIN's src, so without the
        pin the gates would happily check code the branch never changed (the
        same trick alexa_service uses to run a staged build). Raises
        DispatchError when nothing here can run the checks at all."""
        uv = self.resolve_uv(uv_exe)
        if uv is not None:
            return GateRunner([uv, "run"])
        prior = os.environ.get("PYTHONPATH", "")
        env = {
            **os.environ,
            "PYTHONPATH": str(gates_dir / "src") + (os.pathsep + prior if prior else ""),
        }
        for python, where in (
            (self._venv_python(gates_dir), "the branch's own virtual environment"),
            (sys.executable, "her own Python"),
        ):
            if python and await self._runs_the_gate_tools(python, gates_dir):
                return GateRunner(
                    [python, "-m"],
                    env=env,
                    note=f"uv was not found, so the checks ran with {where}",
                )
        raise DispatchError(
            "the checks could not run at all: uv was not found on PATH, at UV_EXE, or in "
            "its WinGet folder, and no Python here has both ruff and pytest. Install uv "
            "and set UV_EXE in .env, or install ruff and pytest into the branch's "
            "virtual environment."
        )

    # ── merge gates ─────────────────────────────────────────────────────────

    async def merge_branch(
        self, *, merge_ref: str, gates_dir: Path, title: str, uv_exe: str = ""
    ) -> tuple[bool, str]:
        """Clean main, lint + tests on the branch, merge --no-ff, push."""
        code, out = await self._cmd(
            ["git", "status", "--porcelain", "--untracked-files=no"], self.root
        )
        if out.strip():
            return False, "the main checkout has uncommitted changes — merge blocked until it's clean"
        note = ""  # how the checks ran, when it wasn't the usual `uv run`
        if (gates_dir / "pyproject.toml").exists():
            try:
                gate = await self.resolve_gate_runner(gates_dir, uv_exe)
            except DispatchError as exc:
                return False, f"merge blocked on {merge_ref}: {exc}"
            note = gate.note
            for label, check in (
                ("lint", ["ruff", "check", "src", "scripts", "tests"]),
                ("tests", ["pytest", "-q"]),
            ):
                code, out = await self._cmd(
                    gate.command(*check), gates_dir, timeout=600.0, env=gate.env
                )
                if code != 0:
                    aside = f" ({note})" if note else ""
                    return False, (
                        f"merge blocked: {label} failed on {merge_ref}{aside}:\n{out[-500:]}"
                    )
        code, out = await self._cmd(
            ["git", "merge", "--no-ff", merge_ref, "-m",
             f"Merge {merge_ref}: {title} (voice-approved by owner)"],
            self.root,
        )
        if code != 0:
            await self._cmd(["git", "merge", "--abort"], self.root)
            return False, f"merge conflict — aborted cleanly; a human needs to look:\n{out[-400:]}"
        push_code, push_out = await self._cmd(["git", "push"], self.root)
        push_note = "" if push_code == 0 else f" (push failed: {push_out[-120:]})"
        aside = f" ({note})" if note else ""
        return True, f"merged {merge_ref} into main and pushed{push_note} — checks passed{aside}."

    async def fetch_remote_branch(self, branch: str, slug: str) -> tuple[str, Path]:
        """A cloud session pushed `branch`: fetch it and check it out detached
        under .worktrees/merge-<slug> so the gates can run on it."""
        code, out = await self._cmd(["git", "fetch", "origin", branch], self.root)
        if code != 0:
            raise DispatchError(f"could not fetch branch {branch!r} from origin: {out[-300:]}")
        ref = f"origin/{branch}"
        path = self.root / ".worktrees" / f"merge-{slug}"
        code, out = await self._cmd(
            ["git", "worktree", "add", "--detach", str(path), ref], self.root
        )
        if code != 0:
            raise DispatchError(f"could not check out {ref} for verification: {out[-300:]}")
        return ref, path

    # ── cloud routines ──────────────────────────────────────────────────────

    @property
    def cloud_enabled(self) -> bool:
        return bool(self._routine_id and self._routine_token)

    @property
    def own_routine(self) -> tuple[str, str]:
        return self._routine_id, self._routine_token

    def extra_repo_names(self) -> list[str]:
        """Other repositories with a dispatch routine configured."""
        return sorted(self._extra_routines)

    def resolve_repo(self, repo: str) -> dict:
        """Spoken names arrive with spaces ("my side project"); compare
        alphanumerics only so they match hyphenated repo names."""

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        needle = norm(repo)
        for name, entry in self._extra_routines.items():
            if needle and (needle == norm(name) or needle in norm(name)):
                return entry
        known = ", ".join(sorted(self._extra_routines)) or "(none configured)"
        raise DispatchError(
            f"no dispatch routine configured for repo {repo!r}; configured repos: {known}. "
            f"Add one in data/{CLOUD_ROUTINES_FILE} (see README)."
        )

    async def fire_cloud(
        self, *, request: str, title: str, routine_id: str, token: str
    ) -> dict[str, str]:
        """Fire a routine: a claude.ai/code session Will can open, watch live,
        and continue in any Claude Code surface."""
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
        return {
            "session_id": str(payload.get("claude_code_session_id", "")),
            "session_url": str(payload.get("claude_code_session_url", "")),
        }

    async def refresh_cloud(self, session_id: str) -> str:
        """Message a cloud session and ask for its real status — the only way
        to learn how a fire-and-forget session is doing. Returns one line
        starting WORKING/DONE/BLOCKED (or an honest failure sentence)."""
        proc = await asyncio.create_subprocess_shell(
            self._cloud_status_cmd.format(session_id=session_id),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.root,
            creationflags=NO_WINDOW,
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
            return (
                f"BLOCKED: the cloud session didn't answer within "
                f"{self._refresh_timeout_s:.0f}s — open it live instead"
            )
        text = out.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 or not text:
            note = err.decode(errors="replace").strip()[-200:] or "no output"
            return f"BLOCKED: status check failed ({note})"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return next(
            (ln for ln in lines if ln.upper().startswith(("WORKING", "DONE", "BLOCKED"))),
            lines[0] if lines else "BLOCKED: empty reply",
        )
