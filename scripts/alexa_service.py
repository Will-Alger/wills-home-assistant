"""Always-On service: keeps Alexa running forever, invisibly.

Launched at Windows logon (see scripts/install_autostart.ps1) via pythonw so
no console appears. Runs the voice app in a loop: if it crashes, it restarts
with backoff; output goes to logs/alexa.log.

Staging (the self-improvement loop): if data/active_checkout.json points at a
task's worktree, the app is launched FROM that worktree — its code, its docs —
while .env, data/ and logs/ stay here (ALEXA_HOME). A staged build that dies
twice in a row within two minutes is rolled back to main automatically, and
she announces it. This file always runs from main and is stdlib-only, so a
broken branch can never break the thing that rolls it back.

Controls:
    scripts/alexa-stop.cmd     stop the service and the app
    logs/alexa.log             what she's been doing
    data/stop.flag             (how the stop signal travels)
    data/active_checkout.json  which build to run (absent = main)

Testing hook: everything after a `--` argument replaces the child command.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "alexa.log"
PIDFILE = ROOT / "data" / "service.pid"
STOP_FLAG = ROOT / "data" / "stop.flag"
POINTER = ROOT / "data" / "active_checkout.json"
MAX_LOG_BYTES = 10 * 1024 * 1024
HEALTHY_RUN_S = 120.0
CRASHES_BEFORE_ROLLBACK = 2


def read_pointer(root: Path, log: Any = None) -> dict | None:
    """The staged-build pointer, or None for main. Tolerant: a broken file
    or a vanished worktree means main."""
    path = root / "data" / "active_checkout.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as err:
        if log:
            log(f"ignoring unreadable checkout pointer: {err}")
        return None
    worktree = Path(str(data.get("worktree", "")))
    if not data.get("task_id") or not (worktree / "scripts" / "m4_realtime.py").exists():
        if log:
            log(f"ignoring checkout pointer at {worktree}: no runnable app there")
        return None
    data["worktree"] = str(worktree)
    return data


def child_command(root: Path, pointer: dict | None, argv: list[str] | None = None) -> list[str]:
    argv = sys.argv if argv is None else argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    if pointer is not None:
        worktree = Path(pointer["worktree"])
        own_python = worktree / ".venv" / "Scripts" / "python.exe"
        python = own_python if own_python.exists() else root / ".venv" / "Scripts" / "python.exe"
        return [str(python), str(worktree / "scripts" / "m4_realtime.py")]
    return [str(root / ".venv" / "Scripts" / "python.exe"), str(root / "scripts" / "m4_realtime.py")]


def child_env(root: Path, pointer: dict | None, base: dict | None = None) -> dict:
    # Redirected output on Windows defaults to cp1252, which chokes on the
    # app's unicode status glyphs (○ ● ⚙ ✎). Force UTF-8 for the child.
    env = {
        **(os.environ if base is None else base),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "ALEXA_HOME": str(root),  # .env, data/, logs/ always live here
    }
    if pointer is not None:
        worktree = Path(pointer["worktree"])
        env["ALEXA_STAGED_TASK"] = str(pointer["task_id"])
        env["ALEXA_STAGED_ITERATION"] = str(pointer.get("iteration", ""))
        if not (worktree / ".venv" / "Scripts" / "python.exe").exists():
            # main's interpreter, the branch's code: PYTHONPATH beats the
            # editable install's .pth entry
            prior = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(worktree / "src") + (os.pathsep + prior if prior else "")
    return env


def record_exit(state: dict, pointer: dict | None, code: int, ran_for: float) -> str:
    """Count short non-zero exits of a STAGED build; the second in a row
    (same pointer) means roll back. Clean exits and healthy runs reset."""
    if pointer is None or code == 0 or ran_for >= HEALTHY_RUN_S:
        state["crashes"] = 0
        state["pointer_set_at"] = None
        return "ok"
    set_at = pointer.get("set_at")
    if state.get("pointer_set_at") != set_at:
        state["crashes"] = 0
        state["pointer_set_at"] = set_at
    state["crashes"] = state.get("crashes", 0) + 1
    return "rollback" if state["crashes"] >= CRASHES_BEFORE_ROLLBACK else "ok"


def write_inbox(data_dir: Path, text: str, *, kind: str, ref: str, priority: str) -> None:
    """Same drop format as assistant.announce.write_inbox — duplicated here on
    purpose so this file never imports the app."""
    inbox = data_dir / "announcements" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    final = inbox / f"{time.time_ns()}.json"
    tmp = final.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"text": text, "kind": kind, "ref": ref, "priority": priority}),
        encoding="utf-8",
    )
    os.replace(tmp, final)


def rollback(root: Path, pointer: dict, code: int) -> None:
    path = root / "data" / "active_checkout.json"
    failed = path.with_name("active_checkout.failed.json")
    failed.unlink(missing_ok=True)
    path.replace(failed)
    task_id = pointer.get("task_id")
    write_inbox(
        root / "data",
        f"The staged build of task {task_id} crashed twice in a row, exit code {code}, "
        "so I rolled back to the main build. Ask for that task's details to see the error.",
        kind="system",
        ref=f"task:{task_id}:rollback",
        priority="urgent",
    )


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    if LOG.exists() and LOG.stat().st_size > MAX_LOG_BYTES:
        LOG.replace(LOG.with_suffix(".log.old"))

    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    backoff = 5.0
    crash_state: dict = {}
    with LOG.open("a", encoding="utf-8", errors="replace") as log:

        def note(msg: str) -> None:
            log.write(f"\n[service {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            log.flush()

        note(f"service starting (pid {os.getpid()})")
        while not STOP_FLAG.exists():
            pointer = read_pointer(ROOT, note)
            command = child_command(ROOT, pointer)
            if pointer is not None:
                note(f"launching STAGED task {pointer['task_id']} from {pointer['worktree']} with {command[0]}")
            else:
                note("launching main")
            started = time.monotonic()
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=log,
                cwd=ROOT,
                creationflags=creation,
                env=child_env(ROOT, pointer),
            )
            PIDFILE.write_text(f"{os.getpid()}\n{child.pid}", encoding="utf-8")
            code = child.wait()
            ran_for = time.monotonic() - started
            if STOP_FLAG.exists():
                break
            if ran_for > HEALTHY_RUN_S:
                backoff = 5.0  # it was healthy; treat this as a fresh crash
            if record_exit(crash_state, pointer, code, ran_for) == "rollback":
                rollback(ROOT, pointer, code)
                note(f"staged task {pointer['task_id']} crashed twice — rolled back to main")
                backoff = 5.0
                crash_state = {}
            note(f"app exited (code {code}) after {ran_for:.0f}s — restarting in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)
        note("service stopped")
    PIDFILE.unlink(missing_ok=True)
    STOP_FLAG.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
