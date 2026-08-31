"""Always-On service: keeps Alexa running forever, invisibly.

Launched at Windows logon (see scripts/install_autostart.ps1) via pythonw so
no console appears. Runs the voice app in a loop: if it crashes, it restarts
with backoff; output goes to logs/alexa.log.

Controls:
    scripts/alexa-stop.cmd     stop the service and the app
    logs/alexa.log             what she's been doing
    data/stop.flag             (how the stop signal travels)

Testing hook: everything after a `--` argument replaces the child command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "alexa.log"
PIDFILE = ROOT / "data" / "service.pid"
STOP_FLAG = ROOT / "data" / "stop.flag"
MAX_LOG_BYTES = 10 * 1024 * 1024


def child_command() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    return [str(python), str(ROOT / "scripts" / "m4_realtime.py")]


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    if LOG.exists() and LOG.stat().st_size > MAX_LOG_BYTES:
        LOG.replace(LOG.with_suffix(".log.old"))

    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    backoff = 5.0
    with LOG.open("a", encoding="utf-8", errors="replace") as log:
        def note(msg: str) -> None:
            log.write(f"\n[service {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            log.flush()

        note(f"service starting (pid {os.getpid()})")
        while not STOP_FLAG.exists():
            started = time.monotonic()
            child = subprocess.Popen(
                child_command(), stdout=log, stderr=log, cwd=ROOT, creationflags=creation
            )
            PIDFILE.write_text(f"{os.getpid()}\n{child.pid}", encoding="utf-8")
            code = child.wait()
            ran_for = time.monotonic() - started
            if STOP_FLAG.exists():
                break
            if ran_for > 120:
                backoff = 5.0  # it was healthy; treat this as a fresh crash
            note(f"app exited (code {code}) after {ran_for:.0f}s — restarting in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)
        note("service stopped")
    PIDFILE.unlink(missing_ok=True)
    STOP_FLAG.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
