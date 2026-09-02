"""Test double for `claude -p --output-format stream-json`: emits event lines.

Modes (env FAKE_CLAUDE_MODE), with FAKE_CLAUDE_STATE naming a marker file for
the "-once" behaviors:
  ""              normal: init, progress, a MILESTONE line, result
  slow            normal, but sleeps 3 s between init and result
  hang-no-init    prints nothing for 6 s, then exits 0 (a CLI that never came up)
  no-result-once  first run: init + progress then exit 0 with no result;
                  later runs: normal (a Max-window death, then a good retry)
  no-session-once first RESUMED run: exit 1 with no init (session not found);
                  later runs: normal
  ask-once        first run: init + progress, then a trailing QUESTION: line
                  and a result (the agent stopped for the owner); later runs
                  (the resume with his answer): normal
"""

import json
import os
import sys
import time
from pathlib import Path

argv = sys.argv
mode = os.environ.get("FAKE_CLAUDE_MODE", "")
marker = Path(os.environ["FAKE_CLAUDE_STATE"]) if os.environ.get("FAKE_CLAUDE_STATE") else None
resumed = "--resume" in argv
preassigned = argv[argv.index("--session-id") + 1] if "--session-id" in argv else ""


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


if mode == "hang-no-init":
    time.sleep(6)
    sys.exit(0)

sys.stdin.read()  # consume the prompt like the real CLI would

if mode == "no-session-once" and resumed and marker is not None and not marker.exists():
    marker.write_text("used", encoding="utf-8")
    sys.exit(1)

session = "sess-fake"
emit({"type": "system", "subtype": "init", "session_id": session, "model": "claude-fake-1"})
emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "working on the task now"}]}})

if mode == "no-result-once" and marker is not None and not marker.exists():
    marker.write_text("used", encoding="utf-8")
    sys.exit(0)  # died without a result

if mode == "ask-once" and marker is not None and not marker.exists():
    marker.write_text("used", encoding="utf-8")
    emit(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "I need a decision before the greeting can be written.\n"
                        "QUESTION: should the greeting use his first name or his full name?",
                    }
                ]
            },
        }
    )
    emit(
        {
            "type": "result",
            "result": "fake agent: stopped to ask the owner a question.",
            "session_id": session,
            "total_cost_usd": 0.005,
        }
    )
    sys.exit(0)

if mode == "slow":
    time.sleep(3)

emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "MILESTONE: tests are passing.\nMoving on to the commit."}]}})
emit(
    {
        "type": "result",
        "result": "fake agent: did the task, committed."
        + (" (resumed the earlier session)" if resumed else "")
        + (f" (session {preassigned})" if preassigned else ""),
        "session_id": session,
        "total_cost_usd": 0.01,
    }
)
