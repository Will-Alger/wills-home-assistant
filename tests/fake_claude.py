"""Test double for `claude -p --output-format stream-json`: emits event lines."""

import json
import sys

sys.stdin.read()  # consume the prompt like the real CLI would
print(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-fake", "model": "claude-fake-1"}))
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "working on the task now"}]}}))
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "MILESTONE: tests are passing.\nMoving on to the commit."}]}}))
print(
    json.dumps(
        {
            "type": "result",
            "result": "fake agent: did the task, committed.",
            "session_id": "sess-fake",
            "total_cost_usd": 0.01,
        }
    )
)
