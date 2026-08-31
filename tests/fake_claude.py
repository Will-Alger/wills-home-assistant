"""Test double for `claude -p --output-format json`: prints a result envelope."""

import json
import sys

sys.stdin.read()  # consume the prompt like the real CLI would
print(json.dumps({"type": "result", "result": "fake agent: did the task, committed."}))
