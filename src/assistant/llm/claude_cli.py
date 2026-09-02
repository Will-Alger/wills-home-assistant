"""LLM provider backed by the Claude Code CLI (`claude -p`).

Why: Will pays for Claude Max — headless CLI calls bill against that
subscription instead of API tokens. Used for background jobs where a few
seconds of process spin-up don't matter (the reflection pass, and later the
dispatch milestone). Not suitable for latency-critical paths.

Tools are not supported here; response_schema is enforced by instruction
(the CLI has no structured-output parameter) with a tolerant JSON extractor.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import json
import time
from typing import Any

from assistant.llm.base import TurnResult, Usage

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # she runs windowless



class ClaudeCliError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    """Return the first {...} block — tolerates prose or code fences around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ClaudeCliError(f"no JSON object in CLI output: {text[:200]!r}")
    return text[start : end + 1]


class ClaudeCli:
    def __init__(self, timeout_s: float = 180.0) -> None:
        self._timeout_s = timeout_s

    async def turn(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
    ) -> TurnResult:
        if tools:
            raise ClaudeCliError("the CLI provider does not support tool use")
        user_text = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        prompt = f"{system}\n\n{user_text}"
        if response_schema is not None:
            prompt += (
                "\n\nRespond with ONLY a JSON object matching this schema — no "
                f"prose, no code fences:\n{json.dumps(response_schema)}"
            )

        started = time.perf_counter()
        process = await asyncio.create_subprocess_shell(
            "claude -p --output-format json",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=NO_WINDOW,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=self._timeout_s
            )
        except TimeoutError as err:
            process.kill()
            raise ClaudeCliError(f"claude -p timed out after {self._timeout_s}s") from err
        if process.returncode != 0:
            raise ClaudeCliError(
                f"claude -p failed ({process.returncode}): {stderr.decode(errors='replace')[:300]}"
            )
        envelope = json.loads(stdout.decode("utf-8"))
        result_text = str(envelope.get("result", ""))
        if response_schema is not None:
            result_text = _extract_json(result_text)
        return TurnResult(
            stop_reason="end_turn",
            text=result_text,
            tool_calls=(),
            assistant_content=[{"type": "text", "text": result_text}],
            usage=Usage(),  # billed to the Max subscription, not per token
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=str(envelope.get("model", "claude-cli")),
        )
