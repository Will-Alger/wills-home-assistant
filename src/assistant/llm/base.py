"""Provider-agnostic LLM interface.

One `turn()` = one model call inside the agent loop. The agent owns the loop
(tool dispatch, history, metering); providers own wire formats. Message
history is stored in the active provider's native format — a session is bound
to one provider, and a future OpenAI adapter translates at its own boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class TurnResult:
    stop_reason: str  # "tool_use" | "end_turn" | "refusal" | "max_tokens" | ...
    text: str
    tool_calls: tuple[ToolCall, ...]
    assistant_content: Any  # raw provider blocks; append verbatim to history
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    model: str = ""


class LLMProvider(Protocol):
    async def turn(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
    ) -> TurnResult: ...
