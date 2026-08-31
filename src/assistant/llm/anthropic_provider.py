"""Anthropic adapter for the LLM interface.

Design notes (see docs/FEATURES.md):
- Prompt caching: one breakpoint on the system block with a 1-hour TTL —
  commands arrive sporadically, so the default 5-minute TTL would rarely hit.
  The breakpoint caches everything before it (tools render first), so tools +
  system are one cached prefix. Nothing volatile may appear in either.
- Thinking: omitted on purpose — claude-opus-5 runs adaptive thinking by
  default; effort (from config) is the depth control.
- Server-side refusal fallbacks are enabled ("default" routing) so a safety
  decline re-runs on a fallback model inside the same call instead of
  dead-ending the voice loop.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

from assistant.llm.base import ToolCall, TurnResult, Usage


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        effort: str,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        # Zero-arg client resolves ANTHROPIC_API_KEY / an `ant auth login`
        # profile; an explicit key (from .env) wins when provided.
        # Identity-linked keys require the workspace header on every request;
        # SDK 1.2 has no first-class option, so it rides as a default header.
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = model
        self._effort = effort

    async def turn(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
    ) -> TurnResult:
        output_config: dict[str, Any] = {"effort": self._effort}
        if response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": response_schema}

        started = time.perf_counter()
        response = await self._client.beta.messages.create(
            model=self._model,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config=output_config,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            tools=tools,
            messages=messages,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = tuple(
            ToolCall(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        return TurnResult(
            stop_reason=response.stop_reason or "end_turn",
            text=text,
            tool_calls=tool_calls,
            assistant_content=response.content,
            usage=usage,
            latency_ms=latency_ms,
            model=response.model,
        )
