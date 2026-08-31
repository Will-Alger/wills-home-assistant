"""The brain: session state, system prompt, agent loop, end-of-turn intent.

One Agent = one session = one growing message history. The loop:
user text -> LLM turn -> (tool calls? execute all, feed results back, repeat)
-> final structured reply {speech, intent}.

Intent contract (docs/FEATURES.md): "close" ends the interaction, "listen"
re-opens the mic (the assistant asked something), "confirm_close" asks
"anything else?" and waits briefly. In the M2 REPL the intent is displayed;
the M4 audio layer acts on it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from assistant.brain.tools import TOOL_DEFINITIONS, ToolExecutor
from assistant.home.base import HomeApi
from assistant.llm.base import LLMProvider, TurnResult
from assistant.meter import Meter

MAX_HOPS_PER_COMMAND = 8  # circuit breaker; a lighting command should never loop

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speech": {
            "type": "string",
            "description": "What to say out loud. Short, natural, spoken language.",
        },
        "intent": {"type": "string", "enum": ["close", "listen", "confirm_close"]},
    },
    "required": ["speech", "intent"],
    "additionalProperties": False,
}

_SYSTEM_TEMPLATE = """\
You are {name}, the voice assistant for {owner}'s home. Anyone in the room \
may talk to you — {owner} is simply the household owner. You are spoken and \
heard, never read: replies are natural spoken sentences — no markdown, no \
lists, no emoji. Commands get terse confirmations (~200 characters); real \
conversation can breathe a little, but stay a talker, not a lecturer. Never \
speak the phrase "{wake_phrase}".

You control the home through tools, and you are also good company: general \
conversation — questions, opinions, thinking out loud — is just as much your \
job as commands. For home control there are no canned routines: interpret \
intent and decide. "Get the room ready for a party" means you choose the \
lighting (and later, music). Prefer area targets over individual bulbs, and \
batch every change into a single set_lights call. Only fetch live state when \
the answer depends on it. If something is beyond your tools, say so honestly \
and briefly.

Devices:
{device_table}

Stored preferences:
{preferences}

End every reply with the right intent — it controls the microphone, so read \
the room:
- "close": a command was completed and nothing invites more, the speaker used \
a wrap-up phrase ("that's all", "thanks, that's it", "never mind" — \
acknowledge in a few words), or your answer clearly ends the exchange.
- "listen": you asked the speaker a question, OR the conversation is flowing \
— when someone is chatting rather than commanding, keep the mic open and let \
THEM decide when it's over. Never end a lively conversation yourself.
- "confirm_close": a task finished but a quick "anything else?" fits.
Never ask a question and then use "close".
"""


@dataclass(frozen=True)
class AgentReply:
    speech: str
    intent: str
    hops: int
    cost_usd: float
    latency_ms: int


class Agent:
    def __init__(
        self,
        home: HomeApi,
        llm: LLMProvider,
        meter: Meter,
        *,
        name: str = "Jarvis",
        wake_phrase: str = "hey jarvis",
        owner: str = "the owner",
    ) -> None:
        self._home = home
        self._llm = llm
        self._meter = meter
        self._executor = ToolExecutor(home)
        self._name = name
        self._wake_phrase = wake_phrase
        self._owner = owner
        self._system: str | None = None
        self._messages: list[dict[str, Any]] = []

    async def start_session(self) -> None:
        """Build the static, cacheable system prompt from the device registry."""
        from assistant.home.base import device_table

        self._system = _SYSTEM_TEMPLATE.format(
            name=self._name,
            owner=self._owner,
            wake_phrase=self._wake_phrase,
            device_table=device_table(await self._home.get_lights()),
            preferences="(none stored yet)",  # M6 memory feature fills this in
        )
        self._messages = []

    def reset(self) -> None:
        self._messages = []

    async def handle(self, user_text: str) -> AgentReply:
        if self._system is None:
            await self.start_session()
        assert self._system is not None

        self._meter.start_command()
        self._messages.append({"role": "user", "content": user_text})

        result: TurnResult | None = None
        for _hop in range(MAX_HOPS_PER_COMMAND):
            result = await self._llm.turn(
                system=self._system,
                tools=TOOL_DEFINITIONS,
                messages=self._messages,
                response_schema=RESPONSE_SCHEMA,
            )
            self._meter.record(result)

            if result.stop_reason == "tool_use":
                # Echo raw blocks back verbatim (preserves thinking-block replay),
                # run all calls concurrently, return every result in ONE message.
                self._messages.append({"role": "assistant", "content": result.assistant_content})
                outcomes = await asyncio.gather(
                    *(self._executor.execute(call.name, call.input) for call in result.tool_calls)
                )
                self._messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": text,
                                "is_error": is_error,
                            }
                            for call, (text, is_error) in zip(result.tool_calls, outcomes)
                        ],
                    }
                )
                continue
            break

        assert result is not None
        self._messages.append({"role": "assistant", "content": result.assistant_content})

        if result.stop_reason == "refusal":
            speech, intent = "Sorry — I can't help with that one.", "close"
        elif result.stop_reason == "tool_use":
            speech, intent = "Sorry — that took too many steps, I gave up.", "close"
        else:
            speech, intent = _parse_reply(result.text)

        stats = self._meter.current
        return AgentReply(
            speech=speech,
            intent=intent,
            hops=stats.hops,
            cost_usd=stats.cost_usd,
            latency_ms=stats.latency_ms,
        )


def _parse_reply(text: str) -> tuple[str, str]:
    """Structured outputs guarantee valid JSON; degrade gracefully anyway."""
    try:
        data = json.loads(text)
        intent = data.get("intent", "close")
        if intent not in ("close", "listen", "confirm_close"):
            intent = "close"
        return str(data.get("speech", "")).strip() or "Done.", intent
    except (json.JSONDecodeError, AttributeError):
        return text.strip() or "Done.", "close"
