"""The brain layer: the voice hands a hard question to slow reasoning and the
answer comes back as an event she speaks — without ever blocking the voice."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.brain.thinker import Thinker
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.llm.base import TurnResult, Usage
from assistant.memory import MemoryStore


class SlowLLM:
    """Records the prompt, answers after a beat like a real Opus call would."""

    def __init__(self, answer: str = "Go with the Ecobee: it is the cheapest of the three.") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def turn(self, *, system, tools, messages, response_schema=None) -> TurnResult:
        self.calls.append({"system": system, "messages": messages})
        await asyncio.sleep(0.05)
        return TurnResult(
            stop_reason="end_turn", text=self.answer, tool_calls=(),
            assistant_content=[], usage=Usage(), model="claude-fake",
        )


async def test_thinker_builds_context_and_answers_for_speech(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.json")
    memory.add("preference", "keep the living room warm in the evening")
    memory.add("fact", "the thermostat is an Ecobee")
    llm = SlowLLM()
    thinker = Thinker(llm, memory=memory, name="Alexa", owner="Will")
    answer = await thinker.think(
        "which thermostat should I buy", transcript=[("you", "I'm choosing a thermostat"), ("alexa", "Okay.")]
    )
    assert answer.startswith("Go with the Ecobee")
    (call,) = llm.calls
    assert "Answer FOR SPEECH" in call["system"] and "Will" in call["system"]
    prompt = call["messages"][0]["content"]
    assert "keep the living room warm" in prompt and "the thermostat is an Ecobee" in prompt
    assert "I'm choosing a thermostat" in prompt and prompt.endswith("Question: which thermostat should I buy")


async def test_think_tool_returns_at_once_and_the_answer_arrives_as_an_event(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json", quiet_hours="00:00-23:59")  # always quiet: urgent must bypass
    llm = SlowLLM()
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        announcer=announcer, thinker=Thinker(llm, owner="Will"),
    )
    engine._live_transcript = [("you", "which thermostat should I buy?")]
    text, is_error = await engine._execute_brain_tool("think", {"question": "which thermostat should I buy"})
    assert not is_error and "thinking" in text.lower()
    assert not announcer.pending()  # nothing yet: the voice is free to keep talking
    await asyncio.gather(*engine._thinking)
    (item,) = announcer.pending()
    assert item.kind == "thought" and item.priority == "urgent"
    assert "Go with the Ecobee" in item.text
    assert announcer.due()  # spoken even inside quiet hours: the owner asked


async def test_think_without_an_announcer_answers_inline(tmp_path: Path) -> None:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", thinker=Thinker(SlowLLM()),
    )
    text, is_error = await engine._execute_brain_tool("think", {"question": "why is the sky blue"})
    assert not is_error and "Ecobee" in text  # falls back to waiting for the answer


async def test_think_without_a_brain_says_so() -> None:
    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will")
    text, is_error = await engine._execute_brain_tool("think", {"question": "anything"})
    assert is_error and "not available" in text
