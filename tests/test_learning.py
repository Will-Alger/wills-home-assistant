"""Learning Loop tests — stubbed LLM, temp memory store, no network."""

from __future__ import annotations

import json

from assistant.learning import Reflector
from assistant.llm.base import TurnResult, Usage
from assistant.memory import MemoryStore


class StubReflectLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    async def turn(self, *, system, tools, messages, response_schema=None) -> TurnResult:
        self.calls += 1
        return TurnResult(
            stop_reason="end_turn",
            text=json.dumps(self._payload),
            tool_calls=(),
            assistant_content=[],
            usage=Usage(),
            model="claude-opus-5",
        )


TRANSCRIPT = [
    ("you", "play some jazz"),
    ("tool play_music", "ERROR: player 'tv' is ambiguous"),
    ("tool play_music", "Started on Apple TV."),
    ("alexa", "Jazz is on."),
]


async def test_reflection_stores_all_kinds(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "m.json")
    llm = StubReflectLLM(
        {
            "summary": "Played jazz after a player mixup.",
            "lessons": ["Music plays on media_player.living_room_apple_tv, not the TV entity."],
            "observations": ["Asks for jazz in the afternoon."],
        }
    )
    reflection = await Reflector(llm, memory).reflect(TRANSCRIPT)

    assert reflection.lessons and reflection.observations
    assert [x.text for x in memory.items("episode")] == ["Played jazz after a player mixup."]
    assert len(memory.items("lesson")) == 1
    assert len(memory.items("observation")) == 1
    assert "living_room_apple_tv" in memory.lessons_text()
    assert "jazz in the afternoon" in memory.observations_text()


async def test_reflection_dedupes_and_skips_empty_sessions(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "m.json")
    llm = StubReflectLLM(
        {"summary": "Same again.", "lessons": ["Wake the TV before AirPlay."], "observations": []}
    )
    reflector = Reflector(llm, memory)
    await reflector.reflect(TRANSCRIPT)
    await reflector.reflect(TRANSCRIPT)
    assert len(memory.items("lesson")) == 1  # deduped by exact text

    # a transcript with no user turns is not worth an API call
    llm.calls = 0
    await reflector.reflect([("alexa", "hello?")])
    assert llm.calls == 0


def test_lessons_render_is_capped(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "m.json")
    for i in range(25):
        memory.add("lesson", f"lesson number {i}")
    rendered = memory.lessons_text(limit=15)
    assert "lesson number 24" in rendered
    assert "lesson number 0" not in rendered