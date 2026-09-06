"""Learning Loop tests — stubbed LLM, temp memory store, no network."""

from __future__ import annotations

import json

from assistant.learning import Reflector
from assistant.llm.base import TurnResult, Usage
from assistant.memory import (
    INFERENCE_CONFIDENCE,
    LESSON_MIN_CONFIDENCE,
    VERIFIED_CONFIDENCE,
    MemoryStore,
)


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
            "lessons": [
                {
                    "text": "Music plays on media_player.living_room_apple_tv, not the TV entity.",
                    "subject": "music",
                    "evidence": "tool",
                }
            ],
            "observations": ["Asks for jazz in the afternoon."],
        }
    )
    reflection = await Reflector(llm, memory).reflect(TRANSCRIPT)

    assert reflection.lessons and reflection.observations
    assert [x.text for x in memory.items("episode")] == ["Played jazz after a player mixup."]
    assert len(memory.items("lesson")) == 1
    assert len(memory.items("observation")) == 1
    # A tool outcome in the transcript backed it up, so it is injected.
    lesson = memory.items("lesson")[0]
    assert lesson.confidence == VERIFIED_CONFIDENCE
    assert lesson.subject == "music" and lesson.source == "reflection"
    assert "living_room_apple_tv" in memory.lessons_text()
    assert "jazz in the afternoon" in memory.observations_text()


async def test_reflection_dedupes_and_skips_empty_sessions(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "m.json")
    llm = StubReflectLLM(
        {"summary": "Same again.", "lessons": ["Wake the TV before AirPlay."], "observations": []}
    )
    reflector = Reflector(llm, memory)
    await reflector.reflect(TRANSCRIPT)
    once = memory.items("lesson")[0]
    # A payload with no evidence at all (the older shape) is an inference:
    # stored, but held too low to be injected on the strength of one night.
    assert once.confidence == INFERENCE_CONFIDENCE
    assert "AirPlay" not in memory.lessons_text()

    await reflector.reflect(TRANSCRIPT)
    assert len(memory.items("lesson")) == 1  # deduped by exact text
    # Learned a second time is its own evidence: now it is a house truth.
    twice = memory.items("lesson")[0]
    assert twice.confidence >= LESSON_MIN_CONFIDENCE
    assert twice.last_verified and "AirPlay" in memory.lessons_text()

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

OUTAGE = [
    ("you", "play some jazz"),
    ("tool play_music", "ERROR: the music provider timed out"),
    ("alexa", "Music isn't answering right now."),
]


async def test_one_bad_night_does_not_become_a_house_truth(tmp_path) -> None:
    """A temporary outage, reflected once, is stored but never injected —
    even when reflection insists a tool proved it, because nothing in that
    transcript ever worked."""
    memory = MemoryStore(tmp_path / "m.json")
    llm = StubReflectLLM(
        {
            "summary": "Music was down.",
            "lessons": [
                {
                    "text": "Music playback is unavailable in this house.",
                    "subject": "music",
                    "evidence": "tool",
                }
            ],
            "observations": [],
        }
    )
    await Reflector(llm, memory).reflect(OUTAGE)

    lesson = memory.items("lesson")[0]
    assert lesson.confidence == INFERENCE_CONFIDENCE  # claimed, not corroborated
    assert lesson.text not in memory.lessons_text()
    assert memory.lessons_text() == "(none yet)"
