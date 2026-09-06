""""Let me think" is a request for patience: for a while no clock closes the
conversation and the server is asked to wait longer for his turn; his next
real sentence restores the normal pace."""

from __future__ import annotations

import asyncio
import time

from assistant.engines import realtime_engine as mod
from assistant.engines.realtime_engine import RealtimeEngine, is_thinking
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


def test_thinking_phrases_are_recognised_whole() -> None:
    for said in ("Let me think.", "hmm, let me think about that", "Hang on a second.", "Give me a moment", "one sec", "Okay hold on", "I'm thinking"):
        assert is_thinking(said), said
    for said in ("Let me think about whether to move.", "hold on to the railing", "wait a second, what's the weather", "", "yes"):
        assert not is_thinking(said), said


async def test_let_me_think_holds_every_clock_then_his_next_turn_restores_pace(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_THINKING_S", 0.9)
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa",
        info_close_s=0.2, idle_timeout_s=0.4, eagerness="high", turn_detection="semantic_vad",
    )
    client = FakeClient()
    engine._client = client
    conn = client.connection
    ui = QuietUi()

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("Should I move to Denver?", reply="That depends on the commute.")
        await asyncio.sleep(0.3)
        conn.user_says("hmm, let me think", reply="Take your time.")

    started = time.monotonic()
    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    took = time.monotonic() - started
    assert took > 0.45 + 0.9  # the 0.4 s idle timer was held for the whole thinking window
    assert stats.ended_by == "idle timeout"
    updates = [
        e["session"]["audio"]["input"]["turn_detection"]["eagerness"]
        for e in conn.sent
        if e["type"] == "session.update" and "audio" in e["session"] and "turn_detection" in e["session"]["audio"]["input"]
    ]
    assert updates[-1] == "low"  # asked the server to wait longer
    assert any("patient" in n for n in ui.notes)


async def test_his_next_sentence_restores_the_normal_pace(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_THINKING_S", 5.0)
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa",
        idle_timeout_s=0.4, eagerness="high", turn_detection="semantic_vad",
    )
    client = FakeClient()
    engine._client = client
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("let me think", reply="Sure.")
        await asyncio.sleep(0.2)
        conn.user_says("okay, the commute is the problem", reply="Then Denver is a stretch.")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 8)
    await turn
    assert stats.ended_by == "idle timeout"  # closed on the normal clock, not after 5 s
    eager = [
        e["session"]["audio"]["input"]["turn_detection"]["eagerness"]
        for e in conn.sent
        if e["type"] == "session.update" and "audio" in e["session"] and "turn_detection" in e["session"]["audio"]["input"]
    ]
    assert eager[-2:] == ["low", "high"]
