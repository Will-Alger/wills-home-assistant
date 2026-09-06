"""Ending a conversation is the engine's call, not the model's: "that's all"
closes after her goodbye whether or not she remembered the end tool, and a
wrap-up that lands with nothing playing closes without a listening window."""

from __future__ import annotations

import asyncio
import time

from assistant.audio.cues import VoiceCues
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


def make_engine(**kw) -> tuple[RealtimeEngine, FakeClient, VoiceCues]:
    played: list[str] = []
    cues = VoiceCues(play=played.append, render=lambda kind, rate: kind.encode())
    engine = RealtimeEngine(
        api_key="test-key", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa",
        wake_phrase="alexa", cues=cues, **kw,
    )
    client = FakeClient()
    engine._client = client
    return engine, client, cues


async def run(engine: RealtimeEngine, client: FakeClient, *, after: float, say: str, reply: str | None):
    speaker = InstantSpeaker()

    async def owner() -> None:
        await asyncio.sleep(after)
        client.connection.user_says(say, reply=reply)

    turn = asyncio.create_task(owner())
    started = time.monotonic()
    stats = await asyncio.wait_for(
        engine.run_conversation(NeverMic(), speaker, None, QuietUi(), announce=False), 8
    )
    await turn
    return stats, time.monotonic() - started, speaker


async def test_thats_all_closes_after_her_goodbye_without_the_end_tool() -> None:
    engine, client, _cues = make_engine()
    stats, took, speaker = await run(engine, client, after=0.2, say="Okay, that's all, thanks!", reply="Bye for now.")
    assert stats.ended_by == "wrap-up"
    assert took < 1.2  # right after the goodbye played — not a quick-close window, not the idle timer
    assert "wake" not in [c.decode() for c in speaker.chunks]  # no listening ding after the goodbye
    assert ("you", "Okay, that's all, thanks!") in stats.transcript


async def test_a_late_wrapup_with_nothing_playing_closes_after_a_short_grace() -> None:
    engine, client, _cues = make_engine()
    stats, took, _speaker = await run(engine, client, after=0.2, say="That's it.", reply=None)
    assert stats.ended_by == "wrap-up"
    assert 1.4 < took < 4.0  # the grace, then closed — never a 45 s listen


async def test_an_ordinary_question_still_opens_the_follow_up_window() -> None:
    engine, client, _cues = make_engine(info_close_s=0.4)
    stats, took, speaker = await run(engine, client, after=0.2, say="What time is it?", reply="It's noon.")
    assert stats.ended_by == "question answered"
    assert "wake" in [c.decode() for c in speaker.chunks]  # she dinged: your turn
    assert took > 0.5
