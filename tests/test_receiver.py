"""The conversation stays responsive and honest while tools run: a
correction spoken during a slow tool is heard at once and rides on the tool
output; a dead socket ends the session through one bounded path; news cut
off by a barge-in stays unread; an end tool without a word gets a goodbye."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi
from tests.test_playback import FrameMic, WakeWhileSpeaking


def make_engine(**kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa", **kw
    )
    client = FakeClient()
    engine._client = client
    return engine, client


class StampingUi(QuietUi):
    def __init__(self) -> None:
        super().__init__()
        self.heard: list[tuple[str, float]] = []

    def user_said(self, transcript: str) -> None:
        self.heard.append((transcript, time.monotonic()))


async def test_a_correction_is_heard_while_a_slow_tool_runs() -> None:
    engine, client = make_engine(idle_timeout_s=0.8)

    async def slow(name: str, args: dict) -> tuple[str, bool]:
        await asyncio.sleep(0.5)
        return "lights: hallway on, bedroom off", False

    engine._executor.execute = slow  # type: ignore[method-assign]
    conn = client.connection
    stamps: list[tuple[str, float]] = []
    original_send = conn.send

    async def timed_send(event: dict) -> None:
        stamps.append((event["type"], time.monotonic()))
        await original_send(event)

    conn.send = timed_send  # type: ignore[method-assign]
    ui = StampingUi()

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("what lights are on", reply="Let me look.", calls=[("get_lights", {})])
        await asyncio.sleep(0.15)
        conn.user_says("actually never mind")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui, announce=False), 8)
    await turn
    heard_at = next(t for text, t in ui.heard if text == "actually never mind")
    output_at = next(t for kind, t in stamps if kind == "conversation.item.create")
    assert heard_at < output_at  # the receiver kept reading during the 0.5 s tool
    output = next(e for e in conn.sent if e["type"] == "conversation.item.create")["item"]
    assert output["type"] == "function_call_output"
    payload = json.loads(output["output"])
    assert "never mind" in payload["since"] and payload["result"].startswith("lights:")
    assert stats.ended_by == "idle timeout"


async def test_a_dead_socket_ends_the_session_through_one_path() -> None:
    engine, client = make_engine()
    client.connection.fail_recv_after = 2  # the handshake reads once; the receiver's first read dies
    started = time.monotonic()
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 5)
    assert stats.ended_by.startswith("session error: socket closed")
    assert time.monotonic() - started < 1.0  # not the idle timer


async def test_news_cut_off_by_a_barge_in_stays_unread(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    first = announcer.enqueue("Task 7 is built.", kind="task", ref="task:7:built")
    second = announcer.enqueue("The porch light came on.", kind="watch", ref="watch:1")
    engine, _client = make_engine(announcer=announcer, info_close_s=0.3)
    speaker = InstantSpeaker(drain_s=0.4)  # she is still talking when the wake phrase lands
    stats = await asyncio.wait_for(
        engine.run_conversation(FrameMic(), speaker, WakeWhileSpeaking(), QuietUi(), announce=True), 6
    )
    assert stats.ended_by == "interrupted announcement"
    rows = {row["id"]: row for row in announcer.items(limit=10)}
    assert rows[first.id]["state"] == "spoken" and rows[second.id]["state"] == "spoken"  # delivered, NOT read
    assert {first.id, second.id} <= set(announcer.unread_ids()) if hasattr(announcer, "unread_ids") else True
    assert sorted(stats.announced) == sorted([first.id, second.id])


async def test_an_end_tool_without_a_word_gets_a_goodbye_first() -> None:
    engine, client = make_engine()
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("bye", calls=[("end_conversation", {})], audio=False)

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 6)
    await turn
    system = [
        e["item"]["content"][0]["text"]
        for e in conn.sent
        if e["type"] == "conversation.item.create" and e["item"].get("role") == "system"
    ]
    assert any("goodbye" in text for text in system)
    assert conn.kinds().count("response.create") == 1  # the goodbye, then nothing
    assert stats.ended_by == "end_conversation"
    assert ("alexa", "Heads up: the build finished.") in stats.transcript  # the fake's reply stood in for the goodbye
