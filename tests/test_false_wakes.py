"""A microphone beside a speaker, in a room with a vacuum cleaner: the
wake word cannot fire on broadband noise, a wake nobody follows up dies
quietly and raises the bar, and the server hears speech or clean silence —
never the room."""

from __future__ import annotations

import asyncio

import numpy as np

from assistant.engines import realtime_engine as mod
from assistant.engines.realtime_engine import RealtimeEngine, _SpeechGate
from assistant.home.fake import FakeHome
from assistant.wake.detector import WakeBackoff
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


def test_two_false_wakes_in_three_minutes_raise_the_bar_for_ten() -> None:
    now = [1000.0]
    backoff = WakeBackoff(now=lambda: now[0])
    assert backoff.extra == 0.0
    assert not backoff.false_wake()  # one is nothing
    now[0] += 60
    assert backoff.false_wake()  # two inside the window: up goes the bar
    assert backoff.extra == 0.15 and 599 < backoff.seconds_left <= 600
    now[0] += 601
    assert backoff.extra == 0.0  # and back down ten minutes later
    assert not backoff.false_wake()
    now[0] += 200  # the window passed: the earlier one no longer counts
    assert not backoff.false_wake()
    backoff.real_wake()
    now[0] += 1
    assert not backoff.false_wake()  # a real wake cleared the count


def test_the_gate_opens_on_speech_with_preroll_and_closes_after_quiet() -> None:
    gate = _SpeechGate()
    frames = {"quiet": b"q" * 10, "hot1": b"h1", "hot2": b"h2", "speech": b"s"}
    t = 0.0
    for _ in range(20):  # a quiet room: nothing goes up
        assert not gate.update(80, t, frames["quiet"])
        t += 0.08
    assert not gate.update(2500, t, frames["hot1"])  # first hot frame: kept as pre-roll
    t += 0.08
    assert gate.update(2600, t, frames["hot2"])  # second: open
    assert gate.take_preroll()[-1] == frames["hot1"] and gate.take_preroll() == []
    t += 0.08
    assert gate.update(1800, t, frames["speech"])  # speaking
    for _ in range(5):  # pauses inside a sentence: still open
        t += 0.08
        assert gate.update(90, t, frames["quiet"])
    t += 0.5  # …until quiet outlasts the hangover
    assert not gate.update(90, t, frames["quiet"])
    assert not gate.open


def test_the_gate_floor_follows_a_noisy_room() -> None:
    gate = _SpeechGate()
    t = 0.0
    for _ in range(400):  # a vacuum cleaner: loud, steady
        gate.update(900, t, b"v")
        t += 0.08
    assert gate.floor > 850  # the floor rose to meet it
    assert not gate.open  # steady noise never opened it
    assert not gate.update(1500, t, b"x") and not gate.update(1500, t + 0.08, b"x")  # 1.7x the floor: not speech


class LevelMic:
    def __init__(self) -> None:
        self.level = 60

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return np.full(1920, self.level, dtype=np.int16).tobytes()

    def drain(self) -> None: ...


def make(**kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa", **kw
    )
    client = FakeClient()
    engine._client = client
    return engine, client


async def test_a_quiet_room_reaches_the_server_as_silence_and_a_voice_as_itself() -> None:
    engine, client = make(idle_timeout_s=0.8)
    mic = LevelMic()

    async def owner() -> None:
        await asyncio.sleep(0.3)
        mic.level = 3000  # he speaks
        await asyncio.sleep(0.2)
        mic.level = 60

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(mic, InstantSpeaker(), None, QuietUi()), 6)
    await turn
    appended = [e["audio"] for e in client.connection.sent if e["type"] == "input_audio_buffer.append"]
    import base64

    frames = [base64.b64decode(a) for a in appended]
    assert frames, "nothing was streamed"
    silence = mod._SILENCE_FRAME
    loud = [i for i, f in enumerate(frames) if np.frombuffer(f, dtype=np.int16).max() >= 3000]
    assert loud, "his voice never went up"
    first = loud[0]
    assert first > mod._GATE_PREROLL  # the room was streamed for a while before he spoke…
    assert all(f == silence for f in frames[: first - mod._GATE_PREROLL])  # …as clean silence
    quiet_frame = np.full(1920, 60, dtype=np.int16).tobytes()
    assert quiet_frame in frames[first - mod._GATE_PREROLL : first]  # the pre-roll carried the onset


async def test_a_wake_nobody_follows_up_dies_quietly(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_NO_SPEECH_S", 0.3)
    engine, client = make(idle_timeout_s=5.0)
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 6)
    assert stats.ended_by == "nobody spoke"
    assert "response.create" not in client.connection.kinds()  # she answered nothing


async def test_a_wake_he_does_follow_up_is_not_a_false_wake(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_NO_SPEECH_S", 0.3)
    engine, client = make(idle_timeout_s=0.6)
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("turn off the hallway", reply="Done.")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 6)
    await turn
    assert stats.replied and stats.ended_by != "nobody spoke"
