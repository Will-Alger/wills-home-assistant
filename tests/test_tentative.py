"""Talking over her on loudspeakers, tentatively: a sustained rise of the mic
above her own echo holds playback and probes the server; its speech
detection confirms (she stops, his turn), or nothing does and she resumes
where she paused with the probe audio cleared."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from assistant.audio import speaker as speaker_module
from assistant.audio.speaker import Speaker
from assistant.engines import realtime_engine as mod
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, QuietUi
from tests.test_playback import FakeStream


class LevelMic:
    """Frames at whatever level the test sets, one every 10 ms."""

    def __init__(self) -> None:
        self.level = 30  # a quiet room
        self.drained = 0

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return np.full(1920, self.level, dtype=np.int16).tobytes()

    def drain(self) -> None:
        self.drained += 1


class NeverWake:
    def detect(self, frame: bytes) -> bool:
        return False

    def reset(self) -> None: ...


def quick(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_TENTATIVE_COOLDOWN_S", 0.08)
    monkeypatch.setattr(mod, "_CAL_FRAMES", 1)
    monkeypatch.setattr(mod, "_FALSE_ALARM_S", 0.3)


def make(**kw) -> tuple[RealtimeEngine, FakeClient, QuietUi]:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa",
        tentative_interrupt=True, **kw,
    )
    client = FakeClient()
    engine._client = client
    return engine, client, QuietUi()


async def test_a_real_talk_over_is_confirmed_by_the_server_and_takes_his_turn(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(idle_timeout_s=1.0)
    conn = client.connection
    mic, speaker = LevelMic(), InstantSpeaker(played_ms=700, drain_s=2.0)

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("tell me a long story", reply="Once upon a time, in a land far, far away…")
        await asyncio.sleep(0.25)  # past the echo-learning window
        mic.level = 3000  # he talks over her
        deadline = time.monotonic() + 1.0
        while speaker.pauses == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        conn.push("input_audio_buffer.speech_started")  # the server heard speech in the probe
        await asyncio.sleep(0.05)
        mic.level = 30

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, NeverWake(), ui, announce=False), 8)
    await turn
    assert speaker.pauses == 1 and speaker.resumes == 0
    kinds = conn.kinds()
    first_probe = next(i for i, e in enumerate(conn.sent) if e["type"] == "input_audio_buffer.append" and speaker.pauses)
    assert first_probe > kinds.index("response.create") if "response.create" in kinds else True
    truncate = next(e for e in conn.sent if e["type"] == "conversation.item.truncate")
    assert truncate["item_id"] == "item_1" and truncate["audio_end_ms"] == 700
    assert "input_audio_buffer.clear" not in kinds  # confirmed, never cleared
    assert any("possible interruption" in n for n in ui.notes) and any("interrupted by talk-over" in n for n in ui.notes)
    assert stats.ended_by == "idle timeout"


async def test_her_own_echo_confirmed_as_a_talk_over_is_thrown_away(monkeypatch) -> None:
    """The Echo Dot beside the mic: her reply comes back, the guard holds,
    the server's speech detection confirms (it IS speech), and the transcript
    is her own sentence. She used to answer it — and herself — in a loop."""
    quick(monkeypatch)
    engine, client, ui = make(idle_timeout_s=1.0)
    conn = client.connection
    mic, speaker = LevelMic(), InstantSpeaker(played_ms=700, drain_s=2.0)

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("close the panel", reply="Gotcha, the panel is closed again. Nice and tidy.")
        await asyncio.sleep(0.25)
        mic.level = 3000  # her own voice, off the loudspeaker
        deadline = time.monotonic() + 1.0
        while speaker.pauses == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        conn.push("input_audio_buffer.speech_started")  # "confirmed"
        await asyncio.sleep(0.05)
        mic.level = 30
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("response.created")
        conn.push("conversation.item.input_audio_transcription.completed", transcript="The panel is closed again.")
        conn.push_response_done()

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, NeverWake(), ui, announce=False), 8)
    await turn
    assert ("you", "The panel is closed again.") not in stats.transcript
    assert any("her own voice coming back" in n for n in ui.notes)
    assert "response.cancel" in conn.kinds()  # the reply to her own echo never plays


async def test_a_cough_holds_then_resumes_with_the_probe_cleared(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(info_close_s=0.3)
    conn = client.connection
    mic, speaker = LevelMic(), InstantSpeaker(played_ms=700, drain_s=1.2)

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("tell me a long story", reply="Once upon a time, in a land far, far away…")
        await asyncio.sleep(0.25)
        mic.level = 3000  # a bang, three frames long
        await asyncio.sleep(0.05)
        mic.level = 30

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, NeverWake(), ui, announce=False), 8)
    await turn
    assert speaker.pauses == 1 and speaker.resumes == 1
    kinds = conn.kinds()
    assert "input_audio_buffer.clear" in kinds  # the probe audio never became a user turn
    assert "conversation.item.truncate" not in kinds and "response.cancel" not in kinds
    assert any("false alarm" in n for n in ui.notes)
    assert stats.ended_by == "question answered"  # she finished her story


async def test_her_own_echo_never_interrupts_her(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(info_close_s=0.3)
    conn = client.connection
    mic, speaker = LevelMic(), InstantSpeaker(drain_s=0.8)

    async def owner() -> None:
        await asyncio.sleep(0.15)
        mic.level = 1200  # her voice, out of the speaker, into the mic — from the first frame
        conn.user_says("what time is it", reply="It is nine o'clock.")
        await asyncio.sleep(0.9)
        mic.level = 30

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(mic, speaker, NeverWake(), ui, announce=False), 8)
    await turn
    assert speaker.pauses == 0  # steady echo is learned, not mistaken for him


async def test_the_speaker_holds_and_resumes_in_place(monkeypatch) -> None:
    monkeypatch.setattr(speaker_module.sd, "RawOutputStream", FakeStream)
    monkeypatch.setattr(speaker_module.devices, "find", lambda spec, kind: None)
    spk = Speaker(24_000)
    await spk.open()
    spk.begin_item("a")
    spk.enqueue(b"\x01" * 4800)
    spk._consume(2400)
    spk.pause()
    held = spk._consume(2400)  # silence (one LSB of keepalive noise, inaudible), nothing advanced
    assert len(held) == 2400 and int(np.abs(np.frombuffer(held, dtype=np.int16)).max()) <= 1
    assert spk.played_ms("a") == 50
    spk.resume()
    assert spk._consume(2400) == b"\x01" * 2400 and spk.played_ms("a") == 100
    await spk.close()
