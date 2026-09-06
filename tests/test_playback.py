"""What he heard, not what was generated: per-item playback accounting on
the speaker, the truncate event after a barge-in, one microphone serving
idle and talk, and stalls that can no longer hang a persistent stream."""

from __future__ import annotations

import asyncio

from assistant.app import wait_for_trigger
from assistant.audio import speaker as speaker_module
from assistant.audio.cues import VoiceCues
from assistant.audio.speaker import Speaker
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, QuietUi
from tests.test_app import NOISE, WAKE, Source, Wake


class FakeStream:
    def __init__(self, *, device=None, **_kw) -> None:
        self.device = device

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


async def test_speaker_accounts_per_item_what_was_actually_played(monkeypatch) -> None:
    monkeypatch.setattr(speaker_module.sd, "RawOutputStream", FakeStream)
    monkeypatch.setattr(speaker_module.sd, "query_devices", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(speaker_module.devices, "find", lambda spec, kind: None)
    spk = Speaker(24_000)
    await spk.open()
    ms = 48  # bytes per millisecond at 24 kHz int16 mono
    spk.begin_item("a")
    spk.enqueue(b"\x00" * (100 * ms))
    spk._consume(50 * ms)  # the device took 50 ms
    assert spk.played_ms("a") == 50
    spk.begin_item("b")
    spk.begin_item("b")  # idempotent
    spk.enqueue(b"\x00" * (100 * ms))
    spk._consume(100 * ms)  # the rest of a, half of b
    assert spk.played_ms("a") == 100 and spk.played_ms("b") == 50 and spk.played_ms() == 50
    spk.clear()  # barge-in: the unplayed half of b is gone
    assert spk.pending_seconds == 0 and spk.played_ms("b") == 50
    spk._consume(100 * ms)  # silence now
    assert spk.played_ms("b") == 50 and spk.played_ms("nope") == 0
    assert not spk.stalled  # nothing waiting
    await spk.close()
    assert not spk.is_open


async def test_a_dead_speaker_cannot_hang_wait_idle(monkeypatch) -> None:
    monkeypatch.setattr(speaker_module.sd, "RawOutputStream", FakeStream)
    monkeypatch.setattr(speaker_module.devices, "find", lambda spec, kind: None)
    monkeypatch.setattr(speaker_module, "_STALL_S", 0.05)
    spk = Speaker(24_000)
    await spk.open()
    spk.enqueue(b"\x00" * 48_000)  # a second of audio nobody will take
    await asyncio.wait_for(spk.wait_idle(), timeout=2)  # returns instead of waiting forever
    assert spk.pending_seconds == 0
    await spk.close()


class FrameMic:
    """A microphone that keeps producing frames, like the real one."""

    def __init__(self) -> None:
        self.drained = 0

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return b"\x00\x00" * 1920

    def drain(self) -> None:
        self.drained += 1


class WakeWhileSpeaking:
    """Fires once, the first time the engine asks during her speech."""

    def __init__(self) -> None:
        self.fired = 0

    def detect(self, frame: bytes) -> bool:
        self.fired += 1
        return self.fired == 1

    def reset(self) -> None: ...


async def test_barge_in_tells_the_server_how_much_he_heard() -> None:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa",
        wake_phrase="alexa", info_close_s=0.3, cues=VoiceCues(play=lambda k: None, render=lambda k, r: b""),
    )
    client = FakeClient()
    engine._client = client
    # her reply arrives in one burst but takes 0.4 s to play out — the wake
    # phrase lands while it is still audible, after the server is done
    speaker = InstantSpeaker(played_ms=1234, drain_s=0.4)

    async def owner() -> None:
        await asyncio.sleep(0.15)
        client.connection.user_says("tell me a story", reply="Once upon a time, in a land far away")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(
        engine.run_conversation(FrameMic(), speaker, WakeWhileSpeaking(), QuietUi(), announce=False), 6
    )
    await turn
    assert speaker.items == ["item_1"]  # the delta's item was registered before its audio
    truncate = next(e for e in client.connection.sent if e["type"] == "conversation.item.truncate")
    assert truncate == {"type": "conversation.item.truncate", "item_id": "item_1", "content_index": 0, "audio_end_ms": 1234}
    assert "response.cancel" not in client.connection.kinds()  # nothing left to cancel: the response had completed
    assert all(chunk == b"" for chunk in speaker.chunks)  # her audio was cleared; only the (empty) cue remains
    assert stats.ended_by != "unknown"


async def test_one_microphone_serves_idle_through_a_converter() -> None:
    seen: list[bytes] = []

    def convert(frame: bytes) -> bytes:
        seen.append(frame)
        return WAKE if frame == b"raw-wake" else frame

    assert await wait_for_trigger(Source([NOISE, b"raw-wake"]), Wake(), None, convert=convert) == "wake"
    assert seen == [NOISE, b"raw-wake"]


async def test_a_silent_microphone_reports_a_stall() -> None:
    class Never:
        async def get_frame(self) -> bytes:
            await asyncio.sleep(3600)
            return b""

    assert await wait_for_trigger(Never(), Wake(), None, stall_s=0.05) == "stalled"
    assert await wait_for_trigger(Source([WAKE]), Wake(), None, stall_s=0.5) == "wake"  # frames flow: no stall
