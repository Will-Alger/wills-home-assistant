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
    """Without a VAD the level is all there is: a vacuum switched on beside
    a cold gate opens it (louder than the room was — the price of a bar low
    enough to hear an ordinary voice), and the floor climbs until the noise
    IS the room and the gate closes on it, inside half a minute."""
    gate = _SpeechGate()
    t = 0.0
    opened = closed = None
    for i in range(800):  # a vacuum cleaner: loud, steady, for a minute
        gate.update(900, t, b"v")
        if gate.open and opened is None:
            opened = i
        if opened is not None and closed is None and not gate.open:
            closed = i
        t += 0.08
    assert opened is not None and closed is not None and closed * 0.08 < 30
    assert gate.floor > 850 and not gate.open  # the floor rose to meet it
    assert not gate.update(1500, t, b"x") and not gate.update(1500, t + 0.08, b"x")  # 1.7x the floor: not speech


def test_with_a_vad_the_gate_opens_on_quiet_speech_and_only_the_end_of_speech_closes_it() -> None:
    """Recorded at the desk: room floor 330, his ordinary voice 500–700. No
    level ratio tells those apart — Silero does."""
    gate = _SpeechGate()
    t = 0.0
    for _ in range(30):
        assert not gate.update(330, t, b"room", 0.02)
        t += 0.08
    assert not gate.update(520, t, b"a", 0.85)  # first frame of his voice: kept as pre-roll
    t += 0.08
    assert gate.update(540, t, b"b", 0.9)  # second: open, at 1.6x the room
    for _ in range(4):
        t += 0.08
        assert gate.update(500, t, b"c", 0.7)
    t += 0.08
    assert gate.update(900, t, b"vac", 0.05)  # a vacuum after the sentence: loud, not speech — the hangover runs
    t += 0.9
    assert not gate.update(900, t, b"vac", 0.05) and not gate.open  # …and closes on it
    for _ in range(10):  # still loud, still not speech: stays closed
        t += 0.08
        assert not gate.update(900, t, b"vac", 0.05)


def test_without_a_vad_the_level_bar_hears_an_ordinary_voice_over_a_desk_floor() -> None:
    gate = _SpeechGate()
    t = 0.0
    for _ in range(40):
        gate.update(330, t, b"room")
        t += 0.08
    assert 300 < gate.floor < 360
    assert not gate.update(620, t, b"a") and gate.update(640, t + 0.08, b"b")  # 1.9x the floor: him
    gate2 = _SpeechGate()
    for _ in range(40):
        gate2.update(330, t, b"room")
    assert not gate2.update(420, t, b"x") and not gate2.update(420, t + 0.08, b"x")  # the fan getting louder: not


class LevelMic:
    def __init__(self) -> None:
        self.level = 60

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return np.full(1920, self.level, dtype=np.int16).tobytes()

    def drain(self) -> None: ...


class FakeSilero:
    """A wake detector whose VAD calls anything above 400 RMS speech."""

    has_vad = True

    def speech_probability(self, frame16k: bytes) -> float:
        return 0.9 if np.abs(np.frombuffer(frame16k, dtype=np.int16)).max() > 400 else 0.05

    def detect(self, frame16k: bytes) -> bool:
        return False


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


async def test_a_quiet_voice_reaches_the_server_when_silero_says_it_is_one(tmp_path) -> None:
    """Room 330, voice 520: the level alone (3x, the old bar) never opened
    the gate — three commands died that way — and Silero opens it."""
    engine, client = make(idle_timeout_s=0.8)
    mic = LevelMic()
    mic.level = 330
    taps: list[tuple[str, dict]] = []
    engine.tap = lambda kind, **f: taps.append((kind, f))

    async def owner() -> None:
        await asyncio.sleep(0.3)
        mic.level = 520
        await asyncio.sleep(0.2)
        mic.level = 330

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, InstantSpeaker(), None, QuietUi()), 6)
    await turn
    assert not any(k == "gate" for k, _ in taps)  # no VAD: 520 over a 330 floor is not 1.8x — deaf
    assert stats.heard_speech is False

    engine, client = make(idle_timeout_s=0.8)
    mic = LevelMic()
    mic.level = 330
    taps.clear()
    engine.tap = lambda kind, **f: taps.append((kind, f))
    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, InstantSpeaker(), FakeSilero(), QuietUi()), 6)
    await turn
    opened = [f for k, f in taps if k == "gate" and f["open"]]
    assert opened and opened[0]["speech"] == 0.9 and opened[0]["level"] == 520
    assert stats.heard_speech is True
    import base64

    frames = [base64.b64decode(e["audio"]) for e in client.connection.sent if e["type"] == "input_audio_buffer.append"]
    assert any(np.frombuffer(f, dtype=np.int16).max() == 520 for f in frames)  # his voice went up


async def test_a_wake_where_someone_was_heard_is_never_a_false_wake(monkeypatch) -> None:
    """The watch used to kill a long first sentence at twelve seconds, mid-
    word, because no transcript had come yet. The moment anyone is heard —
    by the server, or just by the local gate — it stands down."""
    monkeypatch.setattr(mod, "_NO_SPEECH_S", 0.3)
    engine, _client = make(idle_timeout_s=1.0)
    mic = LevelMic()
    mic.level = 3000  # talking from the first frame, and the server never commits
    stats = await asyncio.wait_for(engine.run_conversation(mic, InstantSpeaker(), None, QuietUi()), 6)
    assert stats.ended_by == "idle timeout" and stats.heard_speech


async def test_her_own_acknowledgment_is_dropped_but_his_words_over_it_are_kept() -> None:
    """The frames of her "Yes?" are flagged, not thrown away: the loud ones
    (her, off a loudspeaker) become silence, the quiet ones (him, talking
    over her) go through — "Alexa, let me think" used to lose "let me think"."""

    class OverlapMic(LevelMic):
        def __init__(self) -> None:
            super().__init__()
            self.script = [2200, 2200, 650, 650, 650, 650] + [60] * 200  # her echo, then him
            self.last_suspect = False

        async def get_frame(self) -> bytes:
            await asyncio.sleep(0.01)
            level = self.script.pop(0) if self.script else 60
            self.last_suspect = len(self.script) > 194  # the six frames of the acknowledgment window
            return np.full(1920, level, dtype=np.int16).tobytes()

    engine, client = make(idle_timeout_s=0.8)
    taps: list[tuple[str, dict]] = []
    engine.tap = lambda kind, **f: taps.append((kind, f))
    await asyncio.wait_for(engine.run_conversation(OverlapMic(), InstantSpeaker(), None, QuietUi()), 6)
    assert [f["level"] for k, f in taps if k == "ack_echo_dropped"] == [2200, 2200]
    import base64

    frames = [base64.b64decode(e["audio"]) for e in client.connection.sent if e["type"] == "input_audio_buffer.append"]
    peaks = [int(np.frombuffer(f, dtype=np.int16).max()) for f in frames]
    assert 2200 not in peaks and 650 in peaks  # her echo never went up; his words did


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
