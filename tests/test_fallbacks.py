"""The four lines she can say with the network unplugged.

Nothing here opens an audio device: playback is a list the test reads back.
The WAVs themselves are the committed ones, so a missing or re-rendered
asset fails here rather than in the living room.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from assistant.audio.fallbacks import FILES, LINES, RATE, SpokenFallbacks, kind_for, voice_dir
from assistant.engines.realtime_engine import RealtimeEngine, SessionStats
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient


class Recorder:
    """Somewhere for the audio to go, instead of a speaker."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []  # (bytes, rate)

    def __call__(self, pcm: bytes, rate: int) -> None:
        self.calls.append((len(pcm), rate))


class FakeSpeaker:
    """The session's open output stream, as far as a fallback can tell."""

    def __init__(self, is_open: bool = True) -> None:
        self.is_open = is_open
        self.queued: list[bytes] = []

    def enqueue(self, pcm: bytes) -> None:
        self.queued.append(pcm)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def player(**kw) -> tuple[SpokenFallbacks, Recorder]:
    recorder = Recorder()
    return SpokenFallbacks(play=recorder, **kw), recorder


# ── the assets ────────────────────────────────────────────────────────────


def test_every_line_is_committed_and_loads_as_speaker_ready_audio() -> None:
    fallbacks, recorder = player()
    assert fallbacks.missing() == []  # all four WAVs are in the repo
    assert fallbacks.note() == ""
    clock = Clock()
    for kind in LINES:
        one, played = player(now=clock)
        assert one.say(kind) == FILES[kind]
        assert len(played.calls) == 1
        length, rate = played.calls[0]
        assert rate == RATE
        assert length % 2 == 0
        assert length / 2 / RATE > 0.5  # a spoken sentence, not a click
    assert recorder.calls == []


def test_the_catalogue_and_the_directory_agree() -> None:
    assert set(LINES) == set(FILES)
    assert sorted(p.name for p in voice_dir().glob("*.wav")) == sorted(FILES.values())


def test_a_line_that_will_not_load_is_silence_not_a_crash(tmp_path) -> None:
    fallbacks, recorder = player(directory=tmp_path)
    assert fallbacks.say("voice_service") == ""
    assert recorder.calls == []
    assert sorted(fallbacks.missing()) == sorted(FILES.values())
    assert "render_fallbacks" in fallbacks.note()


def test_an_unknown_kind_says_nothing() -> None:
    fallbacks, recorder = player()
    assert fallbacks.say("out_of_credits") == ""
    assert recorder.calls == []


# ── where the audio goes ──────────────────────────────────────────────────


def test_an_open_session_speaker_gets_the_line_instead_of_a_new_stream() -> None:
    fallbacks, recorder = player()
    speaker = FakeSpeaker()
    assert fallbacks.say("moment", speaker) == "moment.wav"
    assert len(speaker.queued) == 1
    assert recorder.calls == []  # nothing raced the live PortAudio stream


def test_a_closed_speaker_falls_back_to_its_own_stream() -> None:
    fallbacks, recorder = player()
    speaker = FakeSpeaker(is_open=False)
    assert fallbacks.say("moment", speaker) == "moment.wav"
    assert speaker.queued == []
    assert len(recorder.calls) == 1


# ── one failure, one line ─────────────────────────────────────────────────


def test_a_second_line_for_the_same_failure_is_dropped() -> None:
    clock = Clock()
    fallbacks, recorder = player(now=clock, min_gap_s=8.0)
    assert fallbacks.say("failed") == "failed.wav"
    clock.t += 0.2  # the runner's recovery path sees the same collapse
    assert fallbacks.say("voice_service") == ""
    assert len(recorder.calls) == 1
    clock.t += 10.0  # a new failure, later
    assert fallbacks.say("voice_service") == "voice-service.wav"
    assert fallbacks.played == ["failed.wav", "voice-service.wav"]


# ── which line ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "err",
    [
        ConnectionRefusedError(61, "connection refused"),
        OSError("[Errno 11001] getaddrinfo failed"),
        TimeoutError("timed out"),
        httpx.ConnectError("all connection attempts failed"),
    ],
)
def test_anything_that_could_not_be_reached_is_the_voice_service_line(err: Exception) -> None:
    assert kind_for(err) == "voice_service"


def test_a_wrapped_connection_error_is_still_the_voice_service_line() -> None:
    wrapped = RuntimeError("Realtime session config could not be applied")
    wrapped.__cause__ = ConnectionResetError("connection reset")
    assert kind_for(wrapped) == "voice_service"


def test_anything_else_is_the_check_the_log_line() -> None:
    import sounddevice as sd

    # A microphone that will not open is not the network: PortAudioError
    # subclasses Exception alone, and must never claim the socket is down.
    assert kind_for(sd.PortAudioError("Error opening InputStream")) == "failed"
    assert kind_for(ValueError("nope")) == "failed"


# ── the runner's recovery path ────────────────────────────────────────────


def test_the_recovery_path_speaks_the_voice_service_line_for_a_connection_error() -> None:
    import scripts.m4_realtime as runner

    clock = Clock()
    fallbacks, recorder = player(now=clock)
    speaker = FakeSpeaker()
    err = ConnectionError("[Errno 11001] getaddrinfo failed")

    delay = runner.recover(err, 1, None, fallbacks, speaker)

    assert fallbacks.played == ["voice-service.wav"]
    assert len(speaker.queued) == 1  # through the speaker already open
    assert recorder.calls == []
    assert delay == 5.0


def test_the_recovery_path_plays_the_error_tone_before_the_line() -> None:
    import scripts.m4_realtime as runner

    class Cues:
        def __init__(self) -> None:
            self.errors: list[str] = []

        def error(self, message: str = "", speaker=None) -> None:
            self.errors.append(message)

    cues = Cues()
    fallbacks, _recorder = player(now=Clock())
    runner.recover(OSError("no route to host"), 1, cues, fallbacks, None)
    assert cues.errors and "no route" in cues.errors[0]
    assert fallbacks.played == ["voice-service.wav"]  # the tone still plays too


def test_a_retry_streak_repeats_neither_the_tone_nor_the_line() -> None:
    import scripts.m4_realtime as runner

    clock = Clock()
    fallbacks, recorder = player(now=clock)
    delays = []
    for attempt in range(1, 6):
        clock.t += 60.0  # long past the one-line gap: only the streak stops it
        delays.append(runner.recover(OSError("still down"), attempt, None, fallbacks, None))

    assert fallbacks.played == ["voice-service.wav"]
    assert len(recorder.calls) == 1
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0]  # backs off, never spins


def test_a_dead_microphone_is_not_blamed_on_the_voice_service() -> None:
    import sounddevice as sd

    import scripts.m4_realtime as runner

    fallbacks, _recorder = player(now=Clock())
    runner.recover(sd.PortAudioError("Error opening InputStream"), 1, None, fallbacks, None)
    assert fallbacks.played == ["failed.wav"]


# ── a home that is not answering ──────────────────────────────────────────


class DeadHome(FakeHome):
    """Home Assistant unplugged: nothing reaches it at all."""

    async def get_lights(self):
        raise httpx.ConnectError("All connection attempts failed")


async def test_a_home_command_with_the_hub_down_says_so_off_the_disk() -> None:
    fallbacks, _recorder = player(now=Clock())
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=DeadHome(), owner="Will",
        fallbacks=fallbacks,
    )
    client = FakeClient()
    speaker = FakeSpeaker()
    engine._speaker = speaker

    item = SimpleNamespace(
        type="function_call",
        name="set_lights",
        arguments=json.dumps({"changes": [{"target": "Hallway", "turn": "on"}]}),
        call_id="c1",
    )
    event = SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))
    await engine._handle_response_done(client.connection, event, SessionStats())

    assert fallbacks.played == ["home-down.wav"]
    assert len(speaker.queued) == 1  # through the speaker she is already using
    sent = [
        e for e in client.connection.sent
        if e["type"] == "conversation.item.create"
        and e["item"].get("type") == "function_call_output"
    ]
    payload = json.loads(sent[0]["item"]["output"])
    assert payload["status"] == "unavailable"
    assert payload["details"]["home_unreachable"] is True
    # she said the line already: the model must not read it out a second time
    assert "ALREADY said" in payload["follow_up"]


async def test_a_command_the_hub_refuses_is_not_a_spoken_fallback() -> None:
    fallbacks, recorder = player(now=Clock())
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        fallbacks=fallbacks,
    )
    client = FakeClient()
    item = SimpleNamespace(
        type="function_call",
        name="set_lights",
        arguments=json.dumps({"changes": [{"target": "Narnia", "turn": "on"}]}),
        call_id="c1",
    )
    event = SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))
    await engine._handle_response_done(client.connection, event, SessionStats())

    assert fallbacks.played == []  # a name she cannot resolve is hers to explain
    assert recorder.calls == []
