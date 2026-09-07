"""The cues in her voice instead of tones, an echo threshold that follows
what the speaker is playing, and her own lines never becoming his turn."""

from __future__ import annotations

import asyncio
import random

from assistant.audio.acks import CUE_PHRASES, PHRASES, normalise, spoken_lines
from assistant.audio.cues import VoiceCues
from assistant.engines.realtime_engine import RealtimeEngine, _Levels
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


class FakeSpeaker:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def enqueue(self, pcm: bytes) -> None:
        self.chunks.append(pcm)


MM_HM, MM = b"\x01\x00" * 2400, b"\x02\x00" * 1200  # 0.1 s and 0.05 s at 24 kHz
ONE_MOMENT, STILL = b"\x03\x00" * 4800, b"\x04\x00" * 4800
SORRY = b"\x05\x00" * 24_000


def voiced(**extra) -> tuple[VoiceCues, FakeSpeaker, list[str]]:
    played: list[str] = []
    cues = VoiceCues(
        rate=24_000, play=played.append, render=lambda kind, rate: kind.encode(),
        voices={"listen_end": [MM_HM, MM], "working": [ONE_MOMENT, STILL], "error": [SORRY], **extra},
        rng=random.Random(7),
    )
    return cues, FakeSpeaker(), played


def test_in_her_voice_the_dings_are_lines_and_two_of_them_fall_silent() -> None:
    cues, speaker, played = voiced()
    assert cues.voiced
    assert cues.start(speaker) and speaker.chunks == []  # the wake: the acknowledgment answers it
    assert cues.end(speaker, sound=False)
    seconds = cues.turn_over(speaker)
    assert speaker.chunks[-1] in (MM_HM, MM) and seconds == len(speaker.chunks[-1]) / 2 / 24_000
    for _ in range(6):  # never the same line twice in a row
        cues.turn_over(speaker)
        assert speaker.chunks[-1] != speaker.chunks[-2]
    assert cues.working(speaker) > 0 and speaker.chunks[-1] == ONE_MOMENT
    assert cues.working(speaker) > 0 and speaker.chunks[-1] == STILL
    n = len(speaker.chunks)
    assert cues.working(speaker) == 0.0 and len(speaker.chunks) == n  # then the room is quiet on purpose
    cues.start(speaker)  # a new turn: a slow tool may say so again
    assert cues.working(speaker) > 0 and speaker.chunks[-1] == ONE_MOMENT
    assert cues.error("boom", speaker) > 0 and speaker.chunks[-1] == SORRY and not cues.listening
    n = len(speaker.chunks)
    cues.session_end(speaker)
    assert len(speaker.chunks) == n  # she already said goodbye
    assert played == []  # nothing ever raced a second stream


def test_a_kind_without_a_clip_still_rings_its_tone() -> None:
    cues = VoiceCues(rate=24_000, play=lambda k: None, render=lambda kind, rate: kind.encode(), voices={"listen_end": [MM]})
    speaker = FakeSpeaker()
    assert cues.error("boom", speaker) == 0.0 and speaker.chunks == [b"error"]
    assert cues.working(speaker) == 0.0 and speaker.chunks[-1] == b"working"


def test_without_voices_every_cue_is_a_tone_that_runs_for_nothing() -> None:
    cues = VoiceCues(rate=24_000, play=lambda k: None, render=lambda kind, rate: kind.encode())
    speaker = FakeSpeaker()
    assert not cues.voiced
    cues.start(speaker)
    assert cues.turn_over(speaker) == 0.0 and cues.working(speaker) == 0.0 and cues.error("x", speaker) == 0.0
    assert speaker.chunks == [b"wake", b"listen_end", b"working", b"error"]


def test_her_lines_are_known_normalised() -> None:
    lines = spoken_lines("Will")
    assert normalise("Yes, Will?") in lines and normalise("  mm-hm. ") in lines and "one moment" in lines
    assert normalise("Turn off the hallway light") not in lines
    assert set(CUE_PHRASES) & set(PHRASES) == set()  # two tables, no slug twice


def test_the_expected_echo_follows_what_the_speaker_plays() -> None:
    levels = _Levels()
    levels.new_playback(0.0)
    assert levels.calibrating(0.5, 900)
    levels.echo(1, 0)  # her audio has not reached the device yet: teaches nothing
    assert levels.calibrating(0.5, 0) and levels.coupling == 0
    for _ in range(12):
        levels.echo(500, 1000)  # a second of her audibly playing: the mic hears half of it
    assert not levels.calibrating(5.0, 900) and levels.coupling == 0.5
    assert levels.threshold(900) == 720  # 0.5 x 900 x 1.6: a loud word of hers is not him
    assert levels.threshold(0) == 400  # a pause of hers: only the floor stands
    assert levels.expected_echo(2000) == 1000
    old = _Levels()
    old.new_playback(0.0)
    old.echo(300)  # a speaker that cannot say what it plays: the old rule
    assert old.calibrating(0.5) and not old.calibrating(5.0) and old.threshold() == 480

    # The mic hears the room a beat after the device plays it: a loud mic
    # frame over a quiet played one is alignment, not coupling. Energy over
    # the window, not the worst frame — the Echo Dot logged couplings of 4
    # and 7 the other way, and thresholds of 25,000 nobody could talk over.
    skewed = _Levels()
    skewed.new_playback(0.0)
    for i in range(12):
        skewed.echo(500, 100 if i % 2 else 2000)
    assert abs(skewed.coupling - 6000 / 12600) < 0.01  # not 5.0
    loud = _Levels()
    loud.new_playback(0.0)
    for _ in range(12):
        loud.echo(9000, 1000)
    assert loud.coupling == 2.0  # capped: past this it is a glitch, and she would be deaf to him


async def test_the_transcriber_is_told_the_language() -> None:
    engine, _client, _ui = make()
    transcription = (await engine._session_config("whisper-1"))["audio"]["input"]["transcription"]
    assert transcription["language"] == "en"
    engine, _client, _ui = make(transcribe_language="")
    transcription = (await engine._session_config("whisper-1"))["audio"]["input"]["transcription"]
    assert "language" not in transcription


def make(**kw) -> tuple[RealtimeEngine, FakeClient, QuietUi]:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa", **kw
    )
    client = FakeClient()
    engine._client = client
    return engine, client, QuietUi()


async def test_her_own_line_transcribed_is_never_his_turn() -> None:
    engine, client, ui = make(idle_timeout_s=0.6)
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("Mm-hm.", reply="Did you say something?")  # her cue, back through the mic

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert ("you", "Mm-hm.") not in stats.transcript and not stats.replied
    assert any("heard her own" in n for n in ui.notes)


class FlagMic:
    """Nothing to say, but it remembers being flagged for her echo."""

    def __init__(self) -> None:
        self.deadlines: list[float] = []

    async def get_frame(self) -> bytes:
        await asyncio.sleep(3600)
        return b""

    def drain(self) -> None: ...

    def suspect_before(self, deadline: float) -> None:
        self.deadlines.append(deadline)


async def test_a_voiced_turn_over_flags_the_microphone_for_its_echo() -> None:
    cues = VoiceCues(rate=24_000, play=lambda k: None, render=lambda kind, rate: kind.encode(), voices={"listen_end": [MM_HM]})
    engine, client, _ui = make(idle_timeout_s=2.0, cues=cues)
    conn = client.connection
    conn.auto_reply = False
    mic, speaker = FlagMic(), InstantSpeaker()
    assert cues.start(speaker) and speaker.chunks == []  # the runner's acknowledgment opened the window

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")  # …and nothing follows for a second

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(mic, speaker, None, QuietUi()), 8)
    await turn
    assert MM_HM in speaker.chunks  # "Mm-hm." instead of the falling tone
    assert len(mic.deadlines) == 1  # and the mic was told she was about to be heard
