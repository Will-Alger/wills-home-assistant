"""She answers the wake word in her own voice, and does not answer herself.

Nothing here opens an audio device or a socket: the speaker is a list the
test reads back and the microphone is the real one with PortAudio faked out.
The WAVs are the committed ones, so a missing or re-rendered clip fails here
rather than in the living room.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime
from itertools import pairwise

import pytest

from assistant.audio import mic as mic_module
from assistant.audio.acks import ECHO_TAIL_S, GREETINGS, PHRASES, RATE, WakeAcks, ack_dir, phrase
from assistant.audio.base import FRAME_BYTES
from assistant.audio.cues import VoiceCues
from assistant.audio.fallbacks import read_wav
from assistant.audio.mic import Microphone

MAX_S = 0.9  # the ceiling the render script holds every clip to


class FakeSpeaker:
    """The runner's open output stream, as far as an acknowledgment can tell."""

    def __init__(self) -> None:
        self.is_open = True
        self.queued: list[bytes] = []
        self.marks: list[tuple[int, object]] = []

    @property
    def enqueued(self) -> int:
        return sum(len(pcm) for pcm in self.queued)

    def enqueue(self, pcm: bytes) -> None:
        self.queued.append(pcm)

    def notify_when_played(self, position: int, fn) -> None:
        self.marks.append((position, fn))


class FakeMic:
    """Only the one method the acknowledgment needs from a microphone."""

    def __init__(self) -> None:
        self.deadline = 0.0

    def suspect_before(self, deadline: float) -> None:
        self.deadline = deadline


class FakeTrace:
    """The latency log's stopwatch, reduced to what it is asked for here."""

    def __init__(self) -> None:
        self.stamps: list[str] = []
        self.audibles = 0

    def stamp(self, name: str) -> None:
        self.stamps.append(name)

    def audible(self) -> None:
        self.audibles += 1


def rig(**kw) -> tuple[VoiceCues, WakeAcks, FakeSpeaker, FakeMic, FakeTrace]:
    """A wake, with every earcon rendered as a recognisable stub."""
    cues = VoiceCues(
        rate=RATE,
        play=lambda kind, *_a: None,
        render=lambda kind, rate: kind.encode(),
    )
    directory = kw.pop("directory", ack_dir())
    kw.setdefault("rng", random.Random(7))
    acks = WakeAcks(directory, **kw)
    return cues, acks, FakeSpeaker(), FakeMic(), FakeTrace()


# ── the clips themselves ──────────────────────────────────────────────────


def test_every_acknowledgment_is_committed_and_short_enough_to_be_one() -> None:
    acks = WakeAcks(ack_dir())
    assert acks.missing() == []  # all eight WAVs are in the repo
    assert acks.note() == ""
    for slug in PHRASES:
        pcm = read_wav(ack_dir() / f"{slug}.wav", RATE)
        seconds = len(pcm) / 2 / RATE
        assert 0.1 < seconds <= MAX_S, f"{slug} is {seconds:.2f}s"
        assert len(pcm) % 2 == 0


def test_the_owner_is_named_in_the_one_line_that_names_him() -> None:
    assert phrase("yes-owner", "Will") == "Yes, Will?"
    assert phrase("yes") == "Yes?"


# ── one wake, one answer ──────────────────────────────────────────────────


def test_a_wake_answers_with_exactly_one_clip_and_opens_the_listening_window() -> None:
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig()
    before = time.monotonic()

    spoken = runner.acknowledge(cues, acks, speaker, mic, trace)

    assert len(speaker.queued) == 1  # her voice, and nothing else
    assert 0.0 < spoken <= MAX_S
    assert cues.played == []  # the ding stood down
    assert cues.listening is True  # the flag, the panel and the meter are unchanged
    assert acks.played == [acks.last]
    assert speaker.marks == [(0, trace.audible)]  # heard when the callback pulls byte 0
    assert trace.stamps == ["chime_enqueued"]
    # the microphone is shut until her own voice has left the room, and no longer
    assert before + spoken + ECHO_TAIL_S <= mic.deadline <= time.monotonic() + spoken + ECHO_TAIL_S


def test_she_takes_a_beat_before_answering_her_name() -> None:
    """The clip alone came 30 ms after the wake — "so fast it's almost a
    little unnatural" (Will). A beat of silence goes ahead of it on the same
    stream, a little different each time; the log marks the words, not the
    pause, and the microphone stays shut for both."""
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig(beat_s=0.3)
    before = time.monotonic()
    spoken = runner.acknowledge(cues, acks, speaker, mic, trace)
    assert len(speaker.queued) == 2  # the beat, then her voice
    beat, clip = speaker.queued
    assert not any(beat) and 0.3 <= len(beat) / 2 / RATE <= 0.45 and any(clip)
    assert abs(spoken - (len(beat) + len(clip)) / 2 / RATE) < 1e-9
    assert acks.last_beat_s == len(beat) / 2 / RATE
    assert speaker.marks == [(len(beat), trace.audible)]
    assert before + spoken + ECHO_TAIL_S <= mic.deadline <= time.monotonic() + spoken + ECHO_TAIL_S
    beats = set()
    for _ in range(4):
        speaker.queued.clear()
        runner.acknowledge(cues, acks, speaker, mic, trace)
        beats.add(len(speaker.queued[0]))
    assert len(beats) > 1  # never the same pause twice running


def test_two_wakes_in_a_row_never_get_the_same_answer() -> None:
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig()
    picks = []
    for _wake in range(24):
        cues.reset()  # the conversation between them closed
        runner.acknowledge(cues, acks, speaker, mic, trace)
        picks.append(acks.last)

    assert all(one != nxt for one, nxt in pairwise(picks))
    assert len(set(picks)) > 2  # varied, not a two-clip metronome
    assert len(speaker.queued) == 24


def test_the_ding_setting_rings_the_ding_and_plays_no_clip() -> None:
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig(mode="ding")

    spoken = runner.acknowledge(cues, acks, speaker, mic, trace)

    assert spoken == 0.0
    assert speaker.queued == [b"wake"]  # today's rising chime, nothing else
    assert cues.played == ["wake"]
    assert cues.listening is True
    assert mic.deadline == 0.0  # nothing of hers to ignore
    assert acks.played == []


def test_the_off_setting_is_silence_that_still_listens() -> None:
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig(mode="off")

    assert runner.acknowledge(cues, acks, speaker, mic, trace) == 0.0
    assert speaker.queued == []
    assert cues.played == []
    assert cues.listening is True
    assert speaker.marks == []  # nothing was queued, so nothing was ever audible


def test_missing_clips_fall_back_to_the_ding_with_one_boot_note(tmp_path) -> None:
    import scripts.m4_realtime as runner

    cues, acks, speaker, mic, trace = rig(directory=tmp_path)

    assert acks.missing() == [f"{slug}.wav" for slug in PHRASES]
    note = acks.note()
    assert "render_acks" in note and "ding" in note
    assert runner.acknowledge(cues, acks, speaker, mic, trace) == 0.0
    assert speaker.queued == [b"wake"]
    assert cues.listening is True


def test_a_typo_in_the_setting_is_a_note_not_a_silent_wake() -> None:
    acks = WakeAcks(ack_dir(), mode="Ding")  # case and spacing are forgiven
    assert acks.mode == "ding"
    assert acks.note() == ""
    typo = WakeAcks(ack_dir(), mode="ping")
    assert typo.mode == "voice"  # she still answers
    assert "WAKE_ACK='ping'" in typo.note()
    blank = WakeAcks(ack_dir(), mode="")  # WAKE_ACK= means the default, quietly
    assert blank.mode == "voice"
    assert blank.note() == ""


def test_one_clip_left_on_disk_is_still_better_than_a_ding(tmp_path) -> None:
    (tmp_path / "yes.wav").write_bytes((ack_dir() / "yes.wav").read_bytes())
    acks = WakeAcks(tmp_path)
    speaker = FakeSpeaker()

    assert acks.acknowledge(speaker) > 0
    assert acks.acknowledge(speaker) > 0  # no other clip to alternate with
    assert acks.played == ["yes", "yes"]
    assert len(acks.missing()) == len(PHRASES) - 1


# ── the clock ─────────────────────────────────────────────────────────────


def picks_at(hour: int, wakes: int = 60) -> set[str]:
    def clock() -> datetime:  # only the hour is ever read; the zone keeps ruff happy
        return datetime(2026, 9, 6, hour, 0, tzinfo=UTC)

    acks = WakeAcks(ack_dir(), clock=clock, rng=random.Random(3))
    return {acks.pick() for _ in range(wakes)}


def test_a_greeting_is_only_ever_spoken_when_it_is_true() -> None:
    morning = picks_at(9)
    assert "morning" in morning and "evening" not in morning
    evening = picks_at(20)
    assert "evening" in evening and "morning" not in evening
    afternoon = picks_at(14)
    assert not {"morning", "evening"} & afternoon
    assert len(afternoon) > 1  # the neutral lines still vary


def test_the_greeting_windows_are_the_ones_the_spec_asked_for() -> None:
    assert GREETINGS == {"morning": (0, 11), "evening": (18, 24)}
    assert "morning" in picks_at(10) and "morning" not in picks_at(11)
    assert "evening" in picks_at(18) and "evening" not in picks_at(17)


# ── the catch: her own voice must not become his turn ─────────────────────


class Recorder:
    """PortAudio, as far as the microphone can tell: it keeps the callback."""

    instance: Recorder | None = None

    def __init__(self, *, callback, **_kw) -> None:
        self.callback = callback
        Recorder.instance = self

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


async def open_mic(monkeypatch) -> Microphone:
    monkeypatch.setattr(mic_module.sd, "RawInputStream", Recorder)
    monkeypatch.setattr(mic_module, "default_input_name", lambda: "Microphone (Test)")
    monkeypatch.setattr(mic_module, "describe_device", lambda _spec: "Microphone (Test)")
    return await Microphone("").open()


async def speak(mic: Microphone, frames: int = 1) -> None:
    """The room, arriving through the PortAudio callback as it really does."""
    assert Recorder.instance is not None
    for _frame in range(frames):
        Recorder.instance.callback(b"\x11\x11" * (FRAME_BYTES // 2), FRAME_BYTES // 2, None, None)
    await asyncio.sleep(0)  # call_soon_threadsafe lands on the loop


async def test_frames_captured_while_she_speaks_are_flagged_and_the_next_one_is_not(
    monkeypatch,
) -> None:
    """Nothing is thrown away any more — he says the command right over her
    "Yes?" — but the frames of that moment carry the flag the engine uses to
    drop the ones loud enough to be her."""
    mic = await open_mic(monkeypatch)
    try:
        await speak(mic, frames=3)  # her "Yes?" coming back in through the mic
        mic.suspect_before(time.monotonic() + 0.05)  # the clip's end, plus its echo tail
        await asyncio.sleep(0.08)  # ...which now passes
        await speak(mic)  # his command, said the moment she stopped

        flags = []
        for _ in range(4):
            async with asyncio.timeout(1):
                frame = await mic.get_frame()
            assert len(frame) == FRAME_BYTES
            flags.append(mic.last_suspect)
        assert flags == [True, True, True, False]
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await mic.get_frame()
    finally:
        await mic.close()


async def test_a_deadline_only_ever_moves_forward(monkeypatch) -> None:
    mic = await open_mic(monkeypatch)
    try:
        mic.suspect_before(time.monotonic() + 5.0)
        mic.suspect_before(time.monotonic() - 5.0)  # a stale deadline never shortens it
        await speak(mic, frames=2)
        for _ in range(2):
            async with asyncio.timeout(1):
                await mic.get_frame()
            assert mic.last_suspect
    finally:
        await mic.close()


async def test_a_microphone_nobody_muted_delivers_every_frame(monkeypatch) -> None:
    mic = await open_mic(monkeypatch)
    try:
        await speak(mic, frames=2)
        async with asyncio.timeout(1):
            assert len(await mic.get_frame()) == FRAME_BYTES
            assert len(await mic.get_frame()) == FRAME_BYTES
    finally:
        await mic.close()


# ── the render script's own check ─────────────────────────────────────────


def test_the_render_check_rejects_a_clip_that_runs_long() -> None:
    from scripts.render_acks import MAX_S as CEILING
    from scripts.render_acks import problem

    long_take = b"\x00\x00" * int(RATE * 1.4)
    assert "over the" in problem("Yes?", "Yes?", long_take, RATE)
    assert problem("Yes?", "Yes?", b"\x00\x00" * int(RATE * 0.5), RATE) == ""
    assert problem("Yes?", "", b"", RATE) == "no audio came back"
    assert CEILING == MAX_S


def test_the_render_check_rejects_a_clip_with_words_that_were_not_asked_for() -> None:
    from scripts.render_acks import problem

    short = b"\x00\x00" * int(RATE * 0.5)
    assert problem("Yes?", "Yes? How can I help?", short, RATE).startswith("she said")
    assert problem("Yes?", "  yes!  ", short, RATE) == ""  # punctuation and case are hers
    assert problem("I'm here.", "I'm here", short, RATE) == ""


def test_the_render_script_cuts_the_silence_around_the_words() -> None:
    import numpy as np

    from scripts.render_acks import trim

    quiet = np.zeros(int(RATE * 0.4), dtype="<i2")
    word = (np.random.default_rng(1).normal(0, 3000, int(RATE * 0.3))).astype("<i2")
    take = np.concatenate([quiet, word, quiet]).tobytes()

    cut = trim(take, RATE)

    assert 0.3 < len(cut) / 2 / RATE < 0.45  # the words, plus a pad either side

    all_quiet = np.zeros(int(RATE * 0.5), dtype="<i2").tobytes()
    assert trim(all_quiet, RATE) == all_quiet  # nothing to find is not ours to judge
