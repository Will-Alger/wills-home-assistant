"""Replay: whole conversations at real speed, so heard and generated differ.

Every other engine test plays her voice instantly and delivers each event on
time, which hides the whole class of "she closed before he heard it" bugs.
Here a `DelayedSpeaker` drains at 24 kHz and the script pushes each event at
a moment of its choosing: a transcript that lands a second late, a tool that
takes most of a second while he corrects himself over it, a socket that dies
mid-sentence, an announcement cut in half by the wake word, a wrap-up spoken
under a running tool, a barge-in during the goodbye, the end tool firing
without a word to close on, and a nine-second web search whose silence the
working tick fills at two seconds and again at eight.

Each scenario asserts how the session ended, what went to the server, and —
where it matters — what the announcer believes he has actually heard.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from assistant.announce import Announcer
from assistant.audio.cues import VoiceCues
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.base import LightCommand
from assistant.home.fake import FakeHome
from assistant.status import AssistantStatus
from tests.fake_realtime import DelayedSpeaker, FakeClient, FakeConnection, QuietUi

# ── the room ───────────────────────────────────────────────────────────────


class SlowHome(FakeHome):
    """A house that takes its time: `set_lights` is a real, slow tool call."""

    delay_s: float = 0.8

    async def apply(self, commands: list[LightCommand]) -> None:
        await asyncio.sleep(self.delay_s)
        await super().apply(commands)


class FrameMic:
    """A microphone that keeps producing 80 ms frames, like the real one."""

    def __init__(self) -> None:
        self.drained = 0

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return b"\x00\x00" * 1920

    def drain(self) -> None:
        self.drained += 1


class WakeAfter:
    """The wake phrase, spoken once the room has actually HEARD `after_ms` of
    her — the barge-in is tied to playback, not to a stopwatch."""

    def __init__(self, speaker: DelayedSpeaker, after_ms: int) -> None:
        self._speaker = speaker
        self._after_ms = after_ms
        self.fired = False

    def detect(self, frame: bytes) -> bool:
        if self.fired or self._speaker.played_ms(self._speaker.current_item) < self._after_ms:
            return False
        self.fired = True
        return True

    def reset(self) -> None: ...


class ReplayUi(QuietUi):
    """QuietUi that remembers the moments a scenario needs to react to."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []
        self.listening_windows = 0
        self.barge_in = asyncio.Event()

    def listening(self) -> None:
        self.listening_windows += 1

    def interrupted(self) -> None:
        self.barge_in.set()

    def error(self, message: str) -> None:
        self.errors.append(message)


def make_engine(
    *, home: FakeHome | None = None, cues: VoiceCues | None = None, **kw
) -> tuple[RealtimeEngine, FakeConnection, VoiceCues]:
    if cues is None:
        cues = VoiceCues(play=lambda kind: None, render=lambda kind, rate: b"")  # silent earcons
    engine = RealtimeEngine(
        api_key="test-key", model="m", voice="v", home=home or FakeHome(), owner="Will",
        name="Alexa", wake_phrase="alexa", cues=cues, **kw,
    )
    client = FakeClient()
    engine._client = client
    client.connection.auto_reply = False  # this file scripts every event itself
    return engine, client.connection, cues


# ── scripting one turn ─────────────────────────────────────────────────────


def owner_says(conn: FakeConnection, text: str) -> None:
    """The real event order for a finished user turn."""
    conn.push("input_audio_buffer.speech_started")
    conn.push("input_audio_buffer.speech_stopped")
    conn.push("input_audio_buffer.committed")  # the server ended his turn (the ding plays here)
    conn.push("conversation.item.input_audio_transcription.completed", transcript=text)


def she_says(
    conn: FakeConnection,
    transcript: str = "",
    *,
    ms: int = 0,
    item_id: str = "item_1",
    calls: tuple[tuple[str, str, dict], ...] = (),
) -> None:
    """One assistant response: created, `ms` of audio, its transcript, done —
    with any function calls listed on the done event, as the server does it."""
    conn.push("response.created")
    if ms:
        conn.push_audio(ms, item_id=item_id)
    if transcript:
        conn.push("response.output_audio_transcript.done", transcript=transcript)
    conn.push_response_done(*calls)


def sent(conn: FakeConnection, kind: str) -> list[dict]:
    return [e for e in conn.sent if e["type"] == kind]


def tool_outputs(conn: FakeConnection) -> list[dict]:
    """The function_call_output payloads the engine sent back, decoded."""
    return [
        json.loads(e["item"]["output"])
        for e in sent(conn, "conversation.item.create")
        if e["item"].get("type") == "function_call_output"
    ]


def system_items(conn: FakeConnection) -> list[str]:
    return [
        e["item"]["content"][0]["text"]
        for e in sent(conn, "conversation.item.create")
        if e["item"].get("role") == "system"
    ]


async def replay(engine: RealtimeEngine, script, *, mic=None, speaker=None, wake=None, ui=None,
                 announce: bool = False, timeout: float = 12.0):
    """Run one conversation against a script that pushes events as it likes.
    Returns (stats, seconds it took, speaker, ui)."""
    speaker = speaker if speaker is not None else DelayedSpeaker()
    ui = ui if ui is not None else ReplayUi()
    started = time.monotonic()
    driver = asyncio.create_task(script(speaker, ui))
    try:
        stats = await asyncio.wait_for(
            engine.run_conversation(mic or FrameMic(), speaker, wake, ui, announce=announce),
            timeout,
        )
    finally:
        # A script still mid-sleep when the session ends is simply done with;
        # one that BROKE says so here instead of vanishing.
        driver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await driver
    return stats, time.monotonic() - started, speaker, ui


# ── the instrument itself ──────────────────────────────────────────────────


async def test_the_delayed_speaker_reports_only_what_has_drained() -> None:
    speaker = DelayedSpeaker(scale=8.0)  # 400 ms of speech in 50 ms of test
    speaker.begin_item("item_1")
    speaker.enqueue(b"\x00" * (400 * 48))  # 400 ms at 24 kHz int16
    assert speaker.played_ms("item_1") == 0  # nothing has been heard yet
    await asyncio.sleep(0.025)
    heard = speaker.played_ms("item_1")
    assert 100 < heard < 300, heard  # about half way through
    speaker.clear()  # a barge-in: the tail is gone and never counts
    assert speaker.pending_seconds == 0
    await asyncio.sleep(0.05)
    assert speaker.played_ms("item_1") == heard
    assert speaker.played_ms("nobody") == 0

    speaker.begin_item("item_2")
    speaker.enqueue(b"\x00" * (200 * 48))
    await asyncio.wait_for(speaker.wait_idle(), 1)
    assert speaker.played_ms("item_2") == 200 and speaker.played_ms("item_1") == heard
    assert speaker.items == ["item_1", "item_2"] and speaker.clears == 1


# ── 1. a transcript that lands a second late ───────────────────────────────


async def test_a_wrapup_arriving_a_second_late_still_closes_without_a_listening_window() -> None:
    """His "that's all" is transcribed only after her reply is complete — but
    while it is still playing. The close decision belongs to the end of
    playback, so she must not ding for another turn she is about to abandon."""
    engine, conn, cues = make_engine(idle_timeout_s=6.0, info_close_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        she_says(conn, "Anything else I can do?", ms=1600)
        await asyncio.sleep(1.0)  # a full second after response.done
        conn.push("conversation.item.input_audio_transcription.completed", transcript="that's all")

    stats, took, speaker, ui = await replay(engine, script)

    assert stats.ended_by == "wrap-up"
    assert ("you", "that's all") in stats.transcript
    assert speaker.played_ms("item_1") == 1600  # she was heard out in full
    assert 1.7 < took < 3.0  # closed at the end of the audio, not on a timer
    assert "wake" not in cues.played and ui.listening_windows == 0  # no window offered
    assert not sent(conn, "conversation.item.truncate") and not sent(conn, "response.cancel")
    assert not sent(conn, "response.create")  # nothing more was asked of the model


# ── 2. a slow tool, corrected while it runs ────────────────────────────────


async def test_a_correction_spoken_under_a_slow_tool_rides_out_on_its_result() -> None:
    """The receiver keeps reading while the tool runs, so "actually, the
    kitchen" is known by the time the result goes back — and the idle
    watchdog, set shorter than the tool, must not close the session under it."""
    home = SlowHome()
    engine, conn, _cues = make_engine(home=home, idle_timeout_s=0.4)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "turn on the hallway light")
        she_says(conn, calls=(("call_0", "set_lights",
                               {"changes": [{"target": "hallway", "turn": "on"}]}),))
        await asyncio.sleep(0.4)  # the light is still being set
        owner_says(conn, "actually make it the kitchen instead")
        await asyncio.sleep(0.6)  # the tool has returned; she answers the correction
        she_says(conn, "Kitchen's on.", ms=300, item_id="item_2",
                 calls=(("call_1", "end_conversation", {}),))

    stats, took, speaker, _ui = await replay(engine, script)

    assert stats.ended_by == "end_conversation"  # never "idle timeout", though 0.4 s < 0.8 s
    assert took > 0.9 and home.applied, "the tool was cancelled under the idle watchdog"
    since = tool_outputs(conn)[0]["since"]
    assert "actually make it the kitchen instead" in since and "skip what is now moot" in since
    assert stats.tool_calls == ["set_lights", "end_conversation"]
    assert len(sent(conn, "response.create")) == 1  # the tool's answer, and nothing after it
    assert speaker.played_ms("item_2") == 300  # she was heard out before the close


# ── 3. the socket dies mid-sentence ────────────────────────────────────────


async def test_a_receiver_failure_mid_response_ends_the_session_in_under_a_second() -> None:
    engine, conn, _cues = make_engine(idle_timeout_s=30.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "tell me a long story")
        conn.push("response.created")
        conn.push_audio(3000)  # three seconds of her, barely started
        await asyncio.sleep(0.3)
        conn.fail_now()

    stats, took, speaker, ui = await replay(engine, script)

    assert stats.ended_by.startswith("session error")
    assert "socket closed" in stats.ended_by
    assert took < 1.0  # a bounded path out, not the 30 s idle timer
    assert ui.errors == ["socket closed"]
    assert 0 < speaker.played_ms("item_1") < 3000  # she was cut off mid-sentence


# ── 4. a two-part announcement, cut in half by the wake word ───────────────


async def test_an_interrupted_announcement_keeps_both_items_unread(tmp_path: Path) -> None:
    """She opens with two pieces of news; he says "Alexa" over the second.
    Neither is read: delivered means played, and answering the interruption
    is not the same as having heard what it cut off."""
    announcer = Announcer(tmp_path / "a.json")
    first = announcer.enqueue("Task 7 is built.", ref="task:7:1:built")
    second = announcer.enqueue("The porch light came on at nine.", kind="watch", ref="watch:1")
    engine, conn, cues = make_engine(announcer=announcer, info_close_s=0.3, idle_timeout_s=6.0)
    speaker = DelayedSpeaker()
    wake = WakeAfter(speaker, after_ms=600)

    async def script(spk, ui) -> None:
        await asyncio.sleep(0.1)
        she_says(conn, "Task 7 is built. Also, the porch light came on at nine.", ms=1400)
        await asyncio.wait_for(ui.barge_in.wait(), 5)  # he cut in; now he talks
        owner_says(conn, "what was that about the porch?")
        await asyncio.sleep(0.2)
        she_says(conn, "It came on at nine.", ms=200, item_id="item_2")

    stats, _took, speaker, ui = await replay(
        engine, script, speaker=speaker, wake=wake, announce=True
    )

    assert stats.ended_by == "question answered" and stats.replied
    assert stats.announced == [first.id, second.id]  # both were spoken over
    assert first.state == "spoken" and second.state == "spoken"
    assert first.unread and second.unread, "his reply to the interruption read them"
    assert [a.id for a in announcer.unread()] == [first.id, second.id]
    assert "announcement cut short — kept unread" in ui.notes

    truncate = sent(conn, "conversation.item.truncate")[0]
    assert truncate["item_id"] == "item_1" and truncate["content_index"] == 0
    assert 550 < truncate["audio_end_ms"] < 900, truncate  # what he heard, not the 1400 sent
    assert speaker.clears == 1 and speaker.played_ms("item_1") == truncate["audio_end_ms"]
    # one window for the turn he cut in with, one after her answer to it
    assert cues.played == ["wake", "listen_end", "wake"]


# ── 5. "that's all" under a running tool ───────────────────────────────────


async def test_a_wrapup_under_a_running_tool_closes_after_the_tools_reply_plays() -> None:
    """He wraps up while the lights are still being set. The close has to
    wait for the answer that tool is about to produce — closing on the
    silent tool response would cut her off before she said a word."""
    home = SlowHome()
    engine, conn, cues = make_engine(home=home, idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "turn off the hallway light")
        she_says(conn, calls=(("call_0", "set_lights",
                               {"changes": [{"target": "hallway", "turn": "off"}]}),))
        await asyncio.sleep(0.3)  # the tool is still running
        owner_says(conn, "that's all, thanks")
        await asyncio.sleep(0.7)  # the tool returned and she was asked to answer
        she_says(conn, "Off. Goodnight.", ms=500, item_id="item_2")

    stats, took, speaker, ui = await replay(engine, script)

    assert stats.ended_by == "wrap-up"
    assert home.applied and stats.tool_calls == ["set_lights"]
    assert speaker.played_ms("item_2") == 500, "closed before her goodbye finished"
    assert took > 1.5  # the tool, then the whole reply
    assert len(sent(conn, "response.create")) == 1
    assert cues.played == []  # never a ding: he was not invited to speak again
    assert ui.listening_windows == 1  # half-duplex lifted under the tool, silently
    assert not sent(conn, "conversation.item.truncate")


async def test_an_announcement_that_calls_a_tool_still_closes_on_its_own_words(tmp_path: Path) -> None:
    """The other half of the same rule: a close held for a tool's answer must
    not be forgotten. She opens with news, looks something up, says it, and
    goes back to sleep — no listening window she was never going to use."""
    announcer = Announcer(tmp_path / "a.json")
    item = announcer.enqueue("Task 7 is built.", ref="task:7:1:built")
    engine, conn, cues = make_engine(announcer=announcer, idle_timeout_s=6.0, info_close_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        she_says(conn, calls=(("call_0", "get_lights", {}),))
        await asyncio.sleep(0.3)
        she_says(conn, "Task 7 is built, and the hallway is still on.", ms=400, item_id="item_2")

    stats, took, speaker, ui = await replay(engine, script, announce=True)

    assert stats.ended_by == "announcement delivered"
    assert speaker.played_ms("item_2") == 400 and took < 2.0  # said, then straight to sleep
    assert stats.announced == [item.id] and item.state == "spoken"  # spoken, still unread
    assert cues.played == [] and ui.listening_windows == 1  # no ding: nobody was asked anything


# ── 6. a barge-in during the goodbye ───────────────────────────────────────


async def test_a_barge_in_during_the_goodbye_keeps_the_session_open() -> None:
    """He said "that's all", then thought of one more thing while she was
    signing off. The wrap-up must not win over the wake word."""
    engine, conn, cues = make_engine(idle_timeout_s=6.0)
    speaker = DelayedSpeaker()
    wake = WakeAfter(speaker, after_ms=400)

    async def script(spk, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "that's all")
        she_says(conn, "Goodnight, Will. Talk to you tomorrow.", ms=1200)
        await asyncio.wait_for(ui.barge_in.wait(), 5)
        owner_says(conn, "wait, is the front door locked?")
        await asyncio.sleep(0.2)
        she_says(conn, "It's locked.", ms=300, item_id="item_2")

    stats, _took, speaker, ui = await replay(engine, script, speaker=speaker, wake=wake)

    assert ui.listening_windows == 1, "she closed on the goodbye instead of listening"
    assert ("you", "wait, is the front door locked?") in stats.transcript
    assert ("alexa", "It's locked.") in stats.transcript
    truncate = sent(conn, "conversation.item.truncate")[0]
    assert truncate["item_id"] == "item_1" and 350 < truncate["audio_end_ms"] < 700
    assert speaker.clears == 1 and speaker.played_ms("item_2") == 300  # the answer played out
    assert cues.played.count("wake") == 1
    assert stats.ended_by == "wrap-up"  # his wrap-up still stands, one answer later
    assert not sent(conn, "response.create")


# ── 7. the end tool, with nothing to close on ──────────────────────────────


async def test_the_end_tool_without_audio_asks_for_a_goodbye_and_closes_after_it() -> None:
    engine, conn, cues = make_engine(idle_timeout_s=6.0)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "pause the music")
        she_says(conn, calls=(("call_0", "media_control", {"action": "pause"}),
                              ("call_1", "end_conversation", {})))
        await asyncio.sleep(0.3)  # she was asked for a goodbye; here it comes
        she_says(conn, "Paused. Talk later.", ms=600, item_id="item_2")

    stats, took, speaker, ui = await replay(engine, script)

    assert stats.ended_by == "end_conversation"
    assert system_items(conn)[-1] == "Say a brief goodbye now — a few words, nothing more."
    assert len(sent(conn, "response.create")) == 1  # exactly one: the goodbye
    assert speaker.played_ms("item_2") == 600, "closed before the goodbye finished"
    assert took > 0.9
    assert "wake" not in cues.played and ui.listening_windows == 0
    assert not sent(conn, "conversation.item.truncate")
    assert stats.tool_calls == ["media_control", "end_conversation"]


# ── 8. the working cue: a tool that takes its time in silence ──────────────


class SlowSearch:
    """A web search as slow as the real thing — the silent gap the working
    cue exists for."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.queries: list[str] = []

    async def search(self, query: str) -> str:
        self.queries.append(query)
        await asyncio.sleep(self.delay_s)
        return "The Celtics beat the Knicks 112-104 tonight."


class TimedCues(VoiceCues):
    """Silent earcons that remember WHEN each one went out."""

    def __init__(self, status: AssistantStatus | None = None) -> None:
        super().__init__(status=status, play=lambda kind: None, render=lambda kind, rate: b"")
        self.at: list[tuple[str, float]] = []

    def _sound(self, kind: str, speaker: Any | None = None, on_audible: Any | None = None) -> None:
        self.at.append((kind, time.monotonic()))
        super()._sound(kind, speaker, on_audible)

    def ticks_since(self, mark: float) -> list[float]:
        """Seconds from `mark` to each working tick."""
        return [at - mark for kind, at in self.at if kind == "working"]


async def test_a_slow_tool_ticks_at_two_seconds_and_again_six_seconds_later() -> None:
    """A web search takes the room from her last word to her next one in
    silence. One soft tick two seconds in, another six after that, and not
    one more once she is actually speaking."""
    status = AssistantStatus()
    cues = TimedCues(status)
    web = SlowSearch(delay_s=9.0)
    engine, conn, _cues = make_engine(cues=cues, web=web, idle_timeout_s=20.0, info_close_s=0.3)
    marks: dict[str, float] = {}
    panel_state: list[str] = []  # what the panel said mid-wait

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "search the web for tonight's Celtics score")
        marks["asked"] = time.monotonic()
        she_says(conn, calls=(("call_0", "web_search", {"query": "Celtics score tonight"}),))
        await asyncio.sleep(3.0)  # mid-wait: what the Settings panel says
        marks["panel"] = time.monotonic()
        panel_state.append(str(status.snapshot()["state"]))
        await asyncio.sleep(6.7)  # the search has returned; here is the answer
        marks["audio"] = time.monotonic()
        she_says(conn, "The Celtics won, 112 to 104.", ms=400, item_id="item_2")

    stats, _took, speaker, _ui = await replay(engine, script, timeout=25.0)

    ticks = cues.ticks_since(marks["asked"])
    assert len(ticks) == 2, ticks  # ...and none in the six seconds after the audio
    assert 2.0 <= ticks[0] < 2.8, ticks  # the first, two seconds into the silence
    assert 8.0 <= ticks[1] < 8.9, ticks  # the second, six seconds after that
    assert 5.9 < ticks[1] - ticks[0] < 6.3, ticks
    assert ticks[1] < marks["audio"] - marks["asked"]  # both before she said a word
    assert panel_state == ["working"]
    assert web.queries == ["Celtics score tonight"] and stats.tool_calls == ["web_search"]
    assert speaker.played_ms("item_2") == 400  # the answer she waited for, heard in full
    assert cues.played == ["working", "working", "wake"]  # then the listening ding


async def test_a_tool_that_returns_at_once_never_ticks() -> None:
    """"Lights off" is done in a moment: the room hears the answer, not a
    cue for a wait that never happened."""
    cues = TimedCues()
    engine, conn, _cues = make_engine(cues=cues, idle_timeout_s=6.0, command_close_s=0.3)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "lights off")
        she_says(conn, calls=(("call_0", "set_lights",
                               {"changes": [{"target": "hallway", "turn": "off"}]}),))
        await asyncio.sleep(0.4)
        she_says(conn, "Off.", ms=300, item_id="item_2")

    stats, took, _speaker, _ui = await replay(engine, script)

    assert stats.tool_calls == ["set_lights"] and took < 2.5
    assert "working" not in cues.played


async def test_a_word_before_the_tool_call_silences_the_tick() -> None:
    """She said "Sure." and only then went looking. The room has heard her
    voice, so the wait it is now in gets no tick — the rule is about the
    silence, not about the clock."""
    cues = TimedCues()
    web = SlowSearch(delay_s=3.0)
    engine, conn, _cues = make_engine(cues=cues, web=web, idle_timeout_s=20.0, info_close_s=0.3)

    async def script(speaker, ui) -> None:
        await asyncio.sleep(0.1)
        owner_says(conn, "look up tonight's Celtics score")
        she_says(conn, "Sure.", ms=200, calls=(("call_0", "web_search", {"query": "celtics"}),))
        await asyncio.sleep(3.4)
        she_says(conn, "They won by eight.", ms=300, item_id="item_2")

    stats, took, _speaker, _ui = await replay(engine, script, timeout=15.0)

    assert stats.tool_calls == ["web_search"] and took > 3.0  # the search really ran
    assert "working" not in cues.played
