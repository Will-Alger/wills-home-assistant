"""The turn latency log: the instrument every timing change is judged with.

A scripted session with real delays must come out of logs/turns.jsonl with
the delays in it; the idle path must keep the wake words that almost fired;
and "how fast were you today?" must render a sentence she can say — including
when there is nothing to say it about.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from assistant.app import wait_for_trigger
from assistant.brain.outcome import ToolOutcome, ok
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.latency import LatencyLog, TurnTrace, latency_report, since_label
from assistant.sessions import SessionLog
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


def make_engine(latency: LatencyLog, **kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="test-key", model="m", voice="v", home=FakeHome(), owner="Will",
        name="Alexa", wake_phrase="alexa", latency=latency, **kw,
    )
    client = FakeClient()
    engine._client = client  # no network: scripted Realtime connection
    return engine, client


# ── a whole turn, timed ────────────────────────────────────────────────────


async def test_a_scripted_turn_lands_in_the_log_with_its_delays(tmp_path: Path) -> None:
    log = LatencyLog(tmp_path / "turns.jsonl")
    engine, client = make_engine(log, info_close_s=0.3)

    async def slow_tool(name: str, args: dict) -> ToolOutcome:
        await asyncio.sleep(0.3)  # a light command takes about this long in the house
        return ok("the hallway is on")

    engine._executor.run = slow_tool

    trace = log.wake(0.71)
    trace.stamp("chime_enqueued")
    trace.audible()  # the speaker callback consumed it: it could be heard
    trace.stamp("mic_ready")

    async def owner() -> None:
        connection = client.connection
        await asyncio.sleep(0.10)
        connection.push("input_audio_buffer.speech_started")
        await asyncio.sleep(0.20)
        connection.push("input_audio_buffer.speech_stopped")
        await asyncio.sleep(0.10)
        connection.push(
            "conversation.item.input_audio_transcription.completed",
            transcript="turn on the hallway",
        )
        await asyncio.sleep(0.05)
        connection.push_response_done(("c1", "set_lights", {"changes": []}))

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(
        engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), trace=trace), 10
    )
    await turn
    trace.finish(stats.ended_by, session=7)

    rows = log.read()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "turn" and row["turn"] == 0 and row["session"] == 7
    assert row["wake_score"] == 0.71
    assert row["chime_audible"] is not None and row["mic_ready"] is not None
    assert row["connected"] is not None  # session.updated arrived

    # the steps in the order they really happened, carrying the scripted gaps
    # (loosely: Windows' timer granularity spends a few ms of every sleep)
    assert row["first_speech"] >= 0.08
    assert row["speech_end"] >= row["first_speech"] + 0.15
    assert row["transcript_at"] >= row["speech_end"] + 0.05
    assert row["first_audio"] >= row["transcript_at"] + 0.25  # the tool ran first
    assert row["playback_end"] >= row["first_audio"]

    (name, seconds), = row["tools"]
    assert name == "set_lights" and 0.28 < seconds < 2.0
    assert row["ended_by"] == stats.ended_by == "command complete"


async def test_the_last_turn_rides_home_on_the_session_row(tmp_path: Path) -> None:
    """"How fast were you?" must be answerable from data/sessions.json alone."""
    sessions = SessionLog(tmp_path / "sessions.json")
    row = sessions.start("wake")
    trace = TurnTrace()
    trace.stamp("chime_audible")
    trace.speech_started()
    trace.speech_stopped()
    trace.audio_delta()
    trace.playback_done()
    trace.finish("question answered", session=row.id)

    sessions.finish(row.id, ended_by="question answered", timings=trace.compact())
    reloaded = SessionLog(tmp_path / "sessions.json").get(row.id)
    assert reloaded is not None
    assert "first_audio" in reloaded.timings and "chime_audible" in reloaded.timings
    assert "session" not in reloaded.timings  # the row already knows who it is
    assert sessions.rows()[0]["timings"] == reloaded.timings


def test_a_second_user_turn_starts_a_second_row() -> None:
    """Each turn is its own row, and only the first carries the activation."""
    clock = iter([0.0] + [i / 10 for i in range(1, 200)])
    trace = TurnTrace(now=lambda: next(clock), wall=lambda: 1000.0)
    trace.stamp("chime_enqueued")
    trace.speech_started()
    trace.speech_stopped()
    trace.audio_delta()
    trace.playback_done()
    trace.speech_started()  # he asked a second thing
    trace.speech_stopped()
    trace.audio_delta()
    rows = trace.finish("idle timeout", session=3)

    assert [r["turn"] for r in rows] == [0, 1]
    assert rows[0]["chime_enqueued"] == 0.1
    assert "chime_enqueued" not in rows[1]  # the activation happened once
    assert rows[1]["first_speech"] > rows[0]["playback_end"]
    assert all(r["ended_by"] == "idle timeout" and r["session"] == 3 for r in rows)
    assert trace.console_note() == "first audio 0.1 s"  # speech end → her voice


# ── the idle path: the wakes that almost happened ──────────────────────────


class ScriptedWake:
    """A detector reading its scores off a script instead of a microphone."""

    def __init__(self, scores: list[float], threshold: float = 0.5) -> None:
        self._scores = list(scores)
        self.threshold = threshold
        self.last_score = 0.0

    def detect(self, frame: bytes) -> bool:
        self.last_score = self._scores.pop(0)
        return self.last_score >= self.threshold


class Frames:
    def __init__(self, count: int) -> None:
        self._left = count

    async def get_frame(self) -> bytes:
        self._left -= 1
        if self._left < 0:
            await asyncio.sleep(3600)
        return b"\x00\x00" * 1280


async def test_the_idle_path_keeps_the_wakes_that_almost_fired(tmp_path: Path) -> None:
    ticks = iter(i / 2 for i in range(1, 500))  # every clock read is half a second on
    log = LatencyLog(tmp_path / "turns.jsonl", now=lambda: next(ticks), wall=lambda: 1000.0)
    wake = ScriptedWake([0.05, 0.31, 0.42, 0.05, 0.05, 0.10, 0.70])
    trace: TurnTrace | None = None

    def on_score(score: float, fired: bool) -> None:
        nonlocal trace
        if fired:
            trace = log.wake(score)
        else:
            log.near_miss(score, wake.threshold)

    assert await wait_for_trigger(Frames(7), wake, None, on_score=on_score) == "wake"

    misses = [r for r in log.read() if r["kind"] == "wake_miss"]
    assert len(misses) == 1  # one utterance, one row — not one per frame
    assert misses[0]["wake_score"] == 0.42  # its peak, so thresholds can be compared
    assert misses[0]["threshold"] == 0.5
    assert not [r for r in log.read() if r["kind"] == "turn"]  # the turn is still open

    assert trace is not None
    trace.finish("idle timeout", session=1)
    turns = [r for r in log.read() if r["kind"] == "turn"]
    assert [r["wake_score"] for r in turns] == [0.7]


# ── "how fast were you today?" ─────────────────────────────────────────────


def turn_row(**fields: object) -> dict:
    row = {"kind": "turn", "turn": 0, "ts": time.time(), "ended_by": "wrap-up", "session": 1}
    row.update(fields)
    return row


def test_the_report_names_a_median_and_the_slowest_step() -> None:
    rows = [
        turn_row(chime_audible=0.2, speech_end=2.0, first_audio=2.8, tools=[["get_lights", 0.4]]),
        turn_row(chime_audible=0.4, speech_end=3.0, first_audio=4.0, tools=[["web_search", 4.2]]),
        turn_row(chime_audible=0.3, speech_end=1.0, first_audio=2.2),
        {"kind": "wake_miss", "ts": time.time(), "wake_score": 0.3, "threshold": 0.5},
    ]
    said = latency_report(rows, "today")
    assert said.startswith("Today, my chime came 0.3 seconds after the wake word")
    assert "I started answering 1.0 seconds after you stopped talking" in said
    assert "the median over 3 turns" in said
    assert said.endswith("My slowest step was web_search, at 4.2 seconds.")


def test_an_empty_log_says_so_plainly() -> None:
    assert latency_report([], "today") == "I haven't timed any turns today yet."
    misses = [{"kind": "wake_miss", "ts": time.time(), "wake_score": 0.3, "threshold": 0.5}]
    assert latency_report(misses, "yesterday") == "I haven't timed any turns yesterday yet."


def test_the_window_is_named_the_way_she_would_say_it() -> None:
    assert since_label("today") == "today"
    assert since_label("week") == "this week"
    assert since_label("6") == "in the last 6 hours"
    assert since_label("") == "today"


async def test_the_tool_reads_the_log_and_returns_a_sentence(tmp_path: Path) -> None:
    log = LatencyLog(tmp_path / "turns.jsonl")
    engine, _client = make_engine(log)

    empty, is_error = engine._execute_latency_tool({})
    assert not is_error and empty == "I haven't timed any turns today yet."

    log.append(
        [
            turn_row(chime_audible=0.2, speech_end=1.0, first_audio=1.9, tools=[["think", 2.5]]),
            turn_row(ts=time.time() - 86400 * 3, chime_audible=9.9, speech_end=1.0, first_audio=9.0),
        ]
    )
    said, is_error = engine._execute_latency_tool({"since": "today"})
    assert not is_error
    assert "0.2 seconds after the wake word" in said  # three days ago is not today
    assert "0.9 seconds after you stopped talking" in said
    assert "think, at 2.5 seconds" in said

    bad, is_error = engine._execute_latency_tool({"since": "the day before the flood"})
    assert is_error and "since must be" in bad


def test_rows_stay_small_and_content_light(tmp_path: Path) -> None:
    """No transcripts, no audio: a row is numbers and one reason string."""
    log = LatencyLog(tmp_path / "turns.jsonl")
    trace = log.wake(0.6)
    trace.speech_started()
    trace.speech_stopped()
    trace.audio_delta()
    trace.tool("set_lights", 0.4)
    trace.interrupted(0.12)
    trace.playback_done()
    trace.finish("stop command", session=2)

    line = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").strip()
    assert len(line) < 400
    row = json.loads(line)
    assert row["interruptions"] == 1 and row["interrupt_gaps"] == [0.12]
    assert set(row) <= {
        "kind", "turn", "ts", "session", "ended_by", "wake_score", "chime_enqueued",
        "chime_audible", "mic_ready", "connected", "first_speech", "speech_end",
        "transcript_at", "first_call", "first_audio", "playback_end", "tools", "interruptions",
        "interrupt_gaps",
    }
