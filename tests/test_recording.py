"""Session recordings: from Start to End, both sides of the audio and every
engine decision on one timeline across every conversation in between,
browsable from the panel, with the read-aloud test script stamped into it.
No audio device, no window, no network."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.recording import SCRIPT, Recorder, clock, describe
from assistant.status import AssistantStatus
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi

FRAME = np.full(1920, 60, dtype=np.int16).tobytes()  # 80 ms of quiet room at 24 kHz


class LevelMic:
    """A microphone whose loudness the test sets, tapped like the real one."""

    def __init__(self) -> None:
        self.level = 60
        self.tap = None

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        frame = np.full(1920, self.level, dtype=np.int16).tobytes()
        if self.tap is not None:
            self.tap(frame)
        return frame

    def drain(self) -> None: ...


def make(**kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa", **kw
    )
    client = FakeClient()
    engine._client = client
    return engine, client


def events_of(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def wav_frames(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes(), wav.getframerate()


async def test_one_recording_spans_the_idle_room_and_two_conversations(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec", now=lambda: 1_800_000_000.0)
    name = recorder.start()
    assert name and recorder.active and recorder.start() == name  # a second start is the same one
    for _ in range(10):
        recorder.mic(FRAME)  # the idle room, before anyone says her name
    mic, speaker = LevelMic(), InstantSpeaker()
    mic.tap, speaker.tap = recorder.mic, recorder.spoke

    for text in ("turn off the hallway", "and the porch"):
        engine, client = make(idle_timeout_s=0.6)
        engine.tap = recorder.event
        conn = client.connection
        assert recorder.session_started("wake")
        recorder.event("wake", score=0.71, threshold=0.5)

        async def owner(conn=conn, text=text) -> None:
            await asyncio.sleep(0.15)
            conn.user_says(text, reply="Done.")

        turn = asyncio.create_task(owner())
        stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, None, QuietUi()), 6)
        await turn
        recorder.session_ended(ended_by=stats.ended_by, transcript=stats.transcript, replied=stats.replied)
        assert recorder.active  # the recording outlives the conversation

    assert recorder.elapsed > 0
    summary = recorder.stop()
    assert summary is not None and not recorder.active and recorder.elapsed == 0.0

    folder = tmp_path / "rec" / name
    mic_frames, rate = wav_frames(folder / "mic.wav")
    spk_frames, _ = wav_frames(folder / "speaker.wav")
    assert rate == 24_000 and mic_frames > 10 * 1920 and spk_frames > 0
    assert not list(folder.glob("*.pcm"))  # the raw files became WAVs

    rows = events_of(folder)
    kinds = [row["kind"] for row in rows]
    assert kinds[0] == "recording" and rows[0]["what"] == "started"
    assert kinds[-1] == "recording" and rows[-1]["what"] == "stopped"
    assert kinds.count("session") == 4  # opened, closed, twice
    for expected in ("wake", "speech_started", "you_said", "response_created", "audio_first", "alexa_said", "session_over"):
        assert expected in kinds, expected
    said = [row["text"] for row in rows if row["kind"] == "you_said"]
    assert said == ["turn off the hallway", "and the porch"]

    assert summary["conversations"] == 2 and summary["turns"] == 2
    assert [s["ended_by"] for s in summary["sessions"]] == [stats.ended_by] * 2
    assert ["you", "and the porch"] in summary["sessions"][1]["transcript"]
    assert summary["duration_s"] > 0.8 and summary["ended_by"] == "ended"
    assert "2 conversations" in describe(summary) and "2 turns" in describe(summary)


async def test_the_gates_decisions_are_on_the_timeline_with_their_numbers(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    recorder.start()
    engine, _client = make(idle_timeout_s=2.0)  # long enough for the gate's 0.8 s hangover to close it
    engine.tap = recorder.event
    mic = LevelMic()

    async def owner() -> None:
        await asyncio.sleep(0.3)
        mic.level = 3000
        await asyncio.sleep(0.2)
        mic.level = 60

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(mic, InstantSpeaker(), None, QuietUi()), 6)
    await turn
    summary = recorder.stop()
    gates = [row for row in events_of(tmp_path / "rec" / summary["name"]) if row["kind"] == "gate"]
    assert [g["open"] for g in gates][:2] == [True, False]
    assert gates[0]["level"] >= 3000 and 0 < gates[0]["floor"] < 3000 and gates[0]["speech"] is None


def test_script_steps_are_stamped_while_a_recording_runs(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    recorder.step(1, SCRIPT[0][0])  # nothing is recording: nowhere to put it
    name = recorder.start()
    recorder.step(1, SCRIPT[0][0])
    recorder.step(2, SCRIPT[1][0])
    summary = recorder.stop()
    rows = events_of(tmp_path / "rec" / name)
    assert [(row["n"]) for row in rows if row["kind"] == "script_step"] == [1, 2]
    assert summary["steps"] == 2


def test_nothing_is_written_unless_recording_and_a_tap_of_none_is_free(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    for _ in range(5):
        recorder.mic(FRAME)
    recorder.event("wake")
    recorder.spoke(b"\x00" * 100)
    assert not recorder.session_started("wake")
    recorder.session_ended(ended_by="x")
    assert recorder.stop() is None
    assert recorder.list() == [] and not (tmp_path / "rec").exists()
    engine, _client = make()
    engine._tap("anything", n=1)  # no tap: a no-op, never an error


def test_a_crash_mid_recording_still_leaves_readable_files(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    name = recorder.start()
    for _ in range(25):
        recorder.mic(FRAME)
    recorder.spoke(b"\x00" * 3840)
    recorder.session_started("wake")
    recorder.event("you_said", text="hello")
    del recorder  # no stop(): the process died here
    fixed = Recorder(tmp_path / "rec")
    folder = tmp_path / "rec" / name
    frames, _ = wav_frames(folder / "mic.wav")
    assert frames == 25 * 1920 and not list(folder.glob("*.pcm"))
    summary = fixed.list()[0]
    assert summary["ended_by"] == "crashed" and summary["duration_s"] == 2.0 and summary["conversations"] == 1


def test_a_forgotten_recording_stops_itself(tmp_path: Path) -> None:
    now = [100.0]
    recorder = Recorder(tmp_path / "rec", max_s=60.0, clock=lambda: now[0])
    name = recorder.start()
    recorder.mic(FRAME)
    now[0] += 61
    recorder.mic(FRAME)  # the next frame past the limit ends it
    assert not recorder.active and "stopped itself" in recorder.stopped_by
    assert recorder.list()[0]["name"] == name and "stopped itself" in recorder.list()[0]["ended_by"]


def test_speaker_bytes_stashed_from_the_audio_thread_reach_the_file(tmp_path: Path) -> None:
    """The speaker's tap runs in the PortAudio callback: it only stashes;
    the mic tap (loop thread) and stop() write the stash out."""
    recorder = Recorder(tmp_path / "rec")
    name = recorder.start()
    recorder.spoke(b"\x01\x00" * 1000)
    assert len(recorder._spk_pending) == 2000  # stashed, not written
    recorder.mic(FRAME)
    assert len(recorder._spk_pending) == 0
    assert (tmp_path / "rec" / name / "speaker.pcm").stat().st_size == 2000
    recorder.spoke(b"\x02\x00" * 500)
    recorder.stop()
    frames, _ = wav_frames(tmp_path / "rec" / name / "speaker.wav")
    assert frames == 1500


def test_the_timer_reads_like_a_clock() -> None:
    assert clock(0) == "00:00" and clock(65.7) == "01:05" and clock(3600 + 61) == "1:01:01"


class FakeView:
    def __init__(self, panel: SettingsPanel) -> None:
        self.panel = panel

    def start(self) -> None: ...

    def stop(self) -> None: ...


def test_the_panel_starts_ends_lists_notes_plays_and_deletes(tmp_path: Path) -> None:
    played: list[tuple[int, int]] = []
    stopped: list[bool] = []
    now = [10.0]
    recorder = Recorder(tmp_path / "rec", clock=lambda: now[0])
    panel = SettingsPanel(
        AssistantStatus(mic="Blue Snowball"),
        PanelOverrides(tmp_path / "panel.json"),
        view_factory=FakeView,
        recorder=recorder,
        player=lambda pcm, rate: played.append((len(pcm), rate)),
        stopper=lambda: stopped.append(True),
    )
    assert panel.recording is False and panel.snapshot()["recording"] is False
    assert panel.set_recording(False) == "nothing was recording"
    assert panel.set_recording(True).startswith("recording — ")
    now[0] += 125
    assert panel.recording and panel.snapshot()["recording_elapsed"] == 125
    assert panel.set_recording(True) == "already recording — 02:05 so far"

    name = recorder.current
    recorder.mic(FRAME)
    recorder.session_started("wake")
    recorder.session_ended(ended_by="wrap-up", transcript=[("you", "that's all"), ("alexa", "Bye.")], replied=True)
    assert panel.recordings() == []  # the one running is not listed yet
    ended = panel.set_recording(False)
    assert ended.startswith("recording ended — ") and name in ended and "1 conversation" in ended
    assert not panel.recording
    rows = panel.recordings()
    assert [r["name"] for r in rows] == [name] and rows[0]["turns"] == 1

    assert panel.recording_note(name, "cut me off after 'repaint the'") == f"note saved on {name}"
    assert json.loads((tmp_path / "rec" / name / "summary.json").read_text())["note"] == "cut me off after 'repaint the'"
    assert "cut me off" in describe(panel.recordings()[0])
    assert panel.recording_note("nope", "x").startswith("no recording")

    assert panel.play_recording(name, "mic").startswith("playing the mic side")
    assert played == [(3840, 24_000)]
    assert panel.play_recording(name, "video").startswith("no video audio")
    panel.stop_playback()
    assert stopped == [True]

    assert panel.delete_recording(name) == f"deleted {name}"
    assert panel.recordings() == [] and not (tmp_path / "rec" / name).exists()


def test_the_test_script_starts_a_recording_and_walks_its_steps(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    panel = SettingsPanel(
        AssistantStatus(), PanelOverrides(tmp_path / "panel.json"), view_factory=FakeView, recorder=recorder
    )
    assert panel.script_step() == 0 and len(panel.script()) == 12
    n, say, expect = panel.next_step()
    assert (n, say, expect) == (1, *SCRIPT[0])
    assert panel.recording  # the first click started it
    for _ in range(11):
        assert panel.next_step() is not None
    assert panel.next_step() is None and panel.script_step() == 12
    summary = recorder.stop()
    assert summary["steps"] == 12
    panel.restart_script()
    assert panel.script_step() == 0


def test_the_voice_tools_start_and_end_a_recording(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    panel = SettingsPanel(
        AssistantStatus(), PanelOverrides(tmp_path / "panel.json"), view_factory=FakeView, recorder=recorder
    )
    engine, _client = make(panel=panel)
    text, is_error = engine._execute_panel_tool("start_recording")
    assert not is_error and text.startswith("recording — ") and recorder.active
    text, is_error = engine._execute_panel_tool("stop_recording")
    assert not is_error and text.startswith("recording ended") and not recorder.active
    tools = {t["name"] for t in __import__("assistant.engines.realtime_engine", fromlist=["PANEL_TOOLS"]).PANEL_TOOLS}
    assert {"start_recording", "stop_recording"} <= tools


async def test_a_false_wake_inside_a_recording_is_a_closed_session(tmp_path: Path, monkeypatch) -> None:
    from assistant.engines import realtime_engine as mod

    monkeypatch.setattr(mod, "_NO_SPEECH_S", 0.2)
    recorder = Recorder(tmp_path / "rec")
    recorder.start()
    engine, _client = make(idle_timeout_s=5.0)
    engine.tap = recorder.event
    recorder.session_started("wake")
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 6)
    recorder.session_ended(ended_by=stats.ended_by, transcript=stats.transcript, replied=stats.replied)
    summary = recorder.stop()
    assert summary["sessions"][0]["ended_by"] == "nobody spoke" and summary["turns"] == 0
    kinds = [row["kind"] for row in events_of(tmp_path / "rec" / summary["name"])]
    assert "nobody_spoke" in kinds
