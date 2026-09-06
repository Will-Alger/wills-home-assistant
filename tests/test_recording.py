"""Session recordings: both sides of the audio and every engine decision on
one timeline, browsable from the panel, with the read-aloud test script
stamped into it. No audio device, no window, no network."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.recording import SCRIPT, Recorder, describe
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


async def test_a_recorded_conversation_leaves_both_sides_and_a_timeline(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec", now=lambda: 1_800_000_000.0)
    recorder.arm()
    for _ in range(10):
        recorder.mic(FRAME)  # the idle room, before the wake: the pre-roll ring
    name = recorder.begin("wake")
    assert name and recorder.active
    recorder.event("wake", score=0.71, threshold=0.5)

    engine, client = make(idle_timeout_s=0.6)
    engine.tap = recorder.event
    mic, speaker = LevelMic(), InstantSpeaker()
    mic.tap, speaker.tap = recorder.mic, recorder.spoke
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("turn off the hallway", reply="Done.")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, None, QuietUi()), 6)
    await turn
    summary = recorder.end(ended_by=stats.ended_by, transcript=stats.transcript, replied=stats.replied)
    assert summary is not None and not recorder.active

    folder = tmp_path / "rec" / name
    mic_frames, rate = wav_frames(folder / "mic.wav")
    spk_frames, _ = wav_frames(folder / "speaker.wav")
    assert rate == 24_000
    assert mic_frames > 10 * 1920  # the pre-roll, then the session
    assert spk_frames > 10 * 1920  # silence for the pre-roll, then her audio
    assert not list(folder.glob("*.pcm"))  # the raw files became WAVs

    kinds = [row["kind"] for row in events_of(folder)]
    assert kinds[0] == "recording" and kinds[1] == "wake" and kinds[-1] == "ended"
    for expected in ("speech_started", "you_said", "response_created", "audio_first", "alexa_said", "session_over"):
        assert expected in kinds, expected
    said = next(row for row in events_of(folder) if row["kind"] == "you_said")
    assert said["text"] == "turn off the hallway" and said["t"] > 0.8  # after the 0.8 s pre-roll

    assert summary["turns"] == 1 and summary["replied"] is True
    assert summary["preroll_s"] == 0.8 and summary["duration_s"] > 0.8
    assert summary["ended_by"] == stats.ended_by
    assert ["you", "turn off the hallway"] in summary["transcript"]
    assert "1 turn" in describe(summary)


async def test_the_gates_decisions_are_on_the_timeline_with_their_numbers(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    recorder.arm()
    recorder.begin()
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
    recorder.end(ended_by="idle timeout")
    gates = [row for row in events_of(tmp_path / "rec" / next(iter(recorder.list()))["name"]) if row["kind"] == "gate"]
    assert [g["open"] for g in gates][:2] == [True, False]
    assert gates[0]["level"] >= 3000 and 0 < gates[0]["floor"] < 3000


def test_script_steps_are_stamped_even_when_clicked_before_the_session_opens(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    recorder.arm()
    recorder.step(1, SCRIPT[0][0])  # "say nothing for ten seconds" — nothing is open yet
    name = recorder.begin()
    recorder.step(2, SCRIPT[1][0])
    recorder.end(ended_by="nobody spoke")
    rows = events_of(tmp_path / "rec" / name)
    steps = [(row["n"], row["t"]) for row in rows if row["kind"] == "script_step"]
    assert [n for n, _ in steps] == [1, 2]
    assert rows[1]["kind"] == "script_step"  # step 1 first thing after the header


def test_nothing_is_written_unless_armed_and_a_tap_of_none_is_free(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    for _ in range(5):
        recorder.mic(FRAME)
    assert len(recorder._ring) == 0  # not armed: no ring either
    assert recorder.begin() is None and not recorder.active
    recorder.event("wake")
    recorder.spoke(b"\x00" * 100)
    assert recorder.list() == [] and not any((tmp_path / "rec").glob("*/"))
    engine, _client = make()
    engine._tap("anything", n=1)  # no tap: a no-op, never an error


def test_a_crash_mid_session_still_leaves_readable_files(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    recorder.arm()
    name = recorder.begin()
    for _ in range(25):
        recorder.mic(FRAME)
    recorder.spoke(b"\x00" * 3840)
    recorder.event("you_said", text="hello")
    del recorder  # no end(): the process died here
    fixed = Recorder(tmp_path / "rec")
    folder = tmp_path / "rec" / name
    frames, _ = wav_frames(folder / "mic.wav")
    assert frames == 25 * 1920 and not list(folder.glob("*.pcm"))
    summary = fixed.list()[0]
    assert summary["ended_by"] == "crashed" and summary["duration_s"] == 2.0


class FakeView:
    def __init__(self, panel: SettingsPanel) -> None:
        self.panel = panel

    def start(self) -> None: ...

    def stop(self) -> None: ...


def test_the_panel_arms_lists_notes_plays_and_deletes(tmp_path: Path) -> None:
    played: list[tuple[int, int]] = []
    stopped: list[bool] = []
    recorder = Recorder(tmp_path / "rec")
    panel = SettingsPanel(
        AssistantStatus(mic="Blue Snowball"),
        PanelOverrides(tmp_path / "panel.json"),
        view_factory=FakeView,
        recorder=recorder,
        player=lambda pcm, rate: played.append((len(pcm), rate)),
        stopper=lambda: stopped.append(True),
    )
    assert panel.recording is False and panel.snapshot()["recording"] is False
    assert panel.set_recording(True).startswith("recording sessions")
    assert Recorder(tmp_path / "rec").armed  # persisted: a restart keeps recording

    name = recorder.begin("wake")
    recorder.mic(FRAME)
    recorder.end(ended_by="wrap-up", transcript=[("you", "that's all"), ("alexa", "Bye.")], replied=True)
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
    assert panel.set_recording(False) == "not recording sessions"
    assert not Recorder(tmp_path / "rec").armed


def test_the_test_script_turns_recording_on_and_walks_its_steps(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    panel = SettingsPanel(
        AssistantStatus(), PanelOverrides(tmp_path / "panel.json"), view_factory=FakeView, recorder=recorder
    )
    assert panel.script_step() == 0 and len(panel.script()) == 12
    n, say, expect = panel.next_step()
    assert (n, say, expect) == (1, *SCRIPT[0])
    assert panel.recording  # the first click armed it
    assert recorder._pending and recorder._pending[0]["n"] == 1  # held for the next session
    for _ in range(11):
        assert panel.next_step() is not None
    assert panel.next_step() is None and panel.script_step() == 12
    panel.restart_script()
    assert panel.script_step() == 0


def test_the_voice_tool_records_the_conversation_it_was_said_in(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec")
    panel = SettingsPanel(
        AssistantStatus(), PanelOverrides(tmp_path / "panel.json"), view_factory=FakeView, recorder=recorder
    )
    engine, _client = make(panel=panel)
    engine.recorder = recorder
    text, is_error = engine._execute_panel_tool("start_recording")
    assert not is_error and "recorded from here" in text
    assert recorder.armed and recorder.active and engine.tap == recorder.event
    engine._tap("you_said", text="record this session")
    text, is_error = engine._execute_panel_tool("stop_recording")
    assert not is_error and not recorder.armed and recorder.active  # this one still finishes
    summary = recorder.end(ended_by="wrap-up")
    assert summary is not None and summary["events"] >= 2
    tools = {t["name"] for t in __import__("assistant.engines.realtime_engine", fromlist=["PANEL_TOOLS"]).PANEL_TOOLS}
    assert {"start_recording", "stop_recording"} <= tools


async def test_a_false_wake_recording_ends_as_nobody_spoke(tmp_path: Path, monkeypatch) -> None:
    from assistant.engines import realtime_engine as mod

    monkeypatch.setattr(mod, "_NO_SPEECH_S", 0.2)
    recorder = Recorder(tmp_path / "rec")
    recorder.arm()
    recorder.begin()
    engine, _client = make(idle_timeout_s=5.0)
    engine.tap = recorder.event
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi()), 6)
    summary = recorder.end(ended_by=stats.ended_by)
    assert summary is not None and summary["ended_by"] == "nobody spoke" and summary["turns"] == 0
    kinds = [row["kind"] for row in events_of(tmp_path / "rec" / summary["name"])]
    assert "nobody_spoke" in kinds
