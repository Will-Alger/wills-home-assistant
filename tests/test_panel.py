"""Task 7: the listening cues and the Settings panel — no audio device, no
window, no network."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from assistant.audio.cues import VoiceCues
from assistant.config import wake_phrase
from assistant.panel import PanelOverrides, PanelUnavailable, SettingsPanel
from assistant.status import AssistantStatus


class FakeSpeaker:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def enqueue(self, pcm: bytes) -> None:
        self.chunks.append(pcm)


def make_cues(status: AssistantStatus | None = None) -> tuple[VoiceCues, list[str]]:
    """Cues with the synthesis stubbed out: we assert on which earcon, not
    on samples (tones.py owns that)."""
    played: list[str] = []
    cues = VoiceCues(
        rate=24_000,
        status=status,
        play=played.append,
        render=lambda kind, rate: kind.encode(),
    )
    return cues, played


# ── the cues ──────────────────────────────────────────────────────────────


def test_a_listening_window_dings_open_and_closed() -> None:
    cues, played = make_cues()
    assert cues.start() and cues.listening
    assert cues.end() and not cues.listening
    assert played == ["wake", "listen_end"]


def test_repeat_calls_inside_one_window_are_silent() -> None:
    """A multi-step reply raises 'now listening' several times — one ding."""
    cues, played = make_cues()
    cues.start()
    assert cues.start() is False
    cues.end()
    assert cues.end() is False
    assert played == ["wake", "listen_end"]


def test_tones_go_through_an_open_session_speaker() -> None:
    cues, played = make_cues()
    speaker = FakeSpeaker()
    cues.start(speaker)
    cues.end(speaker)
    assert played == []  # nothing raced a second PortAudio stream
    assert speaker.chunks == [b"wake", b"listen_end"]


def test_a_failure_never_looks_like_listening() -> None:
    status = AssistantStatus(mic="Snowball")
    cues, played = make_cues(status)
    cues.start()
    cues.error("the default input would not open")
    assert played == ["wake", "error"]
    assert not cues.listening and not status.listening
    snapshot = status.snapshot()
    assert snapshot["state"] == "error" and "would not open" in str(snapshot["error"])
    assert any("would not open" in line for line in snapshot["log"])  # type: ignore[union-attr]


def test_session_end_chimes_once_and_reset_is_silent() -> None:
    cues, played = make_cues()
    cues.start()
    cues.session_end()
    assert played == ["wake", "close"] and not cues.listening
    cues.start()
    cues.reset()
    assert not cues.listening and played[-1] == "wake"


def test_a_missing_output_device_is_never_a_crash() -> None:
    def explode(kind: str) -> None:
        raise OSError("no output device")

    cues = VoiceCues(play=explode)
    cues.start()  # would have crashed the whole voice loop
    assert cues.listening


def test_listen_end_is_its_own_earcon() -> None:
    import numpy as np

    from assistant.audio.tones import _SOUNDS

    end, close, wake = _SOUNDS["listen_end"], _SOUNDS["close"], _SOUNDS["wake"]
    assert end.dtype == np.float32 and 2000 < len(end) < len(close)  # short and distinct
    assert np.max(np.abs(end)) <= 0.85

    def dominant_hz(chunk: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        return float(np.fft.rfftfreq(len(chunk), 1 / 22_050)[int(np.argmax(spectrum))])

    assert dominant_hz(end[:1500]) > dominant_hz(end[-2500:-1000])  # falls
    assert dominant_hz(wake[:1500]) < dominant_hz(end[:1500])  # opposite of the wake rise


# ── the status mirror ─────────────────────────────────────────────────────


def test_status_snapshot_carries_what_the_panel_shows() -> None:
    status = AssistantStatus(mic="Blue Snowball", voice="sol", wake_word="alexa")
    status.set_state("idle")
    status.note("● connected — talk")
    status.note("● connected — talk")  # a repeat is not a second line
    status.configure(voice="marin")
    status.set_listening(True)
    snapshot = status.snapshot()
    assert snapshot["mic"] == "Blue Snowball" and snapshot["voice"] == "marin"
    assert snapshot["wake_word"] == "alexa" and snapshot["listening"] is True
    assert str(snapshot["summary"]).startswith("idle")
    assert len(snapshot["log"]) == 1 and "connected" in snapshot["log"][0]  # type: ignore[index]


# ── the panel ─────────────────────────────────────────────────────────────


class FakeView:
    """Stands in for the Tk window."""

    instances: ClassVar[list[FakeView]] = []

    def __init__(self, panel: SettingsPanel) -> None:
        self.panel = panel
        self.started = False
        self.stopped = False
        FakeView.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def make_panel(tmp_path: Path, **kw) -> tuple[SettingsPanel, AssistantStatus]:
    status = AssistantStatus(mic="Blue Snowball", voice="sol", wake_word="alexa")
    panel = SettingsPanel(
        status,
        PanelOverrides(tmp_path / "data" / "panel.json"),
        view_factory=FakeView,
        **kw,
    )
    return panel, status


def test_panel_opens_and_closes_by_voice(tmp_path: Path) -> None:
    FakeView.instances.clear()
    panel, status = make_panel(tmp_path)
    assert not panel.is_open
    assert "desktop" in panel.open() and panel.is_open
    assert FakeView.instances[-1].started
    assert "already open" in panel.open()  # opening twice is not two windows
    assert len(FakeView.instances) == 1

    assert "closed" in panel.close() and not panel.is_open
    assert FakeView.instances[-1].stopped
    assert "isn't open" in panel.close()
    assert any("settings panel opened" in line for line in status.snapshot()["log"])  # type: ignore[union-attr]


def test_a_window_that_will_not_open_is_a_sentence_not_a_crash(tmp_path: Path) -> None:
    def broken(panel: SettingsPanel):
        raise RuntimeError("no display")

    panel, _status = make_panel(tmp_path)
    panel._view_factory = broken
    with pytest.raises(PanelUnavailable, match="no display"):
        panel.open()
    assert not panel.is_open


def test_closing_the_window_itself_is_noticed(tmp_path: Path) -> None:
    panel, _status = make_panel(tmp_path)
    panel.open()
    panel.view_closed()  # he clicked the X
    assert not panel.is_open
    assert "desktop" in panel.open()  # and she can open it again


def test_panel_shows_the_live_session_and_what_is_saved(tmp_path: Path) -> None:
    panel, status = make_panel(tmp_path)
    status.set_listening(True)
    snapshot = panel.snapshot()
    assert snapshot["mic"] == "Blue Snowball" and snapshot["listening"] is True
    assert snapshot["voice"] == "sol" and snapshot["wake_word"] == "alexa"
    assert snapshot["saved_voice"] == "" and "summary" in snapshot
    panel.save(voice="cedar")
    assert panel.snapshot()["saved_voice"] == "cedar"


def test_saving_a_voice_and_wake_word_survives_a_restart(tmp_path: Path) -> None:
    panel, _status = make_panel(tmp_path)
    assert "restart to apply" in panel.save(voice="cedar", wake_word="hey jarvis")

    overrides = PanelOverrides(tmp_path / "data" / "panel.json")
    assert overrides.voice == "cedar" and overrides.wake_model == "hey_jarvis"

    class Settings:  # stands in for the pydantic Settings object
        realtime_voice = "sol"
        wake_model = "alexa"

    settings = Settings()
    assert sorted(overrides.apply(settings)) == ["realtime_voice=cedar", "wake_model=hey_jarvis"]
    assert settings.realtime_voice == "cedar" and settings.wake_model == "hey_jarvis"
    assert overrides.apply(settings) == []  # already there: nothing changed


def test_the_panel_refuses_settings_she_could_not_load(tmp_path: Path) -> None:
    panel, _status = make_panel(tmp_path)
    assert "isn't one of her voices" in panel.save(voice="brian")
    assert "isn't a wake word" in panel.save(wake_word="hey nonsense")
    assert panel.save() == "nothing to save"
    assert PanelOverrides(tmp_path / "data" / "panel.json").voice == ""


def test_a_custom_model_is_offered_by_its_spoken_name(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "hey_gary.onnx").write_bytes(b"")
    panel, _status = make_panel(tmp_path, models_dir=models)
    choices = panel.wake_choices()
    assert choices["hey jarvis"] == "hey_jarvis"
    assert choices["hey gary (custom)"].endswith("hey_gary.onnx")
    assert "restart to apply" in panel.save(wake_word="hey gary (custom)")
    assert PanelOverrides(tmp_path / "data" / "panel.json").wake_model.endswith("hey_gary.onnx")
    assert wake_phrase(panel.snapshot()["saved_wake_word"]) == "hey gary"  # type: ignore[arg-type]


def test_unreadable_overrides_leave_the_env_alone(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text("{ this is not json", encoding="utf-8")
    overrides = PanelOverrides(path)
    assert overrides.voice == "" and overrides.wake_model == ""

    class Settings:
        realtime_voice = "sol"
        wake_model = "alexa"

    assert overrides.apply(Settings()) == []


def test_restart_button_asks_the_runner_to_restart(tmp_path: Path) -> None:
    restarts: list[bool] = []
    panel, status = make_panel(tmp_path, restart=lambda: restarts.append(True))
    assert "fifteen seconds" in panel.restart()
    assert restarts == [True]
    assert any("restart requested" in line for line in status.snapshot()["log"])  # type: ignore[union-attr]

    lonely, _status = make_panel(tmp_path)
    assert "isn't wired up" in lonely.restart()


def test_the_panel_has_no_listening_switch(tmp_path: Path) -> None:
    """Informational and configuration only — she is never muted from here."""
    panel, _status = make_panel(tmp_path)
    assert not [name for name in dir(panel) if "listen" in name and not name.startswith("_")]


# ── the voice tools ───────────────────────────────────────────────────────


def make_engine(panel):
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome

    return RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        name="Alexa", wake_phrase="alexa", panel=panel,
    )


async def test_open_and_close_settings_panel_are_offered_and_work(tmp_path: Path) -> None:
    panel, _status = make_panel(tmp_path)
    engine = make_engine(panel)
    config = await engine._session_config(None)
    names = {tool["name"] for tool in config["tools"]}
    assert {"open_settings_panel", "close_settings_panel"} <= names

    text, is_error = engine._execute_panel_tool("open_settings_panel")
    assert not is_error and "desktop" in text and panel.is_open
    text, is_error = engine._execute_panel_tool("close_settings_panel")
    assert not is_error and not panel.is_open


async def test_panel_tools_are_hidden_and_honest_without_a_panel() -> None:
    engine = make_engine(None)
    config = await engine._session_config(None)
    names = {tool["name"] for tool in config["tools"]}
    assert "open_settings_panel" not in names
    text, is_error = engine._execute_panel_tool("open_settings_panel")
    assert is_error and "isn't available" in text


async def test_a_panel_that_will_not_open_reports_it_out_loud(tmp_path: Path) -> None:
    panel, _status = make_panel(tmp_path)

    def broken(_panel):
        raise PanelUnavailable("this machine has no Tk")

    panel._view_factory = broken
    text, is_error = make_engine(panel)._execute_panel_tool("open_settings_panel")
    assert is_error and "no Tk" in text and "couldn't open" in text


# ── the cues inside a real conversation ───────────────────────────────────


async def test_every_turn_dings_open_and_closed(tmp_path: Path) -> None:
    """She asks him something, he answers: the listening window that opens
    for his reply dings, and closes with the other tone when he stops."""
    import asyncio

    from assistant.announce import Announcer
    from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi

    announcer = Announcer(tmp_path / "a.json")
    announcer.enqueue("Task 7 needs your call: one name or two?", kind="question")
    cues, _played = make_cues(AssistantStatus())
    engine = make_engine(None)
    engine._announcer = engine.announcer = announcer
    engine._cues = cues
    engine._info_close_s = 0.2
    client = FakeClient()
    engine._client = client
    speaker = InstantSpeaker()

    async def he_answers() -> None:
        async with asyncio.timeout(5):
            while not cues.listening:  # he speaks once she has stopped and dinged
                await asyncio.sleep(0.01)
        client.connection.user_says("just the first name")

    side = asyncio.create_task(he_answers())
    stats = await engine.run_conversation(NeverMic(), speaker, None, QuietUi(), announce=True)
    await side

    assert stats.replied
    # opened for his answer, closed when he finished — both through the open
    # session speaker, never a second output stream
    assert cues.played == ["wake", "listen_end"]
    assert b"wake" in speaker.chunks and b"listen_end" in speaker.chunks
    assert not cues.listening  # the session is over; nothing claims to be hearing him


async def test_a_tool_step_does_not_ding_before_she_has_spoken(tmp_path: Path) -> None:
    """A response that only ran a tool is followed by another one: the
    listening ding waits for it."""
    from types import SimpleNamespace

    from tests.fake_realtime import FakeClient

    engine = make_engine(None)
    connection = FakeClient().connection
    stats_module = __import__("assistant.engines.realtime_engine", fromlist=["SessionStats"])

    call = SimpleNamespace(type="function_call", name="get_lights", arguments="{}", call_id="c1")
    event = SimpleNamespace(response=SimpleNamespace(output=[call], usage=None))
    await engine._handle_response_done(connection, event, stats_module.SessionStats())
    assert engine.last_response_followup  # more audio is coming

    spoken = SimpleNamespace(response=SimpleNamespace(output=[], usage=None))
    await engine._handle_response_done(connection, spoken, stats_module.SessionStats())
    assert not engine.last_response_followup  # the floor is his again


# ── the boot guard ────────────────────────────────────────────────────────


def test_a_wake_word_from_the_panel_can_never_brick_the_boot(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import scripts.m4_realtime as runner

    class FakeDetector:
        def __init__(self, model: str, threshold: float = 0.5) -> None:
            if model.endswith(".onnx"):
                raise FileNotFoundError(f"Custom wake model not found: {model}")
            self.model = model

    monkeypatch.setattr(runner, "WakeDetector", FakeDetector)
    monkeypatch.setattr(runner, "load_settings", lambda: SimpleNamespace(wake_model="alexa"))
    overrides = PanelOverrides(tmp_path / "panel.json")
    overrides.set(wake_model="models/hey_gary.onnx")  # trained, then deleted
    settings = SimpleNamespace(
        wake_model="models/hey_gary.onnx", wake_threshold=0.5, wake_phrase="hey gary"
    )

    wake, session_wake = runner.load_wake_detectors(settings, overrides)

    assert wake.model == "alexa" and session_wake.model == "alexa"  # she still wakes up
    assert settings.wake_model == "alexa" and overrides.wake_model == ""  # the bad choice is gone

    with pytest.raises(FileNotFoundError):  # nothing to fall back to: say so loudly
        runner.load_wake_detectors(
            SimpleNamespace(wake_model="models/gone.onnx", wake_threshold=0.5), None
        )
