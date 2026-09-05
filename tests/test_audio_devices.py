"""Choosing the microphone and speaker: distinct names, saved choices, a live
switch without a restart, honest fallbacks — with PortAudio stubbed out."""

from __future__ import annotations

import json
from pathlib import Path

import sounddevice as sd

from assistant.app import wait_for_trigger
from assistant.audio import devices, tones
from assistant.audio import speaker as speaker_module
from assistant.audio.speaker import Speaker
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.status import AssistantStatus
from tests.test_app import NOISE, Source, Wake
from tests.test_panel import FakeView, make_engine

FAKE_DEVICES = [
    {"name": "Microphone (Blue Snowball)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
    {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    {"name": "Headphones (Will's AirPods Pro)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    {"name": "Microphone (Blue Snowball)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2},
    {"name": "Headphones (Will's AirPods Pro)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 2},
    {"name": "Headset (Will's AirPods Pro)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2},
    {"name": "Stereo Mix (Realtek)", "max_input_channels": 2, "max_output_channels": 0, "hostapi": 3},
]
FAKE_APIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]


def fake_query_devices(device=None, kind=None):
    if device is not None:
        return FAKE_DEVICES[int(device)]
    if kind == "input":
        return FAKE_DEVICES[0]
    if kind == "output":
        return FAKE_DEVICES[1]
    return FAKE_DEVICES


def patch_sd(monkeypatch) -> None:
    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(devices.sd, "query_hostapis", lambda: FAKE_APIS)


def test_names_collapse_host_apis_and_prefer_wasapi(monkeypatch) -> None:
    patch_sd(monkeypatch)
    assert devices.names("output") == ["System default", "Headphones (Will's AirPods Pro)", "Speakers (Realtek)"]
    assert devices.names("input") == [
        "System default", "Headset (Will's AirPods Pro)", "Microphone (Blue Snowball)", "Stereo Mix (Realtek)",
    ]
    assert devices.find("airpods", "output") == 4  # the WASAPI entry, not MME's index 2
    assert devices.find("snowball", "input") == 3 and devices.find("3", "input") == 3
    assert devices.find("", "output") is None and devices.find("System default", "input") is None
    assert devices.find("bose", "output") is None
    assert devices.describe("airpods", "output") == "Headphones (Will's AirPods Pro)"
    assert devices.describe("", "input") == "Microphone (Blue Snowball)"
    assert "not found" in devices.describe("bose", "output")
    tones.set_output("AirPods")
    assert tones._OUTPUT == 4
    tones.set_output("")
    assert tones._OUTPUT is None


def test_overrides_save_and_apply_the_audio_devices(tmp_path: Path) -> None:
    overrides = PanelOverrides(tmp_path / "panel.json")
    overrides.set(microphone="Snowball", speaker="AirPods")
    assert overrides.microphone == "Snowball" and overrides.speaker == "AirPods"

    class Settings:
        realtime_voice = "sol"
        wake_model = "alexa"
        audio_input_device = ""
        audio_output_device = ""

    settings = Settings()
    assert sorted(overrides.apply(settings)) == ["audio_input_device=Snowball", "audio_output_device=AirPods"]
    assert settings.audio_output_device == "AirPods"
    overrides.clear("speaker")
    assert overrides.speaker == "" and overrides.microphone == "Snowball"
    assert json.loads((tmp_path / "panel.json").read_text(encoding="utf-8")) == {"audio_input_device": "Snowball"}


def test_panel_switches_devices_live_and_saves_them(tmp_path: Path, monkeypatch) -> None:
    patch_sd(monkeypatch)
    changes: list[tuple[str, str]] = []
    rescans: list[bool] = []
    status = AssistantStatus(mic="Microphone (Blue Snowball)", voice="sol", wake_word="alexa")
    panel = SettingsPanel(
        status,
        PanelOverrides(tmp_path / "panel.json"),
        view_factory=FakeView,
        on_audio_change=lambda mic, spk: changes.append((mic, spk)),
        on_refresh_devices=lambda: rescans.append(True),
        devices=devices,
    )
    assert panel.microphone_choices()[1] == "Headset (Will's AirPods Pro)"
    assert panel.speaker_choices() == devices.names("output")

    text = panel.save(speaker="airpods")
    assert text == "saved speaker airpods — used from the next conversation"
    assert changes == [("", "airpods")]
    assert PanelOverrides(tmp_path / "panel.json").speaker == "airpods"
    assert panel.snapshot()["saved_speaker"] == "airpods"

    text = panel.save(microphone="Headset (Will's AirPods Pro)", speaker="System default")
    assert text.startswith("saved microphone Headset (Will's AirPods Pro), speaker System default")
    assert changes[-1] == ("Headset (Will's AirPods Pro)", "")  # default = no override
    assert PanelOverrides(tmp_path / "panel.json").speaker == ""

    text = panel.save(speaker="bose")
    assert text.startswith("no speaker matches 'bose'") and "Speakers (Realtek)" in text
    assert len(changes) == 2  # nothing applied

    text = panel.save(voice="cedar", speaker="Realtek")
    assert "voice cedar — restart to apply" in text and "speaker Realtek — used from the next conversation" in text
    assert "re-scanning" in panel.refresh_devices() and rescans == [True]
    assert any("saved speaker airpods" in line for line in status.snapshot()["log"])  # type: ignore[union-attr]


async def test_a_device_change_reopens_the_idle_mic() -> None:
    flips = iter([False, True])
    assert await wait_for_trigger(Source([NOISE] * 5), Wake(), None, reconfigure=lambda: next(flips)) == "reconfigure"


async def test_speaker_falls_back_to_the_default_with_a_note(monkeypatch) -> None:
    patch_sd(monkeypatch)
    opened: list[int | None] = []

    class FakeStream:
        def __init__(self, *, device=None, **_kw) -> None:
            if device == 4:
                raise sd.PortAudioError("Error opening RawOutputStream: Device unavailable")
            opened.append(device)

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def close(self) -> None: ...

    monkeypatch.setattr(speaker_module.sd, "RawOutputStream", FakeStream)
    async with Speaker(24_000, device="airpods") as spk:
        assert "would not open" in (spk.device_note or "")
        assert spk.device_in_use == "Speakers (Realtek)" and opened == [None]
    async with Speaker(24_000, device="bose") as spk:
        assert "isn't plugged in" in (spk.device_note or "") and spk.device_in_use == "Speakers (Realtek)"
    async with Speaker(24_000) as spk:
        assert spk.device_note is None and spk.device_in_use == "Speakers (Realtek)"


async def test_audio_devices_by_voice(tmp_path: Path, monkeypatch) -> None:
    patch_sd(monkeypatch)
    changes: list[tuple[str, str]] = []
    panel = SettingsPanel(
        AssistantStatus(), PanelOverrides(tmp_path / "panel.json"), view_factory=FakeView,
        on_audio_change=lambda mic, spk: changes.append((mic, spk)), devices=devices,
    )
    engine = make_engine(panel)
    names = {tool["name"] for tool in (await engine._session_config(None))["tools"]}
    assert {"use_audio_device", "list_audio_devices", "refresh_audio_devices"} <= names

    text, is_error = engine._execute_panel_tool("list_audio_devices", {})
    listed = json.loads(text)
    assert not is_error and "Headphones (Will's AirPods Pro)" in listed["speakers"]
    assert listed["saved"] == {"microphone": "System default", "speaker": "System default"}
    text, is_error = engine._execute_panel_tool("use_audio_device", {"speaker": "airpods", "microphone": "airpods"})
    assert not is_error and text.startswith("saved microphone airpods, speaker airpods")
    assert changes == [("airpods", "airpods")]
    text, is_error = engine._execute_panel_tool("use_audio_device", {"speaker": "bose"})
    assert is_error and "no speaker matches" in text
    text, is_error = engine._execute_panel_tool("refresh_audio_devices", {})
    assert not is_error and "isn't wired up" in text  # no re-scan callback in this test
