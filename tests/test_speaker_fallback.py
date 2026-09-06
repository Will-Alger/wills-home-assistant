"""She must be heard in the room: a device that refuses a sample rate on one
host API is opened through its twin on another, and Windows' default output
being Steam's virtual speakers never leaves her talking into a void."""

from __future__ import annotations

import sounddevice as sd

from assistant.audio import devices
from assistant.audio import speaker as speaker_module
from assistant.audio.speaker import Speaker, output_candidates

HANDS_FREE = "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0 ;(Will's AirPods Pro #2 - Find My))"
OUTPUTS = [
    {"name": "Microsoft Sound Mapper - Output", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    {"name": "Speakers (Steam Streaming Speak", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    {"name": "Speakers (Echo Dot-6FG)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
    {"name": "Speakers (Steam Streaming Speakers)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 2},
    {"name": "Speakers (Echo Dot-6FG)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 2},
    {"name": "G34WQC A (NVIDIA High Definition Audio)", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 2},
    {"name": HANDS_FREE, "max_input_channels": 0, "max_output_channels": 1, "hostapi": 3},
]
APIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]


class Rig:
    def __init__(self, monkeypatch, *, default: int, failing: set[int] = frozenset()) -> None:
        self.default = default
        self.failing = set(failing)
        self.opened: list[int | None] = []
        rig = self

        def query_devices(device=None, kind=None):
            if device is not None:
                return OUTPUTS[int(device)]
            if kind == "output":
                return OUTPUTS[rig.default]
            if kind == "input":
                raise ValueError("no input")
            return OUTPUTS

        class FakeStream:
            def __init__(self, *, device=None, **_kw) -> None:
                if device in rig.failing:
                    raise sd.PortAudioError("Error opening RawOutputStream: Invalid sample rate [PaErrorCode -9997]")
                rig.opened.append(device)

            def start(self) -> None: ...
            def stop(self) -> None: ...
            def close(self) -> None: ...

        monkeypatch.setattr(devices.sd, "query_devices", query_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: APIS)
        monkeypatch.setattr(speaker_module.sd, "RawOutputStream", FakeStream)


def test_twins_are_the_same_device_through_other_host_apis(monkeypatch) -> None:
    Rig(monkeypatch, default=4)
    assert devices.twins(4, "output") == [2]  # the Echo Dot's MME entry
    assert devices.twins(3, "output") == [1]  # Steam's 31-character MME cut counts as the same device
    assert devices.twins(5, "output") == [] and devices.twins(99, "output") == []
    assert output_candidates() == [4, 5]  # Echo Dot before the monitor's HDMI audio; Steam and hands-free never


async def test_a_rate_refusal_opens_the_same_speaker_through_mme(monkeypatch) -> None:
    rig = Rig(monkeypatch, default=4, failing={4})
    async with Speaker(24_000, device="Echo Dot") as spk:
        assert rig.opened == [2] and spk.device_note is None  # no fallback happened
        assert spk.device_in_use == "Speakers (Echo Dot-6FG)"


async def test_a_steam_default_yields_to_real_speakers(monkeypatch) -> None:
    rig = Rig(monkeypatch, default=3)
    async with Speaker(24_000) as spk:
        assert rig.opened == [4] and spk.device_in_use == "Speakers (Echo Dot-6FG)"
        assert spk.device_note == (
            "the default output 'Speakers (Steam Streaming Speakers)' isn't a speaker — using 'Speakers (Echo Dot-6FG)'"
        )
    rig.opened.clear()
    rig.failing.add(6)  # the hands-free link won't take 24 kHz either
    async with Speaker(24_000, device="AirPods") as spk:
        assert rig.opened == [4]
        assert spk.device_note.startswith("speaker 'AirPods' would not open")
        assert spk.device_note.endswith("isn't a speaker — using 'Speakers (Echo Dot-6FG)'")
    rig.opened.clear()
    rig.failing.update({4, 5})
    async with Speaker(24_000) as spk:  # nothing real opens: the default, said plainly
        assert rig.opened == [None] and "nothing else would open" in spk.device_note


async def test_a_missing_speaker_falls_back_to_a_real_default_quietly(monkeypatch) -> None:
    rig = Rig(monkeypatch, default=4, failing={6})
    async with Speaker(24_000, device="AirPods") as spk:  # only the hands-free link is left, and it refuses
        assert rig.opened == [None] and spk.device_in_use == "Speakers (Echo Dot-6FG)"
        assert spk.device_note.startswith("speaker 'AirPods' would not open") and spk.device_note.endswith("using the default")
    rig.opened.clear()
    async with Speaker(24_000, device="Bose") as spk:
        assert rig.opened == [None] and spk.device_note == "speaker 'Bose' isn't plugged in — using the default"
