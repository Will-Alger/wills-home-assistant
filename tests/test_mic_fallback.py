"""The idle microphone must never sit silently on an input that hears nothing.

Windows makes Steam's streaming mic or a headset's hands-free endpoint the
default input without asking; both open fine and deliver silence. When the
saved microphone is unplugged (or none is saved) she tries real microphones
first, says plainly when none opens, and keeps re-scanning so the real one
is picked up the moment it is back.
"""

from __future__ import annotations

import sounddevice as sd

from assistant.app import RescanClock
from assistant.audio import devices
from assistant.audio import mic as mic_module
from assistant.audio.mic import Microphone, input_candidates

STEAM = "Microphone (Steam Streaming Mic"  # MME's 31-character cut of the full name
DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0, "hostapi": 0},
    {"name": STEAM, "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
    {"name": "Headset (Will's AirPods Pro #2 ", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
    {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2},
    {"name": "Microphone (Steam Streaming Microphone)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2},
    {"name": "Headset (Will's AirPods Pro #2 - Find My)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2},
]
SNOWBALL = {"name": "Microphone (Blue Snowball)", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 2}
APIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]


class Rig:
    """A fake PortAudio: a device list, a default input, and which indices refuse to open."""

    def __init__(self, monkeypatch, *, devices_: list[dict], default: int, failing: set[int] = frozenset()) -> None:
        self.devices = devices_
        self.default = default
        self.failing = set(failing)
        self.opened: list[int | None] = []
        rig = self

        def query_devices(device=None, kind=None):
            if device is not None:
                return rig.devices[int(device)]
            if kind == "input":
                return rig.devices[rig.default]
            if kind == "output":
                raise ValueError("no output")
            return rig.devices

        class FakeStream:
            def __init__(self, *, device=None, **_kw) -> None:
                if device in rig.failing:
                    raise sd.PortAudioError("Error opening RawInputStream: Invalid device [PaErrorCode -9996]")
                rig.opened.append(device)

            def start(self) -> None: ...
            def stop(self) -> None: ...
            def close(self) -> None: ...

        monkeypatch.setattr(devices.sd, "query_devices", query_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: APIS)
        monkeypatch.setattr(mic_module.sd, "RawInputStream", FakeStream)


def test_the_truncated_mme_twin_is_one_device(monkeypatch) -> None:
    Rig(monkeypatch, devices_=DEVICES, default=1)
    assert devices.names("input") == [
        "System default", "Headset (Will's AirPods Pro #2 - Find My)",
        "Microphone (Realtek HD Audio Mic input)", "Microphone (Steam Streaming Microphone)",
    ]  # the 31-character "Headset (Will's AirPods Pro #2 " row is gone, and so is Steam's twin
    assert input_candidates() == [3]  # the only real microphone; headsets and Steam are never candidates


async def test_a_steam_default_is_skipped_for_a_real_microphone(monkeypatch) -> None:
    rig = Rig(monkeypatch, devices_=DEVICES, default=1)
    async with Microphone("Snowball") as mic:
        assert rig.opened == [3] and mic.device_in_use == "Microphone (Realtek HD Audio Mic input)"
        assert mic.fallback
        assert mic.device_note == (
            "microphone 'Snowball' isn't plugged in and the default input "
            "'Microphone (Steam Streaming Mic' isn't a microphone — using 'Microphone (Realtek HD Audio Mic input)'"
        )
    rig.opened.clear()
    async with Microphone("") as mic:  # nothing saved: the default alone is not good enough either
        assert rig.opened == [3] and mic.fallback
        assert mic.device_note.startswith("the default input 'Microphone (Steam Streaming Mic' isn't a microphone")


async def test_she_says_so_when_no_real_microphone_opens(monkeypatch) -> None:
    rig = Rig(monkeypatch, devices_=DEVICES, default=1, failing={3})  # the Realtek jack is empty
    async with Microphone("Snowball") as mic:
        assert rig.opened == [None] and mic.fallback  # the default, honestly labelled
        assert "no other microphone would open" in mic.device_note
        assert "can't hear the room" in mic.device_note


async def test_the_saved_microphone_wins_when_it_is_back(monkeypatch) -> None:
    rig = Rig(monkeypatch, devices_=[*DEVICES, SNOWBALL], default=1)
    async with Microphone("Snowball") as mic:
        assert rig.opened == [6] and not mic.fallback and mic.device_note is None
        assert mic.device_in_use == "Microphone (Blue Snowball)"
    rig.opened.clear()
    rig.default = 6  # a real default needs no second-guessing
    async with Microphone("") as mic:
        assert rig.opened == [None] and not mic.fallback and mic.device_note is None


def test_rescan_clock_fires_once_per_period_only_while_on_a_fallback() -> None:
    now = [100.0]
    clock = RescanClock(True, every_s=30, now=lambda: now[0])
    assert not clock.due()
    now[0] = 129.9
    assert not clock.due()
    now[0] = 130.0
    assert clock.due() and not clock.due()  # once, then the next period
    now[0] = 200.0
    assert clock.due()
    idle = RescanClock(False, every_s=30, now=lambda: now[0])
    now[0] = 1000.0
    assert not idle.due()  # on the real microphone: never
