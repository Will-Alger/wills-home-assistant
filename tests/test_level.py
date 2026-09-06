"""Task 15: the live level meter — you can see that she is hearing you.

No audio device and no window: the level is computed from bytes, the status
mirror is a plain object, and the engine test drives a fake mic through a
fake session.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from assistant.audio.cues import VoiceCues
from assistant.audio.level import CEILING, FLOOR, LevelMeter, normalize, rms
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.status import AssistantStatus
from tests.fake_realtime import FakeClient, InstantSpeaker, QuietUi
from tests.test_tentative import LevelMic, NeverWake

FRAME = 1920  # 80 ms at 24 kHz


def tone(amplitude: int, hz: float = 220.0, frames: int = 1) -> bytes:
    """A synthetic voice: a sine at a known amplitude, in 80 ms frames."""
    t = np.arange(FRAME * frames) / 24_000.0
    return (np.sin(2 * math.pi * hz * t) * amplitude).astype(np.int16).tobytes()


# ── the level itself ──────────────────────────────────────────────────────


def test_silence_reads_flat() -> None:
    meter = LevelMeter()
    assert rms(bytes(FRAME * 2)) == 0.0
    assert meter.feed(bytes(FRAME * 2)) == 0.0
    assert rms(b"") == 0.0  # a mic that gave us nothing is not a crash
    # room tone is not speech either: the bar stays down
    assert meter.feed(tone(int(FLOOR) - 50)) == 0.0


def test_a_synthetic_tone_moves_the_bar() -> None:
    meter = LevelMeter()
    quiet = LevelMeter()
    assert rms(tone(8_000)) == pytest.approx(8_000 / math.sqrt(2), rel=0.01)  # a sine's RMS
    loud = meter.feed(tone(8_000))
    assert 0.9 < loud <= 1.0  # a voice close to the mic fills the bar
    soft = quiet.feed(tone(1_500))
    assert 0.15 < soft < loud  # ...and a quiet one only part of it
    assert normalize(CEILING) == 1.0 and normalize(CEILING * 4) == 1.0  # never over full


def test_the_bar_falls_flat_within_a_second_of_silence() -> None:
    meter = LevelMeter()
    meter.feed(tone(8_000))
    silence = bytes(FRAME * 2)
    for _ in range(12):  # one second of 80 ms frames
        meter.feed(silence)
    assert meter.value == 0.0


def test_her_own_voice_never_moves_his_bar() -> None:
    """Muted (half-duplex, she is speaking), the meter only falls."""
    meter = LevelMeter()
    meter.feed(tone(8_000))
    steps = [meter.idle() for _ in range(4)]
    assert steps == sorted(steps, reverse=True) and steps[-1] < steps[0]


# ── what the panel is told ────────────────────────────────────────────────


def test_the_status_snapshot_carries_the_level() -> None:
    status = AssistantStatus()
    status.set_listening(True)
    status.set_level(0.6)
    assert status.level == 0.6
    assert status.snapshot()["level"] == 0.6
    assert status.snapshot()["listening"] is True


def test_a_bar_can_never_outlive_the_listening_flag() -> None:
    status = AssistantStatus()
    status.set_listening(True)
    status.set_level(0.9)
    status.set_listening(False)  # the turn ended: flat at once, not when the API says so
    assert status.level == 0.0 and status.snapshot()["level"] == 0.0
    status.set_level(0.9)  # and nothing can raise it while she isn't listening
    assert status.level == 0.0

    status.set_listening(True)
    status.set_level(0.9)
    status.error("the microphone went away")
    assert status.level == 0.0 and status.listening is False


def test_the_level_is_clamped_to_the_bar() -> None:
    status = AssistantStatus()
    status.set_listening(True)
    status.set_level(4.0)
    assert status.level == 1.0
    status.set_level(-1.0)
    assert status.level == 0.0


def test_the_cues_carry_the_level_to_the_status() -> None:
    status = AssistantStatus()
    cues = VoiceCues(status=status, play=lambda kind: None, render=lambda kind, rate: b"")
    cues.level(0.7)
    assert status.level == 0.0  # no window open yet
    cues.start()
    cues.level(0.7)
    assert status.level == 0.7
    cues.end()  # he stopped talking
    assert status.level == 0.0
    VoiceCues(status=None).level(0.5)  # no status is not a crash


def test_the_panel_reads_the_bar_flat_when_she_is_not_listening(tmp_path) -> None:
    status = AssistantStatus()
    panel = SettingsPanel(status, PanelOverrides(tmp_path / "panel.json"), view_factory=lambda p: None)
    assert panel.meter() == (False, 0.0)
    status.set_listening(True)
    status.set_level(0.5)
    assert panel.meter() == (True, 0.5)
    assert SettingsPanel(None, PanelOverrides(tmp_path / "panel.json")).meter() == (False, 0.0)


# ── end to end: the frames reach the bar ──────────────────────────────────


async def test_the_bar_moves_while_he_talks_and_falls_flat_when_he_stops() -> None:
    status = AssistantStatus()
    cues = VoiceCues(status=status, play=lambda kind: None, render=lambda kind, rate: b"")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa",
        wake_phrase="alexa", idle_timeout_s=1.6, cues=cues,
    )
    engine._client = FakeClient()
    mic, speaker = LevelMic(), InstantSpeaker()
    cues.start()  # the wake ding: the runner opens the window before the session
    seen: dict[str, float] = {}

    async def owner() -> None:
        await asyncio.sleep(0.2)
        mic.level = 4_000  # he is talking
        await asyncio.sleep(0.3)
        seen["talking"] = status.level
        mic.level = 30  # he stops, and says nothing more
        await asyncio.sleep(1.0)
        seen["stopped"] = status.level

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(
        engine.run_conversation(mic, speaker, NeverWake(), QuietUi(), announce=False), 8
    )
    await turn
    assert seen["talking"] > 0.5  # the bar moved with his voice
    assert seen["stopped"] == 0.0  # ...and was flat a second after he stopped
