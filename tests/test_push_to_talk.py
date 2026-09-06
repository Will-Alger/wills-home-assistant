"""Task 10: push to talk — the hotkey, and what a hold does to a session.

No keyboard, no audio device, no network: `PushToTalk` is driven by press()
and release() the way the Win32 poll drives it, and the FakeConnection records
the exact client events the engine puts on the wire.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from assistant.app import wait_for_trigger
from assistant.audio.cues import VoiceCues
from assistant.engines import realtime_engine as mod
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.hotkey import (
    HotkeyError,
    PushToTalk,
    describe_hotkey,
    parse_hotkey,
    windows_key_poll,
)
from assistant.status import AssistantStatus
from tests.fake_realtime import FakeClient, InstantSpeaker, LoudMic, QuietRoomMic, QuietUi

VK_CONTROL, VK_MENU, VK_SHIFT = 0x11, 0x12, 0x10  # winuser.h

_SEMANTIC_VAD = {"type": "semantic_vad", "eagerness": "auto"}  # the engine's default patience
_LONG_ENOUGH = mod._PTT_MIN_HOLD_S + 0.06  # a hold that counts, with margin


# ── the hotkey itself ─────────────────────────────────────────────────────


def test_the_default_hotkey_is_two_modifiers() -> None:
    assert parse_hotkey("ctrl+alt") == [VK_CONTROL, VK_MENU]
    assert describe_hotkey("ctrl+alt") == "Ctrl+Alt"


def test_letters_digits_and_function_keys_are_allowed() -> None:
    assert parse_hotkey("ctrl+shift+k") == [VK_CONTROL, VK_SHIFT, ord("K")]
    assert parse_hotkey("f13") == [0x7C]
    assert parse_hotkey("alt-space") == [VK_MENU, 0x20]  # a dash reads as a plus


def test_a_typo_is_a_sentence_she_can_read_aloud() -> None:
    with pytest.raises(HotkeyError) as err:
        parse_hotkey("ctrl+wiggle")
    assert "wiggle" in str(err.value) and "ctrl" in str(err.value)
    with pytest.raises(HotkeyError):
        parse_hotkey("   ")


def test_the_windows_poll_answers_without_a_keyboard_or_admin() -> None:
    """It must never raise: nothing is pressed in a test runner, and a call
    the OS refuses (an elevated window has focus) reads as 'not held'."""
    assert windows_key_poll([VK_CONTROL, VK_MENU])() is False


async def test_the_poll_loop_turns_a_held_key_into_presses() -> None:
    down = [False, True, True, False, True]
    ptt = PushToTalk(lambda: down.pop(0) if down else False, interval_s=0.001)
    task = asyncio.create_task(ptt.run())
    await asyncio.sleep(0.05)
    task.cancel()
    assert ptt.presses == 2 and not ptt.held  # two holds, both let go
    assert ptt.released_at >= ptt.pressed_at > 0


def test_holding_is_one_press_however_often_it_is_polled() -> None:
    ptt = PushToTalk()
    ptt.press()
    ptt.press()
    assert ptt.presses == 1 and ptt.held
    ptt.release()
    ptt.release()
    assert ptt.presses == 1 and not ptt.held


async def test_the_idle_loop_starts_a_session_on_a_held_key() -> None:
    class Frames:
        async def get_frame(self) -> bytes:
            await asyncio.sleep(0)
            return b""

    class NeverWakes:
        def detect(self, frame: bytes) -> bool:
            return False

    ptt = PushToTalk()
    ptt.press()
    assert await wait_for_trigger(Frames(), NeverWakes(), ptt=ptt) == "ptt"
    ptt.release()
    with pytest.raises(TimeoutError):  # nothing held: nothing happens
        async with asyncio.timeout(0.05):
            await wait_for_trigger(Frames(), NeverWakes(), ptt=ptt)


# ── a hold, end to end through the engine ─────────────────────────────────


def make_engine(**kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="test-key",
        model="m",
        voice="v",
        turn_detection="semantic_vad",  # this suite asserts the semantic shape it was written against
        home=FakeHome(),
        owner="Will",
        name="Alexa",
        wake_phrase="alexa",
        command_close_s=0.2,
        info_close_s=0.2,
        idle_timeout_s=3.0,
        **kw,
    )
    client = FakeClient()
    engine._client = client  # no network: scripted Realtime connection
    return engine, client


def make_cues(status: AssistantStatus | None = None) -> VoiceCues:
    """Cues with the synthesis stubbed out — `cues.played` names the earcons
    whichever way they went out (a fresh stream, or the session speaker)."""
    return VoiceCues(
        rate=24_000, status=status, play=lambda kind: None, render=lambda k, r: k.encode()
    )


async def until(check, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.01)


def turn_detection_of(event: dict) -> object:
    return event["session"]["audio"]["input"]["turn_detection"]


def updates(connection) -> list[dict]:
    return [e for e in connection.sent if e["type"] == "session.update"]


async def test_a_hold_runs_the_turn_and_the_release_asks_for_the_answer() -> None:
    engine, client = make_engine()
    ptt = PushToTalk()
    ptt.press()  # he presses, and the session opens under his thumb
    session = asyncio.create_task(
        engine.run_conversation(
            LoudMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt, ptt_session=True
        )
    )
    conn = client.connection
    await until(lambda: conn.kinds().count("input_audio_buffer.append") >= 3)
    # Turn detection was off from the very first session.update: a pause
    # mid-sentence cannot end his turn, because nothing is watching for one.
    assert turn_detection_of(conn.sent[0]) is None

    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    stats = await asyncio.wait_for(session, timeout=5)
    kinds = conn.kinds()
    assert kinds.index("input_audio_buffer.append") < kinds.index("input_audio_buffer.commit")
    assert kinds.index("input_audio_buffer.commit") < kinds.index("response.create")
    assert kinds.count("input_audio_buffer.commit") == 1  # one hold, one turn
    # She answered, then closed on her own — he never has to say "that's all".
    assert stats.ended_by == "push to talk turn done"


async def test_nothing_is_committed_while_he_is_still_holding() -> None:
    """The complaint this feature exists for: she must not answer his pause."""
    engine, client = make_engine()
    ptt = PushToTalk()
    ptt.press()
    session = asyncio.create_task(
        engine.run_conversation(
            LoudMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt, ptt_session=True
        )
    )
    conn = client.connection
    await until(lambda: "input_audio_buffer.append" in conn.kinds())
    await asyncio.sleep(0.6)  # far past any turn-detection window
    assert "input_audio_buffer.commit" not in conn.kinds()
    assert "response.create" not in conn.kinds()
    assert conn.kinds().count("input_audio_buffer.append") > 5  # still capturing him

    ptt.release()
    await asyncio.wait_for(session, timeout=5)
    assert "response.create" in conn.kinds()


async def test_a_tap_says_nothing_and_costs_nothing() -> None:
    """Pressed and let go inside 300 ms: no commit (the API errors on an empty
    buffer), no response, and no error tone — nothing happened at all."""
    engine, client = make_engine()
    cues = make_cues()
    engine._cues = cues
    ptt = PushToTalk()
    ptt.press()
    ptt.release()
    stats = await asyncio.wait_for(
        engine.run_conversation(
            LoudMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt, ptt_session=True
        ),
        timeout=5,
    )
    kinds = client.connection.kinds()
    assert "input_audio_buffer.commit" not in kinds and "response.create" not in kinds
    assert kinds.count("input_audio_buffer.clear") == 2  # opened the turn, threw it away
    assert "error" not in cues.played and not cues.listening
    assert stats.ended_by == "nothing said"


async def test_a_hold_with_an_empty_room_in_it_is_not_a_question() -> None:
    """He held it long enough but said nothing: she must not answer silence."""
    engine, client = make_engine()
    ptt = PushToTalk()
    ptt.press()
    session = asyncio.create_task(
        engine.run_conversation(
            QuietRoomMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt, ptt_session=True
        )
    )
    await until(lambda: client.connection.kinds().count("input_audio_buffer.append") >= 5)
    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    stats = await asyncio.wait_for(session, timeout=5)
    assert "response.create" not in client.connection.kinds()
    assert stats.ended_by == "nothing said"


async def test_a_press_while_she_is_talking_stops_her_before_it_takes_the_turn() -> None:
    engine, client = make_engine()
    ui, speaker = QuietUi(), InstantSpeaker()
    ptt = PushToTalk()
    ptt.press()
    conn = client.connection
    conn.hold_response = True  # she keeps talking until the test lets her stop
    session = asyncio.create_task(
        engine.run_conversation(LoudMic(), speaker, None, ui, ptt=ptt, ptt_session=True)
    )
    await until(lambda: "input_audio_buffer.append" in conn.kinds())
    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    await until(lambda: bool(speaker.chunks))  # her voice is in the room
    mid_reply = len(conn.sent)

    ptt.press()  # he cuts in
    await until(lambda: "response.cancel" in conn.kinds())
    after = conn.kinds()[mid_reply:]
    assert after.index("response.cancel") < after.index("input_audio_buffer.clear")
    assert not speaker.chunks  # she went quiet at once, not when the API agreed
    assert ui.interruptions == 1
    assert "input_audio_buffer.commit" not in after  # his new turn is still open

    conn.hold_response = False
    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    stats = await asyncio.wait_for(session, timeout=5)
    assert conn.kinds().count("input_audio_buffer.commit") == 2  # two holds, two turns
    assert stats.ended_by == "push to talk turn done"


async def test_a_wake_session_hands_turn_taking_over_and_takes_it_back() -> None:
    """The wake path is untouched — until he reaches for the hotkey mid-chat."""
    engine, client = make_engine()
    ptt = PushToTalk()
    conn = client.connection
    session = asyncio.create_task(
        engine.run_conversation(LoudMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt)
    )
    await until(lambda: "input_audio_buffer.append" in conn.kinds())
    assert turn_detection_of(updates(conn)[0]) == _SEMANTIC_VAD

    ptt.press()
    await until(lambda: len(updates(conn)) == 2)
    assert turn_detection_of(updates(conn)[-1]) is None
    # Only audio.input is resent: `voice` may never appear once she has spoken.
    assert "voice" not in str(updates(conn)[-1])

    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    await until(lambda: len(updates(conn)) == 3)
    assert turn_detection_of(updates(conn)[-1]) == _SEMANTIC_VAD
    assert conn.kinds().count("input_audio_buffer.commit") == 1
    session.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await session


# ── what he can see and hear while it happens ─────────────────────────────


async def test_the_listening_flag_follows_his_thumb_not_the_api() -> None:
    status = AssistantStatus()
    cues = make_cues(status)
    engine, client = make_engine()
    engine._cues = cues
    conn = client.connection
    conn.hold_response = True  # the API confirms nothing while we look
    ptt = PushToTalk()
    ptt.press()
    session = asyncio.create_task(
        engine.run_conversation(
            LoudMic(), InstantSpeaker(), None, QuietUi(), ptt=ptt, ptt_session=True
        )
    )
    await until(lambda: "input_audio_buffer.append" in conn.kinds())
    assert cues.listening and status.listening  # the panel says she is hearing him
    assert cues.played == ["wake"]  # the listening ding, once

    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    await until(lambda: not cues.listening, timeout=0.3)
    assert not status.listening and cues.played[-1] == "listen_end"

    conn.finish_response()
    await asyncio.wait_for(session, timeout=5)
    assert not status.snapshot()["listening"]


def test_the_panel_shows_the_hotkey() -> None:
    status = AssistantStatus(hotkey="Ctrl+Alt")
    assert status.snapshot()["hotkey"] == "Ctrl+Alt"
    assert AssistantStatus().snapshot()["hotkey"] == ""  # push to talk switched off


def test_between_holds_she_is_neither_listening_nor_working() -> None:
    """A push-to-talk session stays open, but the panel must not claim she is
    hearing him — his key is the only way back in."""
    status = AssistantStatus()
    cues = make_cues(status)
    cues.start()
    cues.idle()
    assert not cues.listening and not status.listening
    assert status.snapshot()["state"] == "idle"
    assert cues.played == ["wake"]  # going quiet is silent: a tone would be a lie


# ── the merge with main: a hotkey turn keeps everything else she gained ────


async def test_a_hotkey_session_still_gets_noise_reduction_and_the_vocabulary() -> None:
    """turn_detection null is the ONLY thing push to talk changes about the
    audio input: the desk mic's noise reduction and the transcriber's list of
    house names ride along, or a held turn would be transcribed worse than a
    spoken one."""
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        name="Alexa", noise_reduction="far_field", turn_detection="semantic_vad",
    )
    audio_in = (await engine._session_config("whisper-1", turn_detection=False))["audio"]["input"]
    assert audio_in["turn_detection"] is None
    assert audio_in["noise_reduction"] == {"type": "far_field"}
    assert "Alexa" in audio_in["transcription"]["prompt"]
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24_000}

    # ...and the mid-session toggle resends that block, only flipped.
    back_on = engine.audio_input_update(turn_detection=True)
    assert back_on["turn_detection"] == _SEMANTIC_VAD
    assert back_on["noise_reduction"] == audio_in["noise_reduction"]
    assert back_on["transcription"] == audio_in["transcription"]
    assert "voice" not in back_on  # the API refuses it once she has spoken
    off_again = engine.audio_input_update(turn_detection=False)
    assert off_again["turn_detection"] is None
    assert off_again["transcription"] == audio_in["transcription"]


async def test_the_hotkey_session_uses_the_streams_it_was_given() -> None:
    """One microphone and one speaker stay open across idle and talk (AudioIO):
    a hold opens its session on those, never on streams of its own."""
    engine, client = make_engine()
    mic, speaker = LoudMic(), InstantSpeaker()
    ptt = PushToTalk()
    ptt.press()
    session = asyncio.create_task(
        engine.run_conversation(mic, speaker, None, QuietUi(), ptt=ptt, ptt_session=True)
    )
    await until(lambda: "input_audio_buffer.append" in client.connection.kinds())
    await asyncio.sleep(_LONG_ENOUGH)
    ptt.release()
    await asyncio.wait_for(session, timeout=5)
    assert speaker.chunks  # her reply went to the runner's speaker
    assert mic.drained  # and the same microphone was handed back drained
