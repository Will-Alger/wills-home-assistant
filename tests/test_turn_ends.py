"""A pause is not the end of his turn. The "stopped listening" ding plays
when the server COMMITS the turn, not when it merely notices a pause; and a
turn the server ended at a pause, followed within a second by more of the
same sentence before she said a word, drops the reply to the fragment so
the next reply sees both pieces."""

from __future__ import annotations

import asyncio

from assistant.audio.cues import VoiceCues
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi


def make(**kw) -> tuple[RealtimeEngine, FakeClient, VoiceCues, QuietUi]:
    cues = VoiceCues(play=lambda k: None, render=lambda k, r: k.encode())
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa", wake_phrase="alexa",
        cues=cues, **kw,
    )
    client = FakeClient()
    engine._client = client
    return engine, client, cues, QuietUi()


async def test_a_pause_does_not_ding_but_a_commit_does() -> None:
    engine, client, cues, ui = make(idle_timeout_s=0.6)
    conn = client.connection
    seen: list[list[str]] = []

    async def owner() -> None:
        await asyncio.sleep(0.15)
        cues.start()  # the window opened at the wake
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")  # a breath
        await asyncio.sleep(0.1)
        seen.append(list(cues.played))
        conn.push("input_audio_buffer.speech_started")  # …and on he goes
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")  # now the turn is over
        await asyncio.sleep(0.1)
        seen.append(list(cues.played))

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    after_pause, after_commit = seen
    assert after_pause.count("listen_end") == 0  # the breath got no "stopped listening"
    assert after_commit.count("listen_end") == 1  # the commit did, once


async def test_a_fragment_followed_by_the_rest_drops_the_reply_to_the_fragment() -> None:
    engine, client, _cues, ui = make(info_close_s=0.3, idle_timeout_s=0.6)
    conn = client.connection
    conn.ack_cancel = True  # the server answers the cancel with the cancelled response's done

    async def owner() -> None:
        await asyncio.sleep(0.15)
        # the server ends his turn at a pause and starts answering the fragment
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("conversation.item.input_audio_transcription.completed", transcript="Something just fine first.")
        conn.push("response.created")
        await asyncio.sleep(0.3)  # no audio yet: she has not said a word
        conn.push("input_audio_buffer.speech_started")  # …for like cleaning my apartment
        await asyncio.sleep(0.1)
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("conversation.item.input_audio_transcription.completed", transcript="for like cleaning my apartment.")
        conn._reply("Cleaning music coming up.")  # one reply, for both pieces

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert conn.kinds().count("response.cancel") == 1  # the reply to the fragment was dropped
    assert any("kept talking" in n for n in ui.notes)
    assert [t for r, t in stats.transcript if r == "you"] == [
        "Something just fine first.", "for like cleaning my apartment."
    ]
    assert stats.ended_by != "unknown"


async def test_a_real_second_turn_after_her_reply_is_not_a_fragment() -> None:
    engine, client, _cues, ui = make(idle_timeout_s=0.6)
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("turn off the lights", reply="Done.")  # committed, answered, played
        await asyncio.sleep(0.3)
        conn.user_says("and the fan", reply="Fan's off.")  # a new turn after her reply

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert "response.cancel" not in conn.kinds()  # she had spoken: nothing to drop
