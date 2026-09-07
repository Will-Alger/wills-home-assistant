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
    engine, client, cues, ui = make(idle_timeout_s=3.0)  # long enough for the ding's beat
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
        seen.append(list(cues.played))  # not yet: the ding waits a beat for a continuation
        await asyncio.sleep(1.2)
        seen.append(list(cues.played))

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    after_pause, right_after_commit, a_beat_later = seen
    assert after_pause.count("listen_end") == 0  # the breath got no "stopped listening"
    assert right_after_commit.count("listen_end") == 0  # nor did the commit itself, yet
    assert a_beat_later.count("listen_end") == 1  # it stood: one ding


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


async def test_turns_end_semantically_at_auto_by_default_and_silence_mode_widens_for_let_me_think() -> None:
    default, _client, _cues, _ui = make()
    on = (await default._session_config(None))["audio"]["input"]["turn_detection"]
    assert on == {"type": "semantic_vad", "eagerness": "auto"}  # waits while a sentence sounds unfinished
    assert default._turn_detection(False) is None  # push to talk: the client commits
    engine, client, _cues, ui = make(idle_timeout_s=0.5, turn_detection="server_vad", silence_ms=800)
    conn = client.connection
    assert (await engine._session_config(None))["audio"]["input"]["turn_detection"] == {
        "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 800
    }

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("let me think", reply="Sure.")
        await asyncio.sleep(0.2)
        conn.user_says("okay, the hallway", reply="Hallway it is.")

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    silences = [
        e["session"]["audio"]["input"]["turn_detection"]["silence_duration_ms"]
        for e in conn.sent
        if e["type"] == "session.update" and "audio" in e["session"] and "turn_detection" in e["session"]["audio"]["input"]
    ]
    assert silences[-2:] == [2500, 800]  # patient, then back


async def test_a_loud_room_after_a_commit_never_drops_the_reply_on_its_own() -> None:
    """A guard once cancelled her reply from the local frames alone when the
    mic was loud right after a commit. Recorded: it fired twice on a room
    with nobody talking and a command went unanswered. Only the server
    noticing him again inside the continuation window drops a reply now."""
    import numpy as np

    class LoudMic:
        """Loud at the desk, whatever it is: every frame is a voice or worse."""

        async def get_frame(self) -> bytes:
            await asyncio.sleep(0.01)
            return np.full(1920, 2500, dtype=np.int16).tobytes()

        def drain(self) -> None: ...

    engine, client, _cues, ui = make(idle_timeout_s=0.6)
    conn = client.connection
    conn.ack_cancel = True

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("conversation.item.input_audio_transcription.completed", transcript="Something just fine first.")
        conn.push("response.created")
        await asyncio.sleep(0.4)  # loud frames the whole time: nothing is cancelled
        conn.user_says("for like cleaning my apartment.", reply="Cleaning music coming up.")

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(LoudMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert conn.kinds().count("response.cancel") == 1  # once, when the SERVER heard him again
    assert any("he kept talking" in n for n in ui.notes)
    assert not any("still talking after the turn ended" in n for n in ui.notes)


async def test_a_confirmation_owed_after_a_tool_is_never_dropped() -> None:
    """He said one more word while her "Done." was being generated: the tool
    already ran, so the confirmation is owed — dropping it left him with a
    silent command once."""
    engine, client, _cues, ui = make(idle_timeout_s=0.8)
    conn = client.connection
    conn.auto_reply = False

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.user_says("make the lamp red", calls=[("set_lights", {"room": "living room", "color": "red"})], audio=False)
        conn.push("input_audio_buffer.committed")
        await asyncio.sleep(0.3)  # the tool ran; her confirmation was asked for
        conn.push("response.created")  # ...and is being generated, nothing heard yet
        await asyncio.sleep(0.05)
        conn.push("input_audio_buffer.speech_started")  # "please"
        conn.push("input_audio_buffer.speech_stopped")
        await asyncio.sleep(0.05)
        conn.push_response_done()

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert "set_lights" in [k for k in conn.kinds() if k == "set_lights"] or any(
        e["type"] == "conversation.item.create" for e in conn.sent
    )  # the tool's output went back
    assert "response.cancel" not in conn.kinds()


async def test_a_regurgitated_vocabulary_prompt_is_noise_not_a_turn() -> None:
    engine, client, _cues, ui = make(idle_timeout_s=0.6)
    conn = client.connection
    conn.ack_cancel = True
    # the fake house's own vocabulary prompt (built at connect): what the
    # transcriber spits back on noise
    regurgitated = "Alexa, Will, Bedroom, Hallway, Kitchen, Living Room, Apple TV, Cleveland 10K, Dave Brubeck, Kitchen Strip, Bedroom Lamp"

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("response.created")  # the server is about to answer the room
        conn.push("conversation.item.input_audio_transcription.completed", transcript=regurgitated)
        await asyncio.sleep(0.2)
        conn.user_says("turn off the porch light", reply="Off.")  # a real sentence still counts

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 6)
    await turn
    assert [t for r, t in stats.transcript if r == "you"] == ["turn off the porch light"]
    assert conn.kinds().count("response.cancel") == 1  # the reply to the noise was dropped
    assert any("echoed its vocabulary" in n for n in ui.notes)


async def test_a_dropped_fragment_gets_no_listening_ding_and_a_standing_commit_dings_once_later() -> None:
    engine, client, cues, ui = make(idle_timeout_s=4.0)  # long enough for two beats
    conn = client.connection
    conn.ack_cancel = True
    cues.start()  # the window opened at the wake

    async def owner() -> None:
        await asyncio.sleep(0.15)
        conn.push("input_audio_buffer.speech_started")
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")  # a pause the server took for the end
        conn.push("conversation.item.input_audio_transcription.completed", transcript="I want you to Google if")
        conn.push("response.created")
        await asyncio.sleep(0.2)
        conn.push("input_audio_buffer.speech_started")  # …he kept going: the reply is dropped
        await asyncio.sleep(1.2)  # the turn-over ding's beat passes: nothing should have dinged
        assert cues.played.count("listen_end") == 0 and cues.played.count("wake") == 1
        conn.push("input_audio_buffer.speech_stopped")
        conn.push("input_audio_buffer.committed")
        conn.push("conversation.item.input_audio_transcription.completed", transcript="Aldi uses beef from Venezuela.")
        await asyncio.sleep(1.2)  # nothing follows this commit: the ding stands
        assert cues.played.count("listen_end") == 1
        conn._reply("There's no evidence of that.")

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert cues.played.count("listen_end") == 1  # the beat after her reply never adds one


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
