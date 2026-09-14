"""The GPT-Live engine, offline: the session she starts with, the prompt
split, the backend bridge, the endings that are the engine's to decide, and
the money."""

from __future__ import annotations

import asyncio

import numpy as np

from assistant.engines import live_engine as mod
from assistant.engines.live_engine import LiveEngine, backend_cost, voice_for_live
from assistant.home.fake import FakeHome
from tests.fake_live import FakeLiveClient, LevelSpeaker
from tests.fake_realtime import InstantSpeaker, NeverMic, QuietUi

HALLWAY_OFF = {"changes": [{"target": "Hallway", "turn": "off"}]}


async def test_slow_music_feedback_is_suppressed_after_user_changes_or_cancellation(monkeypatch):
    monkeypatch.setattr(mod, "_MUSIC_ACK_S", 0)
    engine, client, ui = make()
    session = mod._LiveSession(engine, client.connection, NeverMic(), InstantSpeaker(), None, ui,
                               mod.SessionStats(), announce=False, ptt=None, ptt_session=False)
    d = session._delegation("music")
    engine._executor.music.begin()
    await session._music_ack(d)
    events = [e for e in client.connection.sent if e["type"] == "session.commentary.append"]
    assert len(events) == 1
    assert "Do not claim it is playing" in events[0]["content"]
    session.user_turns += 1
    await session._music_ack(d)
    session.user_turns -= 1
    engine._executor.music.cancel_pending()
    await session._music_ack(d)
    assert len([e for e in client.connection.sent if e["type"] == "session.commentary.append"]) == 1


async def test_live_cancel_phrase_cancels_music_preparation():
    engine, client, ui = make()
    session = mod._LiveSession(engine, client.connection, NeverMic(), InstantSpeaker(), None, ui,
                               mod.SessionStats(), announce=False, ptt=None, ptt_session=False)
    request = engine._executor.music.begin()
    session._judge("cancel the music")
    assert request.cancelled.is_set()


async def test_live_backend_exposes_one_call_music_selection():
    engine, _client, _ui = make()
    tool = next(t for t in engine._backend_tools() if t.get("name") == "play_music")
    assert tool["parameters"]["properties"]["selection"]["enum"] == ["exact", "discover"]


async def test_an_unmistakable_play_request_starts_before_the_backend(monkeypatch):
    engine, client, ui = make()
    started: list = []
    woken: list = []

    async def fast_start(intent):
        started.append(intent)
        return {"title": "Back In Black", "verified": True}

    async def prewake():
        woken.append(True)
        return True

    monkeypatch.setattr(engine._executor.music, "fast_start", fast_start)
    monkeypatch.setattr(engine._executor.music, "prewake", prewake)
    session = mod._LiveSession(engine, client.connection, NeverMic(), InstantSpeaker(), None, ui,
                               mod.SessionStats(), announce=False, ptt=None, ptt_session=False)
    session._maybe_prewake("Can you")  # nothing to wake for yet
    session._maybe_prewake("Can you play")  # the TV is woken mid-sentence, once
    session._maybe_prewake("Can you play back in")
    session._finish_user_turn("Can you play back in black by AC/DC")
    await asyncio.sleep(0.02)
    assert woken == [True]
    assert [(i.title, i.artist) for i in started] == [("back in black", "AC/DC")]
    session._finish_user_turn("play some jazz")  # the model's call
    await asyncio.sleep(0.02)
    assert len(started) == 1


def make(**kw) -> tuple[LiveEngine, FakeLiveClient, QuietUi]:
    engine = LiveEngine(
        api_key="k", model="gpt-live-1", voice="sol", home=FakeHome(), owner="Will", name="Alexa",
        wake_phrase="alexa", **kw,
    )
    client = FakeLiveClient()
    engine._client = client
    return engine, client, QuietUi()


def quick(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_OUTPUT_QUIET_S", 0.2)
    monkeypatch.setattr(mod, "_SETTLE_S", 0.15)
    monkeypatch.setattr(mod, "_WRAPUP_GRACE_S", 0.3)
    monkeypatch.setattr(mod, "_FAREWELL_MAX_S", 0.25)


class SteadyMic:
    """Frames at one level, one every 10 ms."""

    def __init__(self, level: int = 300) -> None:
        self.level = level

    async def get_frame(self) -> bytes:
        await asyncio.sleep(0.01)
        return np.full(1920, self.level, dtype=np.int16).tobytes()

    def drain(self) -> None: ...


class WakeOnDemand:
    def __init__(self) -> None:
        self.fire = False

    def detect(self, frame: bytes) -> bool:
        fired, self.fire = self.fire, False
        return fired

    def reset(self) -> None: ...


# ── the session she starts with ─────────────────────────────────────────────


async def test_the_session_she_starts_with() -> None:
    engine, _client, _ui = make()
    assert engine.voice == "marin" and "not a Live voice" in (engine.voice_note or "")  # sol does not exist on Live
    assert voice_for_live("cedar") == "cedar" and voice_for_live("") == "marin"
    cedar, _c, _u = make(live_voice="cedar")
    assert cedar.voice == "cedar" and cedar.voice_note is None
    cfg = await engine._live_session_config()
    assert cfg["model"] == "gpt-live-1" and cfg["store"] is False
    assert cfg["audio"] == {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}}
    backend = cfg["delegation"]
    assert backend["type"] == "responses" and backend["responses"]["model"] == "gpt-5.6-luna"
    assert backend["responses"]["reasoning"] == {"effort": "low"} and backend["responses"]["tool_choice"] == "auto"
    tools = backend["responses"]["tools"]
    names = [t.get("name") for t in tools]
    assert {"type": "web_search"} in tools and "web_search" not in names  # the backend's own search
    assert "set_lights" in names and "end_conversation" in names
    live, back = cfg["instructions"], backend["responses"]["instructions"]
    assert "stop mid-word" in live and 'Never say the word "alexa"' in live and "backend" in live
    assert "say nothing and wait" in live and "no remark on how it was said" in live  # her name alone: one "Yes?"
    assert "Hallway" not in live  # the voice knows no devices: that is the backend's world
    assert back.startswith("You are the reasoning and tool backend of Alexa") and "Hallway" in back
    assert "the function call comes FIRST" in back  # today's tool rules, unchanged, on the backend


async def test_a_hub_that_is_down_does_not_cost_him_her_voice() -> None:
    """Every wake after a reboot used to die rendering the session while Home
    Assistant was still coming up. Now the session opens with the hub marked
    out, and the prompt says what to tell him."""
    class DownHome(FakeHome):
        async def get_lights(self):
            raise ConnectionError("timed out")

    engine = LiveEngine(api_key="k", model="gpt-live-1", voice="marin", home=DownHome(), owner="Will",
                        name="Alexa", wake_phrase="alexa")
    cfg = await engine._live_session_config()
    backend = cfg["delegation"]["responses"]["instructions"]
    assert "the home hub is not answering" in backend and engine._home_down == "ConnectionError"
    up = LiveEngine(api_key="k", model="gpt-live-1", voice="marin", home=FakeHome(), owner="Will",
                    name="Alexa", wake_phrase="alexa")
    cfg = await up._live_session_config()
    assert "not answering" not in cfg["delegation"]["responses"]["instructions"] and up._home_down == ""


async def test_backend_web_search_can_stay_ours() -> None:
    engine, _c, _u = make(backend_web_search=False)
    tools = engine._backend_tools()
    assert {"type": "web_search"} not in tools and "web_search" in [t.get("name") for t in tools]


async def test_the_native_search_refuses_minimal_reasoning_so_low_is_the_floor() -> None:
    """Day one: every tool request failed with "The following tools cannot be
    used with reasoning.effort 'minimal': web_search." and she said sorry."""
    engine, _c, _u = make(backend_reasoning="minimal")
    assert engine._backend_reasoning == "low" and "using low" in (engine.backend_note or "")
    ours, _c, _u = make(backend_reasoning="minimal", backend_web_search=False)
    assert ours._backend_reasoning == "minimal" and ours.backend_note is None


async def test_a_backend_refusal_mid_session_is_healed_for_the_next_request(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(live_idle_timeout_s=0.6)
    engine._backend_reasoning = "minimal"  # as if the guard were not there
    conn = client.connection

    async def server() -> None:
        await asyncio.sleep(0.1)
        conn.push(
            "error",
            error=type("E", (), {
                "message": "The following tools cannot be used with reasoning.effort 'minimal': web_search.",
                "code": "invalid_value", "param": "reasoning.effort", "type": "invalid_request_error",
            })(),
        )
        await asyncio.sleep(0.05)
        conn.push(
            "error",
            error=type("E", (), {
                "message": "Unsupported tool: web_search.", "code": "invalid_value", "param": "tools",
                "type": "invalid_request_error",
            })(),
        )

    task = asyncio.create_task(server())
    await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await task
    updates = [e["session"]["delegation"]["responses"] for e in conn.sent if e["type"] == "session.update"]
    assert updates[0] == {"reasoning": {"effort": "low"}}
    assert {"type": "web_search"} not in updates[1]["tools"] and "web_search" in [t.get("name") for t in updates[1]["tools"]]
    assert engine._backend_reasoning == "low" and not engine._backend_web_search
    assert sum("say it again" in n for n in ui.notes) == 2


def test_the_backend_share_of_the_bill() -> None:
    usage = {"input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 800}}
    assert abs(backend_cost(usage, "gpt-5.6-luna") - 0.000116) < 1e-9
    assert abs(backend_cost(usage, "gpt-5.6-terra") - 0.00116) < 1e-9
    assert backend_cost(usage, "something-else") == 0.0 and backend_cost(None, "gpt-5.6-luna") == 0.0


# ── conversations ───────────────────────────────────────────────────────────


async def test_a_question_is_answered_and_the_window_closes(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(info_close_s=0.4, command_close_s=0.2)
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("what time is it")
        await asyncio.sleep(0.1)
        conn.she_speaks(300, "It is nine o'clock.")  # her first word closes his turn
        conn.usage(12.0, 0.01)

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert ("you", "what time is it") in stats.transcript
    assert ("alexa", "It is nine o'clock.") in stats.transcript  # flushed at the close
    assert stats.ended_by == "question answered" and stats.replied
    assert stats.seconds == 12.0 and abs(stats.cost_usd - 0.01) < 1e-9  # 12 s at $0.05 a minute
    kinds = conn.kinds()
    assert kinds[0] == "session.start" and kinds[-1] == "session.close" and not conn.tool_outputs()


async def test_the_backend_asks_for_a_tool_and_the_command_closes(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(command_close_s=0.25, info_close_s=2.0)
    conn = client.connection
    conn.on_response_create = lambda c: c.backend("d1", created=False, text="Done.", response_id="resp_2")

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("turn off the hallway")
        await asyncio.sleep(0.05)
        conn.backend("d1", calls=[("call_1", "set_lights", HALLWAY_OFF)])
        await asyncio.sleep(0.3)
        conn.she_speaks(200, "Done, the hallway is off.")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    outputs = conn.tool_outputs()
    assert len(outputs) == 1 and outputs[0]["status"] == "success" and "end_conversation" in outputs[0]["follow_up"]
    kinds = conn.kinds()
    assert kinds[kinds.index("response.item.create") + 1] == "response.create"  # outputs, then continue
    assert stats.tool_calls == ["set_lights"] and stats.responses == 1  # she spoke once
    assert stats.backend_cost_usd > 0 and stats.ended_by == "command complete"
    assert any(r == "tool set_lights" for r, _ in stats.transcript)


async def test_the_end_tool_closes_after_her_goodbye(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection
    conn.on_response_create = lambda c: c.backend("d1", created=False, text="Bye!", response_id="resp_2")

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("hallway off and that's it")
        await asyncio.sleep(0.05)
        conn.backend("d1", calls=[("c1", "set_lights", HALLWAY_OFF), ("c2", "end_conversation", {})])
        await asyncio.sleep(0.3)
        conn.she_speaks(200, "Hallway off. Bye!")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert stats.ended_by == "end_conversation"
    assert [o["summary"] for o in conn.tool_outputs()][-1] == "closing after your last words"
    assert stats.tool_calls == ["set_lights", "end_conversation"]


async def test_his_next_sentence_overrules_the_end_tool(monkeypatch) -> None:
    """Will, 18:48: "Make the volume one hundred percent" → the backend
    answered and called the end tool → "Alexa, let's also make the living
    room…" was cut off on the beat after her last word. Talking over her in
    full duplex sets no interrupted flag, so the close must yield to an open
    sentence of his, and be withdrawn once one starts after the end tool."""
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection
    conn.on_response_create = lambda c: c.backend("d1", created=False, text="Sure.", response_id="resp_2")

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("make the volume one hundred percent")
        await asyncio.sleep(0.05)
        conn.backend("d1", calls=[("c1", "media_control", {"action": "volume_set", "volume_pct": 100}),
                                  ("c2", "end_conversation", {})])
        await asyncio.sleep(0.2)
        conn.she_speaks(200, "Volume's at one hundred percent.")
        conn.owner_says("Alexa, let's also make the living room warm", start_ms=5000)  # over her last word
        await asyncio.sleep(0.6)  # long past the beat that used to close it
        assert not engine._live_session_ended  # type: ignore[attr-defined]
        conn.she_speaks(100, "Sure, warmer.")  # she answers; his turn closes on the speaker change
        await asyncio.sleep(0.3)
        conn.owner_says("okay that's all", start_ms=conn._stamp + 600)

    monkeypatch.setattr(mod._LiveSession, "_end", _ending(engine), raising=True)
    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert stats.ended_by == "wrap-up"  # his own wrap-up, not the backend's end tool
    said = [e["content"] for e in conn.sent if e["type"] == "session.instructions.append"]
    assert any("NOT over" in s for s in said)
    assert ("you", "Alexa, let's also make the living room warm") in stats.transcript or any(
        "make the living" in text for who, text in stats.transcript if who == "you"
    )


def _ending(engine):
    """Wrap _end so a test can watch for it without the session's own state."""
    original = mod._LiveSession._end
    engine._live_session_ended = False

    def _end(self, reason: str) -> None:
        engine._live_session_ended = True
        original(self, reason)

    return _end


async def test_an_end_tool_with_no_goodbye_gets_asked_for_one_then_closes_anyway(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection
    conn.on_response_create = lambda c: c.backend("d1", created=False, response_id="resp_2")

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("that will be all")
        await asyncio.sleep(0.05)
        conn.backend("d1", calls=[("c1", "end_conversation", {})])

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert stats.ended_by == "end_conversation"
    assert mod._GOODBYE_LINE in conn.commentary()  # she was asked; the close did not wait forever


async def test_a_held_socket_carries_the_wake_and_is_replaced_afterwards(monkeypatch) -> None:
    """The socket connected while she was idle is the one the conversation
    starts on — no connect at the wake — and once it is over the holder has
    the next one ready."""
    quick(monkeypatch)
    engine, client, ui = make()
    warm = mod.WarmSocket(client, max_age_s=5)
    engine._warm = warm
    holder = asyncio.create_task(warm.run())
    try:
        for _ in range(100):
            if warm.ready:
                break
            await asyncio.sleep(0.01)
        assert warm.ready and engine.warm_ready and client.connects == 1
        assert engine.answers_wake(0.98) and engine.answers_wake(None) and not engine.answers_wake(0.56)
        conn = client.connection

        async def owner() -> None:
            await asyncio.sleep(0.1)
            conn.owner_says("that's all")

        turn = asyncio.create_task(owner())
        stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
        await turn
        assert stats.ended_by == "wrap-up"
        assert client.connects == 1 and warm.uses == 1  # the held socket carried it
        for _ in range(100):
            if client.connects == 2 and warm.ready:
                break
            await asyncio.sleep(0.01)
        assert client.connects == 2 and warm.ready  # the next wake has its socket already
    finally:
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)


async def test_a_dead_held_socket_falls_back_to_connecting_cold(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    warm = mod.WarmSocket(client, max_age_s=5)
    engine._warm = warm
    holder = asyncio.create_task(warm.run())
    try:
        for _ in range(100):
            if warm.ready:
                break
            await asyncio.sleep(0.01)
        conn = client.connection
        conn.dead_once = True  # the server dropped it quietly while it was held

        async def owner() -> None:
            await asyncio.sleep(0.15)
            conn.owner_says("that's all")

        turn = asyncio.create_task(owner())
        stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
        await turn
        assert stats.ended_by == "wrap-up"
        assert warm.uses == 1 and client.connects >= 2  # the cold connect carried it
        assert any("connecting cold" in n for n in ui.notes)
    finally:
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)


async def test_thats_all_closes_after_her_closing_word(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("okay that's all thanks")
        await asyncio.sleep(0.05)
        conn.she_speaks(200, "Bye for now.")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert stats.ended_by == "wrap-up" and "response.create" not in conn.kinds()


async def test_a_wrap_up_she_never_answers_closes_after_the_grace(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("never mind")  # settles: nobody answers it

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await turn
    assert stats.ended_by == "wrap-up" and ("you", "never mind") in stats.transcript


async def test_stop_is_instant_and_local(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection
    speaker = InstantSpeaker()

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.she_speaks(600, "Once upon a time in a land far away")
        await asyncio.sleep(0.05)
        conn.owner_says("alexa stop")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), speaker, None, ui), 8)
    await turn
    assert stats.ended_by == "stop command" and speaker.chunks == []  # the tail never played


async def test_idle_money_stops_and_a_session_has_a_cap(monkeypatch) -> None:
    quick(monkeypatch)
    engine, _client, ui = make(live_idle_timeout_s=0.3)
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    assert stats.ended_by == "idle timeout"
    engine, _client, ui = make(live_idle_timeout_s=5.0, max_session_s=0.3)
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    assert stats.ended_by == "session cap"


async def test_the_server_closing_the_session_is_reported(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection

    async def server() -> None:
        await asyncio.sleep(0.1)
        conn.seconds = 42.0
        conn.close("expired")

    task = asyncio.create_task(server())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await task
    assert stats.ended_by == "session expired" and stats.seconds == 42.0
    assert "session.close" not in conn.kinds()  # nothing to close any more


async def test_a_backend_that_rejects_reasoning_gets_one_retry_without(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(live_idle_timeout_s=0.2)
    client.connection.reject_start = {
        "message": "Unknown parameter", "code": "unknown_parameter", "param": "session.delegation.responses.reasoning",
    }
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    assert client.connects == 2 and stats.ended_by == "idle timeout"
    starts = [e["session"] for e in client.connection.sent if e["type"] == "session.start"]
    assert "reasoning" in starts[0]["delegation"]["responses"] and "reasoning" not in starts[1]["delegation"]["responses"]
    assert any("rejected reasoning" in n for n in ui.notes)


async def test_a_dead_socket_ends_the_session_through_one_path(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make()
    conn = client.connection

    async def die() -> None:
        await asyncio.sleep(0.1)
        conn.fail_now()

    task = asyncio.create_task(die())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await task
    assert stats.ended_by.startswith("session error")


# ── the room ────────────────────────────────────────────────────────────────


async def test_a_loud_speaker_is_gated_and_the_wake_word_cuts_in(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(echo_policy="gated", live_idle_timeout_s=5.0)
    conn = client.connection
    mic, speaker, wake = SteadyMic(400), InstantSpeaker(), WakeOnDemand()

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("tell me a story")
        for _ in range(6):  # she talks for a while
            conn.she_speaks(100, "Once" if _ == 0 else "")
            await asyncio.sleep(0.05)
        wake.fire = True  # "alexa" over her
        await asyncio.sleep(0.15)
        conn.owner_says("stop it")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, wake, ui), 8)
    await turn
    frames = conn.audio_frames()
    silent = [f for f in frames if not any(f)]
    assert silent and len(silent) < len(frames)  # gated while she spoke, raw before and after the barge-in
    assert mod._STOP_LINE in conn.instructions() and ui.interruptions == 1
    assert speaker.chunks == [] and stats.ended_by == "stop command"


async def test_a_quiet_speaker_means_full_duplex(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(live_idle_timeout_s=0.6)
    conn = client.connection
    mic, speaker = SteadyMic(150), LevelSpeaker(level=1000)  # she plays at 1000, the mic hears 150: coupling 0.15

    async def owner() -> None:
        await asyncio.sleep(0.05)
        conn.she_speaks(300, "Hello there")

    turn = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, None, ui), 8)
    await turn
    assert any("full duplex" in n for n in ui.notes)
    frames = conn.audio_frames()
    assert frames and not any(f for f in frames[-10:] if not any(f))  # the last frames went up raw
    assert stats.ended_by == "idle timeout"
    loud, _client2, ui2 = make(live_idle_timeout_s=0.6)
    stats = await asyncio.wait_for(loud.run_conversation(SteadyMic(600), LevelSpeaker(level=1000), None, ui2), 8)
    assert any("gated" in n for n in ui2.notes)  # 0.6: the speaker is louder at the mic than he is
