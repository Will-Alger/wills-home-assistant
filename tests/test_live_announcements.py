"""Announcements and push to talk on the Live engine: she opens with the news
through commentary, delivered means played, read means he answered, and a
hold is exactly the stretch of microphone that goes up."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.engines import live_engine as mod
from assistant.hotkey import PushToTalk
from tests.fake_realtime import InstantSpeaker, NeverMic
from tests.test_live_engine import SteadyMic, WakeOnDemand, make, quick


async def test_an_opener_is_spoken_through_commentary_and_closes_unread(tmp_path: Path, monkeypatch) -> None:
    quick(monkeypatch)
    announcer = Announcer(tmp_path / "announcements.json")
    item = announcer.enqueue("'greeting file' is built and ready for your test.", ref="job:x:done")
    engine, client, ui = make(announcer=announcer)
    conn = client.connection

    async def she() -> None:
        await asyncio.sleep(0.1)
        conn.she_speaks(300, "The greeting file is built and ready for your test.")

    task = asyncio.create_task(she())
    stats = await asyncio.wait_for(
        engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui, announce=True), 8
    )
    await task
    assert "opening this conversation" in conn.instructions()[0]
    assert conn.commentary() == [item.text]  # the news itself, never fake user speech
    assert stats.ended_by == "announcement delivered" and ("event", item.text) in stats.transcript
    row = announcer.get(item.id)
    assert row is not None and row.delivered is not None and row.read is None  # spoken is not read
    assert not announcer.pending()


async def test_an_opener_he_answers_is_read(tmp_path: Path, monkeypatch) -> None:
    quick(monkeypatch)
    announcer = Announcer(tmp_path / "announcements.json")
    item = announcer.enqueue("Task 7 is built.", ref="task:7:built")
    engine, client, ui = make(announcer=announcer, info_close_s=0.3)
    conn = client.connection

    async def both() -> None:
        await asyncio.sleep(0.1)
        conn.she_speaks(200, "Task seven is built.")
        conn.owner_says("great thanks")  # right on her last word

    task = asyncio.create_task(both())
    stats = await asyncio.wait_for(
        engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui, announce=True), 8
    )
    await task
    row = announcer.get(item.id)
    assert stats.replied and row is not None and row.delivered is not None and row.read is not None
    assert stats.ended_by == "question answered"


async def test_an_opener_cut_short_by_the_wake_word_stays_unread(tmp_path: Path, monkeypatch) -> None:
    quick(monkeypatch)
    announcer = Announcer(tmp_path / "announcements.json")
    item = announcer.enqueue("Two things: the build finished, and the porch light came on.", ref="x")
    engine, client, ui = make(announcer=announcer, echo_policy="duplex", info_close_s=0.3, live_idle_timeout_s=0.5)
    conn = client.connection
    mic, speaker, wake = SteadyMic(150), InstantSpeaker(), WakeOnDemand()

    async def she() -> None:
        await asyncio.sleep(0.1)
        for i in range(5):
            conn.she_speaks(100, "Two things" if i == 0 else "")
            await asyncio.sleep(0.05)
        wake.fire = True  # "alexa" over the first half

    task = asyncio.create_task(she())
    stats = await asyncio.wait_for(engine.run_conversation(mic, speaker, wake, ui, announce=True), 8)
    await task
    row = announcer.get(item.id)
    assert row is not None and row.delivered is not None and row.read is None
    assert ui.interruptions == 1 and any("cut short" in n for n in ui.notes)
    assert stats.ended_by != "announcement delivered"  # he wanted to say something: the session stayed open


async def test_news_mid_conversation_is_commentary_and_counts_as_heard(tmp_path: Path, monkeypatch) -> None:
    quick(monkeypatch)
    monkeypatch.setattr(mod, "_INJECT_QUIET_S", 0.15)
    monkeypatch.setattr(mod, "_INJECT_MIN_AGE_S", 0.1)
    announcer = Announcer(tmp_path / "announcements.json")
    engine, client, ui = make(announcer=announcer, live_idle_timeout_s=1.0, info_close_s=2.0)
    conn = client.connection
    holder: dict[str, int] = {}

    async def owner() -> None:
        await asyncio.sleep(0.1)
        conn.owner_says("how are you")
        await asyncio.sleep(0.05)
        conn.she_speaks(200, "Doing fine.")
        await asyncio.sleep(0.1)
        holder["id"] = announcer.enqueue("Heads up: the build finished.", ref="job:y").id
        await asyncio.sleep(0.6)  # the watch sees it, the commentary goes out
        conn.she_speaks(200, "Oh, and the build finished.")

    task = asyncio.create_task(owner())
    stats = await asyncio.wait_for(engine.run_conversation(NeverMic(), InstantSpeaker(), None, ui), 8)
    await task
    assert conn.commentary() == ["Heads up: the build finished."]
    assert any("mid-conversation" in line for line in conn.instructions())
    row = announcer.get(holder["id"])
    assert row is not None and row.delivered is not None and row.read is not None  # slipped into a live chat: heard
    assert holder["id"] in stats.announced


async def test_a_hold_is_the_stretch_of_microphone_that_goes_up(monkeypatch) -> None:
    quick(monkeypatch)
    engine, client, ui = make(command_close_s=0.25)
    conn = client.connection
    ptt = PushToTalk()
    ptt.press()
    mic, speaker = SteadyMic(400), InstantSpeaker()
    session = asyncio.create_task(engine.run_conversation(mic, speaker, None, ui, ptt=ptt, ptt_session=True))
    await asyncio.sleep(0.4)  # long enough to be a real hold
    ptt.release()
    await asyncio.sleep(0.15)
    raw_before_release = [f for f in conn.audio_frames() if any(f)]
    conn.she_speaks(600, "Once upon a time there was a light")
    await asyncio.sleep(0.15)
    ptt.press()  # he cuts in mid-reply
    await asyncio.sleep(0.4)
    ptt.release()
    await asyncio.sleep(0.05)
    conn.she_speaks(100, "Yes?")
    stats = await asyncio.wait_for(session, 8)
    frames = conn.audio_frames()
    assert raw_before_release and any(not any(f) for f in frames)  # raw while held, silence between holds
    assert ui.interruptions == 1 and mod._STOP_LINE in conn.instructions()
    assert "response.create" not in conn.kinds()  # no commits, no turn requests: a hold only opens the mic
    assert stats.ended_by == "push to talk turn done"
