"""Announcements end to end without audio or network: a finished job queues
one, the engine opens a session by speaking it, and marks it delivered."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.tasks import TaskBoard
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi
from tests.test_dispatch import fake_runner, make_repo


def make_engine(announcer: Announcer) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="test-key",
        model="m",
        voice="v",
        home=FakeHome(),
        owner="Will",
        name="Alexa",
        wake_phrase="alexa",
        announcer=announcer,
    )
    client = FakeClient()
    engine._client = client  # no network: scripted Realtime connection
    return engine, client


async def test_finished_build_queues_a_spoken_announcement(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    announcer = Announcer(repo / "data" / "announcements.json")
    board = TaskBoard(repo, runner=fake_runner(repo, model="opus"), announcer=announcer)
    task = board.draft("greeting file", "add a greeting file")
    await board.start(task.id)
    async with asyncio.timeout(15):
        while board.get(task.id).state == "building":
            await asyncio.sleep(0.1)

    texts = [a.text for a in announcer.pending()]
    assert any(t.startswith("Progress on task 1, 'greeting file': tests are passing") for t in texts)
    assert any("Task 1, 'greeting file', is built and ready for your test" in t for t in texts)
    assert announcer.due()


async def test_announcement_opens_a_session_speaks_and_closes(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "announcements.json")
    item = announcer.enqueue("'greeting file' is built and ready for your test.", ref="job:x:done")
    engine, client = make_engine(announcer)

    stats = await engine.run_conversation(
        NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True
    )

    kinds = client.connection.kinds()
    # handshake, then the announcement as a SYSTEM item, then a response request
    assert kinds[0] == "session.update"
    item_event = next(e for e in client.connection.sent if e["type"] == "conversation.item.create")
    assert item_event["item"]["role"] == "system"
    assert "nobody has spoken" in item_event["item"]["content"][0]["text"]
    assert "greeting file" in item_event["item"]["content"][0]["text"]
    assert kinds.index("conversation.item.create") < kinds.index("response.create")

    assert stats.ended_by == "announcement delivered"
    assert ("event", item.text) in stats.transcript
    assert not announcer.pending()  # marked delivered only after response.done
    assert announcer.history()[0]["id"] == item.id


async def test_announce_session_with_nothing_queued_ends_immediately(tmp_path: Path) -> None:
    engine, client = make_engine(Announcer(tmp_path / "announcements.json"))
    stats = await engine.run_conversation(
        NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True
    )
    assert stats.ended_by == "nothing to announce"
    assert "response.create" not in client.connection.kinds()


def test_announce_chime_exists() -> None:
    from assistant.audio.tones import _SOUNDS, pcm

    assert "announce" in _SOUNDS
    assert len(pcm("announce", 24_000)) > 10_000
