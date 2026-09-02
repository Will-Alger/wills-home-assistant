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


def make_engine(announcer: Announcer, **kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="test-key",
        model="m",
        voice="v",
        home=FakeHome(),
        owner="Will",
        name="Alexa",
        wake_phrase="alexa",
        announcer=announcer,
        **kw,
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
    assert any("Task 1, 'greeting file', is built and ready for your test" in t for t in texts)
    assert announcer.due()
    rows = announcer.items(limit=50)
    assert any(r["kind"] == "milestone" and r["state"] == "resolved" for r in rows)  # built superseded it


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


# ── milestone 11: spoken is not read ───────────────────────────────────────


async def test_opener_without_reply_stays_unread(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    item = announcer.enqueue("Task 7 is built.", ref="task:7:1:built")
    engine, _client = make_engine(announcer)
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True)
    assert stats.ended_by == "announcement delivered" and not stats.replied
    assert stats.announced == [item.id]
    assert item.state == "spoken" and item.unread  # she said it; nobody proved they heard it
    assert announcer.unread_summary().startswith("1 unread since")


async def test_opener_with_a_reply_is_read(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    item = announcer.enqueue("Task 7 is built.", ref="task:7:1:built")
    engine, client = make_engine(announcer)
    client.connection.say_after_response = "nice, thanks"
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True)
    assert stats.replied and ("you", "nice, thanks") in stats.transcript
    assert item.state == "read" and announcer.unread() == []


async def test_question_opener_asks_and_waits(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    announcer.enqueue(
        "Task 7 needs your call: first name or full name?", kind="question", ref="task:7:1:question"
    )
    engine, client = make_engine(announcer, info_close_s=0.2)
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True)
    lead = next(
        e for e in client.connection.sent if e["type"] == "conversation.item.create"
    )["item"]["content"][0]["text"]
    assert "ASK him" in lead and "answer_task" in lead
    assert stats.ended_by == "no reply"  # she waited the question window, then closed
    assert announcer.get(1).state == "spoken"  # still unread: he never answered


async def test_arrival_batch_leads_with_a_welcome_and_waits(tmp_path: Path) -> None:
    announcer = Announcer(tmp_path / "a.json")
    announcer.enqueue("Task 7 is built.", kind="task", ref="task:7:1:built")
    announcer.enqueue("Will just got home.", kind="presence", ref="presence:arrived:1")
    engine, client = make_engine(announcer, info_close_s=0.2)
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True)
    lead = next(
        e for e in client.connection.sent if e["type"] == "conversation.item.create"
    )["item"]["content"][0]["text"]
    assert lead.startswith("EVENT — Will just walked in")
    assert lead.index("just got home") < lead.index("Task 7 is built")  # the welcome comes first
    assert stats.ended_by == "no reply"
    assert announcer.get(2).state == "read"  # the marker is ephemeral
    assert announcer.get(1).state == "spoken"  # the news still awaits his acknowledgement


async def test_a_slow_tool_call_does_not_get_idled_out(tmp_path: Path, monkeypatch) -> None:
    """The merge gates take minutes; the idle watchdog used to close the
    session under them and cancel the merge half-way."""
    announcer = Announcer(tmp_path / "a.json")
    announcer.enqueue("Task 9 is built.", ref="task:9:1:built")
    engine, _client = make_engine(announcer, idle_timeout_s=0.3)
    finished: list[float] = []

    async def slow_tool(connection, event, stats):
        await asyncio.sleep(0.9)  # three idle timeouts long
        finished.append(asyncio.get_running_loop().time())
        return False

    monkeypatch.setattr(engine, "_handle_response_done", slow_tool)
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=True)
    assert finished, "the tool call was cancelled by the idle watchdog"
    assert stats.ended_by == "announcement delivered"  # not "idle timeout"


async def test_mid_session_injection_is_read_at_once(tmp_path: Path, monkeypatch) -> None:
    from assistant.engines import realtime_engine as mod

    monkeypatch.setattr(mod, "_INJECT_QUIET_S", 0.05)
    monkeypatch.setattr(mod, "_INJECT_MIN_AGE_S", 0.05)
    announcer = Announcer(tmp_path / "a.json")
    engine, client = make_engine(announcer, idle_timeout_s=1.0)

    async def later() -> None:
        await asyncio.sleep(0.2)
        announcer.enqueue("The porch light came on.", kind="watch")

    side = asyncio.create_task(later())
    stats = await engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi())
    await side
    assert stats.ended_by == "idle timeout"
    item = announcer.get(1)
    assert item is not None and item.state == "read" and announcer.unread() == []
    lead = next(
        e for e in client.connection.sent if e["type"] == "conversation.item.create"
    )["item"]["content"][0]["text"]
    assert "mid-conversation" in lead and stats.announced == [1]
