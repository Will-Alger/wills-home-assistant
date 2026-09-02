"""Follow-ups by trigger (time, arrival, departure, next conversation),
and the waiting_on roll-up — fake clock, no audio."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.followups import FollowUpStore
from assistant.home.fake import FakeHome
from assistant.presence import Transition
from assistant.scheduler import Scheduler
from assistant.tasks import Iteration, TaskBoard
from tests.test_dispatch import fake_runner, make_repo

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def at(hour: int, minute: int = 0) -> float:
    return datetime(2026, 9, 2, hour, minute, tzinfo=LOCAL).timestamp()


def test_triggers_time_arrival_departure_conversation(tmp_path: Path) -> None:
    clock = Clock(at(9))
    announcer = Announcer(tmp_path / "a.json", quiet_hours="00:00-24:00", now=clock)
    store = FollowUpStore(tmp_path / "f.json", announcer=announcer, now=clock)
    chicken = store.add("take the chicken out", trigger="arrival", context="dinner")
    lock = store.add("lock the back door", trigger="departure")
    demo = store.add("ask how the demo went", trigger="next_conversation")
    tea = store.add("your tea has steeped", trigger="time", in_seconds=240)
    six = store.add("call the vet", trigger="time", at="18:00")
    assert demo.trigger == "conversation" and six.fire_at == at(18)
    assert [r["when"] for r in store.describe()][:3] == ["when he gets home", "when he leaves", "next time you talk"]
    assert "[id 3] ask how the demo went" in store.text()

    assert store.tick() == []
    clock.at += 241
    assert [f.id for f in store.tick()] == [tea.id]
    (raised,) = announcer.pending()
    assert raised.kind == "followup" and raised.priority == "urgent" and announcer.due()  # beats quiet hours
    assert raised.text == "Follow-up: your tea has steeped"

    assert [f.id for f in store.on_presence(Transition("left", clock.at))] == [lock.id]
    assert [f.id for f in store.on_presence(Transition("arrived", clock.at, 3600.0))] == [chicken.id]
    assert any(a.text == "Follow-up: take the chicken out (dinner)" for a in announcer.pending())
    assert store.on_presence(Transition("arrived", clock.at)) == []  # once
    assert [f.id for f in store.for_conversation()] == [demo.id]
    assert store.mark_raised([demo.id]) == 1 and store.for_conversation() == []
    assert [f.id for f in store.active()] == [six.id]
    assert store.cancel(six.id) is not None and store.cancel(six.id) is None

    again = FollowUpStore(tmp_path / "f.json", now=clock)
    assert again.get(chicken.id).fired == clock.at and again.active() == []
    with pytest.raises(ValueError, match="trigger"):
        store.add("x", trigger="whenever")
    with pytest.raises(ValueError, match="18:00"):
        store.add("x", trigger="time", at="soon")


async def test_engine_follow_up_tools_and_waiting_on(tmp_path: Path) -> None:
    clock = Clock(at(9))
    announcer = Announcer(tmp_path / "a.json", now=clock)
    store = FollowUpStore(tmp_path / "f.json", announcer=announcer, now=clock)
    sched = Scheduler(tmp_path / "s.json", announcer=announcer, now=clock)
    repo = make_repo(tmp_path)
    board = TaskBoard(repo, runner=fake_runner(repo))
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        announcer=announcer, followups=store, scheduler=sched, task_board=board,
    )
    text, is_error = engine._execute_followup_tool("waiting_on", {})
    assert not is_error and text == "nothing is waiting on you"

    text, is_error = engine._execute_followup_tool(
        "follow_up", {"what": "take the chicken out", "when": "arrival", "context": "dinner"}
    )
    assert not is_error and text == "follow-up 1 set: 'take the chicken out' when he gets home"
    text, is_error = engine._execute_followup_tool("follow_up", {"what": "ask about the demo", "when": "next_conversation"})
    assert not is_error and "next time you talk" in text
    text, is_error = engine._execute_followup_tool("follow_up", {"what": "x", "when": "time"})
    assert is_error and "18:00" in text

    stuck = board.draft("Greeting", "greet")
    stuck.state = "needs_input"
    stuck.iterations.append(Iteration(n=1, status="done", question="first or full name?"))
    board._save()
    announcer.enqueue("Task 1 needs your call", kind="question", ref="task:1:1:question")
    sched.schedule(kind="action", label="lock up", in_seconds=1, confirm=True,
                   action={"tool": "set_lights", "input": {"changes": []}})
    clock.at += 2
    await sched.tick()

    waiting = json.loads(engine._execute_followup_tool("waiting_on", {})[0])
    assert [f["what"] for f in waiting["follow_ups"]] == ["take the chicken out", "ask about the demo"]
    assert waiting["tasks_needing_answers"] == [{"id": 1, "title": "Greeting", "question": "first or full name?"}]
    assert [q["kind"] for q in waiting["unanswered_questions"]] == ["question", "question"]
    assert waiting["awaiting_your_yes"] == [{"id": 1, "label": "lock up"}]

    config = await engine._session_config(None)
    assert "[id 2] ask about the demo" in config["instructions"]
    assert {"follow_up", "waiting_on", "confirm_action"} <= {t["name"] for t in config["tools"]}
    assert engine._raised_followups == [2]
    assert "dropped" in engine._execute_followup_tool("cancel_follow_up", {"id": 1})[0]
    assert json.loads(engine._execute_followup_tool("list_follow_ups", {})[0])[0]["id"] == 2
