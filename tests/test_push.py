"""The phone channel: card payloads, clears, and taps that approve, answer,
acknowledge — through FakeHome's generic_call, no Home Assistant."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.home.fake import FakeHome
from assistant.push import PhoneActions, PhonePusher, action_id, parse_action
from tests.test_dispatch import make_repo
from tests.test_phase4 import env_mode, settle
from tests.test_tasks import SPEC, make_board, wait_state


def notify_calls(home: FakeHome) -> list[dict]:
    return [data for domain, service, data in home.generic_calls if domain == "notify"]


async def test_payload_shape_actions_levels_and_clear(tmp_path: Path) -> None:
    home = FakeHome()
    pusher = PhonePusher(home, "notify.mobile_app_my_phone", name="Alexa")
    assert pusher.service == "mobile_app_my_phone"
    a = Announcer(tmp_path / "a.json")
    built = a.enqueue("Task 7 is built.", kind="task", context={"task_id": 7}, actions=["approve", "later"])
    await pusher.push(built)
    (domain, service, data) = home.generic_calls[0]
    assert (domain, service) == ("notify", "mobile_app_my_phone")
    assert data["title"] == "Alexa: Task" and data["message"] == "Task 7 is built."
    assert data["data"]["tag"] == "alexa-1" and data["data"]["group"] == "alexa"
    assert [x["action"] for x in data["data"]["actions"]] == ["alexa:approve:1", "alexa:later:1"]
    assert data["data"]["actions"][0]["title"] == "Approve & merge"
    assert data["data"]["push"]["interruption-level"] == "active"

    urgent = a.enqueue("Rolled back to main.", kind="system", priority="urgent")
    await pusher.push(urgent)
    card = notify_calls(home)[-1]
    assert card["data"]["push"]["interruption-level"] == "time-sensitive"
    assert [x["title"] for x in card["data"]["actions"]] == ["Got it"]  # the default button

    question = a.enqueue("Task 7 needs your call: first or full name?", kind="question", actions=["answer"])
    await pusher.push(question, level="passive")
    card = notify_calls(home)[-1]
    assert card["data"]["actions"][0]["behavior"] == "textInput" and card["title"] == "Alexa: Question"
    assert card["data"]["push"]["interruption-level"] == "passive"

    await pusher.clear(built)
    assert notify_calls(home)[-1] == {"message": "clear_notification", "data": {"tag": "alexa-1"}}
    pusher.clear_later(built)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert notify_calls(home)[-1]["message"] == "clear_notification"
    await pusher.say("Task 7 merged — restarting now.")
    assert notify_calls(home)[-1]["message"].startswith("Task 7 merged")

    assert parse_action("alexa:approve:12") == ("approve", 12)
    assert parse_action("nope") is None and parse_action("alexa:approve:x") is None
    assert action_id("later", 3) == "alexa:later:3"


async def test_phone_approve_merges_resolves_and_asks_for_restart(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)
    task = board.draft("Startup greeting", SPEC)
    await board.start(task.id)
    await wait_state(board, task.id)
    (Path(task.worktree) / "GREETING.md").write_text("hello", encoding="utf-8")
    await board._runner.commit_all(Path(task.worktree), "add greeting")
    built = next(a for a in announcer.pending() if a.kind == "task")
    assert built.actions == ["approve", "later"] and built.context == {"task_id": task.id}

    home = FakeHome()
    restarts: list[int] = []
    actions = PhoneActions(
        announcer, board=board, pusher=PhonePusher(home, "mobile_app_my_phone"),
        request_restart=lambda: restarts.append(1),
    )
    actions.handle_event({"action": action_id("approve", built.id)})
    assert actions.busy
    await actions.drain()
    assert board.get(task.id).state == "merged" and (repo / "GREETING.md").exists()
    assert built.state == "resolved" and restarts == [1]
    messages = [c.get("message", "") for c in notify_calls(home)]
    assert any(m.startswith("Task 1 merged") for m in messages) and "clear_notification" in messages
    assert any("merged from your phone" in a.text for a in announcer.pending())  # she'll say so too

    actions.handle_event({"action": action_id("approve", built.id)})  # a second tap
    await actions.drain()
    assert "already been handled" in notify_calls(home)[-1]["message"]


async def test_read_later_and_unknown_are_honest(tmp_path: Path) -> None:
    a = Announcer(tmp_path / "a.json")
    item = a.enqueue("The porch light came on.", kind="watch")
    a.take_due()
    a.mark_delivered([item.id])
    a.mark_pushed(item.id)
    home = FakeHome()
    actions = PhoneActions(a, pusher=PhonePusher(home, "svc"))

    actions.handle_event({"action": action_id("later", item.id)})
    await actions.drain()
    assert item.unread and notify_calls(home)[-1]["message"] == "clear_notification"
    actions.handle_event({"action": action_id("read", item.id)})
    await actions.drain()
    assert item.state == "read"
    actions.handle_event({"action": "alexa:read:99"})
    await actions.drain()
    assert "no longer have" in notify_calls(home)[-1]["message"]
    actions.handle_event({"action": "garbage"})
    assert not actions.busy
    actions.handle_event({"action": action_id("approve", item.id)})  # a watch has no task
    await actions.drain()
    assert "already been handled" in notify_calls(home)[-1]["message"]  # it was read: not live


async def test_answer_tap_routes_reply_text_to_the_task(tmp_path: Path, monkeypatch) -> None:
    env_mode(monkeypatch, tmp_path, "ask-once")
    repo = make_repo(tmp_path)
    board, announcer = make_board(repo)
    task = board.draft("Greeting", "greet by name")
    await board.start(task.id)
    await settle(board, task.id)
    question = next(a for a in announcer.pending() if a.kind == "question")
    home = FakeHome()
    actions = PhoneActions(announcer, board=board, pusher=PhonePusher(home, "svc"))
    actions.handle_event({"action": action_id("answer", question.id), "reply_text": ""})
    await actions.drain()
    assert "came through empty" in notify_calls(home)[-1]["message"]
    actions.handle_event({"action": action_id("answer", question.id), "reply_text": "first name"})
    await actions.drain()
    assert board.get(task.id).state == "revising" and question.state == "resolved"
    assert notify_calls(home)[-1]["message"].startswith("Sent to task 1")
    await settle(board, task.id)
    assert board.get(task.id).state == "built"
