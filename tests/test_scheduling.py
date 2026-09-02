"""Scheduling & routines: timers, alarms, scheduled actions with a fake clock;
routines applied deterministically to tool calls."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from assistant.announce import Announcer
from assistant.brain.tools import ToolExecutor
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.routines import RoutineStore
from assistant.scheduler import Scheduler, spoken_time

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def wed(hour: int, minute: int = 0, day: int = 2) -> float:  # 2026-09-02 is a Wednesday
    return datetime(2026, 9, day, hour, minute, tzinfo=LOCAL).timestamp()


async def test_timer_fires_once_as_an_urgent_announcement(tmp_path: Path) -> None:
    clock = Clock(wed(12))
    announcer = Announcer(tmp_path / "a.json", quiet_hours="00:00-23:59", now=clock)
    sched = Scheduler(tmp_path / "schedule.json", announcer=announcer, now=clock)
    item = sched.set_timer(20 * 60, "pasta")
    assert sched.describe()[0]["next"] == spoken_time(item.fire_at)
    assert await sched.tick() == []
    clock.at += 20 * 60 + 1
    assert await sched.tick() == ["Your pasta is up."]
    assert not sched.active() and announcer.due()  # urgent: even in quiet hours
    assert announcer.pending()[0].kind == "timer"
    with pytest.raises(ValueError):
        sched.set_timer(0)


async def test_alarm_recurs_on_its_days_and_can_be_snoozed(tmp_path: Path) -> None:
    clock = Clock(wed(6))
    sched = Scheduler(tmp_path / "schedule.json", now=clock)
    alarm = sched.set_alarm("07:00", days=["mon", "tue", "wed", "thu", "fri"], label="work alarm")
    assert sched.next_fire(alarm) == wed(7)
    clock.at = wed(7, 0) + 1
    (said,) = await sched.tick()
    assert said.startswith("It's 7:00 AM") and "work alarm" in said
    assert alarm.active and sched.next_fire(alarm) == wed(7, 0, day=3)  # Thursday
    snoozed = sched.snooze(None, 10)  # the one that just fired
    assert snoozed is alarm and sched.next_fire(alarm) == clock.at + 600
    clock.at += 601
    assert len(await sched.tick()) == 1  # the snooze fires, then back to the rota
    assert sched.next_fire(alarm) == wed(7, 0, day=3)

    clock.at = wed(20, 0, day=4)  # Friday evening → next allowed day is Monday
    sched.set_alarm("07:00", days=["mon", "tue", "wed", "thu", "fri"], label="x")
    assert datetime.fromtimestamp(sched.next_fire(sched.active()[-1])).astimezone().weekday() == 0

    again = Scheduler(tmp_path / "schedule.json", now=clock)  # persisted
    assert [i.label for i in again.active()] == ["work alarm", "x"]
    assert again.cancel(alarm.id) is not None and again.cancel(alarm.id) is None


async def test_scheduled_action_runs_a_home_tool_and_reports(tmp_path: Path) -> None:
    clock = Clock(wed(17, 55))
    home = FakeHome()
    announcer = Announcer(tmp_path / "a.json", now=clock)
    sched = Scheduler(tmp_path / "schedule.json", announcer=announcer, executor=ToolExecutor(home), now=clock)
    item = sched.schedule(
        kind="action",
        label="porch light on",
        at="18:00",
        repeat=True,
        action={"tool": "set_lights", "input": {"changes": [{"target": "Hallway", "turn": "on"}]}},
    )
    reminder = sched.schedule(kind="reminder", label="call mom", in_seconds=120, message="call your mom")
    clock.at = wed(18, 0) + 1
    said = await sched.tick()
    assert "Done: porch light on." in said and "Reminder: call your mom" in said
    assert home.applied and home.applied[0].entity_id == "light.hallway"
    assert item.active and not reminder.active  # daily action keeps going; the reminder was one-shot
    kinds = {a.kind: a.priority for a in announcer.pending()}
    assert kinds == {"action": "normal", "reminder": "urgent"}
    with pytest.raises(ValueError, match="needs a tool"):
        sched.schedule(kind="action", label="x", at="10:00")


def test_routines_apply_defaults_and_overrides_deterministically(tmp_path: Path) -> None:
    evening = Clock(wed(18))
    store = RoutineStore(tmp_path / "routines.json", now=evening)
    store.add(
        "after 5pm, lights I ask for come on warm orange",
        tool="set_lights", after="17:00", before="06:00", defaults={"rgb_color": [255, 140, 40]},
    )
    store.add(
        "TV volume defaults to 65%",
        tool="media_control", match={"action": "volume_set"}, defaults={"volume_pct": 65},
    )
    out, applied = store.apply(
        "set_lights", {"changes": [{"target": "Living Room", "turn": "on"}, {"target": "Hallway", "turn": "off"}]}
    )
    assert out["changes"][0]["rgb_color"] == [255, 140, 40]  # filled in for the light turning on
    assert "rgb_color" not in out["changes"][1]  # lights going off are left alone
    assert next(r.description for r in applied).startswith("after 5pm")

    out, _ = store.apply("set_lights", {"changes": [{"target": "Living Room", "turn": "on", "color_temp_kelvin": 6000}]})
    assert "rgb_color" not in out["changes"][0]  # an explicit color wins over the default

    out, applied = store.apply("media_control", {"action": "volume_set"})
    assert out["volume_pct"] == 65 and len(applied) == 1
    out, applied = store.apply("media_control", {"action": "volume_set", "volume_pct": 30})
    assert out["volume_pct"] == 30 and len(applied) == 1  # a default never overrides what was said
    out, applied = store.apply("media_control", {"action": "pause"})
    assert applied == []  # match filter

    noon = RoutineStore(tmp_path / "routines.json", now=Clock(wed(12)))
    out, applied = noon.apply("set_lights", {"changes": [{"target": "Living Room", "turn": "on"}]})
    assert applied == [] and "rgb_color" not in out["changes"][0]  # outside the window
    # applied twice this evening (the explicit-color call still matched the window)
    assert noon.describe()[0]["applied"] == 2 and "after 17:00 before 06:00" in noon.describe()[0]["window"]
    assert "[id 1]" in noon.text()
    assert noon.remove(1) is not None and noon.remove(1) is None


async def test_executor_applies_routines_and_reports_them(tmp_path: Path) -> None:
    store = RoutineStore(tmp_path / "routines.json")
    store.add("living room lights are always 40%", tool="set_lights", overrides={"brightness_pct": 40})
    home = FakeHome()
    executor = ToolExecutor(home, routines=store)
    text, is_error = await executor.execute(
        "set_lights", {"changes": [{"target": "Living Room", "turn": "on", "brightness_pct": 100}]}
    )
    assert not is_error, text
    assert all(cmd.brightness_pct == 40 for cmd in home.applied)  # override beat the request
    assert executor.last_routines == ["living room lights are always 40%"]


def test_schedule_and_routine_tools_by_voice(tmp_path: Path) -> None:
    clock = Clock(wed(9))
    sched = Scheduler(tmp_path / "schedule.json", now=clock)
    routines = RoutineStore(tmp_path / "routines.json", now=clock)
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", scheduler=sched, routines=routines
    )
    text, is_error = engine._execute_schedule_tool("set_timer", {"seconds": 600, "label": "tea"})
    assert not is_error and "timer 1 ('tea')" in text
    text, is_error = engine._execute_schedule_tool(
        "set_alarm", {"at": "07:00", "days": ["mon", "tue", "wed", "thu", "fri"], "label": "work alarm"}
    )
    assert not is_error and "repeating mon, tue, wed, thu, fri" in text
    text, is_error = engine._execute_schedule_tool(
        "schedule", {"kind": "action", "label": "porch light", "at": "18:30", "repeat": True, "tool": "set_lights",
                     "tool_input": {"changes": [{"target": "Hallway", "turn": "on"}]}}
    )
    assert not is_error and "action 3" in text and "repeating" in text
    rows = json.loads(engine._execute_schedule_tool("list_schedule", {})[0])
    assert {r["kind"] for r in rows} == {"timer", "alarm", "action"}
    text, is_error = engine._execute_schedule_tool("set_alarm", {"at": "25:00"})
    assert is_error and "07:00" in text
    assert "cancelled" in engine._execute_schedule_tool("cancel_schedule", {"id": 1})[0]

    text, is_error = engine._execute_routine_tool(
        "add_routine",
        {"description": "TV volume defaults to 65%", "tool": "media_control",
         "match": {"action": "volume_set"}, "defaults": {"volume_pct": 65}},
    )
    assert not is_error and "routine 1 added" in text and engine._instructions_stale
    assert json.loads(engine._execute_routine_tool("list_routines", {})[0])[0]["rule"] == "TV volume defaults to 65%"
    text, is_error = engine._execute_routine_tool("add_routine", {"description": "no effect"})
    assert is_error and "defaults or overrides" in text
    assert "removed" in engine._execute_routine_tool("remove_routine", {"id": 1})[0]


async def test_confirm_actions_ask_before_running(tmp_path: Path) -> None:
    clock = Clock(wed(21, 59))
    home = FakeHome()
    announcer = Announcer(tmp_path / "a.json", now=clock)
    sched = Scheduler(tmp_path / "schedule.json", announcer=announcer, executor=ToolExecutor(home), now=clock)
    lock = sched.schedule(
        kind="action", label="lock up", at="22:00", repeat=True, confirm=True,
        action={"tool": "set_lights", "input": {"changes": [{"target": "Hallway", "turn": "off"}]}},
    )
    assert sched.describe()[0]["asks first"] is True
    clock.at = wed(22, 0) + 1
    (asked,) = await sched.tick()
    assert asked == "It's 10:00 PM. Shall I lock up? Say yes or no."
    assert home.applied == []  # nothing ran yet
    (question,) = announcer.pending()
    assert question.kind == "question" and question.actions == ["yes", "no"]
    assert question.context == {"schedule_id": lock.id, "fired_at": clock.at}
    assert sched.awaiting_confirmation() == [lock] and sched.describe()[0]["awaiting your yes"] is True

    assert await sched.confirm(lock.id, True) == "Done: lock up."
    assert home.applied and home.applied[0].entity_id == "light.hallway"
    assert question.state == "resolved" and sched.awaiting_confirmation() == []
    assert await sched.confirm(lock.id, True) == "nothing is waiting on a yes for that"  # idempotent
    assert lock.active  # the nightly rule keeps going

    clock.at = wed(22, 0, day=3) + 1
    await sched.tick()
    assert await sched.confirm(lock.id, False) == "Okay, skipping lock up."
    assert len(home.applied) == 1

    clock.at = wed(22, 0, day=4) + 1
    await sched.tick()
    clock.at += 1801  # he never answered: skipped quietly
    await sched.tick()
    assert sched.awaiting_confirmation() == [] and len(home.applied) == 1

    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", scheduler=sched)
    text, is_error = await engine._execute_confirm_action({"id": lock.id, "yes": True})
    assert is_error and "nothing is waiting" in text
    text, is_error = engine._execute_schedule_tool(
        "schedule", {"kind": "action", "label": "heat on", "in_seconds": 60, "confirm": True,
                     "tool": "set_lights", "tool_input": {"changes": []}}
    )
    assert not is_error and text.endswith("it will ask first")


async def test_briefing_kind_speaks_the_composed_summary(tmp_path: Path) -> None:
    clock = Clock(wed(7, 25))
    announcer = Announcer(tmp_path / "a.json", quiet_hours="23:00-08:00", now=clock)
    sched = Scheduler(tmp_path / "schedule.json", announcer=announcer, now=clock)

    async def compose() -> str:
        return "Good morning, Will. Today: run club at 6:30 PM. Awaiting your approval: task 1."

    sched.briefing = compose
    item = sched.schedule(kind="briefing", label="morning briefing", at="07:30", repeat=True, days=["mon", "tue", "wed", "thu", "fri"])
    clock.at = wed(7, 30) + 1
    (said,) = await sched.tick()
    assert said.startswith("Good morning, Will.") and "task 1" in said
    assert item.active and announcer.due()  # inside quiet hours, but the owner scheduled it
    assert announcer.pending()[0].kind == "briefing"
