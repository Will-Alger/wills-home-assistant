"""Pure-function tests for the realtime engine — no network."""

from __future__ import annotations

import numpy as np

from assistant.engines.realtime_engine import (
    FRAME_SAMPLES_24K,
    downsample_24k_to_16k,
    realtime_tools,
)


def test_tools_convert_to_realtime_shape() -> None:
    tools = realtime_tools()
    names = [tool["name"] for tool in tools]
    assert "set_lights" in names
    assert "get_lights" in names
    assert names[-1] == "end_conversation"
    for tool in tools:
        assert tool["type"] == "function"
        assert "parameters" in tool  # realtime name for the schema
        assert "input_schema" not in tool  # anthropic name must not leak


def test_calendar_tools_appear_only_when_a_calendar_is_configured() -> None:
    without = [tool["name"] for tool in realtime_tools()]
    assert "list_calendar_events" not in without  # never advertise a calendar we can't read
    with_calendar = [tool["name"] for tool in realtime_tools(calendar=True)]
    assert "list_calendar_events" in with_calendar
    assert "create_calendar_event" in with_calendar
    assert with_calendar[-1] == "end_conversation"


async def test_session_config_renders_jobs_and_repos(tmp_path) -> None:
    """The instructions template must format cleanly with a dispatcher wired
    in — a stray placeholder here would break every wake."""
    from assistant.dispatch import Dispatcher
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.tasks import TaskBoard

    runner = Dispatcher(
        tmp_path,
        routine_id="trig_x",
        routine_token="tok_x",
        extra_routines={"side-project": {"routine_id": "trig_s", "token": "tok_s"}},
    )
    board = TaskBoard(tmp_path, runner=runner)
    engine = RealtimeEngine(
        api_key="test-key",
        model="m",
        voice="v",
        home=FakeHome(),
        owner="Will",
        name="Alexa",
        wake_phrase="alexa",
        task_board=board,
    )
    config = await engine._session_config(None)
    text = config["instructions"]
    assert "Open tasks right now: none open" in text
    assert "side-project" in text  # she knows which other repos she can work on
    tool_names = {t["name"] for t in config["tools"]}
    assert {
        "draft_task", "start_task", "list_tasks", "task_detail", "search_tasks",
        "approve_task", "abandon_task", "switch_build", "revise_task", "answer_task",
    } <= tool_names
    states = next(t for t in config["tools"] if t["name"] == "list_tasks")
    assert "needs_input" in states["parameters"]["properties"]["states"]["items"]["enum"]
    assert "web_search" in tool_names
    assert "list_notifications" not in tool_names  # no announcer wired here
    assert "develop_feature" not in tool_names


async def test_session_config_renders_unread_and_no_stale_denial(tmp_path) -> None:
    from assistant.announce import Announcer
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome

    announcer = Announcer(tmp_path / "a.json", quiet_hours="00:00-24:00")  # always held → unread
    announcer.enqueue("Task 7 is built.", kind="task")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", announcer=announcer
    )
    config = await engine._session_config(None)
    text = config["instructions"]
    assert "Right now: 1 unread since" in text and "Task 7 is built." in text
    assert "cannot yet react to events" not in text  # she can, since M3
    names = {t["name"] for t in config["tools"]}
    assert {"list_notifications", "mark_notifications"} <= names
    assert "announcement_history" not in names


async def test_read_tools_are_answered_in_one_breath(tmp_path) -> None:
    """A lookup's result carries the no-preamble nudge; a command carries the
    close nudge; the instructions say to call the tool first and speak once."""
    import json
    from types import SimpleNamespace

    from assistant.engines.realtime_engine import RealtimeEngine, SessionStats
    from assistant.home.fake import FakeHome
    from tests.fake_realtime import FakeClient

    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will")
    connection = FakeClient().connection

    def call(name: str, args: dict) -> SimpleNamespace:
        item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id="c1")
        return SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))

    await engine._handle_response_done(connection, call("get_lights", {}), SessionStats())
    output = json.loads(next(e for e in connection.sent if e["type"] == "conversation.item.create")["item"]["output"])
    assert "one breath" in output["note"] and "result" in output
    connection.sent.clear()
    await engine._handle_response_done(
        connection, call("set_lights", {"changes": [{"target": "Hallway", "turn": "on"}]}), SessionStats()
    )
    output = json.loads(next(e for e in connection.sent if e["type"] == "conversation.item.create")["item"]["output"])
    assert "end_conversation" in output["note"]
    text = (await engine._session_config(None))["instructions"]
    assert "call the tool FIRST" in text and "let me pull that up" in text


def test_command_tools_cover_home_actions_only() -> None:
    """COMMAND_TOOLS drives the engine's one-shot auto-close: action tools
    only — info/chat tools must not trigger it."""
    from assistant.engines.realtime_engine import COMMAND_TOOLS

    assert {"set_lights", "media_control", "play_music", "launch_app"} <= COMMAND_TOOLS
    for info_tool in ("browse_music", "search_entities", "get_entity", "check_work",
                      "list_memories", "get_lights"):
        assert info_tool not in COMMAND_TOOLS


def test_downsample_produces_wake_sized_frames() -> None:
    frame_24k = np.zeros(FRAME_SAMPLES_24K, dtype=np.int16).tobytes()
    out = downsample_24k_to_16k(frame_24k)
    assert len(out) == 1280 * 2  # exactly one 80ms wake frame at 16 kHz


def test_downsample_preserves_a_tone() -> None:
    t = np.arange(FRAME_SAMPLES_24K) / 24_000
    tone = (10_000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    out = np.frombuffer(downsample_24k_to_16k(tone.tobytes()), dtype=np.int16)
    assert out.astype(np.float32).std() > 1000  # energy survived the resample
