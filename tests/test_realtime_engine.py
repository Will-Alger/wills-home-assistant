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
    assert "one breath" in output["follow_up"] and output["status"] == "success"
    connection.sent.clear()
    await engine._handle_response_done(
        connection, call("set_lights", {"changes": [{"target": "Hallway", "turn": "on"}]}), SessionStats()
    )
    output = json.loads(next(e for e in connection.sent if e["type"] == "conversation.item.create")["item"]["output"])
    assert "end_conversation" in output["follow_up"]
    text = (await engine._session_config(None))["instructions"]
    assert "the function call comes FIRST" in text and "let me check your schedule" in text
    # the lookups themselves say so — the model weighs a tool's description above the prose
    by_name = {tool["name"]: tool["description"] for tool in realtime_tools(calendar=True)}
    assert by_name["list_calendar_events"].endswith("the first thing said is the answer it returns.")
    assert "CALL THIS BEFORE SPEAKING" in by_name["get_lights"] and "BEFORE SPEAKING" in by_name["web_search"]
    assert "BEFORE SPEAKING" not in by_name["set_lights"]  # commands are confirmed after, not narrated before


async def test_spoken_reply_rules_are_in_the_instructions() -> None:
    """Endings and reading aloud: no dead-end yes/no, plant a seed instead,
    and machine forms like "18:30" get spoken, not read out."""
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome

    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will")
    text = (await engine._session_config(None))["instructions"]
    assert 'not "want me to explain more?"' in text and "plant a seed instead" in text
    assert "Speaking aloud" in text and '"six thirty", not "18:30"' in text
    assert "the id as the plain name of the thing" in text
    assert "a sentence or two unless asked to go deeper" in text  # brevity rules untouched


async def test_noise_reduction_is_sent_only_when_configured() -> None:
    import json

    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome

    plain = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will")
    assert "noise_reduction" not in json.dumps(await plain._session_config(None))
    desk = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", noise_reduction="far_field")
    assert '"noise_reduction": {"type": "far_field"}' in json.dumps(await desk._session_config(None))
    odd = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", noise_reduction="loud")
    assert "noise_reduction" not in json.dumps(await odd._session_config(None))  # never an invalid value


async def test_the_transcriber_is_handed_the_house_s_names() -> None:
    """audio.input.transcription.prompt (SDK: AudioTranscriptionParam.prompt) —
    only when a transcriber is actually running, and only names."""
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.vocabulary import MAX_VOCABULARY_CHARS

    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa"
    )
    await engine.music_names.refresh()  # the boot warm-up, done inline

    off = await engine._session_config(None)
    assert "transcription" not in off["audio"]["input"]  # no transcriber, no prompt

    on = await engine._session_config("gpt-4o-mini-transcribe")
    transcription = on["audio"]["input"]["transcription"]
    assert transcription["model"] == "gpt-4o-mini-transcribe"
    prompt = transcription["prompt"]
    assert len(prompt) <= MAX_VOCABULARY_CHARS
    assert prompt.startswith("Alexa, Will, Bedroom, Hallway, Kitchen, Living Room, ")
    names = prompt.split(", ")
    for earlier, later in (
        ("Apple TV", "Kitchen Strip"),  # players before plain light names
        ("Cleveland 10K", "Kitchen Strip"),  # so is the owner's library
        ("Dave Brubeck", "Bedroom Lamp"),
    ):
        assert names.index(earlier) < names.index(later)


async def test_open_task_titles_reach_the_transcriber_and_closed_ones_do_not(tmp_path) -> None:
    from assistant.dispatch import Dispatcher
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.tasks import TaskBoard

    board = TaskBoard(tmp_path, runner=Dispatcher(tmp_path, routine_id="r", routine_token="t"))
    board.draft("Transcriber vocabulary", "spec")
    board.draft("Delayed playback replay tests", "spec").closed = True
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", task_board=board
    )
    prompt = (await engine._session_config("whisper-1"))["audio"]["input"]["transcription"]["prompt"]
    assert "Transcriber vocabulary" in prompt
    assert "Delayed playback replay tests" not in prompt


def test_wrapup_phrases_are_recognised_whole_not_by_fragment() -> None:
    from assistant.engines.realtime_engine import is_wrapup

    for said in (
        "That's all.", "that’s it", "OK, that's all, thanks!", "No, that's all Alexa.", "Thanks, bye!",
        "Nothing else, thank you.", "Alright, I'm good.", "Never mind.", "Good night", "that'll be all for now",
        "Okay thanks that's it for tonight.", "Bye bye.",
    ):
        assert is_wrapup(said), said
    for said in (
        "That's all for the lights, now play some music.", "Is that all you can do?", "Bye the way, what time is it",
        "Good night mode please", "", "thanks", "ok", "turn it off",
    ):
        assert not is_wrapup(said), said


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


async def test_a_correction_is_one_call_and_leaves_one_answer(tmp_path) -> None:
    """The voice test: 3000 kelvin, then "actually make that 2700" — one
    answer afterwards, in the tools and in the instructions."""
    from assistant.engines.realtime_engine import MEMORY_TOOLS, RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.memory import MemoryStore

    memory = MemoryStore(tmp_path / "m.json")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", memory=memory
    )
    said, is_error = engine._execute_memory(
        "remember",
        {"kind": "preference", "text": "the living room at 3000 kelvin", "subject": "lights"},
    )
    assert not is_error and engine._instructions_stale
    stored = memory.items("preference")[0]

    said, is_error = engine._execute_memory(
        "update_memory", {"id": stored.id, "text": "the living room at 2700 kelvin"}
    )
    assert not is_error and "retired" in said

    listed, is_error = engine._execute_memory("list_memories", {"subject": "lights"})
    assert not is_error and "2700" in listed and "3000" not in listed
    instructions = (await engine._session_config(None))["instructions"]
    assert "the living room at 2700 kelvin" in instructions and "3000" not in instructions

    # A house default is stored and rendered apart from his own preference.
    engine._execute_memory(
        "remember",
        {"kind": "house", "text": "the porch light goes off at midnight", "subject": "house"},
    )
    instructions = (await engine._session_config(None))["instructions"]
    assert "House defaults" in instructions and "porch light goes off" in instructions

    names = [tool["name"] for tool in MEMORY_TOOLS]
    assert "update_memory" in names  # advertised, or the model falls back to two steps
    missing, is_error = engine._execute_memory("update_memory", {"id": 99, "text": "x"})
    assert is_error and "99" in missing  # an invented id is an honest error, not a write


async def test_instructions_carry_the_subjects_in_play(tmp_path) -> None:
    from assistant.engines.realtime_engine import RealtimeEngine
    from assistant.home.fake import FakeHome
    from assistant.memory import INJECT_WHOLE_BELOW, MemoryStore

    memory = MemoryStore(tmp_path / "m.json")
    # An old preference about the lights, buried under a pile of newer ones.
    memory.add("preference", "the reading lamp at 2350 kelvin", subject="lights")
    for i in range(INJECT_WHOLE_BELOW + 4):
        memory.add("preference", f"a calendar rule numbered {i}", subject="calendar")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", memory=memory
    )

    cold = (await engine._session_config(None))["instructions"]
    assert "a calendar rule numbered 0" not in cold  # a big store is never recited
    assert "2350" not in cold  # nothing about lights is in play yet

    engine._tools_in_play.append("set_lights")
    warm = (await engine._session_config(None))["instructions"]
    assert "2350" in warm  # touch a lamp and the lights preference is retrieved
    assert "a calendar rule numbered 0" not in warm
