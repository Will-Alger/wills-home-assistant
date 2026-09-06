"""Working context: the last few minutes survive the close, and only those.

"A little dimmer" in a brand-new session has to find the lamps she just set;
half an hour later it must find nothing at all, so she asks instead of
guessing at the wrong room.
"""

from __future__ import annotations

import asyncio

from assistant.app import record_session
from assistant.context import (
    WorkingContext,
    pending_question,
    question_options,
    temporary_override,
)
from assistant.engines.realtime_engine import RealtimeEngine, SessionStats
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient, InstantSpeaker, NeverMic, QuietUi

MINUTE = 60.0


class Clock:
    def __init__(self, at: float = 1_000_000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at

    def tick(self, minutes: float) -> None:
        self.at += minutes * MINUTE


def make_engine(path, **kw) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="test-key", model="m", voice="v", home=FakeHome(), owner="Will", name="Alexa",
        wake_phrase="alexa", context=WorkingContext(path), **kw,
    )
    client = FakeClient()
    engine._client = client
    return engine, client


# ── across the close ───────────────────────────────────────────────────────


async def test_a_light_command_binds_the_lamps_for_the_next_session(tmp_path) -> None:
    """Set the living-room lamps, let the session close, and the NEXT
    session's instructions still name them — that is what "a little dimmer"
    lands on."""
    path = tmp_path / "context.json"
    engine, client = make_engine(path, command_close_s=0.3)

    async def owner() -> None:
        await asyncio.sleep(0.1)
        client.connection.user_says(
            "Make the living room cozy",
            reply="Cozy.",
            calls=[
                (
                    "set_lights",
                    {"changes": [{"target": "Living Room", "turn": "on", "brightness_pct": 30}]},
                )
            ],
        )

    turn = asyncio.create_task(owner())
    await asyncio.wait_for(
        engine.run_conversation(NeverMic(), InstantSpeaker(), None, QuietUi(), announce=False), 8
    )
    await turn

    rendered = WorkingContext(path).text()
    assert "light.living_room_lamp" in rendered
    assert "light.living_room_ceiling" in rendered
    assert "Living Room Lamp" in rendered  # the friendly name, for her own words
    assert "light.kitchen_strip" not in rendered  # untouched rooms are not candidates
    assert "set_lights" in rendered
    # the level she chose, so "a little brighter" is a step from 30, not a guess
    assert '"brightness_pct":30' in rendered
    assert "Done: 2 light(s) updated" in rendered

    # A brand-new session (a fresh engine, fresh store) carries it in.
    later, _client = make_engine(path)
    instructions = (await later._session_config(None))["instructions"]
    assert rendered in instructions
    assert "with ONE fresh candidate, just act on it" in instructions


async def test_without_a_context_the_instructions_still_render(tmp_path) -> None:
    engine = RealtimeEngine(
        api_key="test-key", model="m", voice="v", home=FakeHome(), owner="Will",
        name="Alexa", wake_phrase="alexa",
    )
    assert "(not kept between sessions)" in (await engine._session_config(None))["instructions"]


# ── the clocks ─────────────────────────────────────────────────────────────


def test_everything_is_gone_after_thirty_one_minutes(tmp_path) -> None:
    clock = Clock()
    ctx = WorkingContext(tmp_path / "context.json", now=clock)
    ctx.note_topic("making the living room cozy")
    ctx.note_entities([("light.living_room_lamp", "Living Room Lamp")])
    ctx.note_action("set_lights", {"changes": []}, "Done: 2 light(s) updated.")
    ctx.note_override("just for tonight, keep the volume down")
    ctx.note_job("think", "thinking over 'where to plant the fig'")
    assert "light.living_room_lamp" in ctx.text()

    clock.tick(31)
    assert ctx.snapshot() == {}
    assert ctx.text().startswith("nothing recent")
    # ...and a restart reads the same emptiness off disk, not the stale file.
    assert WorkingContext(tmp_path / "context.json", now=clock).text().startswith("nothing recent")


def test_a_pending_question_goes_cold_before_the_lamps_do(tmp_path) -> None:
    clock = Clock()
    ctx = WorkingContext(tmp_path / "context.json", now=clock)
    ctx.note_entities([("light.hallway", "Hallway Light")])
    ctx.ask("Warm white or amber?", question_options("Warm white or amber?"))

    text = ctx.text()
    assert 'you asked "Warm white or amber?" and got no answer' in text
    assert "the options were: Warm white, amber" in text

    clock.tick(11)  # the question's own clock (10 minutes) has run out
    text = ctx.text()
    assert "got no answer" not in text
    assert "light.hallway" in text  # the lamps are still fair game for another 19


# ── what goes in ───────────────────────────────────────────────────────────


def test_entities_merge_so_two_candidates_stay_two(tmp_path) -> None:
    ctx = WorkingContext(tmp_path / "context.json")
    ctx.note_entities([("light.living_room_lamp", "Living Room Lamp")])
    ctx.note_entities([("media_player.living_room_speakers", "Living Room Speakers")])
    text = ctx.text()
    assert "media_player.living_room_speakers" in text  # newest first
    assert text.index("media_player") < text.index("light.living_room_lamp")


def test_an_old_lamp_does_not_ride_along_on_a_fresh_speaker(tmp_path) -> None:
    """Merging must not resurrect: each entity ages on its own moment, or
    "turn it off" half an hour later still offers a lamp nobody mentioned."""
    clock = Clock()
    ctx = WorkingContext(tmp_path / "context.json", now=clock)
    ctx.note_entities([("light.living_room_lamp", "Living Room Lamp")])
    clock.tick(31)
    ctx.note_entities([("media_player.living_room_speakers", "Living Room Speakers")])
    text = ctx.text()
    assert "media_player.living_room_speakers" in text
    assert "light.living_room_lamp" not in text


def test_a_job_stops_being_outstanding_when_it_lands(tmp_path) -> None:
    ctx = WorkingContext(tmp_path / "context.json")
    ctx.note_job("think", "thinking over 'the fig tree'")
    assert "still running: thinking over 'the fig tree'" in ctx.text()
    ctx.clear_job("think")
    assert "still running" not in ctx.text()


def test_options_are_only_read_off_a_real_choice() -> None:
    assert question_options("Would you like warm white or amber?") == ["warm white", "amber"]
    assert question_options("Should I dim them?") == []
    assert question_options("Anything else?") == []


def test_a_question_he_answered_is_not_pending() -> None:
    asked = [("you", "lights on"), ("alexa", "Warm white or amber?")]
    assert pending_question(asked) == ("Warm white or amber?", ["Warm white", "amber"])
    assert pending_question([*asked, ("you", "amber")]) is None
    assert pending_question([("alexa", "Done, hallway's dimmed.")]) is None
    # only her LAST sentence counts — a statement after the question is an answer
    assert pending_question([("alexa", "Which lamp? Never mind, doing both.")]) is None


def test_only_an_explicit_just_for_now_is_a_temporary_override() -> None:
    assert temporary_override("Just for tonight, keep the volume down")
    assert temporary_override("Only this once, skip the warm colour")
    assert not temporary_override("That's all for now, thanks")  # a goodbye, not a rule


# ── session end ────────────────────────────────────────────────────────────


async def test_the_end_of_a_session_leaves_the_topic_the_question_and_the_rule(tmp_path) -> None:
    ctx = WorkingContext(tmp_path / "context.json")
    stats = SessionStats(
        transcript=[
            ("you", "Just for tonight, keep the lamps low"),
            ("alexa", "Warm white or amber?"),
        ]
    )
    await record_session(None, None, stats, None, None, context=ctx)
    text = ctx.text()
    assert "Just now (a moment ago): Just for tonight, keep the lamps low" in text
    assert 'you asked "Warm white or amber?" and got no answer' in text
    assert "just for this stretch: Just for tonight, keep the lamps low" in text


async def test_answering_in_the_next_session_retires_the_question(tmp_path) -> None:
    ctx = WorkingContext(tmp_path / "context.json")
    ctx.ask("Warm white or amber?", ["warm white", "amber"])
    answered = SessionStats(transcript=[("you", "amber"), ("alexa", "Amber it is.")])
    await record_session(None, None, answered, None, None, context=ctx)
    assert "got no answer" not in ctx.text()


async def test_an_announcement_nobody_heard_leaves_the_question_alone(tmp_path) -> None:
    ctx = WorkingContext(tmp_path / "context.json")
    ctx.ask("Warm white or amber?", ["warm white", "amber"])
    await record_session(None, None, SessionStats(), None, None, context=ctx)
    assert "got no answer" in ctx.text()
