"""Thinker jobs belong to the current topic: an id, a status, cancellation,
supersession, and a freshness check before the answer is ever spoken."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.announce import Announcer
from assistant.brain.thinker import Thinker
from assistant.context import WorkingContext
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome
from assistant.journal import Journal
from assistant.llm.base import TurnResult, Usage
from assistant.thoughts import (
    ThoughtBook,
    announcement,
    assumption_change,
    moved_on,
    same_topic,
    topic_of,
)

BIKE = "help me think through buying a bike"
REVISED = "should I buy a bike if I wait a year"
DINNER = "what should I make for dinner tonight"


class GatedLLM:
    """Answers only once released, so a second think can arrive mid-flight.
    Answers are keyed by a word in the question, never by call order — a
    superseded job may be cancelled before it ever reaches the model."""

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers = answers or {"": "Buy the bike now."}
        self.gate = asyncio.Event()
        self.calls: list[str] = []

    async def turn(self, *, system, tools, messages, response_schema=None) -> TurnResult:
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        await self.gate.wait()
        answer = next(
            (text for key, text in self.answers.items() if key and key in prompt),
            self.answers.get("", "Buy the bike now."),
        )
        return TurnResult(
            stop_reason="end_turn", text=answer, tool_calls=(),
            assistant_content=[], usage=Usage(), model="claude-fake",
        )


def make_engine(tmp_path: Path, llm: GatedLLM, **extra) -> tuple[RealtimeEngine, Announcer]:
    announcer = Announcer(tmp_path / "a.json")
    engine = RealtimeEngine(
        api_key="k", model="m", voice="v", home=FakeHome(), owner="Will",
        announcer=announcer, thinker=Thinker(llm, owner="Will"), **extra,
    )
    engine._live_conversation = 1  # a conversation is open: an answer may interrupt
    return engine, announcer


async def settle(engine: RealtimeEngine) -> None:
    await asyncio.gather(*engine._thinking, return_exceptions=True)


# ── the topic, on its own ──────────────────────────────────────────────────


def test_a_revised_question_is_the_same_topic_and_a_new_subject_is_not() -> None:
    assert same_topic(BIKE, REVISED)
    assert same_topic(BIKE, "is a bike really worth buying")
    assert not same_topic(BIKE, DINNER)
    assert not same_topic(BIKE, "which thermostat should I buy")  # only the buying is shared
    assert topic_of(BIKE) == "buying a bike"


def test_moved_on_is_looser_than_supersession_because_dropping_costs_more() -> None:
    # One word still in common is enough to keep an answer he asked for...
    assert not moved_on("the bike again", "buying a bike", BIKE)
    assert not moved_on("which thermostat should I buy", "buying a bike", BIKE)
    assert not moved_on("", "buying a bike", BIKE)
    # ...but a room talking about dinner has plainly left the bike behind.
    assert moved_on(DINNER, "buying a bike", BIKE)


def test_the_premise_moving_is_heard_without_any_word_overlap() -> None:
    assert assumption_change("Actually, assume I wait a year.") == "assume I wait a year"
    assert assumption_change("okay, suppose the rent goes up") == "suppose the rent goes up"
    assert assumption_change("what if I move instead") == "what if I move instead"
    assert assumption_change("turn the kitchen lights down") == ""
    assert assumption_change("hang on a second") == ""


def test_a_job_carries_an_id_status_and_times_and_survives_a_restart(tmp_path: Path) -> None:
    clock = [1000.0]
    path = tmp_path / "thoughts.json"
    book = ThoughtBook(path, now=lambda: clock[0])
    thought, replaced = book.start(BIKE, conversation=4)
    assert not replaced
    assert (thought.id, thought.status, thought.started) == (1, "running", 1000.0)
    assert thought.topic == "buying a bike" and thought.conversation == 4
    clock[0] = 1030.0
    assert book.finish(thought.id, "Buy it.") is not None
    assert thought.status == "done" and thought.finished == 1030.0

    # A job still running when she shut down is not running when she wakes.
    running, _ = book.start(DINNER)
    reopened = ThoughtBook(path, now=lambda: clock[0])
    assert [t.status for t in reopened.recent()] == ["done", "cancelled"]
    assert reopened.get(running.id).status == "cancelled"
    assert reopened.start("something else")[0].id == running.id + 1  # ids never repeat


def test_a_second_think_on_the_topic_supersedes_the_first(tmp_path: Path) -> None:
    book = ThoughtBook(tmp_path / "t.json")
    first, _ = book.start(BIKE)
    second, replaced = book.start(REVISED, said="Actually, assume I wait a year")
    assert [t.id for t in replaced] == [first.id]
    assert first.status == "superseded" and first.superseded_by == second.id
    assert second.restarted_from == first.id and second.changed == "assume I wait a year"
    # Its answer is never spoken, however long the old job takes to arrive.
    assert book.finish(first.id, "Buy it now.") is None
    assert book.finish(second.id, "Wait the year.") is second


def test_an_unrelated_question_leaves_the_running_one_alone(tmp_path: Path) -> None:
    book = ThoughtBook(tmp_path / "t.json")
    first, _ = book.start(BIKE)
    _, replaced = book.start(DINNER)
    assert not replaced and first.running and len(book.running()) == 2
    # "Actually, assume…" only speaks for a single running job; with two on
    # the go it takes the words to say which, and neither is guessed at.
    _, replaced = book.start("what about a scooter", said="actually, assume I wait a year")
    assert not replaced and len(book.running()) == 3


def test_a_stale_answer_is_kept_while_the_topic_holds_and_dropped_once_it_moves(
    tmp_path: Path,
) -> None:
    clock = [1000.0]
    book = ThoughtBook(tmp_path / "t.json", now=lambda: clock[0])
    thought, _ = book.start(BIKE)
    clock[0] += 40 * 60  # forty minutes: past the half-hour freshness bar
    book.finish(thought.id, "Buy the bike.")
    assert book.deliverable(thought, current_topic="")[0]  # nothing else on the table
    assert book.deliverable(thought, current_topic="the bike again")[0]
    speak, why = book.deliverable(thought, current_topic=DINNER)
    assert not speak and "moved on" in why and "buying a bike" in why


def test_the_bridge_is_required_only_when_the_question_was_restarted() -> None:
    book = ThoughtBook()
    plain, _ = book.start(BIKE)
    book.finish(plain.id, "Buy it.")
    assert announcement(plain, "Will") == f"Your deeper reasoning on '{BIKE}': Buy it."

    running, _ = book.start(BIKE)  # asked again, and this time the premise moves
    revised, replaced = book.start(REVISED, said="actually, assume I wait a year")
    assert [t.id for t in replaced] == [running.id]
    book.finish(revised.id, "Wait the year.")
    text = announcement(revised, "Will")
    assert "SHORT BRIDGE" in text and "assume I wait a year" in text and "Wait the year." in text


# ── through the engine, the way the voice uses it ──────────────────────────


async def test_two_thinks_on_one_topic_speak_only_the_second_with_a_bridge(
    tmp_path: Path,
) -> None:
    llm = GatedLLM({"wait a year": "Wait the year, then buy.", "": "Buy the bike now."})
    engine, announcer = make_engine(tmp_path, llm)
    engine._live_transcript = [("you", BIKE)]
    first, _ = await engine._execute_brain_tool(
        "think", {"question": BIKE, "topic": "buying a bike"}
    )
    assert "thought 1" in first

    engine._live_transcript.append(("you", "Actually, assume I wait a year."))
    second, _ = await engine._execute_brain_tool(
        "think", {"question": REVISED, "topic": "buying a bike"}
    )
    assert "supersedes thought 1" in second

    llm.gate.set()
    await settle(engine)
    (item,) = announcer.pending()  # one answer only, and it is the revised one
    assert "Wait the year, then buy." in item.text and "Buy the bike now." not in item.text
    assert "SHORT BRIDGE" in item.text and "assume I wait a year" in item.text
    assert [t.status for t in engine._thoughts.recent()] == ["superseded", "done"]


async def test_cancel_thought_stops_the_answer_from_ever_arriving(tmp_path: Path) -> None:
    llm = GatedLLM()
    engine, announcer = make_engine(tmp_path, llm)
    await engine._execute_brain_tool("think", {"question": BIKE})
    text, is_error = await engine._execute_brain_tool("cancel_thought", {"id": 1})
    assert not is_error and "buying a bike" in text

    llm.gate.set()
    await settle(engine)
    assert not announcer.pending()
    assert engine._thoughts.get(1).status == "cancelled"
    # Cancelling something that is not running says so instead of pretending.
    assert (await engine._execute_brain_tool("cancel_thought", {"id": 1}))[1]
    assert (await engine._execute_brain_tool("cancel_thought", {"id": 99}))[1]


async def test_an_old_answer_after_the_topic_moved_on_is_journaled_not_spoken(
    tmp_path: Path,
) -> None:
    clock = [1000.0]
    journal = Journal(tmp_path / "journal")
    llm = GatedLLM()
    engine, announcer = make_engine(
        tmp_path,
        llm,
        thoughts=ThoughtBook(tmp_path / "t.json", now=lambda: clock[0]),
        journal=journal,
    )
    engine._live_transcript = [("you", BIKE)]
    await engine._execute_brain_tool("think", {"question": BIKE})

    clock[0] += 45 * 60  # she took three quarters of an hour, and he moved on
    engine._live_transcript = [("you", DINNER)]
    llm.gate.set()
    await settle(engine)

    assert not announcer.pending()
    (row,) = [e for e in journal.query(kinds=["thought"]) if "dropped" in e.text]
    assert "buying a bike" in row.text and "moved on" in row.text
    assert row.data["answer"] == "Buy the bike now."


async def test_the_working_context_topic_beats_the_last_thing_said(tmp_path: Path) -> None:
    clock = [1000.0]
    context = WorkingContext(tmp_path / "c.json")
    context.note_topic(DINNER)  # the room is on dinner, whatever was said last
    llm = GatedLLM()
    engine, announcer = make_engine(
        tmp_path,
        llm,
        thoughts=ThoughtBook(tmp_path / "t.json", now=lambda: clock[0]),
        context=context,
    )
    engine._live_transcript = [("you", BIKE)]
    _, is_error = await engine._execute_brain_tool("think", {"question": BIKE})
    engine._note_context("think", {"question": BIKE}, "")
    assert not is_error and "think:1" in str(context.snapshot().get("jobs"))
    clock[0] += 45 * 60
    llm.gate.set()
    await settle(engine)
    assert not announcer.pending()  # stale, and the room left the bike behind


async def test_a_thought_is_urgent_only_while_that_conversation_is_open(
    tmp_path: Path,
) -> None:
    llm = GatedLLM()
    engine, announcer = make_engine(tmp_path, llm)
    await engine._execute_brain_tool("think", {"question": BIKE})
    engine._live_conversation = 0  # he said "that's all" while she was thinking
    llm.gate.set()
    await settle(engine)
    (item,) = announcer.pending()
    assert item.kind == "thought" and item.priority == "normal"


async def test_list_thoughts_reads_back_ids_topics_and_statuses(tmp_path: Path) -> None:
    llm = GatedLLM()
    engine, _ = make_engine(tmp_path, llm)
    await engine._execute_brain_tool("think", {"question": BIKE, "topic": "buying a bike"})
    await engine._execute_brain_tool("think", {"question": REVISED, "topic": "buying a bike"})
    listing, is_error = await engine._execute_brain_tool("list_thoughts", {})
    assert not is_error
    assert "2: 'buying a bike' — running, asked just now" in listing
    assert "1: 'buying a bike' — superseded" in listing and "replaced by thought 2" in listing
    llm.gate.set()
    await settle(engine)


async def test_think_still_answers_when_the_question_is_missing_or_the_brain_is_off(
    tmp_path: Path,
) -> None:
    engine, _ = make_engine(tmp_path, GatedLLM())
    assert (await engine._execute_brain_tool("think", {"question": "  "}))[1]
    assert not engine._thoughts.recent()
    bare = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will")
    text, is_error = await bare._execute_brain_tool("think", {"question": BIKE})
    assert is_error and "not available" in text
    assert (await bare._execute_brain_tool("list_thoughts", {}))[0].startswith("you have not")
