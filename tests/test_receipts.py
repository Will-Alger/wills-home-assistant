"""Receipts and undo: "Done" only when it is, and a way back.

The house here is FakeHome, whose `unresponsive` set is a bulb that never
answers — that is how the middle of three lights is made to fail.
"""

from __future__ import annotations

import json
from dataclasses import replace

from assistant.brain.tools import TOOL_DEFINITIONS, ToolExecutor
from assistant.home.fake import FakeHome
from assistant.receipts import ActionReceipt, EntityOutcome, ReceiptBook

LIVING_ROOM_RED = {"changes": [{"target": "Living Room", "turn": "on", "rgb_color": [255, 0, 0]}]}


class Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, minutes: float) -> None:
        self.now += minutes * 60


def make(home: FakeHome | None = None, **kw) -> tuple[ToolExecutor, FakeHome, ReceiptBook]:
    house = home if home is not None else FakeHome()
    book = ReceiptBook(**kw)
    return ToolExecutor(house, receipts=book), house, book


# ── partial success is said as partial success ─────────────────────────────


async def test_a_bulb_that_never_answers_is_named_and_the_rest_still_change() -> None:
    home = FakeHome()
    home.unresponsive = {"light.bedroom_lamp"}
    tools, home, book = make(home)

    text, is_error = await tools.execute(
        "set_lights",
        {
            "changes": [
                {"target": "light.living_room_lamp", "turn": "on", "brightness_pct": 40},
                {"target": "light.bedroom_lamp", "turn": "on", "brightness_pct": 40},
                {"target": "light.kitchen_strip", "turn": "on", "brightness_pct": 40},
            ]
        },
    )

    assert not is_error  # two of them worked; this is not a failed call
    assert not text.startswith("Done")
    assert "2 of 3 lights changed" in text
    assert "Bedroom Lamp did not respond" in text
    # and the house really is in that half-changed state
    assert home.lights["light.living_room_lamp"].on
    assert home.lights["light.kitchen_strip"].on
    assert not home.lights["light.bedroom_lamp"].on

    receipt = book.last()
    assert receipt is not None and receipt.tool == "set_lights"
    assert receipt.targets == [
        "light.living_room_lamp",
        "light.bedroom_lamp",
        "light.kitchen_strip",
    ]
    assert [o.entity_id for o in receipt.failed] == ["light.bedroom_lamp"]
    assert receipt.failed[0].error  # the honest reason, kept for the ledger
    # every outcome carries where it was and what was asked of it
    assert receipt.changed[0].before == {"on": False}
    assert receipt.changed[0].requested == {"on": True, "brightness_pct": 40}


async def test_a_room_where_nothing_answered_never_claims_a_change() -> None:
    home = FakeHome()
    home.unresponsive = {"light.living_room_lamp", "light.living_room_ceiling"}
    tools, _home, _book = make(home)

    text, is_error = await tools.execute("set_lights", LIVING_ROOM_RED)
    assert not is_error
    assert text.startswith("Nothing changed")
    assert "Living Room Lamp" in text and "Living Room Ceiling" in text
    assert "did not respond" in text  # the reason, when the whole room is silent

    # nothing changed, so there is nothing to put back — and she says that
    text, _ = await tools.execute("undo_last", {})
    assert text == "Nothing actually changed there, so there is nothing to put back."


async def test_all_of_it_worked_still_says_done() -> None:
    tools, _home, _book = make()
    text, is_error = await tools.execute("set_lights", LIVING_ROOM_RED)
    assert not is_error and text == "Done: 2 light(s) updated."


# ── undo ───────────────────────────────────────────────────────────────────


async def test_undo_puts_the_lights_back_exactly_as_they_were() -> None:
    """The voice test: turn the living room red, then undo that."""
    tools, home, _book = make()
    assert not home.lights["light.living_room_lamp"].on  # the lamp was off,
    assert home.lights["light.living_room_ceiling"].brightness_pct == 100  # the ceiling warm at 100

    await tools.execute("set_lights", LIVING_ROOM_RED)
    assert home.colors["light.living_room_ceiling"]["rgb_color"] == [255, 0, 0]

    text, is_error = await tools.execute("undo_last", {})
    assert not is_error
    assert "back the way they were" in text

    assert not home.lights["light.living_room_lamp"].on  # off again
    ceiling = home.lights["light.living_room_ceiling"]
    assert ceiling.on and ceiling.brightness_pct == 100
    assert home.colors["light.living_room_ceiling"] == {
        "color_mode": "color_temp",
        "color_temp_kelvin": 2700,
    }  # its warm white, not red


async def test_undo_restores_the_light_that_worked_when_another_failed() -> None:
    home = FakeHome()
    home.unresponsive = {"light.bedroom_lamp"}
    tools, home, _book = make(home)

    await tools.execute(
        "set_lights",
        {
            "changes": [
                {"target": "light.living_room_lamp", "turn": "on", "brightness_pct": 40},
                {"target": "light.bedroom_lamp", "turn": "on", "brightness_pct": 40},
                {"target": "light.hallway", "turn": "on", "brightness_pct": 40},
            ]
        },
    )
    text, is_error = await tools.execute("undo_last", {})

    assert not is_error
    assert not home.lights["light.living_room_lamp"].on  # back to off
    assert home.lights["light.hallway"].brightness_pct == 80  # back to 80
    # the bulb that never changed is not claimed as restored
    assert "Bedroom Lamp" not in text
    assert "Living Room Lamp" in text and "Hallway Light" in text


async def test_undo_after_a_media_command_is_refused_in_a_sentence() -> None:
    tools, home, _book = make()
    home.players = [
        replace(player, state="playing") if player.kind == "music" else player
        for player in home.players
    ]

    _text, is_error = await tools.execute("media_control", {"action": "pause"})
    assert not is_error

    text, is_error = await tools.execute("undo_last", {})
    assert not is_error  # a refusal is an answer, not a tool failure
    assert text == (
        "The last thing I did was paused Living Room Speakers, and that can't be "
        "undone. Undo covers lights for now."
    )
    assert home.lights["light.living_room_ceiling"].on  # it touched nothing


async def test_a_media_command_that_acted_on_nothing_leaves_no_receipt() -> None:
    """Both players idle: she declines to act, so the ledger must not claim
    she did — "undo that" would otherwise misname the last action."""
    tools, _home, book = make()
    text, is_error = await tools.execute("media_control", {"action": "pause"})
    assert not is_error and "nothing is playing" in text
    assert book.last() is None


async def test_undo_after_a_calendar_delete_is_refused_too() -> None:
    tools, _home, book = make()
    book.record(ActionReceipt(tool="delete_calendar_event", note="deleted a calendar event"))
    text, is_error = await tools.execute("undo_last", {})
    assert not is_error
    assert "deleted a calendar event, and that can't be undone" in text


async def test_undoing_the_undo_is_refused_rather_than_ping_ponging() -> None:
    tools, home, _book = make()
    await tools.execute("set_lights", LIVING_ROOM_RED)
    await tools.execute("undo_last", {})

    text, _ = await tools.execute("undo_last", {})
    assert text.startswith("The last thing I did was put ")
    assert "Living Room Lamp" in text and "Living Room Ceiling" in text
    assert "back the way they were, and that can't be undone" in text
    assert not home.lights["light.living_room_lamp"].on  # still where the undo left it


async def test_nothing_to_undo_says_so() -> None:
    tools, _home, _book = make()
    text, is_error = await tools.execute("undo_last", {})
    assert not is_error and text == "There is nothing recent to undo."


async def test_a_light_change_goes_cold_after_half_an_hour() -> None:
    """A before-state from an hour ago describes a room that has moved on."""
    clock = Clock()
    tools, _home, _book = make(now=clock)
    await tools.execute("set_lights", LIVING_ROOM_RED)

    clock.tick(31)
    text, is_error = await tools.execute("undo_last", {})
    assert not is_error
    assert "more than half an hour ago" in text


# ── the ledger itself ──────────────────────────────────────────────────────


def test_the_last_twenty_survive_a_restart(tmp_path) -> None:
    path = tmp_path / "receipts.json"
    book = ReceiptBook(path)
    for i in range(25):
        book.record(
            ActionReceipt(
                tool="set_lights",
                note=f"changed lamp {i}",
                outcomes=[EntityOutcome(f"light.lamp_{i}", f"Lamp {i}", before={"on": False})],
                reversible=True,
            )
        )
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 20
    assert len(book.recent(100)) == 25  # the session itself keeps more

    reopened = ReceiptBook(path)
    last = reopened.last()
    assert last is not None and last.note == "changed lamp 24"
    assert last.outcomes[0].before == {"on": False}
    assert reopened.for_undo()[0] is last  # and it is still undoable


def test_a_corrupt_ledger_is_ignored_rather_than_fatal(tmp_path) -> None:
    path = tmp_path / "receipts.json"
    path.write_text("{not json", encoding="utf-8")
    assert ReceiptBook(path).last() is None


def test_undo_last_is_offered_as_a_tool() -> None:
    tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "undo_last")
    assert tool["input_schema"]["properties"] == {}  # no arguments to get wrong
