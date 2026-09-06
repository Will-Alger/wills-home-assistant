"""One contract for what a tool reports.

Every capability answers with the same five fields, and the model acts on
`status` instead of parsing prose. These tests hold two lines: every tool she
is ADVERTISED (realtime_tools) renders a valid outcome for a real call, and a
half-worked lighting change comes back `partial` — the voice test, where one
bulb is unplugged and she must not say "done".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from assistant.brain.outcome import STATUSES, ToolOutcome, adapt, ok, unavailable
from assistant.brain.tools import ToolExecutor
from assistant.calendar.base import local_tz
from assistant.calendar.fake import FakeCalendar
from assistant.engines.realtime_engine import RealtimeEngine, SessionStats, realtime_tools
from assistant.home.fake import FakeHome
from tests.fake_realtime import FakeClient

CONTRACT_KEYS = ["status", "summary", "details", "reversible", "follow_up"]

TOMORROW = (datetime.now(tz=local_tz()) + timedelta(days=1)).replace(microsecond=0).isoformat()

# One real call per advertised tool. A new tool with no entry here fails the
# roll-call below — the contract is not optional for the next capability.
SAMPLE_CALLS: dict[str, dict] = {
    "set_lights": {"changes": [{"target": "Hallway", "turn": "on", "brightness_pct": 40}]},
    "get_lights": {},
    "undo_last": {},
    "browse_music": {"media_type": "playlist"},
    "play_music": {"media_id": "Cleveland Rocks", "media_type": "playlist"},
    "media_control": {"action": "turn_on"},
    "web_search": {"query": "who won"},  # no key configured: unavailable, honestly
    "search_entities": {"query": "thermostat"},
    "get_entity": {"entity_id": "climate.bedroom"},
    "ha_call_service": {
        "domain": "switch",
        "service": "turn_on",
        "data": {"entity_id": "switch.desk_fan"},
    },
    "show_me": {"url": "https://example.com"},
    "project_status": {},
    "read_roadmap": {},
    "read_history": {},
    "launch_app": {"app": "YouTube"},
    "list_calendar_events": {},
    "create_calendar_event": {"summary": "Dinner", "start": TOMORROW},
    "delete_calendar_event": {"uid": "nope"},  # unconfirmed: a question, not a failure
    "restart_self": {},
}

ADVERTISED = [tool["name"] for tool in realtime_tools(calendar=True)]


def make_engine(home: FakeHome | None = None) -> tuple[RealtimeEngine, FakeClient]:
    engine = RealtimeEngine(
        api_key="k",
        model="m",
        voice="v",
        home=home if home is not None else FakeHome(),
        owner="Will",
        calendar=FakeCalendar(),
    )
    return engine, FakeClient()


async def render(engine: RealtimeEngine, connection, name: str, args: dict) -> dict:
    """Call one tool the way the session does, and decode what the model sees."""
    item = SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(args), call_id="c1"
    )
    event = SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))
    await engine._handle_response_done(connection, event, SessionStats())
    sent = [
        e
        for e in connection.sent
        if e["type"] == "conversation.item.create"
        and e["item"].get("type") == "function_call_output"
    ]
    return json.loads(sent[0]["item"]["output"]) if sent else {}


def test_every_advertised_tool_has_a_sample_call() -> None:
    # end_conversation is control flow, not a capability: it reports nothing.
    assert set(ADVERTISED) - {"end_conversation"} == set(SAMPLE_CALLS)


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
async def test_every_tool_reports_one_valid_outcome(name: str, monkeypatch) -> None:
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)  # show_me opens nothing here
    engine, client = make_engine()
    payload = await render(engine, client.connection, name, SAMPLE_CALLS[name])

    assert list(payload)[:5] == CONTRACT_KEYS, f"{name} rendered {list(payload)}"
    assert payload["status"] in STATUSES
    assert payload["summary"].strip(), f"{name} said nothing she could speak"
    assert isinstance(payload["details"], dict)
    assert isinstance(payload["reversible"], bool)
    assert isinstance(payload["follow_up"], str)


async def test_end_conversation_reports_nothing_and_closes() -> None:
    engine, client = make_engine()
    item = SimpleNamespace(type="function_call", name="end_conversation", arguments="{}", call_id="c1")
    event = SimpleNamespace(response=SimpleNamespace(output=[item], usage=None))
    assert await engine._handle_response_done(client.connection, event, SessionStats()) is True
    assert not [e for e in client.connection.sent if e["type"] == "conversation.item.create"]


# ── the voice test: one bulb unplugged ─────────────────────────────────────


async def test_a_half_worked_lighting_change_reports_partial() -> None:
    """"Turn on everything" with one bulb unplugged: status says partial, the
    summary names the one that stayed dark, and nothing says "Done"."""
    home = FakeHome()
    home.unresponsive = {"light.bedroom_lamp"}
    engine, client = make_engine(home)

    payload = await render(
        engine,
        client.connection,
        "set_lights",
        {"changes": [{"target": "all", "turn": "on"}]},
    )

    assert payload["status"] == "partial"
    assert "Bedroom Lamp" in payload["summary"]
    assert not payload["summary"].startswith("Done")
    assert payload["details"]["unresponsive"] == ["Bedroom Lamp"]
    assert len(payload["details"]["changed"]) == 4
    assert payload["reversible"] is True  # the rest can still be put back
    assert "never say it is done" in payload["follow_up"]


async def test_a_whole_room_that_never_answered_is_unavailable() -> None:
    home = FakeHome()
    home.unresponsive = {"light.living_room_lamp", "light.living_room_ceiling"}
    tools = ToolExecutor(home)

    outcome = await tools.run("set_lights", {"changes": [{"target": "Living Room", "turn": "on"}]})

    assert outcome.status == "unavailable" and not outcome.reversible
    assert outcome.summary.startswith("Nothing changed")
    assert not outcome.is_error  # a spoken answer, not a tool failure to retry


async def test_a_confirmation_gate_asks_rather_than_fails() -> None:
    tools = ToolExecutor(FakeHome(), FakeCalendar())

    outcome = await tools.run("delete_calendar_event", {"uid": "abc"})

    assert outcome.status == "needs_clarification"
    assert "explicit yes" in outcome.summary  # asked verbatim, not wrapped in "Tool failed"
    assert not outcome.summary.startswith("Tool failed")


async def test_a_light_change_that_worked_is_reversible_success() -> None:
    tools = ToolExecutor(FakeHome())
    outcome = await tools.run("set_lights", {"changes": [{"target": "Hallway", "turn": "off"}]})
    assert outcome.status == "success" and outcome.reversible
    assert outcome.details["changed"] == ["Hallway Light"] and not outcome.details["unresponsive"]


# ── the dataclass itself ───────────────────────────────────────────────────


def test_the_payload_is_the_five_keys_and_nothing_else() -> None:
    outcome = ToolOutcome("pending", "it's underway", {"id": 3}, follow_up="don't poll")
    assert outcome.payload() == {
        "status": "pending",
        "summary": "it's underway",
        "details": {"id": 3},
        "reversible": False,
        "follow_up": "don't poll",
    }
    assert "is_error" not in outcome.payload()  # the legacy flag is ours, not hers


def test_an_unknown_status_is_refused_at_the_door() -> None:
    with pytest.raises(ValueError, match="unknown tool status"):
        ToolOutcome("mostly fine", "hm")


def test_old_pairs_adapt_without_being_rewritten() -> None:
    assert adapt("Done.", False).status == "success"
    assert adapt("no key configured", True).status == "unavailable"
    # the engine may know better than the flag: background work is pending
    underway = adapt("task 4 started", False, status="pending")
    assert underway.status == "pending" and underway.summary == "task 4 started"
    assert ok("fine").as_pair() == ("fine", False)
    assert unavailable("no").as_pair() == ("no", True)
    assert unavailable("that can't be undone", is_error=False).as_pair()[1] is False
