"""Agent-loop mechanics with a scripted stub LLM — runs free, no network."""

from __future__ import annotations

import json
from typing import Any

import pytest

from assistant.brain.agent import Agent
from assistant.brain.tools import ToolExecutor, _resolve_target
from assistant.home.fake import FakeHome
from assistant.llm.base import ToolCall, TurnResult, Usage
from assistant.meter import Meter


class StubLLM:
    """Returns pre-scripted TurnResults in order."""

    def __init__(self, results: list[TurnResult]) -> None:
        self._results = list(results)
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def turn(self, *, system, tools, messages, response_schema=None) -> TurnResult:
        self.seen_messages.append([dict(m) for m in messages])
        return self._results.pop(0)


def tool_turn(*calls: ToolCall) -> TurnResult:
    return TurnResult(
        stop_reason="tool_use",
        text="",
        tool_calls=calls,
        assistant_content=[{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls],
        usage=Usage(input_tokens=1000, output_tokens=50),
        latency_ms=400,
        model="claude-opus-5",
    )


def final_turn(speech: str, intent: str = "close") -> TurnResult:
    return TurnResult(
        stop_reason="end_turn",
        text=json.dumps({"speech": speech, "intent": intent}),
        tool_calls=(),
        assistant_content=[{"type": "text", "text": json.dumps({"speech": speech, "intent": intent})}],
        usage=Usage(input_tokens=1200, output_tokens=30),
        latency_ms=300,
        model="claude-opus-5",
    )


@pytest.mark.asyncio
async def test_tool_loop_applies_changes_and_reports_intent():
    home = FakeHome()
    llm = StubLLM(
        [
            tool_turn(
                ToolCall(
                    id="t1",
                    name="set_lights",
                    input={"changes": [{"target": "Living Room", "turn": "off"}]},
                )
            ),
            final_turn("Living room is off.", "confirm_close"),
        ]
    )
    agent = Agent(home, llm, Meter())
    reply = await agent.handle("kill the living room lights")

    assert home.entities_touched() == {"light.living_room_lamp", "light.living_room_ceiling"}
    assert all(cmd.turn == "off" for cmd in home.applied)
    assert reply.speech == "Living room is off."
    assert reply.intent == "confirm_close"
    assert reply.hops == 2
    assert reply.cost_usd > 0

    # Tool results must ride back in ONE user message with matching ids
    followup = llm.seen_messages[1]
    results_msg = followup[-1]
    assert results_msg["role"] == "user"
    assert [b["tool_use_id"] for b in results_msg["content"]] == ["t1"]
    assert results_msg["content"][0]["is_error"] is False


@pytest.mark.asyncio
async def test_bad_target_surfaces_as_tool_error_not_crash():
    home = FakeHome()
    llm = StubLLM(
        [
            tool_turn(
                ToolCall(id="t1", name="set_lights", input={"changes": [{"target": "Garage", "turn": "on"}]})
            ),
            final_turn("There's no garage light.", "close"),
        ]
    )
    agent = Agent(home, llm, Meter())
    reply = await agent.handle("garage on")

    assert home.applied == []
    followup = llm.seen_messages[1][-1]
    assert followup["content"][0]["is_error"] is True
    assert "Garage" in followup["content"][0]["content"]
    assert reply.intent == "close"


@pytest.mark.asyncio
async def test_refusal_produces_safe_speech():
    home = FakeHome()
    refusal = TurnResult(
        stop_reason="refusal",
        text="",
        tool_calls=(),
        assistant_content=[],
        usage=Usage(),
        model="claude-opus-5",
    )
    agent = Agent(home, StubLLM([refusal]), Meter())
    reply = await agent.handle("do something sketchy")
    assert reply.intent == "close"
    assert reply.speech


def test_target_resolution_rules():
    lights = FakeHome().lights.values()
    lights = sorted(lights, key=lambda light: light.entity_id)
    assert {x.entity_id for x in _resolve_target("all", lights)} == {x.entity_id for x in lights}
    assert [x.entity_id for x in _resolve_target("light.hallway", lights)] == ["light.hallway"]
    assert {x.entity_id for x in _resolve_target("living room", lights)} == {
        "light.living_room_lamp",
        "light.living_room_ceiling",
    }
    assert _resolve_target("attic", lights) == []


@pytest.mark.asyncio
async def test_play_music_defaults_to_the_music_player():
    home = FakeHome()
    executor = ToolExecutor(home)
    text, is_error = await executor.execute(
        "play_music", {"media_id": "Chill Vibes", "media_type": "playlist"}
    )
    assert not is_error, text
    (call,) = home.played
    assert call["entity_id"] == "media_player.living_room_speakers"
    assert call["media_id"] == "Chill Vibes"


@pytest.mark.asyncio
async def test_play_music_wakes_a_sleeping_tv_first():
    from dataclasses import replace as dc_replace

    home = FakeHome()
    home.players = [
        home.players[0],
        dc_replace(home.players[1], state="off"),
    ]
    executor = ToolExecutor(home)
    text, is_error = await executor.execute(
        "play_music", {"media_id": "Focus Beats", "media_type": "playlist"}
    )
    assert not is_error, text
    assert ("media_player.living_room_tv", "turn_on") in home.media_commands
    assert "Woke the TV" in text
    assert home.played  # and the music still started


@pytest.mark.asyncio
async def test_browse_music_lists_playlists_and_filters():
    executor = ToolExecutor(FakeHome())
    text, is_error = await executor.execute("browse_music", {})
    assert not is_error
    names = [i["name"] for i in json.loads(text)]
    assert "Cleveland 10K" in names and "Chill Vibes" in names

    text, is_error = await executor.execute("browse_music", {"search": "cleveland"})
    assert not is_error
    assert [i["name"] for i in json.loads(text)] == ["Cleveland 10K"]

    text, is_error = await executor.execute("browse_music", {"search": "zzz"})
    assert not is_error and "nothing matching" in text


@pytest.mark.asyncio
async def test_launch_app_targets_the_tv():
    home = FakeHome()
    executor = ToolExecutor(home)
    text, is_error = await executor.execute("launch_app", {"app": "YouTube"})
    assert not is_error, text
    assert home.launched == [("media_player.living_room_tv", "YouTube")]


@pytest.mark.asyncio
async def test_playback_verbs_route_to_whats_actually_playing():
    """The live bug: the model says player='Apple TV' (the TV's friendly name)
    while music streams on another entity — pause must hit the stream."""
    from dataclasses import replace as dc_replace

    home = FakeHome()
    home.players = [dc_replace(home.players[0], state="playing"), home.players[1]]
    executor = ToolExecutor(home)
    text, is_error = await executor.execute(
        "media_control", {"action": "pause", "player": "Apple TV"}
    )
    assert not is_error, text
    assert home.media_commands == [("media_player.living_room_speakers", "pause")]

    # volume while music plays → the stream too
    text, is_error = await executor.execute(
        "media_control", {"action": "volume_set", "volume_pct": 50, "player": "Apple TV"}
    )
    assert not is_error
    assert home.media_commands[-1][0] == "media_player.living_room_speakers"

    # power verbs always mean the TV, whatever else is happening
    text, is_error = await executor.execute("media_control", {"action": "turn_off"})
    assert not is_error
    assert home.media_commands[-1] == ("media_player.living_room_tv", "turn_off")


@pytest.mark.asyncio
async def test_pause_with_nothing_playing_is_honest_not_an_error():
    executor = ToolExecutor(FakeHome())  # both players idle
    text, is_error = await executor.execute("media_control", {"action": "pause"})
    assert not is_error
    assert "nothing is playing" in text


@pytest.mark.asyncio
async def test_failed_play_suggests_real_library_names():
    class BrokenPlayHome(FakeHome):
        async def play_music(self, *a, **kw):
            raise RuntimeError("HA API error 500 on /api/services/music_assistant/play_media")

    executor = ToolExecutor(BrokenPlayHome())
    text, is_error = await executor.execute(
        "play_music", {"media_id": "cleveland running mix", "media_type": "playlist"}
    )
    assert is_error
    assert "Cleveland 10K" in text and "retry play_music" in text


@pytest.mark.asyncio
async def test_show_me_opens_urls_and_rejects_non_http(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda target: opened.append(target))
    executor = ToolExecutor(FakeHome())

    _text, is_error = await executor.execute(
        "show_me", {"url": "https://claude.ai/code/session_x"}
    )
    assert not is_error and opened == ["https://claude.ai/code/session_x"]

    _text, is_error = await executor.execute("show_me", {"url": "file:///C:/windows"})
    assert is_error  # only http(s)

    _text, is_error = await executor.execute(
        "show_me", {"text": "milk\neggs", "title": "Groceries"}
    )
    assert not is_error and len(opened) == 2 and opened[1].startswith("file://")


@pytest.mark.asyncio
async def test_self_awareness_tools_read_the_real_repo():
    executor = ToolExecutor(FakeHome())
    status, is_error = await executor.execute("project_status", {})
    assert not is_error
    data = json.loads(status)
    assert data["branch"]  # a real branch name from this very repo
    assert data["recent_commits"]
    roadmap, is_error = await executor.execute("read_roadmap", {})
    assert not is_error
    assert "Feature backlog" in roadmap


@pytest.mark.asyncio
async def test_escape_hatch_calls_any_service():
    home = FakeHome()
    executor = ToolExecutor(home)
    text, is_error = await executor.execute(
        "ha_call_service",
        {"domain": "climate", "service": "set_temperature",
         "data": {"entity_id": "climate.bedroom", "temperature": 70}},
    )
    assert not is_error, text
    assert home.generic_calls == [
        ("climate", "set_temperature", {"entity_id": "climate.bedroom", "temperature": 70})
    ]


@pytest.mark.asyncio
async def test_escape_hatch_denies_infrastructure():
    home = FakeHome()
    executor = ToolExecutor(home)
    for domain, service in (
        ("hassio", "addon_stop"),
        ("homeassistant", "restart"),
        ("shell_command", "anything"),
        ("light", "reload_all"),
    ):
        _text, is_error = await executor.execute(
            "ha_call_service", {"domain": domain, "service": service, "data": {}}
        )
        assert is_error, f"{domain}.{service} should be denied"
    assert home.generic_calls == []


@pytest.mark.asyncio
async def test_search_finds_non_light_devices():
    home = FakeHome()
    executor = ToolExecutor(home)
    text, is_error = await executor.execute("search_entities", {"query": "thermostat"})
    assert not is_error
    assert "climate.bedroom" in text
    # a miss falls back to the full inventory so the model can match by meaning
    text, is_error = await executor.execute("search_entities", {"query": "zzz_nonsense"})
    assert not is_error
    assert "full home inventory" in text and "climate.bedroom" in text
    text, _ = await executor.execute("get_entity", {"entity_id": "climate.bedroom"})
    assert "current_temperature" in text


@pytest.mark.asyncio
async def test_capability_filtering_drops_rgb_on_white_bulbs():
    home = FakeHome()
    executor = ToolExecutor(home)
    text, is_error = await executor.execute(
        "set_lights",
        {"changes": [{"target": "all", "turn": "on", "rgb_color": [255, 0, 0], "color_temp_kelvin": 2700}]},
    )
    assert not is_error, text
    by_entity = {cmd.entity_id: cmd for cmd in home.applied}
    assert by_entity["light.hallway"].rgb_color is None  # color_temp-only bulb
    assert by_entity["light.hallway"].color_temp_kelvin == 2700
    assert by_entity["light.kitchen_strip"].rgb_color == (255, 0, 0)  # rgb-only bulb
    assert by_entity["light.kitchen_strip"].color_temp_kelvin is None
