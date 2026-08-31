"""Golden-command evals against the real Anthropic API + the fake house.

Opt-in (costs real money — cents, not dollars):

    $env:RUN_EVALS="1"; uv run pytest tests/test_evals.py -v

Each eval builds a fresh agent over a fresh FakeHome and asserts on *behavior*
(which entities changed, how) rather than exact wording, so prompt/model
tweaks are measured instead of vibes-tested against live bulbs.
"""

from __future__ import annotations

import os

import pytest

from assistant.brain.agent import Agent
from assistant.config import load_settings
from assistant.home.fake import FakeHome
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.meter import Meter

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EVALS") != "1",
    reason="live-API evals are opt-in: set RUN_EVALS=1 (needs Anthropic credentials)",
)

LIVING_ROOM = {"light.living_room_lamp", "light.living_room_ceiling"}
ALL_LIGHTS = LIVING_ROOM | {"light.bedroom_lamp", "light.hallway", "light.kitchen_strip"}


async def run_command(text: str) -> tuple[FakeHome, object]:
    settings = load_settings()
    home = FakeHome()
    llm = AnthropicProvider(
        model=settings.llm_model,
        effort=settings.llm_effort,
        api_key=settings.anthropic_api_key or None,
    )
    agent = Agent(home, llm, Meter())
    reply = await agent.handle(text)
    return home, reply


@pytest.mark.asyncio
async def test_turn_off_everything():
    home, reply = await run_command("turn off all the lights")
    assert home.entities_touched() == ALL_LIGHTS
    assert all(cmd.turn == "off" for cmd in home.applied)
    assert reply.intent in ("close", "confirm_close")


@pytest.mark.asyncio
async def test_cozy_living_room_stays_in_the_living_room():
    home, _ = await run_command("make the living room cozy")
    assert home.entities_touched() == LIVING_ROOM
    assert all(cmd.turn == "on" for cmd in home.applied)


@pytest.mark.asyncio
async def test_kitchen_red_uses_rgb():
    home, _ = await run_command("turn the kitchen strip red")
    assert home.entities_touched() == {"light.kitchen_strip"}
    (cmd,) = [c for c in home.applied if c.rgb_color]
    r, g, b = cmd.rgb_color
    assert r > 180 and r > g and r > b


@pytest.mark.asyncio
async def test_state_question_does_not_change_lights():
    home, reply = await run_command("is the bedroom light on right now?")
    assert home.applied == []
    assert reply.speech


@pytest.mark.asyncio
async def test_party_mode_touches_multiple_areas():
    home, reply = await run_command("get the apartment ready for a party")
    touched = home.entities_touched()
    areas = {home.lights[e].area for e in touched}
    assert len(areas) >= 2, f"party only touched {areas}"
    assert reply.speech


@pytest.mark.asyncio
async def test_dim_hallway_respects_white_only_bulb():
    home, _ = await run_command("dim the hallway to 20 percent")
    by_entity = {cmd.entity_id: cmd for cmd in home.applied}
    assert set(by_entity) == {"light.hallway"}
    assert by_entity["light.hallway"].brightness_pct == 20
    assert by_entity["light.hallway"].rgb_color is None
