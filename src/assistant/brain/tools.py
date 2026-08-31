"""Tool schemas and the executor that maps tool calls onto HomeApi.

Tools are deliberately batched and area-aware (one set_lights call changes
the whole room) — per the latency design in docs/FEATURES.md, each tool hop
is a full LLM round trip, so per-bulb tools would multiply dead air.
"""

from __future__ import annotations

import json
from typing import Any

from assistant.home.base import HomeApi, Light, LightCommand

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "set_lights",
        "description": (
            "Change lights. Batch EVERY change for the request into ONE call. "
            "A target may be an entity_id, an area name (affects every light "
            "in that area), or 'all'. Only include fields you want to change; "
            "bulbs that lack a capability (e.g. rgb on a white-only bulb) "
            "silently skip that field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "entity_id, area name, or 'all'",
                            },
                            "turn": {"type": "string", "enum": ["on", "off"]},
                            "brightness_pct": {"type": "integer", "minimum": 1, "maximum": 100},
                            "rgb_color": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "color_temp_kelvin": {
                                "type": "integer",
                                "minimum": 1500,
                                "maximum": 6500,
                            },
                            "transition_seconds": {"type": "number", "minimum": 0},
                        },
                        "required": ["target", "turn"],
                    },
                }
            },
            "required": ["changes"],
        },
    },
    {
        "name": "get_lights",
        "description": (
            "Current live state of every light (on/off, brightness). Call this "
            "only when the answer depends on current state; the device list in "
            "your instructions already covers names, areas, and capabilities."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "play_music",
        "description": (
            "Play music via Music Assistant (Spotify library behind it). "
            "media_id is a plain NAME — playlist, artist, track, or album name; "
            "search is built in. Use radio_mode for open-ended vibes ('play "
            "something relaxing' → a fitting artist/track + radio_mode). Omit "
            "player to use the default music player."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "media_id": {"type": "string", "description": "name or URI of what to play"},
                "media_type": {
                    "type": "string",
                    "enum": ["playlist", "artist", "track", "album", "radio"],
                },
                "artist": {"type": "string", "description": "disambiguates track/album names"},
                "album": {"type": "string"},
                "enqueue": {"type": "string", "enum": ["play", "replace", "next", "add"]},
                "radio_mode": {"type": "boolean", "description": "auto-continue with similar music"},
                "player": {"type": "string", "description": "player name or entity_id"},
            },
            "required": ["media_id", "media_type"],
        },
    },
    {
        "name": "media_control",
        "description": "Control playback or a TV: pause/resume/skip/volume/power.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "pause",
                        "resume",
                        "next",
                        "previous",
                        "stop",
                        "volume_set",
                        "turn_on",
                        "turn_off",
                    ],
                },
                "volume_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "player": {"type": "string", "description": "player name or entity_id"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "launch_app",
        "description": (
            "Open an app on the TV (see the TV's app list in your instructions), "
            "e.g. YouTube, Netflix, Spotify."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app": {"type": "string"},
                "player": {"type": "string", "description": "TV name or entity_id"},
            },
            "required": ["app"],
        },
    },
]


class ToolExecutor:
    def __init__(self, home: HomeApi) -> None:
        self._home = home

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Returns (result_text, is_error)."""
        try:
            if name == "get_lights":
                return json.dumps([_light_state_view(x) for x in await self._home.get_lights()]), False
            if name == "set_lights":
                return await self._set_lights(tool_input), False
            if name == "play_music":
                player = await self._resolve_player(tool_input.get("player"), kind="music")
                await self._home.play_music(
                    player.entity_id,
                    str(tool_input["media_id"]),
                    str(tool_input.get("media_type", "playlist")),
                    artist=tool_input.get("artist"),
                    album=tool_input.get("album"),
                    enqueue=tool_input.get("enqueue"),
                    radio_mode=bool(tool_input.get("radio_mode", False)),
                )
                return f"Playing on {player.name}.", False
            if name == "media_control":
                player = await self._resolve_player(tool_input.get("player"), kind=None)
                await self._home.media_command(
                    player.entity_id,
                    str(tool_input["action"]),
                    volume_pct=tool_input.get("volume_pct"),
                )
                return f"Done ({tool_input['action']} on {player.name}).", False
            if name == "launch_app":
                player = await self._resolve_player(tool_input.get("player"), kind="tv")
                await self._home.launch_app(player.entity_id, str(tool_input["app"]))
                return f"Opened {tool_input['app']} on {player.name}.", False
            return f"Unknown tool: {name}", True
        except Exception as err:  # noqa: BLE001 — any tool failure must become an
            # is_error tool_result the model can react to, never a crashed loop
            return f"Tool failed: {err}", True

    async def _set_lights(self, tool_input: dict[str, Any]) -> str:
        lights = await self._home.get_lights()
        commands: list[LightCommand] = []
        for change in tool_input.get("changes", []):
            targets = _resolve_target(str(change.get("target", "")), lights)
            if not targets:
                valid = sorted({light.area for light in lights if light.area} | {light.entity_id for light in lights})
                raise ValueError(
                    f"target {change.get('target')!r} matched nothing. Valid targets: {valid}"
                )
            for light in targets:
                rgb = change.get("rgb_color")
                kelvin = change.get("color_temp_kelvin")
                commands.append(
                    LightCommand(
                        entity_id=light.entity_id,
                        turn=change.get("turn", "on"),
                        brightness_pct=change.get("brightness_pct"),
                        rgb_color=tuple(rgb) if rgb and light.supports_rgb else None,
                        color_temp_kelvin=kelvin if kelvin and light.supports_color_temp else None,
                        transition=change.get("transition_seconds"),
                    )
                )
        await self._home.apply(commands)
        return f"Done: {len(commands)} light(s) updated."

    async def _resolve_player(self, spec, kind: str | None):
        """Pick a media player by name/entity_id fragment; default by kind."""
        players = await self._home.media_players()
        if not players:
            raise ValueError("no media players are set up")
        if spec:
            needle = str(spec).strip().lower()
            for p in players:
                if needle in (p.entity_id.lower(), p.name.lower()):
                    return p
            matches = [p for p in players if needle in p.name.lower() or needle in p.entity_id]
            if len(matches) == 1:
                return matches[0]
            raise ValueError(
                f"player {spec!r} is ambiguous or unknown; players: "
                f"{[(p.entity_id, p.name, p.kind) for p in players]}"
            )
        preferred = [p for p in players if kind is None or p.kind == kind]
        if len(preferred) == 1 or (preferred and kind is not None):
            return preferred[0]
        raise ValueError(
            "specify which player; players: "
            f"{[(p.entity_id, p.name, p.kind) for p in players]}"
        )


def _resolve_target(target: str, lights: list[Light]) -> list[Light]:
    needle = target.strip().lower()
    if needle in ("all", "everywhere", "*"):
        return list(lights)
    by_entity = [light for light in lights if light.entity_id.lower() == needle]
    if by_entity:
        return by_entity
    return [light for light in lights if (light.area or "").lower() == needle]


def _light_state_view(light: Light) -> dict[str, Any]:
    return {
        "entity_id": light.entity_id,
        "area": light.area,
        "on": light.on,
        "brightness_pct": light.brightness_pct,
    }
