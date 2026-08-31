"""In-memory fake house implementing HomeApi.

Used by the REPL's --fake mode (play with the brain before Home Assistant
exists) and by the eval suite (assert tool behavior without real bulbs).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from assistant.home.base import Light, LightCommand, MediaPlayer

_SEED = [
    Light("light.living_room_lamp", "Living Room Lamp", "Living Room", ("rgb", "color_temp"), on=False),
    Light("light.living_room_ceiling", "Living Room Ceiling", "Living Room", ("rgb", "color_temp"), on=True, brightness_pct=100),
    Light("light.bedroom_lamp", "Bedroom Lamp", "Bedroom", ("rgb", "color_temp"), on=False),
    Light("light.hallway", "Hallway Light", "Hallway", ("color_temp",), on=True, brightness_pct=80),
    Light("light.kitchen_strip", "Kitchen Strip", "Kitchen", ("rgb",), on=False),
]


_MEDIA_SEED = [
    MediaPlayer("media_player.living_room_speakers", "Living Room Speakers", "idle", "music"),
    MediaPlayer(
        "media_player.living_room_tv",
        "Apple TV",
        "idle",
        "tv",
        apps=("Spotify", "YouTube", "Netflix", "Music"),
    ),
]


@dataclass
class FakeHome:
    lights: dict[str, Light] = field(
        default_factory=lambda: {light.entity_id: light for light in _SEED}
    )
    applied: list[LightCommand] = field(default_factory=list)
    players: list[MediaPlayer] = field(default_factory=lambda: list(_MEDIA_SEED))
    played: list[dict] = field(default_factory=list)
    media_commands: list[tuple[str, str]] = field(default_factory=list)
    launched: list[tuple[str, str]] = field(default_factory=list)
    extra_entities: list[dict] = field(
        default_factory=lambda: [
            {"entity_id": "climate.bedroom", "name": "Bedroom Thermostat", "state": "heat",
             "domain": "climate", "attributes": {"temperature": 68, "current_temperature": 66}},
            {"entity_id": "switch.desk_fan", "name": "Desk Fan", "state": "off",
             "domain": "switch", "attributes": {}},
        ]
    )
    generic_calls: list[tuple[str, str, dict]] = field(default_factory=list)

    async def get_lights(self) -> list[Light]:
        return sorted(self.lights.values(), key=lambda light: light.entity_id)

    async def apply(self, commands: list[LightCommand]) -> None:
        for cmd in commands:
            if cmd.entity_id not in self.lights:
                raise KeyError(f"unknown entity: {cmd.entity_id}")
            self.applied.append(cmd)
            current = self.lights[cmd.entity_id]
            if cmd.turn == "off":
                self.lights[cmd.entity_id] = replace(current, on=False)
            else:
                self.lights[cmd.entity_id] = replace(
                    current,
                    on=True,
                    brightness_pct=(
                        cmd.brightness_pct
                        if cmd.brightness_pct is not None
                        else current.brightness_pct or 100
                    ),
                )

    def entities_touched(self) -> set[str]:
        return {cmd.entity_id for cmd in self.applied}

    async def media_players(self) -> list[MediaPlayer]:
        return list(self.players)

    async def play_music(
        self,
        entity_id: str,
        media_id: str,
        media_type: str,
        *,
        artist: str | None = None,
        album: str | None = None,
        enqueue: str | None = None,
        radio_mode: bool = False,
    ) -> None:
        self.played.append(
            {
                "entity_id": entity_id,
                "media_id": media_id,
                "media_type": media_type,
                "artist": artist,
                "album": album,
                "enqueue": enqueue,
                "radio_mode": radio_mode,
            }
        )

    async def media_command(
        self, entity_id: str, command: str, volume_pct: int | None = None
    ) -> None:
        self.media_commands.append((entity_id, command))

    async def launch_app(self, entity_id: str, app: str) -> None:
        self.launched.append((entity_id, app))

    def _all_entities(self) -> list[dict]:
        rows = [
            {"entity_id": light.entity_id, "name": light.name,
             "state": "on" if light.on else "off", "domain": "light", "attributes": {}}
            for light in self.lights.values()
        ]
        rows += [
            {"entity_id": p.entity_id, "name": p.name, "state": p.state,
             "domain": "media_player", "attributes": {}}
            for p in self.players
        ]
        rows += self.extra_entities
        return rows

    async def search_entities(self, query: str) -> list[dict]:
        words = query.strip().lower().split()
        out = []
        for row in self._all_entities():
            hay = f"{row['entity_id']} {row['name']}".lower()
            if not words or any(w in hay or w == row["domain"] for w in words):
                out.append({k: row[k] for k in ("entity_id", "name", "state", "domain")})
        return out[:60]

    async def get_entity(self, entity_id: str) -> dict:
        for row in self._all_entities():
            if row["entity_id"] == entity_id:
                return row
        raise KeyError(f"unknown entity: {entity_id}")

    async def generic_call(self, domain: str, service: str, data: dict) -> None:
        self.generic_calls.append((domain, service, data))
