"""In-memory fake house implementing HomeApi.

Used by the REPL's --fake mode (play with the brain before Home Assistant
exists) and by the eval suite (assert tool behavior without real bulbs).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from assistant.home.base import Light, LightCommand, MediaPlayer

_SEED = [
    Light("light.living_room_lamp", "Living Room Lamp", "Living Room", ("rgb", "color_temp"), on=False),
    Light("light.living_room_ceiling", "Living Room Ceiling", "Living Room", ("rgb", "color_temp"), on=True, brightness_pct=100),
    Light("light.bedroom_lamp", "Bedroom Lamp", "Bedroom", ("rgb", "color_temp"), on=False),
    Light("light.hallway", "Hallway Light", "Hallway", ("color_temp",), on=True, brightness_pct=80),
    Light("light.kitchen_strip", "Kitchen Strip", "Kitchen", ("rgb",), on=False),
]


# Colour is not on the Light dataclass (the brain only ever needed on/off and
# brightness there), but a receipt has to be able to put a bulb back exactly
# as it was — so the fake house tracks it the way Home Assistant reports it.
_COLOR_SEED: dict[str, dict[str, Any]] = {
    "light.living_room_ceiling": {"color_mode": "color_temp", "color_temp_kelvin": 2700},
    "light.hallway": {"color_mode": "color_temp", "color_temp_kelvin": 3000},
}


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
    colors: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _COLOR_SEED.items()}
    )
    # Bulbs that never answer — how a test fails the second of three lights.
    unresponsive: set[str] = field(default_factory=set)
    players: list[MediaPlayer] = field(default_factory=lambda: list(_MEDIA_SEED))
    played: list[dict] = field(default_factory=list)
    media_commands: list[tuple[str, str]] = field(default_factory=list)
    launched: list[tuple[str, str]] = field(default_factory=list)
    # The native Apple TV route: pages opened and keys pressed, and what the
    # fake Music app then plays — `native_tracks` maps a page url to its rows
    # (select alone plays the first; Down x n then select plays the nth).
    launched_urls: list[tuple[str, str]] = field(default_factory=list)
    remote_keys: list[tuple[str, list[str]]] = field(default_factory=list)
    native_tracks: dict[str, list[str]] = field(default_factory=dict)
    native_broken: bool = False  # keys land on nothing: the page never opened
    native_focus_offset: int = 0  # -1: the page opened with its focus one row off
    _page: str | None = None
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
            if cmd.entity_id in self.unresponsive:
                raise TimeoutError(f"{cmd.entity_id} did not respond")
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
                if cmd.rgb_color:
                    self.colors[cmd.entity_id] = {
                        "color_mode": "rgb",
                        "rgb_color": list(cmd.rgb_color),
                    }
                elif cmd.color_temp_kelvin:
                    self.colors[cmd.entity_id] = {
                        "color_mode": "color_temp",
                        "color_temp_kelvin": cmd.color_temp_kelvin,
                    }

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
        if command in ("turn_on", "turn_off"):
            self.players = [replace(p, state="idle" if command == "turn_on" else "off")
                            if p.entity_id == entity_id else p for p in self.players]
        if command in ("next", "previous") and self._page is not None:
            # the fake Music app steps along the rows of the page it opened
            rows = self.native_tracks.get(self._page) or []
            for p in self.players:
                if p.entity_id == entity_id and p.app_id == "com.apple.TVMusic" and p.now_playing in rows:
                    i = rows.index(p.now_playing) + (1 if command == "next" else -1)
                    if 0 <= i < len(rows):
                        self.players = [replace(q, now_playing=rows[i]) if q is p else q for q in self.players]

    async def launch_app(self, entity_id: str, app: str) -> None:
        self.launched.append((entity_id, app))

    async def launch_url(self, entity_id: str, url: str) -> None:
        self.launched_urls.append((entity_id, url))
        self._page = None if self.native_broken else url

    async def remote_commands(
        self, remote_entity_id: str, commands: list[str], delay_s: float = 0.1
    ) -> None:
        self.remote_keys.append((remote_entity_id, list(commands)))
        if self._page is None or "select" not in commands:
            return
        rows = self.native_tracks.get(self._page) or [self._page.rsplit("/", 1)[-1]]
        row = commands.count("down") + self.native_focus_offset
        title = rows[min(row, len(rows)) - 1] if row > 0 else rows[0]
        self.players = [
            replace(p, state="playing", now_playing=title, app_id="com.apple.TVMusic")
            if p.remote_entity == remote_entity_id else p
            for p in self.players
        ]

    async def music_library(
        self, media_type: str = "playlist", search: str | None = None, limit: int = 50
    ) -> list[dict]:
        library = {
            "playlist": [
                {"name": "Chill Vibes", "media_type": "playlist", "artists": None},
                {"name": "Workout Mix", "media_type": "playlist", "artists": None},
                {"name": "Cleveland 10K", "media_type": "playlist", "artists": None},
            ],
            "artist": [{"name": "Dave Brubeck", "media_type": "artist", "artists": None}],
        }
        items = library.get(media_type, [])
        if search:
            items = [i for i in items if search.lower() in i["name"].lower()]
        return items[:limit]

    async def music_search(
        self, query: str, media_type: str = "playlist", limit: int = 8
    ) -> list[dict]:
        catalog = {
            "playlist": [
                {"name": "Jazz Chill", "media_type": "playlist",
                 "uri": "apple_music://playlist/pl.jazzchill", "artists": None},
                {"name": "Smooth Jazz Essentials", "media_type": "playlist",
                 "uri": "apple_music://playlist/pl.smoothjazz", "artists": None},
            ],
        }
        items = [
            i for i in catalog.get(media_type, [])
            if query.lower() in i["name"].lower()
        ]
        return items[:limit]

    def _light_attributes(self, light: Light) -> dict[str, Any]:
        """Shaped like Home Assistant's: brightness 0-255, and the colour of
        whichever mode the bulb is actually in."""
        attributes: dict[str, Any] = {"supported_color_modes": list(light.color_modes)}
        if not light.on:
            return attributes
        if light.brightness_pct is not None:
            attributes["brightness"] = round(light.brightness_pct * 255 / 100)
        attributes.update(self.colors.get(light.entity_id, {}))
        return attributes

    def _all_entities(self) -> list[dict]:
        rows = [
            {"entity_id": light.entity_id, "name": light.name,
             "state": "on" if light.on else "off", "domain": "light",
             "attributes": self._light_attributes(light)}
            for light in self.lights.values()
        ]
        rows += [
            {"entity_id": p.entity_id, "name": p.name, "state": p.state,
             "domain": "media_player",
             "attributes": {k: v for k, v in (("app_id", p.app_id), ("media_title", p.now_playing)) if v}}
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
