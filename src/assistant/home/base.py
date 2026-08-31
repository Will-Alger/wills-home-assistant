"""Shared home-control types and the HomeApi protocol.

The brain talks to `HomeApi`; the real Home Assistant client and the fake
in-memory house both implement it, so the LLM layer can be developed and
eval-tested without touching real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

RGB_CAPABLE_MODES = frozenset({"rgb", "rgbw", "rgbww", "hs", "xy"})


@dataclass(frozen=True)
class Light:
    entity_id: str
    name: str
    area: str | None
    color_modes: tuple[str, ...]
    on: bool | None = None  # None = unavailable/unknown
    brightness_pct: int | None = None

    @property
    def supports_rgb(self) -> bool:
        return bool(RGB_CAPABLE_MODES & set(self.color_modes))

    @property
    def supports_color_temp(self) -> bool:
        return "color_temp" in self.color_modes


@dataclass(frozen=True)
class LightCommand:
    entity_id: str
    turn: str  # "on" | "off"
    brightness_pct: int | None = None
    rgb_color: tuple[int, int, int] | None = None
    color_temp_kelvin: int | None = None
    transition: float | None = None


def media_table(players: list[MediaPlayer]) -> str:
    rows = []
    for p in sorted(players, key=lambda x: x.entity_id):
        extra = f" | apps: {', '.join(p.apps)}" if p.apps else ""
        rows.append(f"- {p.entity_id} | {p.name} | {p.kind} | state: {p.state}{extra}")
    return "\n".join(rows) or "- (no media players set up)"


def device_table(lights: list[Light]) -> str:
    """Deterministic, prompt-ready device listing (sorted = cacheable prefix)."""
    rows = [
        f"- {light.entity_id} | {light.name} | area: {light.area or 'unassigned'}"
        f" | modes: {', '.join(light.color_modes) or 'on/off'}"
        for light in sorted(lights, key=lambda light: light.entity_id)
    ]
    return "\n".join(rows) or "- (no lights found yet)"


@dataclass(frozen=True)
class MediaPlayer:
    entity_id: str
    name: str
    state: str  # off / idle / playing / paused ...
    kind: str  # "music" (Music Assistant / speakers) or "tv" (has a remote, launches apps)
    apps: tuple[str, ...] = ()  # launchable sources for kind="tv"
    now_playing: str | None = None


class HomeApi(Protocol):
    async def get_lights(self) -> list[Light]:
        """Registry (name/area/capabilities) merged with live state."""
        ...

    async def apply(self, commands: list[LightCommand]) -> None:
        """Execute per-entity light commands."""
        ...

    async def media_players(self) -> list[MediaPlayer]: ...

    async def search_entities(self, query: str) -> list[dict]:
        """Search ALL entities (any domain) by id/name/area/domain fragment."""
        ...

    async def get_entity(self, entity_id: str) -> dict:
        """Full state + attributes of one entity."""
        ...

    async def generic_call(self, domain: str, service: str, data: dict) -> None:
        """Escape hatch: call any (allowed) Home Assistant service."""
        ...

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
        """Music Assistant play_media — media_id may be a plain name."""
        ...

    async def media_command(
        self, entity_id: str, command: str, volume_pct: int | None = None
    ) -> None: ...

    async def launch_app(self, entity_id: str, app: str) -> None: ...
