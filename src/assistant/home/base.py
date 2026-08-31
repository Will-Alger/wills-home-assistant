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


class HomeApi(Protocol):
    async def get_lights(self) -> list[Light]:
        """Registry (name/area/capabilities) merged with live state."""
        ...

    async def apply(self, commands: list[LightCommand]) -> None:
        """Execute per-entity light commands."""
        ...
