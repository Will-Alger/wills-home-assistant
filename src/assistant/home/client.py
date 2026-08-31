"""Async client for the Home Assistant REST API.

Wraps only what the assistant needs — this is not a general HA library.
Request shapes verified against https://developers.home-assistant.io/docs/api/rest/
and the light service schema (2026-08): auth is `Authorization: Bearer <token>`,
and service data (entity_id + fields like brightness_pct) goes flat in one JSON body.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, Self

import httpx

from assistant.home.base import Light, LightCommand, MediaPlayer


class HomeAssistantError(RuntimeError):
    pass


# Renders "entity_id|area name" lines for every entity assigned to an area
# (directly or via its device). Template functions verified against
# https://www.home-assistant.io/template-functions/ (areas, area_entities,
# area_name). POST /api/template returns plain text.
_AREA_MAP_TEMPLATE = (
    "{% for a in areas() %}{% for e in area_entities(a) %}"
    "{{ e }}|{{ area_name(a) }}\n"
    "{% endfor %}{% endfor %}"
)


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    state: str
    attributes: dict[str, Any]

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]

    @property
    def friendly_name(self) -> str:
        return self.attributes.get("friendly_name", self.entity_id)


class HomeAssistantClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def api_alive(self) -> bool:
        """False only when HA is unreachable; a bad token raises its own clear error."""
        # HA requires the trailing slash: /api/ not /api
        try:
            resp = await self._http.get("/api/")
        except httpx.HTTPError:
            return False
        self._check(resp)
        return True

    async def states(self) -> list[EntityState]:
        resp = await self._http.get("/api/states")
        self._check(resp)
        return [
            EntityState(s["entity_id"], s["state"], s.get("attributes", {}))
            for s in resp.json()
        ]

    async def lights(self) -> list[EntityState]:
        return [s for s in await self.states() if s.domain == "light"]

    async def area_map(self) -> dict[str, str]:
        """entity_id -> area name, for every entity assigned to an area."""
        resp = await self._http.post("/api/template", json={"template": _AREA_MAP_TEMPLATE})
        self._check(resp)
        mapping: dict[str, str] = {}
        for line in resp.text.splitlines():
            entity_id, sep, area = line.strip().partition("|")
            if sep and entity_id:
                mapping[entity_id] = area
        return mapping

    async def get_lights(self) -> list[Light]:
        """HomeApi: registry (name/area/capabilities) merged with live state."""
        areas = await self.area_map()
        lights: list[Light] = []
        for state in await self.lights():
            attrs = state.attributes
            brightness = attrs.get("brightness")  # HA reports 0-255 when on
            lights.append(
                Light(
                    entity_id=state.entity_id,
                    name=state.friendly_name,
                    area=areas.get(state.entity_id),
                    color_modes=tuple(attrs.get("supported_color_modes") or ()),
                    on=None if state.state in ("unavailable", "unknown") else state.state == "on",
                    brightness_pct=round(brightness / 255 * 100) if brightness else None,
                )
            )
        return sorted(lights, key=lambda light: light.entity_id)

    async def apply(self, commands: list[LightCommand]) -> None:
        """HomeApi: execute per-entity light commands."""
        for cmd in commands:
            if cmd.turn == "off":
                await self.light_off(cmd.entity_id, transition=cmd.transition)
            else:
                await self.light_on(
                    cmd.entity_id,
                    brightness_pct=cmd.brightness_pct,
                    rgb_color=cmd.rgb_color,
                    color_temp_kelvin=cmd.color_temp_kelvin,
                    transition=cmd.transition,
                )

    async def search_entities(self, query: str) -> list[dict]:
        """HomeApi: word-OR fragment search; empty query returns everything."""
        words = query.strip().lower().split()
        results = []
        for s in await self.states():
            haystack = f"{s.entity_id} {s.friendly_name}".lower()
            if not words or any(w in haystack or w == s.domain for w in words):
                results.append(
                    {
                        "entity_id": s.entity_id,
                        "name": s.friendly_name,
                        "state": s.state,
                        "domain": s.domain,
                    }
                )
        return results[:60]

    async def get_entity(self, entity_id: str) -> dict:
        resp = await self._http.get(f"/api/states/{entity_id}")
        self._check(resp)
        payload = resp.json()
        return {
            "entity_id": payload["entity_id"],
            "state": payload["state"],
            "attributes": payload.get("attributes", {}),
            "last_changed": payload.get("last_changed"),
        }

    async def generic_call(self, domain: str, service: str, data: dict) -> None:
        await self.call_service(domain, service, data, timeout=20.0)

    async def media_players(self) -> list[MediaPlayer]:
        """HomeApi: media players. Heuristic: an entity with a `remote.` sibling
        of the same suffix is a TV (pyatv-style); others are music players."""
        states = await self.states()
        remote_suffixes = {
            s.entity_id.split(".", 1)[1] for s in states if s.domain == "remote"
        }
        players = []
        for s in states:
            if s.domain != "media_player":
                continue
            suffix = s.entity_id.split(".", 1)[1]
            is_tv = suffix in remote_suffixes
            attrs = s.attributes
            players.append(
                MediaPlayer(
                    entity_id=s.entity_id,
                    name=s.friendly_name,
                    state=s.state,
                    kind="tv" if is_tv else "music",
                    apps=tuple(attrs.get("source_list") or ()),
                    now_playing=attrs.get("media_title"),
                )
            )
        return sorted(players, key=lambda p: p.entity_id)

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
        data: dict[str, Any] = {
            "entity_id": entity_id,
            "media_id": media_id,
            "media_type": media_type,
        }
        if artist:
            data["artist"] = artist
        if album:
            data["album"] = album
        if enqueue:
            data["enqueue"] = enqueue
        if radio_mode:
            data["radio_mode"] = True
        try:
            # MA search + AirPlay spin-up can legitimately take ~20s.
            await self.call_service("music_assistant", "play_media", data, timeout=30.0)
        except httpx.ReadTimeout as err:
            # Do NOT claim success: a hang here usually means Music Assistant
            # is stuck (e.g. Spotify rate-limiting with a long backoff).
            raise HomeAssistantError(
                "the music system did not confirm playback within 30s — it may "
                "be temporarily rate-limited by Spotify or busy; worth trying "
                "again in a little while"
            ) from err

    _MEDIA_COMMANDS: typing.ClassVar[dict[str, str]] = {
        "pause": "media_pause",
        "resume": "media_play",
        "next": "media_next_track",
        "previous": "media_previous_track",
        "stop": "media_stop",
        "turn_on": "turn_on",
        "turn_off": "turn_off",
    }

    async def media_command(
        self, entity_id: str, command: str, volume_pct: int | None = None
    ) -> None:
        if command == "volume_set":
            await self.call_service(
                "media_player",
                "volume_set",
                {"entity_id": entity_id, "volume_level": max(0, min(100, volume_pct or 0)) / 100},
            )
            return
        service = self._MEDIA_COMMANDS.get(command)
        if service is None:
            raise HomeAssistantError(f"unknown media command: {command}")
        await self.call_service("media_player", service, {"entity_id": entity_id})

    async def launch_app(self, entity_id: str, app: str) -> None:
        await self.call_service(
            "media_player", "select_source", {"entity_id": entity_id, "source": app}
        )

    async def call_service(
        self, domain: str, service: str, data: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        resp = await self._http.post(
            f"/api/services/{domain}/{service}",
            json=data,
            timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
        )
        self._check(resp)
        return resp.json()

    async def light_on(
        self,
        entity_id: str,
        *,
        brightness_pct: int | None = None,
        rgb_color: tuple[int, int, int] | None = None,
        color_temp_kelvin: int | None = None,
        transition: float | None = None,
    ) -> Any:
        data: dict[str, Any] = {"entity_id": entity_id}
        if brightness_pct is not None:
            data["brightness_pct"] = brightness_pct
        if rgb_color is not None:
            data["rgb_color"] = list(rgb_color)
        if color_temp_kelvin is not None:
            data["color_temp_kelvin"] = color_temp_kelvin
        if transition is not None:
            data["transition"] = transition
        return await self.call_service("light", "turn_on", data)

    async def light_off(self, entity_id: str, *, transition: float | None = None) -> Any:
        data: dict[str, Any] = {"entity_id": entity_id}
        if transition is not None:
            data["transition"] = transition
        return await self.call_service("light", "turn_off", data)

    def _check(self, resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise HomeAssistantError(
                "Home Assistant rejected the token (401). Re-check HA_TOKEN in .env."
            )
        if resp.is_error:
            raise HomeAssistantError(
                f"HA API error {resp.status_code} on {resp.request.url.path}: {resp.text[:300]}"
            )
