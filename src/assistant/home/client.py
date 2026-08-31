"""Async client for the Home Assistant REST API.

Wraps only what the assistant needs — this is not a general HA library.
Request shapes verified against https://developers.home-assistant.io/docs/api/rest/
and the light service schema (2026-08): auth is `Authorization: Bearer <token>`,
and service data (entity_id + fields like brightness_pct) goes flat in one JSON body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import httpx


class HomeAssistantError(RuntimeError):
    pass


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
        # HA requires the trailing slash: /api/ not /api
        try:
            resp = await self._http.get("/api/")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def states(self) -> list[EntityState]:
        resp = await self._http.get("/api/states")
        self._check(resp)
        return [
            EntityState(s["entity_id"], s["state"], s.get("attributes", {}))
            for s in resp.json()
        ]

    async def lights(self) -> list[EntityState]:
        return [s for s in await self.states() if s.domain == "light"]

    async def call_service(self, domain: str, service: str, data: dict[str, Any]) -> Any:
        resp = await self._http.post(f"/api/services/{domain}/{service}", json=data)
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
