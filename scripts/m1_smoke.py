"""Milestone 1 smoke test: prove Home Assistant sees and controls the lights.

No voice, no LLM — just the plumbing.

    uv run scripts/m1_smoke.py                       # health check + list lights
    uv run scripts/m1_smoke.py --entity light.desk --on --brightness 60 --rgb 255 120 0
    uv run scripts/m1_smoke.py --entity light.desk --kelvin 2700
    uv run scripts/m1_smoke.py --entity light.desk --off
    uv run scripts/m1_smoke.py --entity light.desk --demo   # short color cycle

Entity IDs come from the listing (or the HA UI) — nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from assistant.config import MissingSettingError, load_settings
from assistant.home import EntityState, HomeAssistantClient

console = Console()

DEMO_STEPS: list[tuple[str, dict]] = [
    ("red", {"rgb_color": (255, 0, 0), "brightness_pct": 80}),
    ("green", {"rgb_color": (0, 255, 0), "brightness_pct": 80}),
    ("blue", {"rgb_color": (0, 0, 255), "brightness_pct": 80}),
    ("warm white", {"color_temp_kelvin": 2700, "brightness_pct": 60}),
]


def print_lights(lights: list[EntityState]) -> None:
    table = Table(title="Lights known to Home Assistant")
    table.add_column("entity_id", style="cyan")
    table.add_column("name")
    table.add_column("state")
    table.add_column("color modes", style="dim")
    for light in lights:
        modes = light.attributes.get("supported_color_modes") or []
        table.add_row(light.entity_id, light.friendly_name, light.state, ", ".join(modes))
    console.print(table)


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.require("ha_url", "ha_token")

    async with HomeAssistantClient(settings.ha_url, settings.ha_token) as ha:
        if not await ha.api_alive():
            console.print(
                f"[red]Cannot reach Home Assistant at {settings.ha_url}.[/red] "
                "Is the container running? Try: docker compose up -d"
            )
            return 1
        console.print(f"[green]✓[/green] Home Assistant API is up at {settings.ha_url}")

        if args.entity is None:
            lights = await ha.lights()
            if lights:
                print_lights(lights)
                console.print(
                    "\nNext: uv run scripts/m1_smoke.py --entity <entity_id> --demo"
                )
            else:
                console.print(
                    "[yellow]No light entities yet.[/yellow] Add your bulbs in the HA UI "
                    "(Settings → Devices & services) — see README milestone 1."
                )
            return 0

        if args.demo:
            for label, kwargs in DEMO_STEPS:
                console.print(f"  → {args.entity}: {label}")
                await ha.light_on(args.entity, **kwargs)
                await asyncio.sleep(1.5)
            console.print("[green]✓[/green] demo complete")
            return 0

        if args.off:
            await ha.light_off(args.entity)
            console.print(f"[green]✓[/green] {args.entity} off")
            return 0

        await ha.light_on(
            args.entity,
            brightness_pct=args.brightness,
            rgb_color=tuple(args.rgb) if args.rgb else None,
            color_temp_kelvin=args.kelvin,
        )
        console.print(f"[green]✓[/green] {args.entity} on")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", help="light entity_id, e.g. light.desk_lamp")
    parser.add_argument("--on", action="store_true", help="turn on (default when --entity given)")
    parser.add_argument("--off", action="store_true")
    parser.add_argument("--demo", action="store_true", help="short color cycle")
    parser.add_argument("--brightness", type=int, metavar="PCT", help="0-100")
    parser.add_argument("--rgb", type=int, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--kelvin", type=int, help="color temperature, e.g. 2700")
    args = parser.parse_args()
    if (args.off or args.demo or args.on) and not args.entity:
        parser.error("--on/--off/--demo need --entity")
    try:
        return asyncio.run(run(args))
    except MissingSettingError as err:
        console.print(f"[red]{err}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
