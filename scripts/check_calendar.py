"""Milestone 6: prove the Apple Calendar link works, before asking by voice.

    uv run scripts/check_calendar.py                  # list calendars + the next week
    uv run scripts/check_calendar.py --days 14
    uv run scripts/check_calendar.py --write-test     # create a test event, then delete it

Needs ICLOUD_USERNAME and ICLOUD_APP_PASSWORD in .env — the app-specific
password from appleid.apple.com (Sign-In and Security -> App-Specific
Passwords), never the Apple ID password itself.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta

from rich.console import Console

from assistant.calendar.base import CalendarError, local_tz, spoken_now, spoken_when
from assistant.config import load_settings

console = Console()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="how far ahead to read")
    parser.add_argument(
        "--write-test",
        action="store_true",
        help="also create a test event an hour from now and delete it again",
    )
    args = parser.parse_args()

    settings = load_settings()
    settings.require("icloud_username", "icloud_app_password")
    from assistant.calendar.apple import AppleCalendar

    calendar = AppleCalendar(
        settings.icloud_username,
        settings.icloud_app_password,
        url=settings.icloud_caldav_url,
        default_calendar=settings.icloud_calendar_name,
    )

    try:
        names = await calendar.calendar_names()
        console.print(f"[bold]Calendars:[/bold] {', '.join(names)}")
        default = settings.icloud_calendar_name or names[0]
        console.print(f"[dim]reading {default} · now {spoken_now()}[/dim]")

        now = datetime.now(tz=local_tz())
        events = await calendar.list_events(now, now + timedelta(days=args.days))
        console.print(f"[bold]Next {args.days} days:[/bold] {len(events)} event(s)")
        for event in events:
            where = f" @ {event.location}" if event.location else ""
            console.print(f"  · {spoken_when(event)} — {event.summary}{where}")

        if args.write_test:
            start = now.replace(second=0, microsecond=0) + timedelta(hours=1)
            created = await calendar.create_event(
                summary="Alexa calendar test",
                start=start,
                end=start + timedelta(minutes=30),
                description="Written by scripts/check_calendar.py — safe to ignore.",
            )
            console.print(f"[green]created:[/green] {spoken_when(created)} — {created.summary}")
            await calendar.delete_event(created.uid)
            console.print("[green]deleted it again — read and write both work.[/green]")
    except CalendarError as err:
        console.print(f"[red]{err}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
