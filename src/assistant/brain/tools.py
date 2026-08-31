"""Tool schemas and the executor that maps tool calls onto HomeApi.

Tools are deliberately batched and area-aware (one set_lights call changes
the whole room) — per the latency design in docs/FEATURES.md, each tool hop
is a full LLM round trip, so per-bulb tools would multiply dead air.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from assistant.calendar.base import (
    CalendarApi,
    CalendarEvent,
    as_datetime,
    default_end,
    local_tz,
    parse_when,
    spoken_now,
    spoken_when,
)
from assistant.home.base import HomeApi, Light, LightCommand

REPO_ROOT = Path(__file__).resolve().parents[3]  # the assistant's own codebase

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
        "name": "browse_music",
        "description": (
            "Find music. scope='library' (default) lists the owner's own "
            "playlists/artists/albums — use for 'what playlists do I have'. "
            "scope='catalog' searches the ENTIRE streaming catalog (Apple "
            "Music) — use for discovery like 'find me a jazz playlist' or "
            "anything not in the library; it requires a search term. Results "
            "include a uri: pass it as play_music's media_id for an exact "
            "match, no name guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "media_type": {
                    "type": "string",
                    "enum": ["playlist", "artist", "album", "track", "radio"],
                },
                "search": {"type": "string", "description": "name/genre/mood to look for"},
                "scope": {"type": "string", "enum": ["library", "catalog"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        },
    },
    {
        "name": "play_music",
        "description": (
            "Play music via Music Assistant (Apple Music behind it). media_id "
            "is a name OR — better — a uri from browse_music results (exact, "
            "never mis-resolves). Use radio_mode for open-ended vibes ('play "
            "something relaxing' → a fitting artist/track + radio_mode). Omit "
            "player to use the default music player. Unsure what exists? "
            "browse_music first (library or catalog scope)."
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
        "name": "search_entities",
        "description": (
            "Search EVERYTHING in the home (any device type: climate, switches, "
            "sensors, scenes, covers, weather, people...) by name fragment or "
            "domain. The home has more than the lights/media listed in your "
            "instructions — use this to discover entities before ha_call_service, "
            "or to answer 'do I have / what is' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_entity",
        "description": "Full state and attributes of one entity (temperatures, sensor values...).",
        "input_schema": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "ha_call_service",
        "description": (
            "ESCAPE HATCH — use only when no dedicated tool covers the request. "
            "Calls any Home Assistant service (climate.set_temperature, "
            "switch.turn_on, scene.turn_on, cover.close_cover, ...). Workflow: "
            "search_entities first if unsure of the entity_id, then call with "
            "data including entity_id. Standard HA service vocabulary applies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "service": {"type": "string"},
                "data": {
                    "type": "object",
                    "description": "service data, usually including entity_id",
                },
            },
            "required": ["domain", "service", "data"],
        },
    },
    {
        "name": "project_status",
        "description": (
            "Inspect your OWN codebase's recent development: current branch, "
            "recent commits, uncommitted changes. Use when asked how your "
            "development is going or what you recently learned to do."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_roadmap",
        "description": (
            "Read your own feature backlog (docs/FEATURES.md) — every planned, "
            "in-progress, and shipped capability with status. Use to discuss "
            "your roadmap, what you can't do yet, or what's coming."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_history",
        "description": (
            "Read the story of your own creation (docs/HISTORY.md) — how and "
            "why you were built, the key decisions and moments. Use when asked "
            "about your origins or the reasoning behind your design."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_me",
        "description": (
            "Display something on the owner's desktop screen: open an https "
            "URL in the browser (a cloud work session, a pull request, a "
            "dashboard), or render a short text note as a page. Use when asked "
            "to 'show me', 'pull it up', 'open it on my screen'. Offer it "
            "proactively when you're holding a URL worth seeing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "https URL to open"},
                "text": {"type": "string", "description": "short note to display (if no url)"},
                "title": {"type": "string"},
            },
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

# Only offered when an iCloud account is configured (see RealtimeEngine) —
# an assistant that lists a calendar tool it can't reach invents answers.
CALENDAR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_calendar_events",
        "description": (
            "Read what's scheduled on the owner's Apple calendar. Defaults to "
            "the next week from now. ALWAYS call this before answering "
            "anything about the schedule ('what's on today', 'am I free "
            "Thursday', 'when is the dentist') — never answer from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": (
                        "ISO local date or datetime (2026-09-02 or "
                        "2026-09-02T15:00); defaults to now"
                    ),
                },
                "end": {
                    "type": "string",
                    "description": "ISO local date or datetime; defaults to a week after start",
                },
                "calendar": {"type": "string", "description": "calendar name; omit for the default"},
            },
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Add an event to the owner's Apple calendar. Work the exact date "
            "out yourself from the current time in your instructions, restate "
            "title, day and time aloud, and only call this after an explicit "
            "yes. A date-only start makes it all-day; otherwise it lasts an "
            "hour unless you give end or duration_minutes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "what the event is called"},
                "start": {
                    "type": "string",
                    "description": "ISO local datetime (2026-09-02T15:00), or a date for all-day",
                },
                "end": {"type": "string", "description": "ISO local datetime"},
                "duration_minutes": {"type": "integer", "minimum": 1},
                "location": {"type": "string"},
                "description": {"type": "string"},
                "calendar": {"type": "string", "description": "calendar name; omit for the default"},
            },
            "required": ["summary", "start"],
        },
    },
]
_CALENDAR_TOOL_NAMES = frozenset(tool["name"] for tool in CALENDAR_TOOLS)

_DEFAULT_EVENT_MINUTES = 60
_DEFAULT_WINDOW_DAYS = 7
_MAX_EVENTS_REPORTED = 40


# The escape hatch controls the HOME, never the infrastructure.
_DENIED_DOMAINS = frozenset(
    {"hassio", "backup", "update", "shell_command", "python_script", "recorder", "system_log"}
)
_DENIED_SERVICES = frozenset(
    {("homeassistant", "restart"), ("homeassistant", "stop"), ("homeassistant", "check_config")}
)


class ToolExecutor:
    def __init__(self, home: HomeApi, calendar: CalendarApi | None = None) -> None:
        self._home = home
        self._calendar = calendar

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Returns (result_text, is_error)."""
        try:
            if name == "get_lights":
                return json.dumps([_light_state_view(x) for x in await self._home.get_lights()]), False
            if name == "set_lights":
                return await self._set_lights(tool_input), False
            if name == "play_music":
                player = await self._resolve_player(tool_input.get("player"), kind="music")
                woke = await self._wake_tv_if_off()
                media_id = str(tool_input["media_id"])
                media_type = str(tool_input.get("media_type", "playlist"))
                try:
                    await self._home.play_music(
                        player.entity_id,
                        media_id,
                        media_type,
                        artist=tool_input.get("artist"),
                        album=tool_input.get("album"),
                        enqueue=tool_input.get("enqueue"),
                        radio_mode=bool(tool_input.get("radio_mode", False)),
                    )
                except Exception as err:  # noqa: BLE001 — enrich with real names
                    # A failed play is usually a name the library can't resolve.
                    # Hand back what actually exists so the retry can succeed.
                    detail = str(err) or type(err).__name__
                    return f"play failed: {detail}.{await self._media_hint(media_id, media_type)}", True
                note = "Woke the TV first. " if woke else ""
                return (
                    f"{note}Started on {player.name} (audio may take a few seconds to begin).",
                    False,
                )
            if name == "browse_music":
                media_type = str(tool_input.get("media_type", "playlist"))
                search = tool_input.get("search")
                if tool_input.get("scope") == "catalog":
                    if not search:
                        return "catalog scope needs a search term", True
                    items = await self._home.music_search(
                        str(search), media_type=media_type,
                        limit=int(tool_input.get("limit", 8)),
                    )
                    if not items:
                        return "the catalog found nothing for that — try different words", False
                    return json.dumps(items)[:2500], False
                items = await self._home.music_library(
                    media_type=media_type,
                    search=search,
                    limit=int(tool_input.get("limit", 50)),
                )
                if not items:
                    return (
                        "the owner's library has nothing matching that — "
                        "browse_music with scope='catalog' searches all of "
                        "Apple Music instead"
                    ), False
                return json.dumps(items)[:2500], False
            if name == "media_control":
                action = str(tool_input["action"])
                player = await self._media_target(tool_input.get("player"), action)
                if player is None:
                    return "nothing is playing right now — say what to play instead", False
                await self._home.media_command(
                    player.entity_id,
                    action,
                    volume_pct=tool_input.get("volume_pct"),
                )
                return f"Done ({action} on {player.name} [{player.entity_id}]).", False
            if name == "search_entities":
                found = await self._home.search_entities(str(tool_input["query"]))
                if not found:
                    # Literal search missed (words like "temperature" often do).
                    # Hand back the whole inventory — the model matches meaning.
                    inventory = await self._home.search_entities("")
                    return json.dumps(
                        {
                            "note": "no literal matches — full home inventory follows; "
                            "pick semantically",
                            "entities": inventory,
                        }
                    ), False
                return json.dumps(found), False
            if name == "get_entity":
                detail = await self._home.get_entity(str(tool_input["entity_id"]))
                return json.dumps(detail)[:1500], False
            if name == "ha_call_service":
                domain = str(tool_input["domain"]).lower()
                service = str(tool_input["service"]).lower()
                if (
                    domain in _DENIED_DOMAINS
                    or (domain, service) in _DENIED_SERVICES
                    or service.startswith("reload")
                ):
                    return f"service {domain}.{service} is not allowed from voice", True
                await self._home.generic_call(domain, service, dict(tool_input.get("data") or {}))
                return f"called {domain}.{service}", False
            if name in _CALENDAR_TOOL_NAMES:
                return await self._calendar_tool(name, tool_input), False
            if name == "show_me":
                return self._show_me(tool_input), False
            if name == "project_status":
                return await self._project_status(), False
            if name == "read_roadmap":
                roadmap = (REPO_ROOT / "docs" / "FEATURES.md").read_text(encoding="utf-8")
                return roadmap[:10_000], False
            if name == "read_history":
                history = (REPO_ROOT / "docs" / "HISTORY.md").read_text(encoding="utf-8")
                return history[:10_000], False
            if name == "launch_app":
                player = await self._resolve_player(tool_input.get("player"), kind="tv")
                await self._wake_tv_if_off()
                await self._home.launch_app(player.entity_id, str(tool_input["app"]))
                return f"Opened {tool_input['app']} on {player.name}.", False
            return f"Unknown tool: {name}", True
        except Exception as err:  # noqa: BLE001 — any tool failure must become an
            # is_error tool_result the model can react to, never a crashed loop
            # (str(err) can be empty — httpx timeouts — so include the type)
            return f"Tool failed: {str(err) or type(err).__name__}", True

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

    async def _calendar_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        if self._calendar is None:
            raise ValueError(
                "the calendar isn't connected — it needs an iCloud Apple ID and "
                "app-specific password in the settings"
            )
        target = str(tool_input.get("calendar") or "").strip() or None
        if name == "list_calendar_events":
            now = datetime.now(tz=local_tz())
            start_spec = str(tool_input.get("start") or "").strip()
            start = as_datetime(parse_when(start_spec)) if start_spec else now
            end_spec = str(tool_input.get("end") or "").strip()
            end = (
                as_datetime(parse_when(end_spec), end_of_day=True)
                if end_spec
                else start + timedelta(days=_DEFAULT_WINDOW_DAYS)
            )
            if end <= start:
                raise ValueError("that range ends before it starts")
            events = await self._calendar.list_events(start, end, calendar=target)
            return json.dumps(
                {
                    "now": spoken_now(now),
                    "events": [_event_view(event) for event in events[:_MAX_EVENTS_REPORTED]],
                }
            )

        summary = str(tool_input.get("summary") or "").strip()
        if not summary:
            raise ValueError("an event needs a title")
        start_when = parse_when(str(tool_input.get("start") or ""))
        end_spec = str(tool_input.get("end") or "").strip()
        minutes = tool_input.get("duration_minutes")
        if end_spec:
            end_when = parse_when(end_spec)
        elif minutes:
            end_when = default_end(start_when, int(minutes))
        else:
            end_when = default_end(start_when, _DEFAULT_EVENT_MINUTES)
        if end_when is not None and as_datetime(end_when) <= as_datetime(start_when):
            raise ValueError("that event would end before it starts")
        created = await self._calendar.create_event(
            summary=summary,
            start=start_when,
            end=end_when,
            calendar=target,
            location=str(tool_input.get("location") or "").strip() or None,
            description=str(tool_input.get("description") or "").strip() or None,
        )
        return f'Added "{created.summary}" to {created.calendar}: {spoken_when(created)}.'

    def _show_me(self, tool_input: dict[str, Any]) -> str:
        import html
        import time
        import webbrowser

        url = str(tool_input.get("url") or "").strip()
        if url:
            if not url.startswith(("http://", "https://")):
                raise ValueError("only http(s) URLs can be shown")
            webbrowser.open(url)
            return "opened on the desktop screen"
        text = str(tool_input.get("text") or "").strip()
        if not text:
            raise ValueError("give me a url or some text to show")
        title = html.escape(str(tool_input.get("title") or "From Alexa"))
        page = REPO_ROOT / "data" / "shown" / f"note-{int(time.time())}.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            "<meta charset='utf-8'><title>" + title + "</title>"
            "<body style='font:18px/1.6 system-ui;max-width:640px;margin:8vh auto;"
            "padding:0 24px'><h2>" + title + "</h2>"
            "<pre style='white-space:pre-wrap;font:inherit'>" + html.escape(text) + "</pre>",
            encoding="utf-8",
        )
        webbrowser.open(page.as_uri())
        return "showing it on the desktop screen"

    async def _project_status(self) -> str:
        async def git(*args: str) -> str:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=REPO_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            return out.decode(errors="replace").strip()

        return json.dumps(
            {
                "branch": await git("rev-parse", "--abbrev-ref", "HEAD"),
                "recent_commits": (
                    await git("log", "-10", "--pretty=format:%ad · %s", "--date=relative")
                ).splitlines(),
                "uncommitted_files": len(
                    (await git("status", "--porcelain")).splitlines()
                ),
            }
        )

    async def _wake_tv_if_off(self) -> bool:
        """Music rides through the Apple TV — wake it before playing/launching.
        Only auto-wakes when exactly one TV exists and it's clearly not on."""
        tvs_off = [
            p
            for p in await self._home.media_players()
            if p.kind == "tv" and p.state in ("off", "standby", "unavailable", "unknown")
        ]
        if len(tvs_off) != 1:
            return False
        await self._home.media_command(tvs_off[0].entity_id, "turn_on")
        await asyncio.sleep(3)  # give it a beat to wake before streaming at it
        return True

    _PLAYBACK_ACTIONS = frozenset({"pause", "resume", "next", "previous", "stop"})

    async def _media_target(self, spec, action: str):
        """State-aware routing: playback verbs act on what is actually playing
        (preferring the music stream); power verbs act on the TV. Entity names
        lie here — the friendly name 'Apple TV' belongs to the television while
        the music stream is a different entity — but live state doesn't lie.
        Returns None when a playback verb has nothing to act on."""
        players = await self._home.media_players()
        if not players:
            raise ValueError("no media players are set up")
        if action in ("turn_on", "turn_off"):
            tvs = [p for p in players if p.kind == "tv"]
            if tvs:
                return tvs[0]
            return await self._resolve_player(spec, kind=None)
        active = [p for p in players if p.state in ("playing", "paused", "buffering")]
        music_active = [p for p in active if p.kind == "music"]
        if music_active:
            return music_active[0]
        if active:
            return active[0]
        if action == "volume_set":
            tvs = [p for p in players if p.kind == "tv"]
            if tvs:
                return tvs[0]
        if spec:
            return await self._resolve_player(spec, kind=None)
        return None

    async def _media_hint(self, media_id: str, media_type: str) -> str:
        """Closest real names from the library, for retrying a failed play."""
        import difflib

        try:
            items = await self._home.music_library(media_type=media_type, limit=100)
        except Exception:  # noqa: BLE001 — a hint must never mask the real error
            return ""
        names = [i.get("name", "") for i in items if i.get("name")]
        if not names:
            return ""
        close = difflib.get_close_matches(media_id, names, n=5, cutoff=0.3) or names[:8]
        return (
            f" The library's actual {media_type}s include: {', '.join(close)} — "
            "retry play_music with one exact name."
        )

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


def _event_view(event: CalendarEvent) -> dict[str, Any]:
    """Speakable "when" first; ISO start so follow-up calls can be precise."""
    view: dict[str, Any] = {
        "summary": event.summary,
        "when": spoken_when(event),
        "start": event.start.isoformat(),
    }
    if event.all_day:
        view["all_day"] = True
    if event.location:
        view["location"] = event.location
    if event.calendar:
        view["calendar"] = event.calendar
    return view


def _light_state_view(light: Light) -> dict[str, Any]:
    return {
        "entity_id": light.entity_id,
        "area": light.area,
        "on": light.on,
        "brightness_pct": light.brightness_pct,
    }
