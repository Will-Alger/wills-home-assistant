"""Tool schemas and the executor that maps tool calls onto HomeApi.

Tools are deliberately batched and area-aware (one set_lights call changes
the whole room) — per the latency design in docs/FEATURES.md, each tool hop
is a full LLM round trip, so per-bulb tools would multiply dead air.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timedelta
from typing import Any

import httpx

from assistant.brain.outcome import NeedsClarification, ToolOutcome, ok, unavailable
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
from assistant.config import code_root, home_dir
from assistant.dispatch import NO_WINDOW
from assistant.home.base import RGB_CAPABLE_MODES, HomeApi, Light, LightCommand
from assistant.music import MusicClarification, MusicCoordinator, MusicSuperseded
from assistant.receipts import ActionReceipt, EntityOutcome, ReceiptBook

# Her own codebase: the checkout this code runs from (main or a staged
# worktree) for docs and git; the HOME dir for anything written to data/.
CODE_ROOT = code_root()

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "set_lights",
        "description": (
            "Change lights. Batch EVERY change for the request into ONE call. "
            "A target may be an entity_id, an area name (affects every light "
            "in that area), or 'all'. Only include fields you want to change; "
            "bulbs that lack a capability (e.g. rgb on a white-only bulb) "
            "silently skip that field. Give color_temp_kelvin for warm/cool "
            "white OR rgb_color for a colour, never both in one change (a bulb "
            "takes one colour parameter; given both, the kelvin wins)."
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
        "name": "undo_last",
        "description": (
            "Put the last thing you changed back the way it was — for \"undo "
            "that\", \"no, put it back\", \"never mind\". Lights only for now; "
            "it tells you plainly when the last action has no undo (a media "
            "command, a calendar delete), and that sentence is your answer. "
            "Not for corrections that keep part of the change ('keep the "
            "brightness but make it blue') — those are one new set_lights."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browse_music",
        "description": (
            "Find music. scope='library' (default) lists the owner's own "
            "playlists/artists/albums — use for 'what playlists do I have'. "
            "scope='catalog' searches the ENTIRE streaming catalog (Apple "
            "Music) — use when the owner asks for options without playback; "
            "it requires a search term. For find-AND-play use play_music with "
            "selection='discover' in one call instead. Results "
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
            "is an exact title or a known URI. For 'find and play a jazz playlist', "
            "call THIS tool once with media_id='jazz', media_type='playlist', "
            "selection='discover': it searches, picks a relevant result and plays. "
            "No preliminary browse or TV power call is needed. For a named song use "
            "selection='exact', with artist/album when given. Never invent a URI. "
            "Use fresh=true when asked for something new. Reserve radio_mode for "
            "an explicit request for similar-track radio. Omit player for the default."
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
                "selection": {"type": "string", "enum": ["exact", "discover"],
                              "description": "exact title (default), or choose from a catalog search"},
                "fresh": {"type": "boolean", "description": "bypass a previous discovery choice"},
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
        "name": "web_search",
        "description": (
            "Search the web and get a short, current answer with sources — "
            "store hours, news, scores, facts, anything outside the home. Use "
            "it instead of guessing or saying you can't browse. Takes a few "
            "seconds; say you're checking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the question, in full"}},
            "required": ["query"],
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
    {
        "name": "delete_calendar_event",
        "description": (
            "Remove one event from the owner's Apple calendar — this is "
            "PERMANENT. First list_calendar_events to get the event's uid, "
            "restate exactly which event aloud (title and time), and only "
            "after an explicit yes call this with confirmed=true. Never "
            "delete on a guess; if several events could match, ask."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {"type": "string", "description": "from list_calendar_events"},
                "confirmed": {
                    "type": "boolean",
                    "description": "true ONLY after the owner verbally approved deleting this exact event",
                },
                "calendar": {"type": "string", "description": "calendar name; omit for the default"},
            },
            "required": ["uid", "confirmed"],
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
    def __init__(
        self,
        home: HomeApi,
        calendar: CalendarApi | None = None,
        web: Any | None = None,
        routines: Any | None = None,
        receipts: ReceiptBook | None = None,
        music_destinations: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._web = web  # WebSearch, or None when no key is configured
        self._routines = routines  # RoutineStore: deterministic defaults/overrides
        # Every mutation leaves a receipt here; undo_last reads the last one.
        # Without a path it is still a working ledger, just not a durable one.
        self._receipts = receipts if receipts is not None else ReceiptBook()
        self.last_routines: list[str] = []  # descriptions applied on the last call
        # (entity_id, friendly name) the last call actually touched — the
        # working context binds "it" and "that" to these (context.py).
        self.last_entities: list[tuple[str, str]] = []
        self._home = home
        self.music = MusicCoordinator(home, destinations=music_destinations)
        self._calendar = calendar

    async def run(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        """Every tool answers in one shape — status, summary, details,
        reversible, follow_up (brain/outcome.py)."""
        self.last_routines = []
        self.last_entities = []
        if self._routines is not None and isinstance(tool_input, dict):
            try:
                tool_input, applied = self._routines.apply(name, tool_input)
                self.last_routines = [r.description for r in applied]
            except Exception:  # noqa: BLE001 — a routine must never break a command
                self.last_routines = []
        outcome = await self._dispatch(name, tool_input)
        acted = bool(self.last_entities) or name not in _NEEDS_A_TARGET
        if not outcome.is_error and acted and name in _ONE_WAY_TOOLS:
            # It worked and it has no inverse: the ledger records that too, so
            # "undo that" can name what she just did instead of guessing.
            self._receipts.record(
                ActionReceipt(
                    tool=name,
                    note=_one_way_note(name, tool_input, self.last_entities),
                    outcomes=[EntityOutcome(eid, label) for eid, label in self.last_entities],
                )
            )
        return outcome

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """The old (result_text, is_error) pair, for the callers still on it."""
        return (await self.run(name, tool_input)).as_pair()

    async def _dispatch(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        try:
            if name == "get_lights":
                return ok(json.dumps([_light_state_view(x) for x in await self._home.get_lights()]))
            if name == "set_lights":
                return await self._set_lights(tool_input)
            if name == "undo_last":
                return await self._undo_last()
            if name == "play_music":
                return await self._play_music(tool_input)
            if name == "browse_music":
                media_type = str(tool_input.get("media_type", "playlist"))
                search = tool_input.get("search")
                if tool_input.get("scope") == "catalog":
                    if not search:
                        return unavailable("catalog scope needs a search term")
                    items = await self._home.music_search(
                        str(search), media_type=media_type,
                        limit=int(tool_input.get("limit", 8)),
                    )
                    if not items:
                        # A search that found nothing ran fine; the empty
                        # answer is the answer, not a failure to retry.
                        return unavailable(
                            "the catalog found nothing for that — try different words",
                            is_error=False,
                        )
                    self.music.catalog.observe(items)
                    return ok(json.dumps(items)[:2500])
                items = await self._home.music_library(
                    media_type=media_type,
                    search=search,
                    limit=int(tool_input.get("limit", 50)),
                )
                if not items:
                    return unavailable(
                        "the owner's library has nothing matching that — "
                        "browse_music with scope='catalog' searches all of "
                        "Apple Music instead",
                        is_error=False,
                    )
                self.music.catalog.observe(items)
                return ok(json.dumps(items)[:2500])
            if name == "media_control":
                action = str(tool_input["action"])
                cancelled = self.music.cancel_pending() if action in ("stop", "pause") else False
                player = await self._media_target(tool_input.get("player"), action)
                if player is None:
                    if cancelled:
                        return ok("Cancelled the music that was still preparing.")
                    # A refusal she simply says — never a tool failure.
                    return unavailable(
                        "nothing is playing right now — say what to play instead",
                        is_error=False,
                    )
                await self._home.media_command(
                    player.entity_id,
                    action,
                    volume_pct=tool_input.get("volume_pct"),
                )
                self.last_entities = [(player.entity_id, player.name)]
                kept_awake = ""
                if action == "pause" and player.kind == "music":
                    # Apple TVs sleep the instant their AirPlay session pauses
                    # (verified in HA history: paused + TV off in the same
                    # second). Re-wake it so pause means "paused", not "dark".
                    with contextlib.suppress(Exception):  # cosmetic; never fail the pause
                        tvs = [
                            p for p in await self._home.media_players() if p.kind == "tv"
                        ]
                        if tvs:
                            await self._home.media_command(tvs[0].entity_id, "turn_on")
                            kept_awake = " Kept the TV awake."
                return ok(
                    f"Done ({action} on {player.name} [{player.entity_id}]).{kept_awake}",
                    details={"player": player.name, "action": action},
                )
            if name == "web_search":
                if self._web is None:
                    return unavailable("web search isn't configured (it needs OPENAI_API_KEY)")
                answer = await self._web.search(str(tool_input.get("query", "")))
                return ok(str(answer)[:2000])
            if name == "search_entities":
                found = await self._home.search_entities(str(tool_input["query"]))
                if not found:
                    # Literal search missed (words like "temperature" often do).
                    # Hand back the whole inventory — the model matches meaning.
                    inventory = await self._home.search_entities("")
                    return ok(
                        json.dumps(
                            {
                                "note": "no literal matches — full home inventory follows; "
                                "pick semantically",
                                "entities": inventory,
                            }
                        )
                    )
                return ok(json.dumps(found))
            if name == "get_entity":
                detail = await self._home.get_entity(str(tool_input["entity_id"]))
                return ok(json.dumps(detail)[:1500])
            if name == "ha_call_service":
                domain = str(tool_input["domain"]).lower()
                service = str(tool_input["service"]).lower()
                if (
                    domain in _DENIED_DOMAINS
                    or (domain, service) in _DENIED_SERVICES
                    or service.startswith("reload")
                ):
                    return unavailable(f"service {domain}.{service} is not allowed from voice")
                data = dict(tool_input.get("data") or {})
                await self._home.generic_call(domain, service, data)
                target = data.get("entity_id")
                ids = [target] if isinstance(target, str) else list(target or [])
                self.last_entities = [(str(i), str(i)) for i in ids if i]
                return ok(f"called {domain}.{service}")
            if name in _CALENDAR_TOOL_NAMES:
                return await self._calendar_tool(name, tool_input)
            if name == "show_me":
                return ok(self._show_me(tool_input))
            if name == "project_status":
                return ok(await self._project_status())
            if name == "read_roadmap":
                roadmap = (CODE_ROOT / "docs" / "FEATURES.md").read_text(encoding="utf-8")
                return ok(roadmap[:10_000])
            if name == "read_history":
                history = (CODE_ROOT / "docs" / "HISTORY.md").read_text(encoding="utf-8")
                return ok(history[:10_000])
            if name == "launch_app":
                player = await self._resolve_player(tool_input.get("player"), kind="tv")
                await self._wake_tv_if_off()
                await self._home.launch_app(player.entity_id, str(tool_input["app"]))
                self.last_entities = [(player.entity_id, player.name)]
                return ok(f"Opened {tool_input['app']} on {player.name}.")
            return unavailable(f"Unknown tool: {name}")
        except NeedsClarification as ask:
            # It cannot act until the owner answers: the message IS the
            # question, so it goes back word for word, unwrapped.
            return ToolOutcome("needs_clarification", str(ask), is_error=True)
        except Exception as err:  # noqa: BLE001 — any tool failure must become an
            # outcome the model can react to, never a crashed loop
            # (str(err) can be empty — httpx timeouts — so include the type)
            detail = f"Tool failed: {str(err) or type(err).__name__}"
            if isinstance(err, httpx.TransportError):
                # Nothing reached Home Assistant at all — a hub that is off,
                # unplugged or not answering, not one command it refused. The
                # engine says that out loud in her own recorded voice rather
                # than leaving dead air while the model reads "Tool failed".
                return unavailable(detail, details={"home_unreachable": True})
            return unavailable(detail)

    async def _play_music(self, args: dict[str, Any]) -> ToolOutcome:
        if not str(args.get("media_id") or "").strip():
            return unavailable("Name the music you want to play.")
        if args.get("selection", "exact") not in ("exact", "discover"):
            return unavailable("Music selection must be exact or discover.")
        request = self.music.begin()
        try:
            result = await self.music.play(args, request)
            player = result["player"]
            self.last_entities = [(player.entity_id, player.name)]
            prefix = "Woke the TV first. " if result["woke"] else ""
            outcome = ok(
                f"{prefix}Playback requested for {result['title']} on {player.name}.",
                details={"player": player.name, "media_id": result["media_id"]},
                follow_up="The service accepted the request; audible playback is not verified. Do not submit it again.",
            )
        except MusicClarification as err:
            outcome = ToolOutcome("needs_clarification", str(err), is_error=True)
        except MusicSuperseded:
            outcome = unavailable("The pending music request was cancelled or replaced; it was not played.", is_error=False)
        except Exception as err:  # noqa: BLE001 — never retry an uncertain mutation
            outcome = unavailable(
                f"Music request failed: {str(err) or type(err).__name__}",
                follow_up=("Playback may still start. Check the queue/player before another play call; do not retry blindly."
                           if request.submitted else "No playback was submitted. Explain the issue briefly."),
            )
        request.stamp("finished")
        outcome.details["music_trace"] = request.details()
        return outcome

    async def _set_lights(self, tool_input: dict[str, Any]) -> ToolOutcome:
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
                if rgb and not any(rgb):
                    rgb = None  # black is not a colour (a backend sent [0, 0, 0] with a warm kelvin)
                use_rgb = bool(rgb) and light.supports_rgb
                use_kelvin = bool(kelvin) and light.supports_color_temp
                if use_rgb and use_kelvin:
                    # Home Assistant takes ONE colour parameter per call and
                    # answers 400 to both (the whole living room "did not
                    # respond" that way). The temperature is the deliberate
                    # warm/cool instruction; the rgb was its guess at the same thing.
                    use_rgb = False
                commands.append(
                    LightCommand(
                        entity_id=light.entity_id,
                        turn=change.get("turn", "on"),
                        brightness_pct=change.get("brightness_pct"),
                        rgb_color=tuple(rgb) if use_rgb else None,
                        color_temp_kelvin=kelvin if use_kelvin else None,
                        transition=change.get("transition_seconds"),
                    )
                )
        by_id = {light.entity_id: light for light in lights}
        touched = list(dict.fromkeys(c.entity_id for c in commands))  # ordered, deduped
        before = await self._snapshot(touched)
        # One entity at a time: a bulb that never answers must not take the
        # rest of the room down with it, and each gets its own verdict.
        outcomes: dict[str, EntityOutcome] = {}
        for command in commands:
            entity_id = command.entity_id
            error = await self._try_apply(command)
            prior = outcomes.get(entity_id)
            outcomes[entity_id] = EntityOutcome(
                entity_id=entity_id,
                name=by_id[entity_id].name if entity_id in by_id else entity_id,
                before=prior.before if prior else before.get(entity_id, {}),
                requested=_requested(command),
                ok=not error and (prior is None or prior.ok),
                error=error or (prior.error if prior else ""),
            )
        receipt = self._receipts.record(
            ActionReceipt(
                tool="set_lights",
                note=_light_note(list(outcomes.values())),
                outcomes=list(outcomes.values()),
                reversible=True,  # a bulb's before-state fully describes it
            )
        )
        # "it" and "that" bind to what actually changed, not what was aimed at.
        self.last_entities = [(o.entity_id, o.name) for o in receipt.changed]
        summary = receipt.summary("light")
        details = {
            "changed": [o.name or o.entity_id for o in receipt.changed],
            "unresponsive": [o.name or o.entity_id for o in receipt.failed],
        }
        if receipt.failed and receipt.changed:
            # The status carries the half-worked news; the model must not
            # round it up, and the summary already names who went silent.
            return ToolOutcome(
                "partial",
                summary,
                details,
                reversible=True,
                follow_up=(
                    "name the light that did not respond and never say it is done; "
                    "undo_last would put the rest back"
                ),
            )
        if receipt.failed:
            # Nothing worked at all: the reason is usually the whole story
            # (a bad token, an unreachable hub), so hand it over rather than
            # leaving her to guess at five silent bulbs.
            return unavailable(
                f"{summary} ({receipt.failed[0].error[:200]})",
                details=details,
                is_error=False,  # she says it; there is nothing to retry blindly
            )
        return ok(summary, details=details, reversible=True)

    async def _snapshot(self, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        """What these lights look like right now — the before-state a receipt
        needs. Fetched together, because one round trip per bulb before every
        command is exactly the dead air the tool design avoids."""
        details = await asyncio.gather(
            *(self._home.get_entity(entity_id) for entity_id in entity_ids),
            return_exceptions=True,
        )
        found: dict[str, dict[str, Any]] = {}
        for entity_id, detail in zip(entity_ids, details, strict=True):
            if isinstance(detail, dict):
                state = _before_state(detail)
                if state:
                    found[entity_id] = state
        return found

    async def _try_apply(self, command: LightCommand) -> str:
        """Apply one command; the error string if the bulb did not take it."""
        try:
            await self._home.apply([command])
        except Exception as err:  # noqa: BLE001 — one dead bulb is an outcome, not a crash
            return str(err) or type(err).__name__
        return ""

    async def _undo_last(self) -> ToolOutcome:
        receipt, refusal = self._receipts.for_undo()
        if receipt is None:
            # "That can't be undone" is an answer she reads out, not a failure.
            return unavailable(refusal, is_error=False)
        lights = {light.entity_id: light for light in await self._home.get_lights()}
        outcomes: list[EntityOutcome] = []
        for entry in receipt.restorable():
            light = lights.get(entry.entity_id)
            if light is None:
                outcomes.append(
                    EntityOutcome(
                        entry.entity_id, entry.name, ok=False, error="it is no longer in the house"
                    )
                )
                continue
            error = await self._try_apply(_restore_command(light, entry.before))
            outcomes.append(
                EntityOutcome(
                    entity_id=light.entity_id,
                    name=light.name,
                    before=entry.requested,  # the state we are undoing, for the record
                    requested=entry.before,
                    ok=not error,
                    error=error,
                )
            )
        undo = self._receipts.record(
            ActionReceipt(tool="undo_last", note=_undo_note(outcomes), outcomes=outcomes)
        )
        self.last_entities = [(o.entity_id, o.name) for o in undo.changed]
        details = {
            "restored": [o.name or o.entity_id for o in undo.changed],
            "unresponsive": [o.name or o.entity_id for o in undo.failed],
        }
        if not undo.failed:
            return ok(
                f"Put {_names(undo.changed)} back the way {_they_were(undo.changed)}.",
                details=details,
            )
        if not undo.changed:
            return unavailable(
                f"Could not put it back — {_names(undo.failed)} did not respond.",
                details=details,
                is_error=False,
            )
        return ToolOutcome(
            "partial",
            f"Put {len(undo.changed)} of {len(outcomes)} back; "
            f"{_names(undo.failed)} did not respond.",
            details,
            follow_up="name what did not go back; never say it is all back",
        )

    async def _calendar_tool(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
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
            return ok(
                json.dumps(
                    {
                        "now": spoken_now(now),
                        "events": [_event_view(event) for event in events[:_MAX_EVENTS_REPORTED]],
                    }
                )
            )

        if name == "delete_calendar_event":
            if not tool_input.get("confirmed"):
                raise NeedsClarification(
                    "not deleted: restate exactly which event (title and time) to "
                    "the owner and get an explicit yes, then retry with confirmed=true"
                )
            uid = str(tool_input.get("uid") or "").strip()
            if not uid:
                raise ValueError("deleting needs the event's uid — list_calendar_events shows them")
            await self._calendar.delete_event(uid, calendar=target)
            return ok("deleted — it's off the calendar")

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
        return ok(
            f'Added "{created.summary}" to {created.calendar}: {spoken_when(created)}.',
            details={"uid": created.uid, "calendar": created.calendar},
        )

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
        page = home_dir() / "data" / "shown" / f"note-{int(time.time())}.html"
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
                cwd=CODE_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=NO_WINDOW,
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
            raise NeedsClarification(
                f"player {spec!r} is ambiguous or unknown; players: "
                f"{[(p.entity_id, p.name, p.kind) for p in players]}"
            )
        preferred = [p for p in players if kind is None or p.kind == kind]
        if len(preferred) == 1 or (preferred and kind is not None):
            return preferred[0]
        raise NeedsClarification(
            "specify which player; players: "
            f"{[(p.entity_id, p.name, p.kind) for p in players]}"
        )


def _requested(command: LightCommand) -> dict[str, Any]:
    """What a command asked of one bulb — the receipt's "after", as intended."""
    asked: dict[str, Any] = {"on": command.turn == "on"}
    if command.brightness_pct is not None:
        asked["brightness_pct"] = command.brightness_pct
    if command.rgb_color:
        asked["rgb_color"] = list(command.rgb_color)
    if command.color_temp_kelvin:
        asked["color_temp_kelvin"] = command.color_temp_kelvin
    return asked


def _before_state(detail: dict[str, Any]) -> dict[str, Any]:
    """A light's live state, reduced to what it takes to put it back. HA
    reports rgb_color and color_temp_kelvin side by side, but a bulb is in
    ONE colour mode at a time and asking for both at once is rejected — so
    color_mode decides which of the two is the truth worth keeping."""
    state = str(detail.get("state", ""))
    if state in ("", "unavailable", "unknown"):
        return {}  # unknown before-state = nothing honest to restore
    before: dict[str, Any] = {"on": state == "on"}
    if not before["on"]:
        return before
    attributes = detail.get("attributes") or {}
    brightness = attributes.get("brightness")  # HA reports 0-255 when on
    if brightness:
        before["brightness_pct"] = max(1, round(float(brightness) / 255 * 100))
    mode = str(attributes.get("color_mode") or "")
    rgb = attributes.get("rgb_color")
    kelvin = attributes.get("color_temp_kelvin")
    if kelvin and mode not in RGB_CAPABLE_MODES:
        before["color_temp_kelvin"] = int(kelvin)
    elif isinstance(rgb, (list, tuple)) and len(rgb) == 3:
        before["rgb_color"] = [int(channel) for channel in rgb]
    return before


def _restore_command(light: Light, before: dict[str, Any]) -> LightCommand:
    if not before.get("on"):
        return LightCommand(entity_id=light.entity_id, turn="off")
    rgb = before.get("rgb_color")
    kelvin = before.get("color_temp_kelvin")
    return LightCommand(
        entity_id=light.entity_id,
        turn="on",
        brightness_pct=before.get("brightness_pct"),
        rgb_color=tuple(rgb) if rgb and light.supports_rgb else None,
        color_temp_kelvin=kelvin if kelvin and light.supports_color_temp else None,
    )


def _names(outcomes: list[EntityOutcome]) -> str:
    labels = [o.name or o.entity_id for o in outcomes]
    if len(labels) > 3:
        return f"{len(labels)} lights"
    if len(labels) <= 1:
        return labels[0] if labels else "nothing"
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _they_were(outcomes: list[EntityOutcome]) -> str:
    return "it was" if len(outcomes) == 1 else "they were"


def _light_note(outcomes: list[EntityOutcome]) -> str:
    changed = [o for o in outcomes if o.ok]
    if not changed:
        return f"try to change {_names(outcomes)}"
    return f"changed {_names(changed)}"


def _undo_note(outcomes: list[EntityOutcome]) -> str:
    changed = [o for o in outcomes if o.ok]
    if not changed:
        return f"try to put {_names(outcomes)} back"
    return f"put {_names(changed)} back the way {_they_were(changed)}"


# Mutations with no reliable inverse. They still go in the ledger, so "undo
# that" can name what she actually just did instead of guessing.
_ONE_WAY_TOOLS = frozenset(
    {
        "media_control",
        "play_music",
        "launch_app",
        "ha_call_service",
        "create_calendar_event",
        "delete_calendar_event",
    }
)

# media_control can decline to act ("nothing is playing right now"); a ledger
# entry for something she did not do would make "undo that" misstate the last
# action, so no target means no receipt.
_NEEDS_A_TARGET = frozenset({"media_control"})

_MEDIA_PAST = {
    "pause": "paused",
    "resume": "resumed",
    "next": "skipped a track on",
    "previous": "went back a track on",
    "stop": "stopped",
    "volume_set": "set the volume on",
    "turn_on": "turned on",
    "turn_off": "turned off",
}


def _one_way_note(name: str, args: dict[str, Any], entities: list[tuple[str, str]]) -> str:
    """The action in past tense, so the refusal reads as one sentence:
    "the last thing I did was <note>, and that can't be undone"."""
    target = entities[0][1] if entities else ""
    where = f" on {target}" if target else ""
    if name == "media_control":
        verb = _MEDIA_PAST.get(str(args.get("action", "")), "sent a command to")
        return f"{verb} {target or 'the player'}"
    if name == "play_music":
        return f"started {args.get('media_id') or 'music'}{where}"
    if name == "launch_app":
        return f"opened {args.get('app') or 'an app'}{where}"
    if name == "ha_call_service":
        return f"called {args.get('domain')}.{args.get('service')}{where}"
    if name == "create_calendar_event":
        return f'added "{args.get("summary") or "an event"}" to the calendar'
    if name == "delete_calendar_event":
        return "deleted a calendar event"
    return f"used {name}"


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
        "uid": event.uid,  # the handle delete_calendar_event needs
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
