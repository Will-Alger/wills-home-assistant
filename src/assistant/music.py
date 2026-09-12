"""Resolve music and start it where he will hear it, as fast as the room allows.

Two routes. The native one: the Apple TV opens the Music app page for the
resolved track, album or playlist (a music.apple.com link over Companion), a
remote press starts it, and Home Assistant's state confirms the title. Apple
streams it itself, so a warm start is about two seconds from the request
(measured 2026-09-12: page ready in under a second, exact track playing
1.8 s after the link). The Music Assistant one: play_media on the MA player,
which resolves the provider stream and pushes AirPlay — 20 s and more with
the Apple Music provider that day, in HA history and MA's own log. Native
when the destination TV has a remote and the item has an Apple id; MA
otherwise, and whenever a native start cannot be confirmed from the TV.

No background playback, credentials or persisted audio URLs. The bounded
metadata cache is shared with library browsing and the transcriber's
background warmup.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from assistant.apple_catalog import MUSIC_APP, AppleCatalog, AppleItem, normalise, parse_uri
from assistant.home.base import HomeApi, MediaPlayer

__all__ = [
    "PLAY_WORD", "MusicCatalog", "MusicClarification", "MusicCoordinator", "MusicRequest",
    "MusicSuperseded", "MusicUnavailable", "PlayIntent", "catalog_for", "exact_match", "normalise",
    "parse_play_request",
]


class MusicClarification(ValueError):
    """No action until the speaker identifies the intended item or destination."""


class MusicUnavailable(ValueError):
    """Resolution or preparation failed before submitting playback."""


class MusicSuperseded(Exception):
    """A newer request replaced work that had not submitted playback."""


class MusicCatalog:
    def __init__(self, *, ttl_s: float = 21600, limit: int = 512,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s, self.limit, self.clock = ttl_s, limit, clock
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._choices: OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]] = OrderedDict()

    def observe(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            uri = str(item.get("uri") or "")
            # Cache provider/library identifiers only, never signed stream URLs.
            if not uri or uri.startswith(("http:", "https:")) or "://" not in uri:
                continue
            self._items[uri] = (self.clock(), dict(item))
            self._items.move_to_end(uri)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def exact(self, name: str, kind: str, artist: str = "", album: str = "") -> dict | None:
        candidates = [item for at, item in self._items.values() if self.clock() - at < self.ttl_s]
        return exact_match(candidates, name, kind, artist, album)

    def exact_uri(self, uri: str) -> dict | None:
        entry = self._items.get(str(uri or ""))
        return entry[1] if entry and self.clock() - entry[0] < self.ttl_s else None

    def choice(self, query: str, kind: str) -> dict | None:
        entry = self._choices.get((normalise(query), kind))
        return entry[1] if entry and self.clock() - entry[0] < self.ttl_s else None

    def chose(self, query: str, kind: str, item: dict) -> None:
        if str(item.get("uri", "")).startswith(("http:", "https:")):
            return
        key = (normalise(query), kind)
        self._choices[key] = (self.clock(), dict(item))
        self._choices.move_to_end(key)
        while len(self._choices) > self.limit:
            self._choices.popitem(last=False)


def catalog_for(home: Any) -> MusicCatalog:
    catalog = getattr(home, "music_catalog", None)
    if catalog is None:
        catalog = MusicCatalog()
        home.music_catalog = catalog
    return catalog


def exact_match(items: list[dict], name: str, kind: str,
                artist: str = "", album: str = "") -> dict | None:
    candidates = []
    for item in items:
        if not item.get("uri") or item.get("media_type") != kind:
            continue
        if normalise(str(item.get("name", ""))) != normalise(name):
            continue
        artists = [str(a.get("name", "")) if isinstance(a, dict) else str(a)
                   for a in item.get("artists") or []]
        if artist and normalise(artist) not in [normalise(a) for a in artists]:
            continue
        record = item.get("album") or ""
        if isinstance(record, dict):
            record = record.get("name", "")
        if album and normalise(album) != normalise(str(record)):
            continue
        candidates.append(item)
    # Duplicate provider/library references to the same recording are fine;
    # different artists or album versions require an actual choice.
    identities = {(normalise(str(i.get("artists") or "")), normalise(str(i.get("album") or "")))
                  for i in candidates}
    if len(identities) > 1:
        raise MusicClarification(f"Which version of {name} do you want? Please name the artist or album.")
    return candidates[0] if candidates else None


# ── what he said, before the model has said anything ─────────────────────────

@dataclass(frozen=True)
class PlayIntent:
    """An unmistakable spoken request: a title AND an artist. Anything vaguer
    ("play some jazz", "play it again") is the model's to interpret."""

    title: str
    artist: str
    destination: str = ""


_PLAY_RE = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:alexa[\s,!.]+)?"
    r"(?:(?:can|could|would|will)\s+you\s+|please\s+|go\s+ahead\s+and\s+|just\s+)*"
    r"(?:play|put\s+on|start)\s+(?P<title>.+?)"
    r"\s+by\s+(?P<artist>.+?)"
    r"(?:\s+(?:on|in|at|through|to)\s+(?:the\s+|my\s+)?(?P<dest>[\w\s]+?))?"
    r"[\s.?!,]*$",
    re.IGNORECASE,
)
_VAGUE_TITLES = ("some", "something", "a ", "an ", "the next", "next", "it", "that", "this",
                 "music", "my ", "me", "anything", "whatever", "songs", "stuff")
_QUOTES = " \"'“”‘’"
# The word that means a TV is about to be needed, heard mid-sentence.
PLAY_WORD = re.compile(r"\b(?:play|put on)\b", re.IGNORECASE)


def parse_play_request(text: str) -> PlayIntent | None:
    match = _PLAY_RE.match(str(text or "").strip())
    if match is None:
        return None
    title = match.group("title").strip(_QUOTES)
    artist = match.group("artist").strip(_QUOTES).rstrip(",")
    if not title or not artist or len(title) > 80 or len(artist) > 60:
        return None
    lowered = title.casefold()
    if lowered in _VAGUE_TITLES or lowered.startswith(tuple(v for v in _VAGUE_TITLES if v.endswith(" "))):
        return None
    return PlayIntent(title, artist, (match.group("dest") or "").strip())


def _request_key(args: dict[str, Any]) -> tuple[str, str] | None:
    """What makes two play requests the same request: an exact track title
    and artist. Only those can join a start that is already under way."""
    if str(args.get("media_type") or "playlist") != "track" or args.get("selection", "exact") != "exact":
        return None
    title = str(args.get("media_id") or "").strip()
    if not title or "://" in title:
        return None
    return normalise(title), normalise(str(args.get("artist") or ""))


@dataclass
class MusicRequest:
    id: int
    origin: str = "tool"  # tool | fast_start
    key: tuple[str, str] | None = None
    started: float = field(default_factory=time.monotonic)
    stages: dict[str, float] = field(default_factory=dict)
    submitted: bool = False
    native: bool = False
    verified: bool = False  # Home Assistant reported the requested title playing
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    runner: asyncio.Task | None = None  # the fast start's task; a same-key tool call joins it

    def stamp(self, name: str) -> None:
        self.stages[name] = round((time.monotonic() - self.started) * 1000, 1)

    def details(self) -> dict:
        return {"request_id": self.id, "origin": self.origin, "stages_ms": dict(self.stages),
                "submitted": self.submitted, "native": self.native,
                "playing_verified": self.verified, "audible_verified": False}


class MusicCoordinator:
    def __init__(self, home: HomeApi, *, destinations: dict[str, dict[str, str]] | None = None,
                 ready_timeout_s: float = 5, search_timeout_s: float = 8,
                 poll_s: float = 0.15, native: bool = True, storefront: str = "us",
                 native_ready_s: float = 0.9, native_verify_s: float = 4.0,
                 key_delay_s: float = 0.1, apple: AppleCatalog | None = None) -> None:
        self.home = home
        self.catalog = catalog_for(home)
        self.destinations = destinations or {}
        self.ready_timeout_s, self.search_timeout_s, self.poll_s = ready_timeout_s, search_timeout_s, poll_s
        self.native = native
        self.storefront = storefront or "us"
        self.native_ready_s, self.native_verify_s, self.key_delay_s = native_ready_s, native_verify_s, key_delay_s
        self.apple = apple or AppleCatalog(self.storefront)
        self._next = 0
        self.active: MusicRequest | None = None
        self._submission_lock = asyncio.Lock()
        self._prewoke_at = float("-inf")

    def cancel_pending(self) -> bool:
        if self.active is None or self.active.submitted:
            return False
        self.active.cancelled.set()
        return True

    def begin(self, args: dict[str, Any] | None = None, *, origin: str = "tool") -> MusicRequest:
        key = _request_key(args or {})
        active = self.active
        if (origin == "tool" and key is not None and active is not None and active.key == key
                and active.runner is not None and not active.cancelled.is_set()):
            active.stamp("joined")
            return active  # the fast start is already on it: join it, never start twice
        self.cancel_pending()
        self._next += 1
        request = MusicRequest(self._next, origin=origin, key=key)
        self.active = request
        return request

    # ── the destination ─────────────────────────────────────────────────────

    def _destination(self, players: list[MediaPlayer], spec: str) -> tuple[MediaPlayer, MediaPlayer | None]:
        music = [p for p in players if p.kind == "music"]
        tvs = [p for p in players if p.kind == "tv"]
        binding = next((v for k, v in self.destinations.items()
                        if normalise(k) == normalise(spec or "default")), None)
        if binding:
            player = next((p for p in music if p.entity_id == binding.get("player")), None)
            power_id = binding.get("power", "")
            power = next((p for p in tvs if p.entity_id == power_id), None)
            if player is None or (power_id and power is None):
                raise MusicUnavailable("The saved music destination is unavailable; check its player mapping.")
            return player, power
        if spec:
            matches = [p for p in players if spec.casefold() in (p.name.casefold(), p.entity_id.casefold())]
            if not matches:
                matches = [p for p in players if spec.casefold() in p.name.casefold()
                           or spec.casefold() in p.entity_id.casefold()]
            if len(matches) != 1:
                raise MusicClarification("Which music player do you mean? " + ", ".join(p.name for p in music))
            selected = matches[0]
            if selected.kind == "tv":
                if len(music) == 1 and len(tvs) == 1:
                    return music[0], selected
                raise MusicClarification("Which music destination belongs to that TV? Set its music destination mapping.")
        else:
            if len(music) != 1:
                raise MusicClarification("Which music player? " + ", ".join(p.name for p in music))
            selected = music[0]
        # Only the unambiguous single-TV/single-player house can infer a pair.
        power = tvs[0] if len(tvs) == 1 and len(music) == 1 else None
        return selected, power

    async def _prepare(self, args: dict, request: MusicRequest, player: MediaPlayer,
                       power: MediaPlayer | None) -> bool:
        """Wake the destination if it needs it; True when it did."""
        if player.state in ("unavailable", "unknown"):
            raise MusicUnavailable(f"{player.name} is unavailable; no music was submitted.")
        woke = False
        if args.get("enqueue") in ("next", "add"):
            request.stamp("destination_ready")
            return False  # editing the queue must not wake the TV
        if power and power.state in ("off", "standby"):
            await self.home.media_command(power.entity_id, "turn_on")
            request.stamp("wake_sent")
            woke = True
            async def wait_ready() -> None:
                while True:
                    state = str((await self.home.get_entity(power.entity_id)).get("state", ""))
                    if state in ("idle", "on", "playing", "paused", "buffering"):
                        return
                    await asyncio.sleep(self.poll_s)
            try:
                await asyncio.wait_for(wait_ready(), self.ready_timeout_s)
            except TimeoutError as err:
                raise MusicUnavailable(f"{power.name} has not confirmed it is awake; no music was submitted.") from err
        elif power and power.state in ("unavailable", "unknown"):
            raise MusicUnavailable(f"{power.name} is unavailable; no music was submitted.")
        request.stamp("destination_ready")
        return woke

    async def prewake(self) -> bool:
        """Wake the one TV while he is still talking: the Apple TV takes about
        five seconds to come back, most of which his sentence can cover."""
        if not self.native or time.monotonic() - self._prewoke_at < 30:
            return False
        self._prewoke_at = time.monotonic()
        players = await self.home.media_players()
        tvs = [p for p in players if p.kind == "tv"]
        music = [p for p in players if p.kind == "music"]
        binding = self.destinations.get("default") or {}
        power = next((p for p in tvs if p.entity_id == binding.get("power")), None)
        if power is None and len(tvs) == 1 and len(music) == 1:
            power = tvs[0]
        if power is None or power.state not in ("off", "standby"):
            return False
        await self.home.media_command(power.entity_id, "turn_on")
        return True

    # ── what to play ────────────────────────────────────────────────────────

    def _native_wanted(self, args: dict) -> bool:
        return (self.native and str(args.get("media_type") or "playlist") in ("track", "album", "playlist")
                and args.get("enqueue") not in ("next", "add") and not args.get("radio_mode"))

    async def _apple(self, work: Coroutine[Any, Any, AppleItem | None], request: MusicRequest,
                     stage: str) -> AppleItem | None:
        """Apple's catalog is the fast path, never the only one: a miss or a
        network failure just hands the question to Music Assistant."""
        try:
            return await asyncio.wait_for(work, self.search_timeout_s)
        except Exception:  # noqa: BLE001 — see above
            request.stamp(f"{stage}_failed")
            return None

    async def _with_page(self, item: dict | None, request: MusicRequest) -> dict | None:
        """A track from Music Assistant names its id but not its album page or
        row; one lookup fills those in so the Apple TV can open it."""
        if not item or item.get("album_id"):
            return item
        parsed = parse_uri(str(item.get("uri") or ""))
        if parsed is None or parsed[0] != "track":
            return item
        found = await self._apple(self.apple.track(parsed[1]), request, "apple_lookup")
        if found is None:
            return item
        enriched = {**item, "apple_id": found.id, "album_id": found.album_id, "track_number": found.track_number}
        self.catalog.observe([enriched])
        return enriched

    async def _resolve(self, args: dict, request: MusicRequest, native: bool) -> tuple[str, dict | None]:
        """`native`: the destination can open Apple's pages, so an Apple id
        with its album page is worth a lookup; otherwise only Music Assistant
        is asked, exactly as before."""
        query = str(args.get("media_id") or "").strip()
        kind = str(args.get("media_type") or "playlist")
        selection = args.get("selection", "exact")
        if not query:
            raise MusicUnavailable("Name the music you want to play.")
        if selection not in ("exact", "discover"):
            raise MusicUnavailable("Music selection must be exact or discover.")
        if "://" in query:
            request.stamp("resolved_uri")
            cached = self.catalog.exact_uri(query)
            return query, (await self._with_page(cached or {"uri": query, "name": ""}, request) if native else None)
        artist, album = str(args.get("artist") or ""), str(args.get("album") or "")
        cached = (self.catalog.choice(query, kind) if selection == "discover" else
                  self.catalog.exact(query, kind, artist, album))
        if cached and not args.get("fresh"):
            request.stamp("cache_hit")
            return cached["uri"], (await self._with_page(cached, request) if native else cached)
        if native and selection == "exact" and kind in ("track", "album"):
            request.stamp("apple_search_started")
            found = await self._apple(
                self.apple.song(query, artist, album) if kind == "track" else self.apple.album(query, artist),
                request, "apple_search",
            )
            if found is not None and found.openable:
                item = found.as_item()
                self.catalog.observe([item])
                request.stamp("resolved")
                return item["uri"], item
        # MA already resolves library playlist/album names; do not add a browse
        # round trip to that existing exact-name path unless the Apple TV could
        # open the page itself. Tracks need strict artist matching and benefit
        # from caching the resolved ID after this one search.
        if selection == "exact" and kind != "track" and not native:
            request.stamp("provider_name_resolution")
            return query, None
        request.stamp("search_started")
        search = f"{query} {artist}".strip()
        try:
            items = await asyncio.wait_for(self.home.music_search(search, media_type=kind, limit=8),
                                           self.search_timeout_s)
        except TimeoutError as err:
            raise MusicUnavailable("The music search is taking too long; no playback was submitted.") from err
        request.stamp("search_finished")
        self.catalog.observe(items)
        if selection == "exact":
            chosen = exact_match(items, query, kind, artist, album)
            if chosen is None:
                if kind != "track":
                    request.stamp("provider_name_resolution")
                    return query, None  # a library name Music Assistant resolves itself
                raise MusicUnavailable(f"I couldn't find an exact match for {query}" +
                                       (f" by {artist}." if artist else "."))
        else:
            candidates = [i for i in items if i.get("uri") and i.get("media_type") == kind]
            # The provider ranks relevance; keep its order. Discovery is used
            # only when the speaker asks us to choose, not for precise titles.
            if args.get("fresh") and cached:
                candidates = [i for i in candidates if i["uri"] != cached["uri"]]
            if not candidates:
                raise MusicUnavailable("I couldn't find a suitable new music selection.")
            chosen = candidates[0]
        request.stamp("resolved")
        return chosen["uri"], (await self._with_page(chosen, request) if native else chosen)

    # ── starting it ─────────────────────────────────────────────────────────

    @staticmethod
    def already_playing(power: MediaPlayer | None, title: str) -> bool:
        """The TV's own Music app is already on that title: nothing to press."""
        return bool(power is not None and power.app_id == MUSIC_APP and power.state == "playing"
                    and power.now_playing and normalise(power.now_playing) == normalise(title))

    async def _play_native(self, item: AppleItem, power: MediaPlayer, request: MusicRequest,
                           woke: bool) -> tuple[bool, str]:
        """Open the page, press the keys, and believe only what the TV reports."""
        before = (power.state, power.now_playing or "")
        async with self._submission_lock:
            if request.cancelled.is_set() or request is not self.active:
                raise MusicSuperseded()
            await self.home.launch_url(power.entity_id, item.url(self.storefront))
            request.stamp("native_launched")
            # The page is up well inside a second on a warm TV; give one that
            # just woke twice that. A cancel here costs nothing: no key was sent.
            await asyncio.sleep(self.native_ready_s * (2 if woke else 1))
            if request.cancelled.is_set() or request is not self.active:
                raise MusicSuperseded()
            request.submitted = True
            request.native = True
            assert power.remote_entity is not None
            await self.home.remote_commands(power.remote_entity, item.keys(), delay_s=self.key_delay_s)
            request.stamp("native_keys_sent")
        deadline = time.monotonic() + self.native_verify_s
        while True:
            entity = await self.home.get_entity(power.entity_id)
            attrs = entity.get("attributes") or {}
            seen = str(attrs.get("media_title") or "")
            playing = entity.get("state") == "playing" and attrs.get("app_id") == MUSIC_APP
            if item.kind == "track":
                confirmed = playing and normalise(seen) == normalise(item.title)
            else:
                confirmed = playing and (before[0] != "playing" or seen != before[1])
            if confirmed:
                request.stamp("native_playing")
                request.verified = True
                return True, seen
            if time.monotonic() >= deadline:
                request.stamp("native_unconfirmed")
                return False, seen
            await asyncio.sleep(self.poll_s)

    async def play(self, args: dict, request: MusicRequest) -> dict:
        if request.runner is not None and asyncio.current_task() is not request.runner:
            # The fast start has this title in hand: wait for it rather than
            # pressing the same keys twice. If it gave up, start over properly.
            result = await asyncio.shield(request.runner)
            if result is not None:
                return result
            return await self.play(args, self.begin(args))

        async def prepare_and_resolve() -> tuple:
            # The destination first (one state read): whether its TV can open
            # Apple's pages decides how the title is resolved. Then the wake,
            # which is the long pole, and the resolution run side by side.
            players = await self.home.media_players()
            if request.cancelled.is_set():
                raise MusicSuperseded()
            player, power = self._destination(players, str(args.get("player") or ""))
            request.stamp("destination_resolved")
            native = self._native_wanted(args) and power is not None and bool(power.remote_entity)
            tasks = [asyncio.create_task(self._prepare(args, request, player, power)),
                     asyncio.create_task(self._resolve(args, request, native))]
            try:
                woke, resolved = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            return player, power, native, woke, resolved

        work = asyncio.create_task(prepare_and_resolve())
        cancelled = asyncio.create_task(request.cancelled.wait())
        try:
            await asyncio.wait([work, cancelled], return_when=asyncio.FIRST_COMPLETED)
            if request.cancelled.is_set():
                raise MusicSuperseded()
            player, power, native, woke, (media_id, item) = await work
            title = str(item.get("name") or args["media_id"]) if item else str(args["media_id"])
            outcome = {"woke": woke, "media_id": media_id, "title": title, "already": False}
            if native and self.already_playing(power, title):
                request.stamp("already_playing")
                request.native = request.verified = True
                return {**outcome, "player": power, "native": True, "verified": True, "already": True}
            apple = AppleItem.from_item(item) if native and item else None
            if apple is not None and apple.openable:
                verified, seen = await self._play_native(apple, power, request, woke)
                if verified:
                    self._remember(args, item)
                    return {**outcome, "player": power, "native": True, "verified": True,
                            "title": seen if apple.kind == "track" else title}
                request.stamp("native_fallback")
            # Don't cancel a mutation already sent. Serialize local submissions;
            # a remote timeout still leaves its actual outcome uncertain.
            async with self._submission_lock:
                if request.cancelled.is_set() or request is not self.active:
                    raise MusicSuperseded()
                request.submitted = True
                request.stamp("play_submitted")
                await self.home.play_music(
                    player.entity_id, media_id, str(args.get("media_type") or "playlist"),
                    artist=args.get("artist"), album=args.get("album"),
                    enqueue=args.get("enqueue"), radio_mode=bool(args.get("radio_mode", False)),
                )
                request.stamp("service_returned")
            self._remember(args, item)
            return {**outcome, "player": player, "native": False, "verified": False}
        finally:
            for task in (work, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, cancelled, return_exceptions=True)
            if self.active is request:
                self.active = None

    def _remember(self, args: dict, item: dict | None) -> None:
        if item and args.get("selection") == "discover":
            self.catalog.chose(str(args["media_id"]), str(args.get("media_type") or "playlist"), item)

    async def fast_start(self, intent: PlayIntent) -> dict | None:
        """He said "play <title> by <artist>": start it now, before the model
        has decided anything. Its own play_music call for the same title joins
        this request instead of starting twice. None means nothing happened
        (no Apple id, an unknown destination, superseded) — the model's call
        then runs the ordinary way."""
        if not self.native:
            return None
        args: dict[str, Any] = {"media_id": intent.title, "artist": intent.artist,
                                "media_type": "track", "selection": "exact"}
        if intent.destination:
            players = await self.home.media_players()
            words = set(normalise(intent.destination).split()) - {"the", "my", "a", "tv", "television"}
            named = [p for p in players if words & set(normalise(p.name).split())]
            if len(named) == 1:
                args["player"] = named[0].name
            elif not (len([p for p in players if p.kind == "tv"]) == 1
                      and len([p for p in players if p.kind == "music"]) == 1):
                return None  # a room we cannot place: the model asks
        request = self.begin(args, origin="fast_start")
        request.runner = asyncio.current_task()
        try:
            return await self.play(args, request)
        except (MusicClarification, MusicUnavailable, MusicSuperseded):
            return None
