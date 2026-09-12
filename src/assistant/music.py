"""Resolve music and prepare its destination concurrently, then submit once.

No background playback, credentials or persisted audio URLs. The bounded metadata
cache is shared with library browsing, including the transcriber's background warmup.
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from assistant.home.base import HomeApi, MediaPlayer


def normalise(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in value if not unicodedata.combining(c))))


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


@dataclass
class MusicRequest:
    id: int
    started: float = field(default_factory=time.monotonic)
    stages: dict[str, float] = field(default_factory=dict)
    submitted: bool = False
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    def stamp(self, name: str) -> None:
        self.stages[name] = round((time.monotonic() - self.started) * 1000, 1)

    def details(self) -> dict:
        return {"request_id": self.id, "stages_ms": dict(self.stages),
                "submitted": self.submitted, "audible_verified": False}


class MusicCoordinator:
    def __init__(self, home: HomeApi, *, destinations: dict[str, dict[str, str]] | None = None,
                 ready_timeout_s: float = 5, search_timeout_s: float = 8,
                 poll_s: float = 0.15) -> None:
        self.home = home
        self.catalog = catalog_for(home)
        self.destinations = destinations or {}
        self.ready_timeout_s, self.search_timeout_s, self.poll_s = ready_timeout_s, search_timeout_s, poll_s
        self._next = 0
        self.active: MusicRequest | None = None
        self._submission_lock = asyncio.Lock()

    def cancel_pending(self) -> bool:
        if self.active is None or self.active.submitted:
            return False
        self.active.cancelled.set()
        return True

    def begin(self) -> MusicRequest:
        self.cancel_pending()
        self._next += 1
        request = MusicRequest(self._next)
        self.active = request
        return request

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

    async def _prepare(self, args: dict, request: MusicRequest) -> tuple[MediaPlayer, bool]:
        players = await self.home.media_players()
        if request.cancelled.is_set():
            raise MusicSuperseded()
        player, power = self._destination(players, str(args.get("player") or ""))
        request.stamp("destination_resolved")
        if player.state in ("unavailable", "unknown"):
            raise MusicUnavailable(f"{player.name} is unavailable; no music was submitted.")
        woke = False
        if args.get("enqueue") in ("next", "add"):
            request.stamp("destination_ready")
            return player, False  # editing the queue must not wake the TV
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
        return player, woke

    async def _resolve(self, args: dict, request: MusicRequest) -> tuple[str, dict | None]:
        query = str(args.get("media_id") or "").strip()
        kind = str(args.get("media_type") or "playlist")
        selection = args.get("selection", "exact")
        if not query:
            raise MusicUnavailable("Name the music you want to play.")
        if selection not in ("exact", "discover"):
            raise MusicUnavailable("Music selection must be exact or discover.")
        if "://" in query:
            request.stamp("resolved_uri")
            return query, None
        artist, album = str(args.get("artist") or ""), str(args.get("album") or "")
        cached = (self.catalog.choice(query, kind) if selection == "discover" else
                  self.catalog.exact(query, kind, artist, album))
        if cached and not args.get("fresh"):
            request.stamp("cache_hit")
            return cached["uri"], cached
        # MA already resolves library playlist/album names; do not add a browse
        # round trip to that existing exact-name path. Tracks need strict artist
        # matching and benefit from caching the resolved ID after this one search.
        if selection == "exact" and kind != "track":
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
        return chosen["uri"], chosen

    async def play(self, args: dict, request: MusicRequest) -> dict:
        async def prepare_and_resolve() -> tuple:
            tasks = [asyncio.create_task(self._prepare(args, request)),
                     asyncio.create_task(self._resolve(args, request))]
            try:
                return tuple(await asyncio.gather(*tasks))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        work = asyncio.create_task(prepare_and_resolve())
        cancelled = asyncio.create_task(request.cancelled.wait())
        try:
            await asyncio.wait([work, cancelled], return_when=asyncio.FIRST_COMPLETED)
            if request.cancelled.is_set():
                raise MusicSuperseded()
            (player, woke), (media_id, item) = await work
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
            if item and args.get("selection") == "discover":
                self.catalog.chose(str(args["media_id"]), str(args.get("media_type") or "playlist"), item)
            return {"player": player, "woke": woke, "media_id": media_id,
                    "title": item.get("name", args["media_id"]) if item else args["media_id"]}
        finally:
            for task in (work, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, cancelled, return_exceptions=True)
            if self.active is request:
                self.active = None
