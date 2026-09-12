"""Apple's public catalog, and the Music app pages the Apple TV can open.

Siri plays music on an Apple TV by asking its own Music app, which streams
straight from Apple. This module is the half of that we can reach from Home
Assistant: the iTunes Search API (no key, the same ids Apple Music uses,
~300 ms) resolves a spoken title to a track/album id, and a music.apple.com
link opens the matching page in the Music app through the Apple TV's
Companion protocol. Verified on the living-room Apple TV 2026-09-12: the
`https://music.apple.com/...` form opens the page, `music://` does nothing,
the page is ready for keys within a second, "select" presses Play, and on an
album page Down x track number then select plays that exact track.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import httpx

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"
MUSIC_APP = "com.apple.TVMusic"  # app_id the Apple TV reports while its Music app plays

# Music Assistant's Apple Music uris, with or without the provider instance:
# apple_music://track/574050602, apple_music--in6cbKAC://playlist/pl.42fa7a…
_MA_URI = re.compile(r"^apple_music(?:--[\w-]+)?://(track|album|playlist|artist)/([^/?#]+)$")


def normalise(value: str) -> str:
    """Case, accents and punctuation folded away: what "the same title" means."""
    value = unicodedata.normalize("NFKD", str(value).casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in value if not unicodedata.combining(c))))


def parse_uri(uri: str) -> tuple[str, str] | None:
    """(kind, id) of an Apple Music uri from Music Assistant, else None."""
    match = _MA_URI.match(str(uri or ""))
    return (match.group(1), match.group(2)) if match else None


@dataclass(frozen=True)
class AppleItem:
    """One thing the Music app can open: a track (with its album page and row),
    an album, or a catalog playlist."""

    kind: str  # track | album | playlist
    id: str
    title: str
    artist: str = ""
    album: str = ""
    album_id: str = ""
    track_number: int = 0

    @property
    def uri(self) -> str:
        return f"apple_music://{self.kind}/{self.id}"

    def url(self, storefront: str = "us") -> str:
        """The page the Apple TV opens. The slug segment is ignored by Apple."""
        base = f"https://music.apple.com/{storefront}"
        if self.kind == "track":
            return f"{base}/album/x/{self.album_id}?i={self.id}"
        if self.kind == "album":
            return f"{base}/album/x/{self.id}"
        return f"{base}/playlist/x/{self.id}"

    def keys(self) -> list[str]:
        """Remote presses once the page is up: Play is focused on every page;
        on an album page each Down moves one row into the track list."""
        if self.kind == "track" and self.track_number > 0:
            return ["down"] * self.track_number + ["select"]
        return ["select"]

    @property
    def openable(self) -> bool:
        return bool(self.id) and (self.kind != "track" or bool(self.album_id and self.track_number > 0))

    def as_item(self) -> dict[str, Any]:
        """The media-item shape the catalog cache and the tools already speak."""
        return {
            "name": self.title,
            "media_type": self.kind,
            "uri": self.uri,
            "artists": [self.artist] if self.artist else None,
            "album": self.album or None,
            "apple_id": self.id,
            "album_id": self.album_id,
            "track_number": self.track_number,
        }

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> AppleItem | None:
        """Rebuild from a cached item: ours (apple_id/album_id/track_number)
        or a bare Music Assistant uri, which still names the kind and id."""
        parsed = parse_uri(str(item.get("uri") or ""))
        if parsed is None:
            return None
        kind, item_id = parsed
        if kind == "artist":
            return None
        artists = item.get("artists") or []
        artist = artists[0] if artists else ""
        artist = artist.get("name", "") if isinstance(artist, dict) else str(artist or "")
        album = item.get("album") or ""
        album = album.get("name", "") if isinstance(album, dict) else str(album or "")
        return cls(
            kind=kind,
            id=str(item.get("apple_id") or item_id),
            title=str(item.get("name") or ""),
            artist=artist,
            album=album,
            album_id=str(item.get("album_id") or ""),
            track_number=int(item.get("track_number") or 0),
        )


def _artist_matches(wanted: str, found: str) -> bool:
    if not wanted:
        return True
    a, b = normalise(wanted), normalise(found)
    return bool(a) and bool(b) and (a in b or b in a)


def _pick(results: list[dict[str, Any]], title: str, artist: str, album: str, *, kind: str) -> dict | None:
    """The exact title (then the same title with a suffix) by the named
    artist. Never the "closest" thing: a wrong song is worse than a search."""
    name_key = "trackName" if kind == "track" else "collectionName"
    artist_key = "artistName"
    want = normalise(title)
    candidates = [
        r for r in results
        if r.get(name_key) and _artist_matches(artist, str(r.get(artist_key, "")))
        and (not album or normalise(album) == normalise(str(r.get("collectionName", ""))))
    ]
    exact = [r for r in candidates if normalise(str(r[name_key])) == want]
    if exact:
        return exact[0]
    prefixed = [r for r in candidates if normalise(str(r[name_key])).startswith(want + " ")]
    return prefixed[0] if prefixed else None


def _track(result: dict[str, Any]) -> AppleItem:
    return AppleItem(
        kind="track",
        id=str(result.get("trackId") or ""),
        title=str(result.get("trackName") or ""),
        artist=str(result.get("artistName") or ""),
        album=str(result.get("collectionName") or ""),
        album_id=str(result.get("collectionId") or ""),
        track_number=int(result.get("trackNumber") or 0),
    )


def _album(result: dict[str, Any]) -> AppleItem:
    return AppleItem(
        kind="album",
        id=str(result.get("collectionId") or ""),
        title=str(result.get("collectionName") or ""),
        artist=str(result.get("artistName") or ""),
    )


class AppleCatalog:
    """The iTunes Search API, which answers for Apple Music's catalog too."""

    def __init__(
        self,
        storefront: str = "us",
        *,
        timeout_s: float = 4.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.storefront = storefront or "us"
        self._http = http or httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        resp = await self._http.get(url, params={**params, "country": self.storefront})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        return [r for r in results if isinstance(r, dict)]

    async def song(self, title: str, artist: str = "", album: str = "") -> AppleItem | None:
        """The catalog track for a spoken title (+ artist), or None when the
        title does not appear verbatim in the top results."""
        term = " ".join(part for part in (title, artist) if part).strip()
        if not term:
            return None
        results = await self._get(SEARCH_URL, {"term": term, "entity": "song", "limit": 10, "media": "music"})
        chosen = _pick(results, title, artist, album, kind="track")
        return _track(chosen) if chosen else None

    async def album(self, title: str, artist: str = "") -> AppleItem | None:
        term = " ".join(part for part in (title, artist) if part).strip()
        if not term:
            return None
        results = await self._get(SEARCH_URL, {"term": term, "entity": "album", "limit": 10, "media": "music"})
        chosen = _pick(results, title, artist, "", kind="album")
        return _album(chosen) if chosen else None

    async def track(self, track_id: str) -> AppleItem | None:
        """A track by id (what Music Assistant's search hands back) with the
        album page and row the Apple TV needs."""
        if not str(track_id).isdigit():
            return None
        results = await self._get(LOOKUP_URL, {"id": track_id, "entity": "song"})
        rows = [r for r in results if r.get("wrapperType") == "track"]
        return _track(rows[0]) if rows else None
