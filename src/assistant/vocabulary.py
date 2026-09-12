"""What the transcriber should expect to hear.

The Realtime model listens to the audio itself, but the *written* transcript
comes from a second, separate transcription model that has never heard of the
Cleveland 10K playlist and cheerfully writes "living room apple tv". That
transcript is what the stop-phrase match, the journal, reflection and the
session log all read, so its spelling is the house's spelling. Handing the
transcriber the proper names it is about to hear fixes them, and changes
nothing about what the model itself understands.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from assistant.home.base import HomeApi, Light, MediaPlayer
from assistant.music import catalog_for

MAX_VOCABULARY_CHARS = 1000
_MAX_NAME_CHARS = 60  # a sentence-long task title helps no transcriber
_LIBRARY_LIMIT = 60  # playlists (and artists) to pull per refresh
_REFRESH_S = 6 * 3600.0  # a music library changes slowly
_RETRY_S = 300.0  # ...but one that is not answering yet deserves another look


def _clean(name: Any) -> str:
    """A speakable name, or "" for anything not worth a transcriber's time."""
    text = " ".join(str(name or "").split())
    return text if 1 < len(text) <= _MAX_NAME_CHARS else ""


def vocabulary(
    *,
    name: str = "",
    owner: str = "",
    lights: Sequence[Light] = (),
    players: Sequence[MediaPlayer] = (),
    playlists: Sequence[str] = (),
    artists: Sequence[str] = (),
    tasks: Sequence[str] = (),
    limit: int = MAX_VOCABULARY_CHARS,
) -> str:
    """The house's proper names, most useful first, deduplicated and capped.

    The order decides what the cap throws away, so the names the owner says
    out loud and the transcriber gets wrong come first: hers and his, the
    areas, the players ("living room apple tv" was the complaint), then the
    library. Plain light names and task titles bring up the rear — English
    already spells "Kitchen Strip" correctly.

    Every source is optional. A house with no media, no music library and no
    task board still yields her name and the owner's, and no error.
    """
    groups: list[Iterable[Any]] = [
        [name, owner],
        [light.area for light in lights or ()],
        [player.name for player in players or ()],
        playlists,
        artists,
        [light.name for light in lights or ()],
        tasks,
    ]
    seen: set[str] = set()
    picked: list[str] = []
    width = 0
    for group in groups:
        for raw in group or ():
            text = _clean(raw)
            if not text or text.casefold() in seen:
                continue
            cost = len(text) + (2 if picked else 0)  # ", "
            if width + cost > limit:
                return ", ".join(picked)  # the rest matters less than staying under
            seen.add(text.casefold())
            picked.append(text)
            width += cost
    return ", ".join(picked)


class MusicNames:
    """Playlist and artist names from the owner's library, kept in memory.

    Browsing Music Assistant is a round trip through Home Assistant, and the
    session config sits on the wake-word critical path — so a session spends
    whatever the last refresh left behind (nothing at all, on the first wake
    after a restart) and starts the next one in the background.
    """

    def __init__(
        self,
        home: HomeApi,
        *,
        limit: int = _LIBRARY_LIMIT,
        refresh_s: float = _REFRESH_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._home = home
        self._limit = limit
        self._refresh_s = refresh_s
        self._clock = clock
        self.playlists: list[str] = []
        self.artists: list[str] = []
        self._due = 0.0  # when the next refresh is allowed
        self._task: asyncio.Task[None] | None = None

    def refresh_soon(self) -> asyncio.Task[None] | None:
        """Start a due refresh in the background; hands back the task, if any.

        Never blocks and never raises: its callers stand between the wake word
        and the first word she hears.
        """
        running = self._task is not None and not self._task.done()
        if running or self._clock() < self._due:
            return None
        try:
            self._task = asyncio.create_task(self.refresh())
        except RuntimeError:  # called with no event loop: nothing to warm
            return None
        return self._task

    async def refresh(self) -> None:
        """Re-read the library. One that is down keeps the last good names."""
        playlists = await self._names("playlist")
        artists = await self._names("artist")
        if playlists or artists:
            self.playlists, self.artists = playlists, artists
            self._due = self._clock() + self._refresh_s
        else:
            self._due = self._clock() + _RETRY_S

    async def _names(self, media_type: str) -> list[str]:
        """Library names in the order Music Assistant returns them."""
        try:
            items = await self._home.music_library(media_type=media_type, limit=self._limit)
        except Exception:  # noqa: BLE001 — no music library is not a session failure
            return []
        catalog_for(self._home).observe(items or [])
        return [
            str(item.get("name", ""))
            for item in items or ()
            if isinstance(item, dict) and item.get("name")
        ]
