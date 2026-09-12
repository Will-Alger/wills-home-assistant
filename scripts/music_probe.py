"""Time the music tool directly, without Live/model costs.

Default: offline fake search/wake with 200 ms delay each; --repeat 2 exposes
overlap and caching. --live explicitly plays on real devices; no recordings.
The service's return time is NOT the time the first music becomes audible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace

from assistant.brain.tools import ToolExecutor
from assistant.config import load_settings
from assistant.home.client import HomeAssistantClient
from assistant.home.fake import FakeHome


class DelayedHome(FakeHome):
    def __init__(self) -> None:
        super().__init__()
        self.players = [replace(p, state="off") if p.kind == "tv" else p for p in self.players]
        self._waiting = True

    async def music_search(self, query: str, media_type: str = "playlist", limit: int = 8) -> list[dict]:
        await asyncio.sleep(0.2)
        return [{"name": "Take Five", "artists": ["Dave Brubeck"], "album": "Time Out",
                 "media_type": "track", "uri": "apple_music://track/fake-take-five"}]

    async def get_entity(self, entity_id: str) -> dict:
        if self._waiting:
            self._waiting = False
            await asyncio.sleep(0.2)
        return await super().get_entity(entity_id)


async def probe(executor: ToolExecutor, args: argparse.Namespace) -> int:
    call = {"media_id": args.media_id, "media_type": args.media_type,
            "selection": args.selection, "fresh": args.fresh}
    if args.artist:
        call["artist"] = args.artist
    if args.player:
        call["player"] = args.player
    failed = False
    for trial in range(args.repeat):
        outcome = await executor.run("play_music", call)
        print(json.dumps({"mode": "live" if args.live else "fake", "trial": trial + 1,
                          **outcome.payload()}, ensure_ascii=False))
        if outcome.is_error:
            failed = True
            break  # no automatic retry after an uncertain play
    return int(failed)


async def run(args: argparse.Namespace) -> int:
    if not args.live:
        return await probe(ToolExecutor(DelayedHome()), args)
    settings = load_settings()
    if not settings.ha_token:
        raise ValueError("Set HA_TOKEN before running a live playback probe")
    async with HomeAssistantClient(settings.ha_url, settings.ha_token) as home:
        return await probe(ToolExecutor(home, music_destinations=settings.music_destinations), args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="send real music playback requests")
    parser.add_argument("--media-id", default="Take Five", help="exact title, discovery query, or known URI")
    parser.add_argument("--artist", default="")
    parser.add_argument("--media-type", default="track", choices=["track", "playlist", "album", "artist", "radio"])
    parser.add_argument("--selection", default="exact", choices=["exact", "discover"])
    parser.add_argument("--player", default="")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="explicit number of play requests, 1–20")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 20:
        parser.error("--repeat must be between 1 and 20")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
