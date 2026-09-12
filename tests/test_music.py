"""Music startup contracts using controlled fake readiness and search events."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from assistant.brain.tools import ToolExecutor
from assistant.home.base import MediaPlayer
from assistant.home.client import HomeAssistantError
from assistant.home.fake import FakeHome
from assistant.music import MusicCatalog, catalog_for
from assistant.vocabulary import MusicNames

TRACK = {"name": "Take Five", "media_type": "track", "artists": ["Dave Brubeck"],
         "album": "Time Out", "uri": "apple_music://track/take-five"}


class ObservedHome(FakeHome):
    def __init__(self, *, asleep=False):
        super().__init__()
        self.player_reads = 0
        self.searches = 0
        self.library_reads = 0
        self.search_started = asyncio.Event()
        self.wake_started = asyncio.Event()
        self.ready = asyncio.Event()
        self.search_release = asyncio.Event()
        self.ready.set()
        self.search_release.set()
        self.results = [dict(TRACK)]
        if asleep:
            self.players = [replace(p, state="off") if p.kind == "tv" else p for p in self.players]

    async def media_players(self):
        self.player_reads += 1
        return await super().media_players()

    async def media_command(self, entity_id, command, volume_pct=None):
        if command == "turn_on":
            self.wake_started.set()
        return await super().media_command(entity_id, command, volume_pct)

    async def get_entity(self, entity_id):
        if entity_id == "media_player.living_room_tv":
            await self.ready.wait()
        return await super().get_entity(entity_id)

    async def music_search(self, query, media_type="playlist", limit=8):
        self.searches += 1
        self.search_started.set()
        await self.search_release.wait()
        return list(self.results)

    async def music_library(self, *args, **kwargs):
        self.library_reads += 1
        return await super().music_library(*args, **kwargs)


def song(**kw):
    return {"media_id": "Take Five", "media_type": "track", "artist": "Dave Brubeck", **kw}


async def test_exact_song_is_cached_and_reuses_one_destination_snapshot():
    home = ObservedHome()
    executor = ToolExecutor(home)
    for _ in range(2):
        result = await executor.run("play_music", song())
        assert not result.is_error, result
    assert home.searches == 1
    assert home.player_reads == 2  # once for each request, never two per play
    assert all(p["media_id"] == TRACK["uri"] for p in home.played)
    assert result.details["music_trace"]["audible_verified"] is False
    assert "cache_hit" in result.details["music_trace"]["stages_ms"]


async def test_search_and_wake_overlap_and_immediate_readiness_has_no_sleep():
    home = ObservedHome(asleep=True)
    home.ready.clear()
    home.search_release.clear()
    executor = ToolExecutor(home)
    task = asyncio.create_task(executor.run("play_music", song()))
    try:
        # Both must start before either dependency finishes. Sequential work
        # deadlocks here, so this proves overlap without timing tiny operations.
        await asyncio.wait_for(home.search_started.wait(), 1)
        await asyncio.wait_for(home.wake_started.wait(), 1)
        assert home.played == []
        home.ready.set()
        home.search_release.set()
        result = await asyncio.wait_for(task, 1)  # the old fixed 3 s sleep fails
        assert not result.is_error, result
        assert len(home.played) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_discovery_searches_plays_and_caches_in_one_call():
    home = ObservedHome()
    home.results = [{"name": "Jazz Chill", "uri": "apple_music://playlist/jazz", "media_type": "playlist"}]
    executor = ToolExecutor(home)
    args = {"media_id": "jazz", "media_type": "playlist", "selection": "discover"}
    assert not (await executor.run("play_music", args)).is_error
    assert home.played[0]["media_id"] == home.results[0]["uri"]
    assert not home.played[0]["radio_mode"]
    assert not (await executor.run("play_music", args)).is_error
    assert home.searches == 1


async def test_fresh_discovery_does_not_reuse_the_previous_pick():
    home = ObservedHome()
    home.results = [{"name": str(i), "uri": f"apple_music://playlist/{i}", "media_type": "playlist"}
                    for i in range(2)]
    executor = ToolExecutor(home)
    args = {"media_id": "jazz", "media_type": "playlist", "selection": "discover"}
    await executor.run("play_music", args)
    await executor.run("play_music", {**args, "fresh": True})
    assert home.searches == 2
    assert home.played[0]["media_id"] != home.played[1]["media_id"]


@pytest.mark.parametrize("change", [
    {"artists": ["Someone Else"]}, {"name": "Take Five (Live)"}, {"album": "Other Album"},
])
async def test_exact_artist_title_and_album_are_never_substituted(change):
    home = ObservedHome()
    home.results = [{**TRACK, **change}]
    result = await ToolExecutor(home).run("play_music", song(album="Time Out"))
    assert result.is_error
    assert home.played == []


async def test_ambiguous_artist_requires_clarification():
    home = ObservedHome()
    home.results += [{**TRACK, "artists": ["Another Artist"], "uri": "apple_music://track/other"}]
    result = await ToolExecutor(home).run("play_music", {"media_id": "Take Five", "media_type": "track"})
    assert result.status == "needs_clarification"
    assert home.played == []


async def test_explicit_tv_maps_to_music_entity():
    home = ObservedHome()
    result = await ToolExecutor(home).run("play_music", song(player="Apple TV"))
    assert not result.is_error
    assert home.played[0]["entity_id"] == "media_player.living_room_speakers"


async def test_multiple_tvs_never_wake_an_unrelated_one():
    home = ObservedHome()
    home.players.append(MediaPlayer("media_player.bedroom_tv", "Bedroom TV", "off", "tv"))
    result = await ToolExecutor(home).run("play_music", song())
    assert not result.is_error
    assert not home.media_commands


async def test_mapping_wakes_only_its_tv():
    home = ObservedHome(asleep=True)
    home.players.append(MediaPlayer("media_player.bedroom_tv", "Bedroom TV", "off", "tv"))
    executor = ToolExecutor(home, music_destinations={
        "living room": {"player": "media_player.living_room_speakers", "power": "media_player.living_room_tv"},
    })
    result = await executor.run("play_music", song(player="living room"))
    assert not result.is_error, result
    assert home.media_commands == [("media_player.living_room_tv", "turn_on")]


async def test_unknown_destination_does_not_play_or_wake():
    home = ObservedHome(asleep=True)
    result = await ToolExecutor(home).run("play_music", song(player="attic"))
    assert result.status == "needs_clarification"
    assert home.played == home.media_commands == []


async def test_wake_timeout_is_bounded_and_does_not_play():
    home = ObservedHome(asleep=True)
    home.ready.clear()
    executor = ToolExecutor(home)
    executor.music.ready_timeout_s = 0.02
    result = await asyncio.wait_for(executor.run("play_music", song()), 1)
    assert result.is_error
    assert home.played == []
    assert not result.details["music_trace"]["submitted"]


async def test_search_timeout_cancels_preparation_and_does_not_play():
    home = ObservedHome(asleep=True)
    home.ready.clear()
    home.search_release.clear()
    executor = ToolExecutor(home)
    executor.music.search_timeout_s = 0.02
    result = await asyncio.wait_for(executor.run("play_music", song()), 1)
    assert result.is_error
    assert home.played == []


async def test_new_music_supersedes_pending_search():
    home = ObservedHome()
    home.search_release.clear()
    executor = ToolExecutor(home)
    old = asyncio.create_task(executor.run("play_music", song()))
    await asyncio.wait_for(home.search_started.wait(), 1)
    new = await executor.run("play_music", {"media_id": "apple_music://track/new", "media_type": "track"})
    result = await asyncio.wait_for(old, 1)
    assert "cancelled or replaced" in result.summary
    assert not new.is_error
    assert [p["media_id"] for p in home.played] == ["apple_music://track/new"]


async def test_stop_cancels_pending_search_even_when_nothing_is_playing():
    home = ObservedHome()
    home.search_release.clear()
    executor = ToolExecutor(home)
    old = asyncio.create_task(executor.run("play_music", song()))
    await asyncio.wait_for(home.search_started.wait(), 1)
    result = await executor.run("media_control", {"action": "stop"})
    await asyncio.wait_for(old, 1)
    assert "Cancelled" in result.summary
    assert home.played == []


async def test_submitted_commands_are_not_cancelled_or_overtaken():
    entered, release = asyncio.Event(), asyncio.Event()
    class SlowPlay(ObservedHome):
        async def play_music(self, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return await super().play_music(*args, **kwargs)
    home = SlowPlay()
    executor = ToolExecutor(home)
    old = asyncio.create_task(executor.run("play_music", {"media_id": "apple_music://track/old", "media_type": "track"}))
    await asyncio.wait_for(entered.wait(), 1)
    assert not executor.music.cancel_pending()
    new = asyncio.create_task(executor.run("play_music", {"media_id": "apple_music://track/new", "media_type": "track"}))
    await asyncio.sleep(0)
    assert home.played == []
    release.set()
    await asyncio.wait_for(asyncio.gather(old, new), 1)
    assert [p["media_id"] for p in home.played] == ["apple_music://track/old", "apple_music://track/new"]


async def test_timeout_never_fetches_library_or_suggests_a_retry():
    class BrokenPlay(ObservedHome):
        async def play_music(self, *args, **kwargs):
            raise HomeAssistantError("Playback may still start; do not retry immediately")
    home = BrokenPlay()
    result = await ToolExecutor(home).run("play_music", song())
    assert result.is_error
    assert home.library_reads == 0
    assert "Check the queue" in result.follow_up
    assert "retry play_music" not in result.summary
    assert result.details["music_trace"]["submitted"]


async def test_browse_and_background_names_warm_the_shared_uri_cache():
    class Library(ObservedHome):
        async def music_library(self, media_type="playlist", **kwargs):
            return [{"name": "Chill Vibes", "uri": "library://playlist/1", "media_type": "playlist"}]
    home = Library()
    await MusicNames(home).refresh()
    executor = ToolExecutor(home)
    result = await executor.run("play_music", {"media_id": "Chill Vibes", "media_type": "playlist"})
    assert not result.is_error
    assert home.played[0]["media_id"] == "library://playlist/1"
    assert home.searches == 0
    await executor.run("browse_music", {"scope": "catalog", "search": "Take Five", "media_type": "track"})
    await executor.run("play_music", song())
    assert home.searches == 1  # browse only


def test_cache_expires_is_bounded_and_excludes_audio_urls():
    now = [0.0]
    cache = MusicCatalog(ttl_s=10, limit=2, clock=lambda: now[0])
    for i in range(3):
        cache.observe([{**TRACK, "name": str(i), "uri": f"apple_music://track/{i}"}])
    assert cache.exact("0", "track") is None
    assert cache.exact("2", "track")
    cache.observe([{**TRACK, "uri": "https://cdn.example/audio?signature=secret"}])
    assert cache.exact("Take Five", "track") is None
    now[0] = 11
    assert cache.exact("2", "track") is None


async def test_enqueue_and_radio_options_survive_the_fast_path():
    home = ObservedHome(asleep=True)
    catalog_for(home).observe([TRACK])
    result = await ToolExecutor(home).run("play_music", song(enqueue="next", radio_mode=True))
    assert not result.is_error
    assert home.played[0]["enqueue"] == "next"
    assert home.played[0]["radio_mode"] is True
    assert home.media_commands == []


async def test_invalid_requests_do_not_wake_the_tv():
    home = ObservedHome(asleep=True)
    executor = ToolExecutor(home)
    result = await executor.run("play_music", {"media_id": "", "media_type": "track"})
    assert result.is_error
    assert home.player_reads == 0
    assert not home.media_commands
