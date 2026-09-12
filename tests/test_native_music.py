"""The native Apple TV route: the Music app opened by link, keys pressed,
the title believed only once the TV reports it — and the fast start that
begins before the backend has spoken."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from assistant.apple_catalog import MUSIC_APP, AppleItem, parse_uri
from assistant.brain.tools import ToolExecutor
from assistant.home.base import MediaPlayer
from assistant.home.fake import FakeHome
from assistant.music import (
    MusicCoordinator,
    PlayIntent,
    catalog_for,
    parse_play_request,
    trace_line,
)

BACK_IN_BLACK = AppleItem("track", "574050602", "Back In Black", "AC/DC", "Back In Black", "574050396", 6)
TIME_OUT = AppleItem("album", "157427923", "Time Out", "The Dave Brubeck Quartet")
JAZZ = AppleItem("playlist", "pl.42fa7a4b", "Relaxing Jazz")


class FakeApple:
    """Apple's catalog without the network: answers by title, on request."""

    def __init__(self, *items: AppleItem) -> None:
        self.items = list(items)
        self.albums: dict[str, list[str]] = {}  # album id -> its rows, for album_tracks
        self.calls: list[tuple[str, str]] = []
        self.release = asyncio.Event()
        self.release.set()

    async def album_tracks(self, album_id):
        self.calls.append(("album", album_id))
        return [AppleItem("track", f"{album_id}-{i + 1}", title, "AC/DC", "Back In Black", album_id, i + 1)
                for i, title in enumerate(self.albums.get(album_id, []))]

    async def song(self, title, artist="", album=""):
        self.calls.append(("song", title))
        await self.release.wait()
        return next((i for i in self.items if i.kind == "track" and i.title.casefold() == title.casefold()), None)

    async def album(self, title, artist=""):
        self.calls.append(("album", title))
        return next((i for i in self.items if i.kind == "album" and i.title.casefold() == title.casefold()), None)

    async def track(self, track_id):
        self.calls.append(("track", track_id))
        return next((i for i in self.items if i.kind == "track" and i.id == track_id), None)


class NativeHome(FakeHome):
    """A house whose TV has a remote, like the living room does."""

    def __init__(self, *, asleep: bool = False) -> None:
        super().__init__()
        self.players = [
            MediaPlayer("media_player.living_room_speakers", "Living Room Speakers", "idle", "music"),
            MediaPlayer("media_player.living_room_tv", "Apple TV", "off" if asleep else "idle", "tv",
                        apps=("Music",), remote_entity="remote.living_room_tv"),
        ]
        self.native_tracks = {
            BACK_IN_BLACK.url(): ["Hells Bells", "Shoot to Thrill", "What Do You Do for Money Honey",
                                  "Givin the Dog a Bone", "Let Me Put My Love Into You", "Back In Black"],
            TIME_OUT.url(): ["Blue Rondo à la Turk"],
            JAZZ.url(): ["Say It (Over and Over Again)"],
        }
        self.wakes = 0

    async def media_command(self, entity_id, command, volume_pct=None):
        if command == "turn_on":
            self.wakes += 1
        return await super().media_command(entity_id, command, volume_pct)


def executor(home: NativeHome, *items: AppleItem, **kw) -> ToolExecutor:
    ex = ToolExecutor(home, music_native_ready_s=0.01, **kw)
    ex.music.apple = FakeApple(*(items or (BACK_IN_BLACK, TIME_OUT)))
    ex.music.poll_s = 0.01
    ex.music.native_verify_s = 0.2
    return ex


def tv(home: FakeHome) -> MediaPlayer:
    return next(p for p in home.players if p.kind == "tv")


# ── the pieces ───────────────────────────────────────────────────────────────


def test_apple_items_know_their_page_and_keys() -> None:
    assert BACK_IN_BLACK.url() == "https://music.apple.com/us/album/x/574050396?i=574050602"
    assert BACK_IN_BLACK.url("gb").startswith("https://music.apple.com/gb/")
    assert BACK_IN_BLACK.keys() == ["down"] * 6 + ["select"]
    assert TIME_OUT.url() == "https://music.apple.com/us/album/x/157427923" and TIME_OUT.keys() == ["select"]
    assert JAZZ.url() == "https://music.apple.com/us/playlist/x/pl.42fa7a4b"
    assert parse_uri("apple_music--in6cbKAC://playlist/pl.42fa") == ("playlist", "pl.42fa")
    assert parse_uri("library://playlist/17") is None
    bare = AppleItem.from_item({"uri": "apple_music://track/574050602", "name": "Back In Black"})
    assert bare is not None and not bare.openable  # no album page or row yet
    assert AppleItem.from_item(BACK_IN_BLACK.as_item()) == BACK_IN_BLACK


def test_the_trace_line_names_the_route() -> None:
    native = {"origin": "fast_start", "stages_ms": {"resolved": 407.0, "native_playing": 2891.0}, "playing_verified": True}
    assert trace_line(native) == "music fast_start: resolved 407 → native_playing 2891 ms · native ✓"
    fallen = {"origin": "tool", "stages_ms": {"native_fallback": 4000.0, "service_returned": 24000.0}, "submitted": True}
    assert trace_line(fallen).endswith("· native → MA")
    assert trace_line({"stages_ms": {}, "submitted": False}).endswith("· no playback")


@pytest.mark.parametrize("text, expected", [
    ("Alexa, can you play Back in Black by AC/DC at my living room TV?",
     PlayIntent("Back in Black", "AC/DC", "living room TV")),
    ("Can you play uh American Girls by Harry Styles", PlayIntent("American Girls", "Harry Styles")),
    ("play um, the song Kiwi by, uh, Harry Styles", PlayIntent("Kiwi", "Harry Styles")),
    ("play take five by dave brubeck", PlayIntent("take five", "dave brubeck")),
    ("Put on “Hotel California” by the Eagles.", PlayIntent("Hotel California", "the Eagles")),
    ("play some jazz", None),
    ("play something by Drake", None),
    ("play it by ear", None),
    ("turn off the lights", None),
    ("play the next one", None),
])
def test_only_unmistakable_requests_start_early(text, expected) -> None:
    assert parse_play_request(text) == expected


# ── the native start ─────────────────────────────────────────────────────────


async def test_an_exact_song_plays_in_the_tv_music_app_and_is_confirmed() -> None:
    home = NativeHome()
    result = await executor(home).run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert "Playing Back In Black on Apple TV" in result.summary
    assert result.details["native"] is True
    assert home.launched_urls == [("media_player.living_room_tv", BACK_IN_BLACK.url())]
    assert home.remote_keys == [("remote.living_room_tv", ["down"] * 6 + ["select"])]
    assert home.played == []  # Music Assistant never entered into it
    trace = result.details["music_trace"]
    assert trace["native"] and trace["playing_verified"] and not trace["audible_verified"]
    assert {"native_launched", "native_keys_sent", "native_playing"} <= set(trace["stages_ms"])
    assert tv(home).now_playing == "Back In Black" and tv(home).app_id == MUSIC_APP


async def test_albums_and_catalog_playlists_need_one_press() -> None:
    home = NativeHome()
    ex = executor(home, TIME_OUT)
    result = await ex.run("play_music", {"media_id": "Time Out", "artist": "Dave Brubeck", "media_type": "album"})
    assert not result.is_error, result
    assert home.remote_keys[-1][1] == ["select"] and result.details["native"]
    # A playlist comes back from Music Assistant's catalog search with its Apple id.
    home.music_search = _search([{"name": "Relaxing Jazz", "media_type": "playlist",
                                  "uri": "apple_music--in6cbKAC://playlist/pl.42fa7a4b", "artists": None}])
    result = await ex.run("play_music", {"media_id": "jazz", "media_type": "playlist", "selection": "discover"})
    assert not result.is_error, result
    assert home.launched_urls[-1][1] == JAZZ.url() and home.remote_keys[-1][1] == ["select"]
    assert home.played == []


async def test_a_track_from_music_assistant_gets_its_page_looked_up() -> None:
    home = NativeHome()
    home.music_search = _search([{"name": "Back In Black", "media_type": "track", "artists": ["AC/DC"],
                                  "album": "Back In Black", "uri": "apple_music://track/574050602"}])
    ex = executor(home)
    ex.music.apple = FakeApple(BACK_IN_BLACK)
    ex.music.apple.items = [replace(BACK_IN_BLACK, title="Back In Black (Remastered)")]  # search misses…
    result = await ex.run("play_music", {"media_id": "Back In Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert ("track", "574050602") in ex.music.apple.calls  # …so the id from MA is looked up
    assert home.launched_urls[-1][1] == BACK_IN_BLACK.url()


async def test_a_page_that_opens_one_row_off_is_put_right_from_the_album() -> None:
    home = NativeHome()
    home.native_focus_offset = -1  # the Now Playing screen was up: the first Down was swallowed
    ex = executor(home)
    ex.music.apple.albums = {BACK_IN_BLACK.album_id: home.native_tracks[BACK_IN_BLACK.url()]}
    result = await ex.run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert "Playing Back In Black on Apple TV" in result.summary and result.details["native"]
    assert home.media_commands == [("media_player.living_room_tv", "next")]
    assert "native_corrected" in result.details["music_trace"]["stages_ms"]
    assert home.played == []


async def test_an_unconfirmed_native_start_falls_back_to_music_assistant() -> None:
    home = NativeHome()
    home.native_broken = True
    result = await executor(home).run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert "Playback requested" in result.summary and result.details["native"] is False
    assert home.remote_keys and home.played[0]["media_id"] == BACK_IN_BLACK.uri
    assert "native_fallback" in result.details["music_trace"]["stages_ms"]


async def test_what_is_already_playing_is_not_restarted() -> None:
    home = NativeHome()
    home.players = [replace(p, state="playing", now_playing="Back In Black", app_id=MUSIC_APP)
                    if p.kind == "tv" else p for p in home.players]
    result = await executor(home).run("play_music", {"media_id": "back in black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert "already playing" in result.summary
    assert home.launched_urls == [] and home.remote_keys == [] and home.played == []


async def test_a_sleeping_tv_is_woken_before_the_page_opens() -> None:
    home = NativeHome(asleep=True)
    result = await executor(home).run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert home.wakes == 1 and result.summary.startswith("Woke the TV first.")
    stages = result.details["music_trace"]["stages_ms"]
    assert stages["wake_sent"] <= stages["native_launched"]


async def test_the_queue_and_radio_keep_music_assistant() -> None:
    home = NativeHome()
    ex = executor(home)
    catalog_for(home).observe([BACK_IN_BLACK.as_item()])  # known already: no search either way
    for extra in ({"enqueue": "next"}, {"radio_mode": True}):
        await ex.run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track", **extra})
    assert home.launched_urls == [] and len(home.played) == 2


async def test_native_off_or_no_remote_means_music_assistant() -> None:
    home = NativeHome()
    result = await executor(home, music_native=False).run("play_music", {"media_id": "apple_music://track/1", "media_type": "track"})
    assert not result.is_error and home.launched_urls == [] and home.played
    plain = FakeHome()  # the seeded TV has no remote entity: Apple's catalog is never asked
    plain.music_search = _search([{"name": "Back In Black", "media_type": "track", "artists": ["AC/DC"],
                                   "album": "Back In Black", "uri": BACK_IN_BLACK.uri}])
    ex = executor(plain)
    result = await ex.run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"})
    assert not result.is_error, result
    assert ex.music.apple.calls == []
    assert plain.launched_urls == [] and plain.played[0]["media_id"] == BACK_IN_BLACK.uri


async def test_pause_and_volume_go_to_the_tv_while_its_music_app_plays() -> None:
    home = NativeHome()
    home.players = [replace(p, state="playing", now_playing="Back In Black", app_id=MUSIC_APP)
                    if p.kind == "tv" else replace(p, state="playing") for p in home.players]
    ex = executor(home)
    await ex.run("media_control", {"action": "pause"})
    await ex.run("media_control", {"action": "volume_set", "volume_pct": 40})
    assert [c[0] for c in home.media_commands] == ["media_player.living_room_tv"] * 2


# ── the fast start ───────────────────────────────────────────────────────────


async def test_the_fast_start_and_the_backends_call_press_the_keys_once() -> None:
    home = NativeHome()
    ex = executor(home)
    ex.music.apple.release.clear()  # the catalog answer is still in flight…
    early = asyncio.create_task(ex.music.fast_start(PlayIntent("Back in Black", "AC/DC")))
    await asyncio.sleep(0.02)
    late = asyncio.create_task(ex.run("play_music", {"media_id": "Back in Black", "artist": "AC/DC", "media_type": "track"}))
    await asyncio.sleep(0.02)
    ex.music.apple.release.set()  # …when the backend's own call arrives
    first, result = await asyncio.gather(early, late)
    assert first is not None and first["verified"]
    assert not result.is_error and "Playing Back In Black" in result.summary
    assert len(home.launched_urls) == 1 and len(home.remote_keys) == 1
    assert "joined" in result.details["music_trace"]["stages_ms"]


async def test_a_fast_start_that_finds_nothing_leaves_the_backend_to_it() -> None:
    home = NativeHome()
    ex = executor(home, TIME_OUT)  # no such song in the catalog
    assert await ex.music.fast_start(PlayIntent("Nothing Here", "Nobody")) is None
    assert home.launched_urls == [] and home.played == []
    home.players.append(MediaPlayer("media_player.bedroom_tv", "Bedroom TV", "off", "tv", remote_entity="remote.bedroom_tv"))
    assert await ex.music.fast_start(PlayIntent("Back in Black", "AC/DC", "attic")) is None
    assert home.wakes == 0


async def test_prewake_wakes_the_one_tv_once() -> None:
    home = NativeHome(asleep=True)
    ex = executor(home)
    assert await ex.music.prewake() is True
    assert await ex.music.prewake() is False  # not again for a while
    assert home.wakes == 1 and tv(home).state == "idle"
    awake = NativeHome()
    assert await executor(awake).music.prewake() is False


def _search(items: list[dict]):
    async def music_search(query, media_type="playlist", limit=8):
        return [i for i in items if i["media_type"] == media_type]
    return music_search


def test_coordinator_defaults_are_the_measured_ones() -> None:
    music = MusicCoordinator(FakeHome())
    assert music.native and music.storefront == "us" and music.native_ready_s == 0.9
