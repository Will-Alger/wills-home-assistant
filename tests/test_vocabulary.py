"""The proper names handed to the transcriber — no network, no live house."""

from __future__ import annotations

from assistant.home.base import Light, MediaPlayer
from assistant.home.fake import FakeHome
from assistant.vocabulary import MAX_VOCABULARY_CHARS, MusicNames, vocabulary


def _names(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


async def test_the_house_s_names_come_out_most_useful_first() -> None:
    home = FakeHome()
    text = vocabulary(
        name="Alexa",
        owner="Will",
        lights=await home.get_lights(),
        players=await home.media_players(),
        playlists=["Cleveland 10K"],
        artists=["Dave Brubeck"],
        tasks=["Transcriber vocabulary"],
    )
    names = _names(text)
    assert names[0] == "Alexa" and names[1] == "Will"
    # The ones the transcriber actually mangles beat the plain English ones.
    assert names.index("Apple TV") < names.index("Kitchen Strip")
    assert names.index("Cleveland 10K") < names.index("Kitchen Strip")
    assert {"Living Room", "Living Room Speakers", "Dave Brubeck"} <= set(names)
    assert names[-1] == "Transcriber vocabulary"  # task titles bring up the rear


async def test_names_are_deduplicated_case_insensitively() -> None:
    text = vocabulary(
        name="Alexa",
        owner="Will",
        lights=[Light("light.a", "Hallway", "Hallway", ())],
        players=[MediaPlayer("media_player.a", "hallway", "idle", "music")],
        playlists=["Alexa"],
    )
    assert _names(text) == ["Alexa", "Will", "Hallway"]


def test_every_source_is_optional() -> None:
    assert vocabulary(name="Alexa", owner="Will") == "Alexa, Will"
    assert vocabulary() == ""  # a house with nothing in it is not an error


def test_junk_never_reaches_the_transcriber() -> None:
    text = vocabulary(
        name="Alexa",
        lights=[Light("light.a", "  ", None, ()), Light("light.b", "Desk\n Lamp", None, ())],
        playlists=["", "x", "L" * 61],  # blank, a single letter, a paragraph
    )
    assert _names(text) == ["Alexa", "Desk Lamp"]


def test_the_list_is_capped_and_never_cut_mid_name() -> None:
    playlists = [f"Playlist Number {n}" for n in range(200)]
    text = vocabulary(name="Alexa", owner="Will", playlists=playlists)
    assert len(text) <= MAX_VOCABULARY_CHARS
    assert len(text) > MAX_VOCABULARY_CHARS - 40  # it fills the budget it has
    assert all(name in playlists or name in ("Alexa", "Will") for name in _names(text))

    tight = vocabulary(name="Alexa", owner="Will", playlists=playlists, limit=20)
    assert tight == "Alexa, Will"  # nothing that would overflow, no fragments


async def test_music_names_are_read_from_the_library_and_reused() -> None:
    home = FakeHome()
    clock = [0.0]
    music = MusicNames(home, clock=lambda: clock[0])
    assert music.playlists == [] and music.artists == []  # nothing before the first read

    reading = music.refresh_soon()
    assert reading is not None
    assert music.refresh_soon() is None  # a second caller never starts a second read
    await reading
    assert music.playlists == ["Chill Vibes", "Workout Mix", "Cleveland 10K"]
    assert music.artists == ["Dave Brubeck"]

    assert music.refresh_soon() is None  # fresh names are not re-fetched
    clock[0] += 7 * 3600
    stale = music.refresh_soon()
    assert stale is not None  # ...but stale ones are
    await stale


class _SilentHouse(FakeHome):
    async def music_library(self, media_type="playlist", search=None, limit=50):
        raise RuntimeError("Music Assistant is not set up")


async def test_a_missing_music_library_is_skipped_without_error() -> None:
    clock = [0.0]
    music = MusicNames(_SilentHouse(), clock=lambda: clock[0])
    await music.refresh()  # no raise
    assert music.playlists == [] and music.artists == []
    assert music.refresh_soon() is None  # backs off rather than hammering
    clock[0] += 301
    assert music.refresh_soon() is not None
