# Music the way Siri does it: the Apple TV plays it itself

September 12, 2026. Will's acceptance criterion, verbatim in spirit: *when
Tony Stark says "daddy's home", music starts in two or three seconds. Choosing
an unknown playlist gets more wiggle room. If it would be lame for Tony Stark
to wait fifteen seconds, it is lame here.*

## Where the twenty-eight seconds went

Measured from Home Assistant history, the turn log and the Music Assistant
add-on log for "play Back in Black by AC/DC" (13:34 that day, TV asleep):

| Stage | Time |
| --- | --- |
| End of the request → backend delegation, catalog search, TV wake sent | ~4.4 s |
| The fixed `asyncio.sleep(3)` after the wake (removed by the fast-start merge) | 3.0 s |
| Music Assistant `play_media` → "Streaming to Living Room via AirPlay 2" | ~19 s |
| AirPlay 2 start → the TV reports playing | ~1.3 s |

The warm cases (music already playing, TV awake) took the same 24–29 s in the
tool, so it was never the wake. The AirPlay leg is a second. Everything slow
was Music Assistant getting audio out of its Apple Music provider: Widevine
licence, encrypted HLS, one concurrent stream per provider, and its own
next-track pre-buffer giving up "after 20.00 s, 0 s buffered" a dozen times a
day in the log. Bandwidth never entered into it.

## The route

Siri does not stream to the Apple TV; it asks the TV's own Music app, which
streams from Apple. We can do the same from Home Assistant:

1. **Resolve** the title with Apple's public iTunes Search API (no key, the
   same ids Apple Music uses, ~300 ms): track id, album id, track number.
   Catalog playlists come from Music Assistant's search with their `pl.` id.
2. **Open the page**: `media_player.play_media` with a `url` type on the
   Apple TV entity hands `https://music.apple.com/us/album/x/<album>?i=<track>`
   to pyatv's `launch_app`, which the Companion protocol opens in the Music
   app (~40 ms to accept, the page ready inside a second). `music://` links
   do nothing on tvOS.
3. **Press the keys** with one `remote.send_command`: Play is focused on every
   page, so `select` starts an album or playlist; on an album page each
   `down` moves one row into the track list, so `down` × track number then
   `select` plays that exact track.
4. **Believe the TV, not the call**: poll the Apple TV entity until it reports
   `playing` in `com.apple.TVMusic` with the requested title. Unconfirmed
   within 4 s → the Music Assistant path runs with the same resolved id.

Measured on the living-room Apple TV (HA state as the "playing" clock):

| Case | Link → playing |
| --- | --- |
| Warm, exact track (Back in Black, track 6) | 1.8–2.5 s |
| Warm, album or catalog playlist (one press) | 1.8–2.0 s |
| Asleep: wake reported after 5.5 s, then as above | ~8.5 s from the wake |
| Through the coordinator, tool start → verified (`scripts/music_probe.py --live`) | 2.9 s |
| The same song asked again | 0.16 s, "already playing", nothing pressed |

The Apple TV ignores launches and keys until it reports awake, so a cold start
cannot be overlapped inside the tool. It can be overlapped with the sentence:
the Live transcript arrives word by word, so the engine wakes the TV the moment
it hears "play" (`MusicCoordinator.prewake`), and the remaining words plus the
backend's decision cover most of the five seconds.

## The fast start

"Play *title* by *artist*" needs no model to interpret. When a user turn ends
on that pattern (`parse_play_request`: a title AND an artist, nothing vague),
the Live engine starts it at once through `MusicCoordinator.fast_start`. The
backend's own `play_music` call for the same title arrives a second or two
later and **joins** the request already under way (`begin` returns the active
request when the normalised title and artist match), so the keys are pressed
once and the tool answers "Playing … on Apple TV" from the same verification.
A fast start that finds nothing returns None and the backend's call runs the
ordinary way; anything with a destination it cannot place is left to the model.

## What changed

- `src/assistant/apple_catalog.py` — `AppleCatalog` (iTunes search/lookup),
  `AppleItem` (page url, key sequence, the cached item shape), `parse_uri`
  for Music Assistant's `apple_music(--instance)://` uris, `normalise`.
- `src/assistant/music.py` — the native path in `MusicCoordinator.play`
  (destination first, then wake and resolution side by side, native only when
  the TV has a remote and the item an Apple id), `already_playing`,
  `_play_native` with its verification, `prewake`, `fast_start`, the joining
  `begin(args)`, `parse_play_request` / `PlayIntent` / `PLAY_WORD`.
- `src/assistant/home/base.py` + `client.py` + `fake.py` — `MediaPlayer.remote_entity`
  / `app_id`, `HomeApi.launch_url`, `HomeApi.remote_commands`; the fake plays
  a title per page row so the tests can watch the whole sequence.
- `src/assistant/brain/tools.py` — `play_music` says "Playing X on Apple TV"
  only when the TV confirmed it, "already playing" when it was; pause, skip and
  volume go to the TV while its Music app plays (`_media_target`).
- `src/assistant/engines/live_engine.py` — `_maybe_prewake` on a mid-sentence
  "play", `_maybe_fast_start` when the turn ends, `first_call` stamped on the
  turn row so the backend's decision time is finally measured.
- Config: `MUSIC_NATIVE` (default on), `APPLE_STOREFRONT`, `MUSIC_NATIVE_READY_S`.
- Tests: `tests/test_native_music.py` (the route, the fallback, already
  playing, the cold wake, the join, the parser), a Live engine test for the
  hooks, the latency row.

## What still goes through Music Assistant

- Your own library playlists: Home Assistant's Music Assistant actions return
  them as `library://playlist/N` without the Apple `pl.u-…` id, so the Apple TV
  has nothing to open. MA plays them (with its startup cost) until we read the
  id from MA's own API.
- Queue edits (`enqueue: next/add`), radio mode, and any native start the TV
  did not confirm.
- Volume through the Apple TV entity returned OK but Music Assistant's mirror
  did not move; whether the receiver heard it is for Will's ears.

## The evening's corrections

- **One row off from the Now Playing screen.** Will's "Play American Girls
  by Harry Styles" fell through to Music Assistant twice while my probe of the
  same song landed in 2.2 s. The difference was his earlier "pause": from the
  Music app's Now Playing screen the album page opens with its focus one row
  off (Kiwi, track 7, played track 6). The coordinator now believes the TV and
  corrects: when a *different track of the same album* starts, one catalog
  lookup gives the album's rows and next/previous presses walk to the right
  one, then the title is verified again (`_skip_along_album`, stage
  `native_corrected`).
- **"uh" is not part of the title.** The fast-start parser strips hesitations
  ("play uh American Girls by, um, Harry Styles").
- **Every music request now logs its route**: `music tool: destination_resolved
  16 → … → native_playing 2891 ms · native ✓` on the console, so the log itself
  says where the seconds went.
- **Keys spaced under 0.1 s are dropped by tvOS** (0.05 s landed on the wrong
  track), so the measured 0.1 s stays.

## Voice acceptance

1. "Alexa, play Back in Black by AC/DC." Count from the end of the sentence
   to the first note. The turn row carries `first_call`; the tool result's
   `music_trace` carries `native_launched`, `native_keys_sent`, `native_playing`.
2. The same request again: "it's already on", nothing restarts.
3. "Play Hotel California by the Eagles" with the Apple TV asleep: the TV
   wakes on the word "play"; expect roughly five seconds, not eight.
4. "Find and play a relaxing jazz playlist on the Apple TV": one tool, native.
5. "Pause." / "Next." / "Turn it down." while the Music app plays: the TV
   entity takes them.
6. "Play my Emo playlist": still Music Assistant (library playlist), still slow.
