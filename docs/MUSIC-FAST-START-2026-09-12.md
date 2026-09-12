# Music fast start — implementation and voice test

This build removes avoidable app work from music startup. Actual Siri parity
still requires a comparison on the same Apple TV and audio output.

## Behavior

- `play_music` remains compatible with existing exact names and URIs. It now
  accepts `selection="discover"` to search, select and play within one tool call.
  `fresh=true` bypasses a saved discovery choice and excludes that previous URI.
- Exact tracks get a bounded catalog search on a cache miss. The title, supplied
  artist and supplied album must match. Different recordings/artists prompt a
  clarification instead of silently choosing. A cache hit skips that lookup.
  Exact uncached playlist/album names retain MA's existing name-resolution path.
- The metadata cache is in memory, bounded to 512 entries with six-hour freshness,
  and shared with library browsing and the existing background vocabulary warmup.
  It retains provider/library IDs, never signed HTTP audio URLs. Restart clears it.
- A single destination snapshot and music resolution run concurrently. A sleeping
  TV is woken and checked for readiness with a five-second deadline; an already
  awake TV has no artificial delay. There is no fixed three-second sleep in the
  music path. HA reporting on/idle is a readiness proxy, not acoustic proof.
- With exactly one music player and one TV, an explicit TV name maps to its music
  player. More complex homes can set `MUSIC_DESTINATIONS` to a JSON mapping from
  spoken aliases to `player` (MA entity) and optional `power` (TV entity). A
  `default` entry handles omitted destinations. Ambiguous destinations ask;
  unrelated TVs are never inferred as the target.
- Queue `next`/`add` requests preserve their semantics and do not wake the TV.
  Radio expansion is opt-in. A new play request supersedes pending preparation;
  stop/pause cancels pending preparation even if no music is playing yet.
- Submissions are serialized. A request already sent is not cancelled by a new
  coordinator request or automatically retried after timeout. An explicit stop
  can cancel preparation; stopping an already submitted remote request still
  depends on the player and may race remote processing. Check queue state after
  an uncertain service result.
- Slow Live music tools receive at most one delayed commentary request for a
  brief acknowledgment. This is suppressed after intervening user turns, closing,
  cancellation, or while the assistant is speaking. The model may omit the
  acknowledgment if it already acknowledged the request.

`MUSIC_DESTINATIONS` is optional. Example shape, using placeholders to replace
with actual entity IDs:

```json
{"default":{"player":"media_player.YOUR_MA_PLAYER","power":"media_player.YOUR_TV"},"living room":{"player":"media_player.YOUR_MA_PLAYER","power":"media_player.YOUR_TV"}}
```

## Timing

Each music tool result carries `details.music_trace` with a request ID and
monotonic millisecond offsets for resolution, readiness and play submission/return.
The engine records that trace through its existing tap and action journal.
`submitted` says whether a remote mutation was attempted; `audible_verified` is
always false. “Playback requested” means the service returned, not that the room
has heard music. This build adds no acoustic recording or provider instrumentation.

Offline example (does not contact HA or load settings):

```powershell
uv run scripts/music_probe.py --repeat 2
```

The fake has 200 ms search and 200 ms readiness delays. They should overlap on the
first request; the second should show `cache_hit` and no search or wake delay.

Run a direct real-device probe only while ready for music to play. It bypasses
GPT-live and uses the usual HA configuration. From an isolated worktree, set
`ALEXA_HOME` to the main repository so configuration comes from the normal home.

```powershell
uv run scripts/music_probe.py --live --media-id "Take Five" --artist "Dave Brubeck"
uv run scripts/music_probe.py --live --media-id "jazz" --media-type playlist --selection discover
```

For an exact-URI test, substitute a URI returned by your MA search/library, not an
invented ID or a public Apple Music share page. `--repeat` deliberately issues
multiple play requests; default is one, and a failed request stops the run.

Compare the direct probe, the same request by voice, and Siri on the same output.
Record end-of-request → first audible music separately with a stopwatch or observer.
The probe's service-return timing must not be used as that audible measurement.
Separate TV awake, cold TV, replacement while playing, and resume after pause.

## Voice acceptance

1. “Play Take Five by Dave Brubeck.” Repeat it; the second request should use its ID.
2. “Find and play a jazz playlist on the Apple TV.” One music tool handles it.
3. “Something new this time.” The cached discovery choice is not reused.
4. Try a song shared by different artists without specifying the artist; clarify once.
5. Turn the TV off, then request music; preparation and lookup overlap.
6. Ask for a list of playlists; `browse_music` must not wake or play anything.
7. Cancel while search is pending; no delayed music starts from that request.
8. Replace a pending request with another song; only the pending latest request plays.
9. “Add it next.” The queue is edited without waking an idle TV.
10. Simulate search, wake and playback errors with fakes; no generic retry loop.

## Still requires hardware evidence

No MA version, buffer depth, provider configuration, AirPlay mode, or running
service was changed. The read-only investigation did not inspect `.env`, `data/`,
or `logs/`. Provider/receiver startup can still dominate even after the app gets
faster. If the direct exact-URI comparison is slow, follow the proposal's MA/AirPlay
investigation before declaring this feature comparable to Siri.

The implementation stays on its own branch for voice testing and review; merging
and switching the running service are separate from these source changes.
