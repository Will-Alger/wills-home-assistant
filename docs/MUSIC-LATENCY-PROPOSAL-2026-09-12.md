# Proposal: make music on the living-room TV respond like Siri

September 12, 2026 · wills-home-assistant · reviewed commit `d804577`

**Recommendation**

Build one coordinated music-request path that resolves the music and prepares the destination concurrently, starts playback without an intermediate model round trip, and measures time to audible music. Apply it to both exact songs and playlist discovery. First establish whether direct Music Assistant playback is fast enough; if that path is already slow, prioritize the provider/AirPlay layer alongside the app changes.

Will reports waits usually exceeding fifteen seconds, particularly when finding playlists, and confirms that specific known songs are also too slow. Siri streaming to the same living-room Apple TV feels substantially faster. That is the product benchmark. An earlier spoken acknowledgment improves feedback but does not satisfy the speed requirement.

This proposal replaces assumptions from the September 5 Realtime-era audit for this feature. The current app uses GPT-live with a delegated tool backend; music execution still goes through Home Assistant → Music Assistant → the streaming provider and AirPlay receiver. Source identifies Apple Music as the intended provider. Deployed provider versions, settings, and the precise TV/speaker topology remain unverified.

**What the investigation established**

| Finding | Evidence in current code | Consequence |
| --- | --- | --- |
| A fixed three-second wait precedes playback when a TV appears asleep | `brain/tools.py:906`, `_wake_tv_if_off`: `await asyncio.sleep(3)` | A guaranteed application delay in that branch, even if the device is ready sooner |
| Each play discovers media players twice, sequentially | `_dispatch` → `_resolve_player`, then `_wake_tv_if_off`; each invokes `home.media_players()` → `/api/states` | Two full-state reads before playback; durations are not measured here |
| Playlist discovery requires a search tool result to return to the backend before it chooses and invokes play | `brain/tools.py:106`; inherited music instructions in `engines/realtime_engine.py:162`; `live_engine.py:733–837` | Extra model work on the path to first music |
| Specific songs can still require provider name resolution | `home/client.py:190` sends the supplied name or URI to MA | Knowing the song as a user does not mean the app already holds its playable ID |
| The library vocabulary cache retains names, not URIs | `vocabulary.py`, `MusicNames._names` | Existing background refresh cannot currently serve a resolved-song/playlist playback cache |
| Open-ended requests are encouraged to use radio mode | `play_music` schema and inherited instructions | Recommendation expansion is another possible startup cost; a playlist request should not need it |
| A timeout triggers more library work and contradictory retry advice | `_dispatch` catches every play exception and calls `_media_hint`; `home/client.py:217` says not to retry | Failure can take longer and encourage duplicate playback attempts |
| Explicit player matches bypass the requested kind | `_resolve_player` accepts an exact TV entity even when called with `kind="music"` | “On the Apple TV” can resolve to the remote-control entity instead of its MA music entity; real failure depends on the installation |
| Success is returned before audibility is established | `Started ... (audio may take a few seconds to begin)` | Tool completion and first assistant speech are unsuitable end-to-end music metrics |
| The client already reuses one `httpx.AsyncClient` | `home/client.py:49` | Connection pooling is already present; adding it is not a new optimization |

The wake helper also counts off/unavailable TVs, not all TVs, and does not bind the wake to the selected music player. An unrelated sleeping TV can therefore be woken. Fix the routing relationship while changing wake timing.

**Offline verification performed**

Ran the real `ToolExecutor` against an instrumented `FakeHome`, without live services, configuration secrets, or recordings:

| Probe | Observation |
| --- | --- |
| TV already awake | Two media-player lookups before play; fake execution about 0.0001 seconds |
| TV asleep | Two lookups, `turn_on`, then music submitted at 3.0005 seconds |
| Injected timeout exception, with no real thirty-second wait | An extra library lookup and both “DO NOT retry” and “retry play_music” in the same result |
| Explicit fake TV entity as destination | That TV entity was passed directly to music playback |

The near-zero warm result reflects fake devices, not actual playback speed. The cold probe isolates the hardcoded delay; removing it can eliminate up to three seconds of artificial waiting but cannot remove genuine device wake time.

Existing relevant tests: **6 passed, 13 deselected**, using `tests/test_agent.py -k 'music or playback or launch_app'`. These verify existing fake-house behaviors, not hardware latency. No application code or running services were changed. In keeping with `CLAUDE.md`, `.env`, `data/`, and `logs/` were not read. This investigation therefore identifies confirmed software costs and proposes measurements; it does not claim to have apportioned the reported fifteen-plus seconds.

**The path we should shorten**

Typical current playlist request:

```text
Request ends
  → Live delegates
  → backend chooses browse_music
  → MA catalog search
  → backend selects result and calls play_music
  → discover destination
  → discover TV again
  → wake TV + fixed 3 s wait, when applicable
  → MA resolves playable content and establishes stream
  → receiver buffers / TV audio path becomes ready
  → audible music
```

An exact-song request can skip the explicit browse call, but retains delegation, destination discovery, wake handling, provider resolution and stream startup. This explains why a playlist-only optimization would leave part of Will's complaint unresolved.

Proposed request:

```text
Request ends → Live delegates → one start_music call
                                ├─ resolve exact cached ID or bounded search ─┐
                                └─ resolve destination + prepare if needed ─┤
                                                                           ↓
                                                            submit exact URI once
                                                                           ↓
                                                               audible music
```

For independent resolution and preparation, the ideal preparation portion changes from `Tresolve + Tprepare` to `max(Tresolve, Tprepare)`. Its possible saving is their overlap, not a guaranteed number of seconds. Transport and provider startup remain afterward. Do not add overlapping savings twice when estimating total improvement.

**1. Establish a benchmark that separates the bottlenecks**

Measure from the end of the request to the first audible music. Also record wake-to-ready separately, so a cold Live session cannot be confused with slow music resolution. Use the same song version or playlist, receiver, sound output, volume, network conditions and initial TV state across comparisons. Exclude tracks with long silent introductions or account for that silence.

| Test route | What it isolates |
| --- | --- |
| Siri/HomePod → the same Apple TV/audio output | User's actual reference experience; record the observed source/route if available |
| MA UI → already selected exact track | Provider + MA queue/stream + receiver startup with no app/model lookup |
| HA service → exact MA/provider URI | Adds HA orchestration to that path |
| Current assistant → exact song | Adds Live/backend and app routing; compare with exact URI to expose name lookup |
| Current assistant → find and play a playlist | Adds discovery and selection |
| Direct MA → local test audio, if already available | Separates Apple Music content preparation from the common AirPlay/output path |

Test awake-but-idle, already streaming/replacing, paused/resuming, and TV-asleep separately. Start with five alternating trials per important route to identify the bottleneck; collect at least twenty per priority condition for an acceptance comparison. Report median, p90, maximum, failures and wrong selections. Larger samples are needed for a reliable p95 claim.

Extend the existing `TurnTrace` and tool timing, rather than inventing another general telemetry system. Attach a music request ID and timestamps for delegation start, function ready, executor start, cache/search start and finish, destination resolved, wake sent/readiness observed, play submission/return, and matching player state/queue progression. Provider first bytes and AirPlay readiness belong in optional provider diagnostics where available. `playing` state is a proxy: confirm the final audible boundary with an observer timestamp or a short controlled recording. Never label first assistant audio as first music.

Decision rule: if direct exact-URI playback is already slow, start work on item 4 immediately. If it is fast but assistant playback is slow, prioritize items 2 and 3. Both may contribute; the three-second wait alone cannot explain the reported experience.

**2. Implement a fast route for known songs and destinations**

Extend the existing `play_music` internals or expose one `start_music` operation with a common coordinator. Keep old tools compatible for schedules and other callers.

- Store resolved metadata: canonical title, artist, media type, provider/account/storefront context, URI, aliases and last successful use. Extend background library loading without blocking wake or playback. Cache stable identifiers, not expiring audio URLs or credentials.
- For “play [song] by [artist],” use a high-confidence resolved match immediately. On a miss, do one focused track search or MA name-resolution call. Avoid a separate model browsing cycle when the request is already precise. Do not play the wrong artist/version merely to hit a latency target.
- Use an explicit destination binding: spoken “living-room Apple TV” → MA player/queue plus its corresponding Apple TV power entity. Resolve and validate once; share the snapshot across preparation. Maintain fresh state from the existing event connection where useful; the current `on_state` callback exposes state strings, so richer metadata tracking requires an extension.
- Replace the fixed sleep with readiness-driven preparation and a bounded fallback for unreliable state reporting. Run preparation alongside uncached music resolution after a clear play request. An already-ready destination must incur no artificial wait. Do not simply delete the delay without testing cold starts.
- Keep existing queue semantics: “resume” should resume the current queue, not resolve and restart the song. “Add next” and “queue” must not silently become immediate replacement.

MA's supported play action accepts media identifiers, including URIs, and HA's integration exposes both library lookup and catalog search. Those APIs support this design without replacing the streaming layer. [HA Music Assistant integration](https://www.home-assistant.io/integrations/music_assistant/), [MA play action](https://www.music-assistant.io/faq/massplaymedia/)

**3. Make playlist discovery one coordinated operation**

Use a tool such as `start_music(query, media_type, artist?, destination?, selection_policy, enqueue?)`. The backend interprets the request once; the coordinator performs search, destination preparation, selection and play. Keep `browse_music` for “show me options” and library questions, which should not wake the TV or start anything.

For “find me a relaxing jazz playlist and play it,” search a small candidate set while preparing the requested TV. Select using the requested style, provider availability, known preferences and a clear ranking policy. Start the selected playlist by URI without returning the candidate list to the model solely to choose the obvious result. If metadata is insufficient for meaningful constraints, ask or retain a deliberate model-selection step; a speed improvement must not degrade relevance. “Find something new” must bypass a cached mood favorite.

For familiar broad requests, reuse a previously successful, relevant choice where appropriate. Do not send unbounded library lists or collect metadata the first play does not require. Defer optional discovery/enrichment until music has started. Disable automatic radio expansion for a plain song/playlist start; only add it when requested or established as a preference, and assess whether the installed MA version supports adding it after startup.

Keep the Live receiver responsive—the new engine already runs tool execution in a separate task. Do not reintroduce the old receiver-blocking problem. Give the coordinator request ownership and a latest-request generation check immediately before play submission: “actually, play something else” must supersede pending search. A command already submitted has an uncertain side effect until reconciled; never blindly duplicate it.

The backend currently collects function-call items at `response.output_item.done` but executes them at `response.completed`. Measure that interval before considering earlier dispatch. Streaming tool execution is a later optimization requiring complete validated arguments, call-ID deduplication and verified Live delegation semantics. Likewise, measure backend time before changing models or global reasoning settings; current code deliberately raises low reasoning when native web search is enabled.

**4. Treat the remaining AirPlay/provider delay as its own engineering problem**

Record installed HA, MA, Apple Music provider and receiver firmware versions, then compare the installed implementation with its matching documentation. The current public MA documentation and development source describe newer AirPlay controls, but this investigation did not establish which are installed here.

Use MA's native AirPlay destination rather than inadvertently routing through an imported HA media-player wrapper. HA explicitly recommends native MA player providers where available. [HA integration guidance](https://www.home-assistant.io/integrations/music_assistant/)

Check supported pairing and wake controls, streaming mode, buffer depth, grouping and receiver readiness. Tune one setting at a time. Current MA docs expose automatic streaming mode and buffer-depth controls; deeper buffers trade responsiveness for stable playback. Synchronization offset is for relative alignment and should not be treated as a general startup-speed knob. [MA AirPlay settings](https://www.music-assistant.io/player-support/airplay/)

MA's development AirPlay documentation describes reusing connections across some warm transitions and event-driven start readiness. If the installed version supports that behavior, prefer supported resume/replacement paths that preserve it. Its reported start lead covers only part of startup; it is not an end-to-end latency promise. The source also explains why draining queued audio can delay pause/track changes. Do not prescribe development-branch installation or minimum buffers on that evidence alone. [MA AirPlay implementation](https://github.com/music-assistant/server/blob/dev/music_assistant/providers/airplay/README.md)

Inspect provider resolution separately. MA documents limitations in its Apple Music playback implementation, including possible gaps between tracks. That does not prove the cause of initial playback delay, but warrants testing a local audio source against the same receiver. [MA Apple Music provider](https://www.music-assistant.io/music-providers/apple-music/)

If transport remains slow, compare supported AirPlay modes on this receiver, a single-player route versus existing grouping, and wired versus current networking where available. Do not buy hardware or keep the TV awake indefinitely without evidence. A short supported warm-session policy may help repeated commands, but must release resources and allow normal TV sleep; streaming silence forever is not the default proposal.

**5. Fix the silence and failure behavior alongside the speed**

The current “never narrate mechanics” prompt is sensible for fast commands, but music needs bounded feedback when it is genuinely delayed. Allow one short acknowledgment after roughly one second if music has not begun—“Finding some jazz” or “Starting that now”—and let conversation continue. Use the coordinator's state to supply an actionable update if a stage stalls. Avoid repeated filler, a list of candidates the user did not request, or “playing” before confirmation. This feedback does not count toward the latency success metric.

Represent `resolving`, `preparing`, `starting`, `playing`, `failed`, `cancelled`, and `superseded` distinctly. Tie updates to the current request so an old result cannot announce success for a newer request. The music job should survive ordinary voice-session closure and reconcile its final outcome through the existing delivery mechanism.

Replace generic exception recovery with typed outcomes. A not-found result can get a cached name suggestion; a timeout, unavailable player or provider backoff must not trigger another live library lookup and retry instruction. The current thirty-second exception text speculates about rate limiting/sync; say what is known and inspect state before attributing a cause. A timeout is not proof the music never started. Correlate queue/item state before retrying or compensating.

A search timeout may permit a previously suitable cached choice for an open-ended request, if clearly conveyed. An exact-song request must not quietly substitute unrelated music. Reducing the thirty- or forty-five-second HTTP timeout alone would produce quicker failures, not quicker music.

**When to change architecture**

| Option | Recommendation | Reason |
| --- | --- | --- |
| Keep HA + MA; optimize request coordination, IDs and readiness | First choice | Removes confirmed app overhead while preserving working integrations |
| Direct authenticated MA API/client | Conditional follow-up | Can simplify richer queue/state control and remove the HA hop, but does not bypass provider preparation or AirPlay buffering; justify with measurements |
| Native Apple playback bridge | Bounded feasibility experiment if optimized MA still misses the benchmark | Could offer a different content/playback route, but unattended exact-song and arbitrary-playlist control must be demonstrated before proposing migration |
| Replace everything with direct `pyatv` calls | Not justified by current evidence | Documented remote control, app launching and file/URL streaming do not establish a ready replacement for authenticated Apple Music catalog playback |

MA publishes an authenticated API with instance-specific documentation, so a direct adapter is feasible to investigate against the deployed version. [MA API](https://www.music-assistant.io/api/)

The native Apple option needs a real proof of operation on available hardware. Do not assume a public Apple Music share URL is a playable audio URL, or that launching Music selects and plays the requested item. The documented pyatv capabilities support controls and streaming, but do not establish this particular end-to-end replacement. [pyatv supported features](https://pyatv.dev/documentation/supported_features/)

**Success criteria and stop/go gates**

Primary acceptance: on the same receiver and initial state, assistant music-start latency should be comparable to the measured Siri benchmark. A provisional tolerance is median within one second and p90 within two seconds of Siri; confirm that this difference is actually acceptable by listening. Track exact-song and discovery cases separately, and count failures and wrong selections rather than dropping them from statistics.

Provisional internal targets for an awake destination: known-song start median at most three seconds, playlist discovery median at most five seconds; both p90 at most eight seconds. These are engineering targets, not measured capability or promises. Passing them is insufficient if Siri remains materially faster. Cold-TV results must include genuine wake/HDMI/audio readiness and be compared with Siri starting from the same state.

| Gate | Required evidence |
| --- | --- |
| Before implementation | Baseline exact URI, exact spoken song, discovery, and Siri on the same output; identify dominant stage |
| Coordinator ready | No fixed wait on the ready path; shared destination lookup; overlap proven in fake delayed search/wake tests; one play submission |
| Correctness ready | Exact title/artist preserved; ambiguity handled; targeted TV only; timeout never instructs blind retry; superseded search cannot play late |
| Hardware ready | Audible timing, reliable cold starts, acceptable pause/resume and no dropouts after tuning |
| Ship | Meets the Siri-relative target for everyday requests; otherwise continue with the dominant provider/transport issue or evaluate the native Apple route |

Regression cases: familiar song, uncached exact song, familiar playlist, new mood playlist, “find options” without playback, awake TV, asleep TV, unavailable TV, multiple destinations, paused queue, replace current track, cancel during search, change request during wake, service timeout followed by late playback, and provider unavailable. Include silence duration as a separate UX measure.

**Proposed work packages**

1. **Measurement and immediate correctness fixes:** add music-stage traces and the benchmark harness; fix destination binding and timeout recovery. Capture a baseline before modifying behavior. Rough effort: half to one development day, plus hardware trials.
2. **Fast known-song and discovery coordinator:** share destination resolution, overlap readiness and content lookup, retain resolved IDs, support one-call selection/play and cancellation. Rough effort: two to four days including meaningful fake-service tests; exact scope depends on MA readiness signals.
3. **Provider/AirPlay tuning:** identify deployed versions, compare direct URI startup, test supported session reuse/modes/buffering and source preparation. Rough effort: half to two days of controlled experiments, with further work determined by results.
4. **Alternative route only if needed:** a timeboxed native Apple playback or direct-MA experiment judged by the same audible benchmark, before committing to a migration.

Application implementation would touch `brain/tools.py`, a proposed `music.py` coordinator, `home/client.py` and `home/base.py`, `vocabulary.py` or a shared catalog cache, `latency.py`, the existing event integration, and music-specific prompt/tool descriptions inherited by Live. The Live conversation engine itself should change only where timing, feedback or request identity requires it. Run the repository's Ruff and pytest gates for implementation changes, then the controlled hardware comparison.

The first deliverable should demonstrate exactly where the reported wait is spent and remove the confirmed artificial work. The final deliverable is audible music starting as promptly as the Siri experience Will already has.
