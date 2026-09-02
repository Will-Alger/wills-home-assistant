# Feature backlog — working document

The living list of everything the assistant should eventually do. Edit freely;
the README milestone table stays the short version. Statuses: `idea` →
`planned` (has a milestone) → `building` → `testing` → `done`.

**North star (Will, 2026-08-31):** conversational ability is co-equal with
command execution — "that's where the strength comes into play for a custom
solution; Alexa isn't conversational at all." Any tradeoff that makes it a
better command box but a worse conversation partner is the wrong tradeoff.

## Checkpoint 2026-09-02 — where things stand, what's on deck

**Done:** the self-improvement loop (Phases 1–4: announcements, task board, staging +
`switch_build`, revisions via `--resume`, detached builds that survive restarts), web
search, Apple Music, calendar (read/create/delete), earcons, one-shot auto-close. She
runs `main` via the watchdog. Board: task 1 (birth certificate) built and unmerged —
the ideal `approve_task` test; task 5 (staging drill) built — the ideal `switch_build`
test. **Not yet voice-tested by Will: Phases 2–4.** Quiet hours are disabled in `.env`
for late-night development — restore `ANNOUNCE_QUIET_HOURS=23:00-08:00`.

**On deck, in order:**
0. Drive the loop end to end by voice with a real feature (greet-by-name, a timer):
   draft → build → "switch to task N" → revise → "ship it". First real use will
   surface friction no test can.
1. Phase 5 polish: announcement history ("what did you tell me this morning?"), board
   search ranking, cloud dispatch → watch-mode only, README/runbook, wake-time status
   line mentions pending approvals.
2. **Brain layer** (true-jarvis pattern): a `think(question)` tool → Opus via `claude -p`
   (Max-billed) with memory + board context, answer injected mid-conversation after
   "let me think about that". ~1 day now that injection exists; ideal first
   self-commission.
3. Event reactivity: HA event subscription → announcement queue ("when the door opens
   after 11pm, tell me"); needs Eve Door/Motion sensors (Thread via the Apple TV).
4. Satellites = Home Assistant Voice PE pucks + an audio-stream bridge (replaces the
   Pi plan; a June-2026 project proved gpt-realtime on Voice PE).
5. Longer arcs: music stopwatch verdict on Apple Music, playlist authoring, YouTube
   deep-links, calendar RESCHEDULE (update tool), voice ID, Telegram/webhook front
   door, HA MCP server eval, scoping her to a dedicated calendar for privacy.

**Engineering rules learned the hard way (2026-09-02):** she runs windowless, so every
subprocess must pass `CREATE_NO_WINDOW`; never `DETACHED_PROCESS` (it opens a visible
terminal per child); run dev commands from a shell that has a console (PowerShell),
not a console-less one; `tests/conftest.py` hides test subprocesses.

## Hardware reality (governs sequencing)

**Now:** a USB microphone + the desktop and/or a laptop. Everything through
milestone 7 and milestone 9 runs on exactly this — no purchases required.

**Later, only if the assistant earns it:** Raspberry Pi(s) as wake-word
satellites around the apartment. Note the satellite architecture (M8) can be
**prototyped with zero new hardware**: desktop = brain + satellite #1,
laptop = satellite #2. If closest-responds arbitration works between two
rooms, the Pis are just cheaper, smaller copies of a proven thing.

The brain should eventually live on the desktop (always on, and it's where
Claude Code dispatch runs anyway); pipeline code stays host-agnostic.

**Device buying rule (Will, 2026-08-31 — staying in the Apple ecosystem):**
buy HomeKit-compatible devices that are also **Matter-certified**. Matter
multi-admin means: pair to Apple Home first (normal Apple experience), then
share to Home Assistant (30 s) — both control it, nothing is given up. Apple
Home stays the household's face; HA is invisible plumbing that gives our
assistant its API. Non-Matter oddballs (e.g. WiZ) go the reverse way via
HA's HomeKit Bridge into Apple Home. Thread devices need a border router —
an Apple TV 4K / HomePod counts. Apple Home itself has no API; this is the
closest legitimate thing to "just pair it to Apple Home."

## Core pipeline

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Control lights via HA | **DONE 2026-08-31**: HAOS 18.2 in Hyper-V VM (Wi-Fi-bridged eth0 at 192.168.1.114 + host-only eth1 at 172.28.144.50 as backup lane), first Matter bulb (light.living_room_plant_light_1, shared from Apple Home) ran the full color demo, visually confirmed. Gotchas recorded in README: port 80 on new HAOS, use IP not .local, post-onboarding update downtime | 1 | $0 | **done** |
| LLM brain, no canned routines | Claude tool use (`claude-opus-5`, effort low), swappable provider interface; batched area-aware `set_lights`; end-of-turn intent via structured output; refusal fallbacks on | 2 | ~1–3¢/command (meter will tell) | testing |
| Cost meter, caching, low effort | per-hop cost + latency, `.usage.jsonl` log, session totals in REPL; static prefix cached (1 h TTL), live state via tool | 2 | saves money | testing |
| Audio-reality spike | `scripts/m3_spike.py`: device list, live score monitor (detections/hour stats), wav capture. First datapoint (2026-08-31, Blue Snowball, quiet room, 45 s): 0 false accepts, noise floor 0.01. Will runs the TV/music + say-it-10× protocol | 2.5 | $0 | testing |
| Wake-word activation | openWakeWord 0.6 via ONNX, live-verified on the desktop (tflite-runtime dep overridden — no py3.12 wheels; revisit at Pi) | 3 | $0 | testing |
| Streaming STT | Deepgram Flux via listen-v2 websocket (EndOfTurn events = native turn detection); swappable adapter. Needs Will's Deepgram signup | 3 | ~0.1¢/command ($200 credit first) | testing |
| Session state machine | `voice.py`: IDLE→wake→LISTENING→THINKING with intent-driven follow-up windows, single-flight, drain-before-listen, command audio retained (voice ID), earcons as the listening cue, loop survives any stage failure | 3 | $0 | testing |
| Voice engine bake-off | **BUILT + live-verified 2026-08-31** (`engines/realtime_engine.py`, `scripts/m4_realtime.py`): gpt-realtime-2.1 over websocket, semantic VAD, our HA tools bridged, `end_conversation` tool implements the close contract, wake-gated sessions (idle = $0), idle timeout, wake-phrase barge-in (half-duplex: no self-hearing on open speakers), per-response cost logging. Probes: hallway-off = 1 tool call, right entity, spoken reply, $0.012; party = 1 batched call, 4 entities, left the bedroom alone, $0.040. **GPT-Live (full-duplex) still NOT in API** — waitlist: openai.com/form/gpt-live-1-in-the-api; engine-seam swap when it lands. Will's favorite voice **Sol is org-gated** (API says "not available for your organization" — likely unlocks with GPT-Live access; marin/cedar meanwhile). `REALTIME_TALK_OVER=true` enables interrupt-by-speaking with headphones | 4 | ~1–4¢/exchange measured | testing |
| Pipeline mouth (ElevenLabs) | alternate engine + provider-independence/Pi fallback: Flash v2.5 streaming TTS over WebSocket. **Custom voice clone: Will demoted this to least-important** — parked until wanted (Starter $6/mo unlocks it) | 4b | $6/mo when used | idea |
| One-shot vs open conversation | one state machine, not two modes: the LLM ends each turn with an intent — `close` / `listen` / `confirm_close` ("anything else?" then waits) — deciding whether the mic re-opens without a wake word. Session = growing message history; tools available every turn. **Will's field feedback (2026-08-31)**: general chit-chat is a first-class use — prompt keeps flowing conversations on `listen` (only the speaker ends a live conversation); listen windows time out only until speech STARTS, then STT end-of-turn governs (no more mid-sentence cutoffs); `STT_EOT_THRESHOLD` tunes pause patience | 3–4 | in-session context growth, cache-absorbed | testing |
| Wake-word interrupt | while SPEAKING, full STT is off (half-duplex) but the wake detector keeps running — wake phrase mid-reply cuts TTS and returns to LISTENING; TTS is never allowed to say the wake phrase | 4 | $0 | planned |
| Barge-in (interrupt by just talking) | VAD-based open-mic interruption during playback — one extra transition in the M4 state machine, not a rewrite | 8 | $0 | idea |

## Smart home & media

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| "Party mode"-style reasoning | emergent from M2 tools — the whole point | 2 | — | planned |
| Matter bulbs (Linkind/AiDot) | unblocked by the HAOS VM: Matter Server add-on + share from Apple Home (multi-admin, no unpairing) — M1 stretch step | 1 | $0 | planned |
| Spotify → Apple TV speakers | **CORE DONE 2026-08-31**: Music Assistant add-on (Spotify provider + AirPlay player) + Apple TV integration; brain tools `play_music` (names, radio_mode, enqueue), `media_control` (pause/skip/volume/power), `launch_app` (TV apps). Live-verified: "find a good jazz playlist and play it on the apple tv" → Coffee Table Jazz streaming, both players `playing`. **Rate-limit architecture (verified in MA source 2026-08-31)**: with Will's dev key wired in, MA allows itself 45 req/30s on the dev token — but recommendations (`get_similar_tracks` → radio_mode + Endless Mix!), Spotify-owned/editorial playlists, category playlists, and Liked Songs are hardcoded to MA's SHARED built-in client id (throttled 1 req/2s, chronically 429'd with hour-long server Retry-After). A 429'd resolve sleeps ~1h inside `throttle_with_retries` HOLDING THE PLAYBACK LOCK — every later play waits 30s then dies quietly; only an add-on restart clears it. Rules: keep Endless Mix + Library Recommendations plugins DISABLED; prefer artists/tracks/own playlists (dev-key path, instant); radio_mode touches the shared key — use sparingly; NEVER retry into a timeout. **RESOLUTION 2026-08-31: Will switched the MA provider to APPLE MUSIC entirely** — its limiter backs off in seconds (1–2s, healthy), no shared-token architecture, track resolution verified ("Take Five" resolved from Apple's catalog). First ~15 min after adding the provider = initial library sync; streams starve (0 bytes buffered → skip → idle spinner) until it settles. Spotify provider benched with the shared-token pathology documented above | 5 | $0 | testing |
| YouTube search → play on Apple TV | YouTube Data API (free) for search; deep-link via pyatv/HA app launch — experimental; fallback: launch app + remote (launch_app already opens it) | 5 stretch | $0 | idea |
| Playlist authoring by voice | Will asked 2026-08-31: create_playlist / add_to_playlist / save-this-song tools via Music Assistant's own server API (port 8095 — richer than the HA integration; `music/playlists/add_playlist_tracks` etc.). "Add this to my chill playlist" works mid-song (MA knows current track). Verify at build: do new playlists sync into the Spotify app or live in MA's library (playable by voice either way) | 5b | $0 | idea |

## Assistant intelligence

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Web search ("google it") | Anthropic server-side `web_search` tool — declare it, zero code | 6 | ~$10/1k searches | planned |
| Apple Calendar events | **BUILT 2026-08-31, awaiting live credentials** (`src/assistant/calendar/`): Python `caldav` straight to iCloud (`https://caldav.icloud.com`, Apple ID + app-specific password) — NOT through HA, which would need config this repo doesn't own. Create-event support was the open question: it's real and generic — `Calendar.add_event(summary=…, dtstart=…, dtend=…)` builds the VEVENT via `caldav.lib.vcal.create_ical` and PUTs it (verified against the installed caldav 3.2 source). Two tools, offered only when `ICLOUD_USERNAME`/`ICLOUD_APP_PASSWORD` are set: `list_calendar_events` (default window: the next week, recurrences expanded) and `create_calendar_event` (date-only start = all-day; 60 min default length). Her instructions carry the current local date/time — a Realtime session has no clock — and require a spoken read-back before writing. iCloud reminder lists are filtered out of calendar choice. Tests stub only the HTTP transport, so the real ICS builder and mapping are exercised; live check: `uv run scripts/check_calendar.py --write-test` (creates a test event, reads it back, deletes it) | 6 | $0 | built, needs Will's app-specific password |
| Long-term memory + preferences | **BUILT + live-verified 2026-08-31** (`memory.py` + realtime engine): local JSON store (`data/memory.json`, gitignored) behind remember/list_memories/forget tools; preferences render into session instructions and refresh mid-session on change; facts recalled on demand. Verified cross-session: "from now on when I say movie time…" stored, then a fresh session's bare "movie time" set the living room warm at 20%. Deliberate-storage-only policy in instructions. "When X *happens*" event triggers remain a later HA-event-stream feature (she says so honestly) | 6 | pennies | testing |
| Voice ID | local speaker embeddings (SpeechBrain ECAPA), enroll Will once, cosine-match per command. Personalization, NOT security — this project literally clones voices | 7 | $0 | idea |
| Sensitive memories gated by voice ID | **re-scoped by review**: far-field 1–3 s speaker match has error rates too high to gate data, and our own TTS clone defeats it. Voice ID routes *preferences* (whose Spotify/calendar); nothing goes into memory that a houseguest shouldn't extract by voice; truly private recall would need a non-voice factor (phone push / button) | 7 | $0 | idea |

## Multi-device & desktop integration

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Satellites, closest one responds | client/server split; satellites report wake confidence + mic RMS in a ~1 s window, brain picks winner. **Will's target topology (2026-08-31): desktop brain + two Raspberry Pi satellites around the apartment.** Prototype desktop+laptop first. `show_me` (built) already routes displays to the desktop regardless of which mic hears | 8 | 2× Pi later | idea |
| Custom wake phrase | train openWakeWord model (Colab, ~1–2 h, synthetic speech) | 8 | $0 | idea |
| Voice-dispatch Claude Code — **the endgame, staged 2026-08-31** | **Stage 1 DONE + live-verified 2026-08-31**: `project_status` + `read_roadmap` tools — she accurately narrated her own git history and backlog, and states the can't-self-modify-yet boundary unprompted. **Stage 2 *Hands* DONE + live-verified 2026-08-31**: `develop_feature` (confirmed=true only after spoken approval) → headless Claude Code in a sandboxed worktree (Max-billed, hard timeout, commits to alexa/* branch, no push/merge) + `check_work`; first self-commission completed in 30s (docs/BIRTH.md on alexa/birth-certificate-4d550f, awaiting Will's first review/merge). Stage 3 *Initiative*: reflection friction-patterns become consent-gated dev proposals ("want me to build myself a blinds tool?"). Review constraints stand: dispatch never shares a web-content context. Prereq for 2: Always-On. First commission: "Alexa, build yourself the calendar integration" | 9 | $/task, metered | staged |
| Project status by voice | read-only tools: git log, `gh pr status`, running-session transcripts, summarized aloud | 9 | pennies | idea |
| Job lifecycle across days | **BUILT 2026-08-31**: jobs are open until explicitly closed. `close_work` archives a job Will considers dealt with (merge auto-closes); `check_work` reports open jobs by default (`include_closed` for the archive). Cloud jobs are fire-and-forget, so `check_work refresh=true` messages the live session (`claude -p --cloud <id>`, Max-billed) asking for one WORKING/DONE/BLOCKED line — runs in the background, answer stored on the job record (~1–2 min), a DONE reply flips status. Wake-time instructions carry an open-jobs snapshot ("done 6.2h ago, not yet closed") so she can lead with what finished while Will was away | 9 | Max-billed refresh | built |
| Announcements (self-improvement loop, Phase 1) | **BUILT 2026-09-01**: `announce.py` persisted queue (quiet hours, backoff, dedupe by ref, file inbox for other processes); the idle loop (`app.wait_for_trigger`) checks it every 80 ms; a due item opens a session where SHE speaks first (system item + response.create), chime through the session speaker, closes after the audio drains; mid-conversation items are injected when quiet. Sources: local job done/failed, `MILESTONE:` lines from the coding agent (max 3), cloud DONE. Coding agents now run on Opus (`DISPATCH_MODEL`). Next: Phase 2 task board | 9 | ~1¢/announcement | testing |
| Task board (self-improvement loop, Phase 2) | **BUILT 2026-09-02**: `tasks.py` TaskBoard — every request becomes a Task with a spec (`data/specs/<slug>.md`, committed into the branch as `docs/tasks/<slug>.md` = the agent's prompt target and a permanent record), states drafting→building→built→merged (+failed/abandoned; staged/revising land in Phases 3–4), iterations with agent summary/cost/session id, full history. Tools: draft_task, start_task (confirmed), list_tasks (today/yesterday/week/ISO windows), task_detail, search_tasks, approve_task (merge gates), abandon_task — replacing develop_feature/check_work/merge_work/close_work. `dispatch.py` is now a pure runner (worktrees, `claude -p --model opus` streaming, merge gates, cloud fire/refresh); old `data/jobs.json` imported once into `data/tasks.json`. Announcements: milestones, built, failed, cloud DONE | 9 | Max-billed | testing |
| Staging + switch_build (self-improvement loop, Phase 3) | **BUILT 2026-09-02**: `data/active_checkout.json` points the watchdog at a task's worktree; `alexa_service.py` launches from there (its own venv, or main's python + PYTHONPATH) with `ALEXA_HOME` so .env/data/logs stay on main (`config.home_dir()` vs `code_root()`). `switch_build(target)` — a task id or main — is free and reversible (uv-syncs the branch first); approve_task uv-syncs main before restarting onto it; abandon leaves staging. Watchdog rolls a staged build back after two quick crashes and drops an urgent announcement; `startup_maintenance` absorbs rollbacks, announces a staged start, and removes merged worktrees. She knows when she is staged (`{staged}` paragraph) | 9 | $0 | testing |
| Web search | **BUILT 2026-09-02**: `web.py` — OpenAI Responses API `web_search` tool on `WEB_SEARCH_MODEL` (gpt-4.1-mini: 4.4 s live; gpt-5-mini answered in 29 s, gpt-5-nano in 121 s) with the existing OPENAI_API_KEY; `web_search` voice tool answers in 1–3 spoken sentences + a source. Store hours, news, scores, facts | 6 | ~1¢/query | testing |
| Revisions + detached builds (self-improvement loop, Phase 4) | **BUILT 2026-09-02**: `revise_task(id, feedback)` appends a Revision section to the spec (data copy + branch copy, committed) and resumes the SAME agent session (`claude -p --resume`); a missing session falls back to a fresh build on the branch. Agents run in their own hidden console (CREATE_NEW_PROCESS_GROUP + CREATE_NO_WINDOW — never DETACHED_PROCESS, which opens visible terminals), output to `logs/tasks/<slug>-<n>.log`, pid recorded; the board tails the log (`Dispatcher.follow`) and re-attaches after a restart (`startup_maintenance`), so switching builds no longer kills a build. A run that dies after starting is resumed once after `DISPATCH_RESUME_DELAY_S`; a CLI that never reports a session fails within 90 s. Announcements: revision built, failed | 9 | Max-billed | testing |
| Phase 5 polish (overnight 2026-09-02) | **BUILT**: `announcement_history` tool (today / yesterday / N hours — "what did you tell me this morning?"), ranked task search (title hits first, recency tiebreak), wake-time status line leads with "N awaiting your approval", cloud dispatch reframed as opt-in watch mode, README runbook for the loop | 9 | $0 | testing |
| Brain layer (overnight 2026-09-02) | **BUILT**: `brain/thinker.py` + `think(question)` tool — the voice hands a hard question to Opus via `claude -p --model opus` (Max-billed, `BRAIN_MODEL`) with her preferences, facts, lessons, task board and the last 24 transcript lines as context; the tool returns at once ("thinking it over"), the answer is queued as an urgent `thought` announcement and delivered through the same system-item injection path — spoken mid-conversation when quiet, or at the next idle moment if the session ended. true-jarvis pattern: voice ↔ brain ↔ hands, one persona. Home tools stay on the voice | 9 | Max-billed | testing |
| Event reactivity (overnight 2026-09-02) | **BUILT**: `events.py` — `EventWatcher` keeps HA's websocket open (auth → subscribe_events state_changed; verified live on HA 2026.8.3; reconnects with backoff) and evaluates standing `Watch` rules deterministically (entity or fragment, to/from state, HH:MM window with overnight wrap, days, once/keep, normal/urgent); hits become announcements (quiet hours apply unless urgent). Voice: watch_for / list_watches / cancel_watch — "tell me when the front door opens after 11pm". Persisted in data/watches.json | 6 | $0 | testing |
| Scheduling & routines — NEW milestone (overnight 2026-09-02) | **BUILT**: `scheduler.py` — timers ("20 minute timer"), alarms (once or repeating on days; work/sleep; snooze; fire even in quiet hours), scheduled reminders and ACTIONS (any home tool at a time / after a delay / repeating: "porch light on at 6:30 every night"), persisted in data/schedule.json, 1 s tick loop in the app, results announced. `routines.py` — structured rules applied DETERMINISTICALLY in the tool executor: WHEN tool (+ match fields) AND time window/days THEN defaults (fill what the speaker left out) / overrides (always win): "after 5pm lights come on warm orange", "TV volume defaults to 65%"; listed in her instructions; tool results carry routines_applied. Voice: set_timer, set_alarm, schedule, list_schedule, cancel_schedule, snooze, add_routine, list_routines, remove_routine | 10 | $0 | testing |
| Multi-repo dispatch | **BUILT 2026-08-31**: `develop_feature` takes an optional `repo`; each claude.ai/code routine pins ONE GitHub repo, so other repos need one routine each, registered in gitignored `data/routines.json` (`{"repo-name": {"routine_id": "trig_…", "token": "sk-ant-oat01-…"}}`). Spoken-name matching normalizes spaces/hyphens. Other-repo jobs are cloud-only, land as branch/PR, can NEVER be voice-merged (merge gate is local+done only). Setup per repo: claude.ai/code → Routines → new routine on that repo, API trigger, Opus, generic instructions ("task from Will via his assistant; branch + PR; never merge"), paste id+token into routines.json | 9 | Max-billed | built, needs routine setup |

## Assistant intelligence — architecture

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Generality upgrade (escape hatch) | **BUILT + live-verified 2026-08-31**: `search_entities` (whole-home discovery, any domain), `get_entity` (full state/attrs), `ha_call_service` (any service, with an infrastructure denylist: no hassio/restart/shell/reload). Selection is description-driven: dedicated tools stay preferred, escape hatch declares itself last-resort. Live probe: "what's the weather?" — no weather tool exists — she searched, found the HAOS weather entity, read it, answered with real conditions. HA **MCP server** evaluation still pending as possible successor | done | $0 | testing |

| Learning Loop | **BUILT + live-verified 2026-08-31** (`learning.py`): after each voice conversation a Claude reflection pass distills the transcript into (a) **lessons** — operational recipes injected into all future instructions (live test produced: "no indoor temp sensors; use weather.forecast_home"); (b) **observations** — consent-gated: she asks before promoting to a preference (confirm→remember/forget flow wired into instructions); (c) **episodes** — journal, recallable via list_memories(kind=episode). Console shows "✎ learned: …" after sessions; deduped; lessons render capped at 15; failures never break the loop. **Reflection runs via `claude -p` = billed to Will's Max subscription, $0 API (USE_CLAUDE_SUBSCRIPTION, default on; API-key fallback kept)** — same mechanism dispatch (Stage 2) will use, so the whole Anthropic side of the endgame rides the $100/mo he already pays. Far arc (Will's dream: "talk to Alexa instead of you for development"): the dispatch milestone gives her hands — voice-commissioned Claude Code runs on this repo in a worktree, human reviews/merges | done | ~1-2¢/session | testing |

## Design notes already locked in

- Audio layer keeps the command audio buffer after STT (voice ID needs it) and
  is a clean frames-in/frames-out boundary (satellites slide a network under it).
- Money is only spent after the wake word — idle listening is local and free.
- No entity IDs or keys in code; everything via `.env` + runtime discovery.
- Budget: ~$20/month soft target (flexible). ElevenLabs sub is the fixed $6;
  the meter (M2) keeps the rest honest.

Added after the 2026-08-30 adversarial review:

- **Half-duplex is an M3/M4 requirement, not M8 polish**: full STT is
  suspended while TTS plays — but the wake detector stays hot, so the wake
  phrase mid-reply cuts TTS and interrupts (and TTS is never allowed to speak
  the wake phrase). HA media volume is ducked during capture. "Issue a
  command while music is playing" is an M5 acceptance test. (VAD-based
  barge-in — interrupting by just talking — stays M8.)
- **Session state machine (M3/M4)**: IDLE → wake → LISTENING → THINKING →
  SPEAKING → then the LLM's end-of-turn intent, a structured flag with three
  values: `close` (one-shot done → IDLE), `listen` (it asked a question — mic
  re-opens, no wake word), or `confirm_close` (it thinks it's done but checks
  — "anything else?" — mic re-opens briefly; explicit "no/that's all" or
  ~6 s silence → IDLE, anything else continues the conversation). "Let's
  chat" pins the session open. One-shot commands and open conversation are
  the same code path; session = growing message history; a soft earcon plays
  whenever the mic re-opens (doubles as the privacy "listening" cue). The
  intent flag is designed and tested in the M2 text REPL before audio exists.
- **Failure contract (M3/M4)**: per-stage timeout budget; a handful of
  pre-rendered local WAVs in the cloned voice ("the internet seems down") so
  the assistant can speak even when the cloud can't; single-flight rule — one
  command in progress, wake events during processing are dropped.
- **Cache design**: only the truly static prefix (system prompt + tool
  schemas + entity registry *without* live state) gets cached; live state is
  fetched via a tool. Live state in the prompt = byte-changed prefix = 0% hit
  rate at cache-write premium prices.
- **Latency design**: speak an instant canned ack ("on it") before the LLM
  round trip; tools are batched and area-aware (`set_lights(area=…, […])`),
  never one call per bulb; the M2 meter logs per-hop latency, not just tokens.
- **M1 done-criteria addition**: every entity gets a human name **and an
  area** in the HA UI — `GET /api/states` carries no room topology, so the
  client grows a registry/area fetch (WebSocket API) before M2, or "get the
  room ready" dims the bedroom too.
- **M2 ships an eval set**: ~10 golden typed commands with expected tool
  calls, so prompt/model tweaks aren't vibes-tested against live bulbs.
- **TTS character budget**: hard cap in the system prompt — ElevenLabs
  Starter's ceiling is ~60k Flash characters/month (~8–13 modest replies/day).
- **Appliance-ness (by M4)**: Task Scheduler at-logon launch + a watchdog;
  the assistant must survive Patch Tuesday unattended.
- **Privacy posture**: a mic mute affordance and audible/visible "listening"
  cue; set retention opt-outs at Deepgram/ElevenLabs; long-term memory stores
  owner-directed facts only (guests' chatter is not data); cloned voice is
  Will's own or has written consent.
- **Multi-user by default (Will, 2026-08-31)**: anyone in the room can use
  the assistant — the wake word is speaker-independent and nothing gates on
  identity. Voice ID (M7) only *personalizes* (whose Spotify/calendar/
  preferences); unknown voices get default behavior, and per-person
  preference profiles become possible on top of it. Owner name is config
  (`OWNER_NAME`), not hardcoded, so the repo works for any household.
- **"Proved itself" gate for hardware purchases**: the M4 loop used daily for
  a month plus one working media feature. M7+ is speculative until then.
  Once the platform question is settled, stand up HA Assist for an afternoon
  as an honest latency/behavior benchmark the DIY pipeline has to beat.

## Open questions

- **Platform — DECIDED 2026-08-30: HAOS in a Hyper-V VM** (Will delegated;
  review's #1 insisted change). `scripts/setup_haos_vm.ps1` creates it (Gen 2,
  Secure Boot off, bridged external switch, per official docs). Unblocks
  Matter (add-on), AirPlay/Music Assistant, all discovery; autostarts
  headless. `docker-compose.yml` stays as the Linux/Pi-era fallback.
- **Default command-path model — DECIDED 2026-08-30: Opus 5 everywhere to
  start** (Will's call: "probably fast enough"). The review's mitigations
  still apply and matter more, not less, on Opus: instant canned ack before
  the LLM round trip, batched area-aware tools, low effort on routine turns.
  The M2 meter logs per-hop latency + cost precisely so this decision gets
  re-examined with real data; the downshift is a one-line `.env` change.
- Whose voice to clone for TTS, and record the 1–2 min sample (Starter-tier instant clone).
- Wake phrase — stock "Hey Jarvis" until custom training day; pick the real phrase.
- Can pyatv deep-link a specific YouTube video, or only launch the app? (Settles M5 stretch scope.)
