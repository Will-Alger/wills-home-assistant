# Feature backlog — working document

The living list of everything the assistant should eventually do. Edit freely;
the README milestone table stays the short version. Statuses: `idea` →
`planned` (has a milestone) → `building` → `testing` → `done`.

**North star (Will, 2026-08-31):** conversational ability is co-equal with
command execution — "that's where the strength comes into play for a custom
solution; Alexa isn't conversational at all." Any tradeoff that makes it a
better command box but a worse conversation partner is the wrong tradeoff.

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
| Spotify → Apple TV speakers | **CORE DONE 2026-08-31**: Music Assistant add-on (Spotify provider + AirPlay player) + Apple TV integration; brain tools `play_music` (names, radio_mode, enqueue), `media_control` (pause/skip/volume/power), `launch_app` (TV apps). Live-verified: "find a good jazz playlist and play it on the apple tv" → Coffee Table Jazz streaming, both players `playing`. No Spotify dev app needed (MA handles auth) | 5 | $0 | testing |
| YouTube search → play on Apple TV | YouTube Data API (free) for search; deep-link via pyatv/HA app launch — experimental; fallback: launch app + remote (launch_app already opens it) | 5 stretch | $0 | idea |
| Playlist authoring by voice | Will asked 2026-08-31: create_playlist / add_to_playlist / save-this-song tools via Music Assistant's own server API (port 8095 — richer than the HA integration; `music/playlists/add_playlist_tracks` etc.). "Add this to my chill playlist" works mid-song (MA knows current track). Verify at build: do new playlists sync into the Spotify app or live in MA's library (playable by voice either way) | 5b | $0 | idea |

## Assistant intelligence

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Web search ("google it") | Anthropic server-side `web_search` tool — declare it, zero code | 6 | ~$10/1k searches | planned |
| Apple Calendar events | iCloud CalDAV + app-specific password (HA CalDAV or Python `caldav` — verify create-event support when building) | 6 | $0 | planned |
| Long-term memory + preferences | **BUILT + live-verified 2026-08-31** (`memory.py` + realtime engine): local JSON store (`data/memory.json`, gitignored) behind remember/list_memories/forget tools; preferences render into session instructions and refresh mid-session on change; facts recalled on demand. Verified cross-session: "from now on when I say movie time…" stored, then a fresh session's bare "movie time" set the living room warm at 20%. Deliberate-storage-only policy in instructions. "When X *happens*" event triggers remain a later HA-event-stream feature (she says so honestly) | 6 | pennies | testing |
| Voice ID | local speaker embeddings (SpeechBrain ECAPA), enroll Will once, cosine-match per command. Personalization, NOT security — this project literally clones voices | 7 | $0 | idea |
| Sensitive memories gated by voice ID | **re-scoped by review**: far-field 1–3 s speaker match has error rates too high to gate data, and our own TTS clone defeats it. Voice ID routes *preferences* (whose Spotify/calendar); nothing goes into memory that a houseguest shouldn't extract by voice; truly private recall would need a non-voice factor (phone push / button) | 7 | $0 | idea |

## Multi-device & desktop integration

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Satellites, closest one responds | client/server split; satellites report wake confidence + mic RMS in a ~1 s window, brain picks winner. Prototype desktop+laptop before buying Pis | 8 | hardware later | idea |
| Custom wake phrase | train openWakeWord model (Colab, ~1–2 h, synthetic speech) | 8 | $0 | idea |
| Voice-dispatch Claude Code | `start_coding_task` tool → headless Claude Code / Agent SDK in a dedicated worktree with constrained permissions. **Review constraints adopted:** dispatch tools never share a context that ingested web content (injection surface); confirmation is a non-voice factor (phone push / button), not voice ID; hard per-task spend cap; no push rights or secrets beyond the worktree. The read-only status half ships first | 9 | $/task — biggest spend item, metered | idea |
| Project status by voice | read-only tools: git log, `gh pr status`, running-session transcripts, summarized aloud | 9 | pennies | idea |

## Assistant intelligence — architecture

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Generality upgrade (escape hatch) | **BUILT + live-verified 2026-08-31**: `search_entities` (whole-home discovery, any domain), `get_entity` (full state/attrs), `ha_call_service` (any service, with an infrastructure denylist: no hassio/restart/shell/reload). Selection is description-driven: dedicated tools stay preferred, escape hatch declares itself last-resort. Live probe: "what's the weather?" — no weather tool exists — she searched, found the HAOS weather entity, read it, answered with real conditions. HA **MCP server** evaluation still pending as possible successor | done | $0 | testing |

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
