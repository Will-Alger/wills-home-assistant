# Feature backlog — working document

The living list of everything the assistant should eventually do. Edit freely;
the README milestone table stays the short version. Statuses: `idea` →
`planned` (has a milestone) → `building` → `testing` → `done`.

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

## Core pipeline

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Control lights via HA | REST client + WiZ manual-IP setup | 1 | $0 | testing |
| LLM brain, no canned routines | Claude tool use (`claude-opus-5`), swappable to OpenAI | 2 | ~1–3¢/command | planned |
| Cost meter, caching, low effort | log usage tokens per command; prompt-cache system+tools+device list | 2 | saves money | planned |
| Audio-reality spike | record the real mic in the real room; measure openWakeWord false accepts/misses vs quiet / TV / Spotify **before** building M3 around it (review finding: this is where identical projects die) | 2.5 | $0 | planned |
| Wake-word activation | openWakeWord, local (Porcupine free tier is dead; project is dormant — we own any Windows/ONNX friction) | 3 | $0 | planned |
| Streaming STT | Deepgram Flux (semantic end-of-turn); swappable adapter | 3 | ~0.1¢/command ($200 credit first) | planned |
| Custom-voice TTS | ElevenLabs Flash v2.5 over WebSocket; Starter plan ($6/mo) unlocks instant clone | 4 | $6/mo flat | planned |
| One-shot vs open conversation | one state machine, not two modes: the LLM ends each turn with an intent — `close` / `listen` / `confirm_close` ("anything else?" then waits) — deciding whether the mic re-opens without a wake word. Session = growing message history (built in the M2 REPL first); tools available every turn, so commands work mid-chat | 4 | in-session context growth, cache-absorbed | planned |
| Wake-word interrupt | while SPEAKING, full STT is off (half-duplex) but the wake detector keeps running — wake phrase mid-reply cuts TTS and returns to LISTENING; TTS is never allowed to say the wake phrase | 4 | $0 | planned |
| Barge-in (interrupt by just talking) | VAD-based open-mic interruption during playback — one extra transition in the M4 state machine, not a rewrite | 8 | $0 | idea |

## Smart home & media

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| "Party mode"-style reasoning | emergent from M2 tools — the whole point | 2 | — | planned |
| Matter bulbs (Linkind/AiDot) | blocked on Windows Docker; needs HAOS VM (Hyper-V, bridged NIC) or a future Pi host | TBD | $0 | idea |
| Spotify → Apple TV speakers | Music Assistant + AirPlay (stock Spotify integration can't start AirPlay playback); needs Premium + dev app. ⚠ AirPlay discovery is mDNS — **blocked under Docker Desktop exactly like Matter**; hangs on the platform decision below | 5 | $0 | planned |
| YouTube search → play on Apple TV | YouTube Data API (free) for search; deep-link via pyatv/HA app launch — experimental; fallback: launch app + remote | 5 stretch | $0 | idea |

## Assistant intelligence

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Web search ("google it") | Anthropic server-side `web_search` tool — declare it, zero code | 6 | ~$10/1k searches | planned |
| Apple Calendar events | iCloud CalDAV + app-specific password (HA CalDAV or Python `caldav` — verify create-event support when building) | 6 | $0 | planned |
| Long-term memory | local SQLite behind remember/recall tools (Anthropic memory-tool shape) | 6 | pennies | planned |
| Voice ID | local speaker embeddings (SpeechBrain ECAPA), enroll Will once, cosine-match per command. Personalization, NOT security — this project literally clones voices | 7 | $0 | idea |
| Sensitive memories gated by voice ID | **re-scoped by review**: far-field 1–3 s speaker match has error rates too high to gate data, and our own TTS clone defeats it. Voice ID routes *preferences* (whose Spotify/calendar); nothing goes into memory that a houseguest shouldn't extract by voice; truly private recall would need a non-voice factor (phone push / button) | 7 | $0 | idea |

## Multi-device & desktop integration

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Satellites, closest one responds | client/server split; satellites report wake confidence + mic RMS in a ~1 s window, brain picks winner. Prototype desktop+laptop before buying Pis | 8 | hardware later | idea |
| Custom wake phrase | train openWakeWord model (Colab, ~1–2 h, synthetic speech) | 8 | $0 | idea |
| Voice-dispatch Claude Code | `start_coding_task` tool → headless Claude Code / Agent SDK in a dedicated worktree with constrained permissions. **Review constraints adopted:** dispatch tools never share a context that ingested web content (injection surface); confirmation is a non-voice factor (phone push / button), not voice ID; hard per-task spend cap; no push rights or secrets beyond the worktree. The read-only status half ships first | 9 | $/task — biggest spend item, metered | idea |
| Project status by voice | read-only tools: git log, `gh pr status`, running-session transcripts, summarized aloud | 9 | pennies | idea |

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
- **"Proved itself" gate for hardware purchases**: the M4 loop used daily for
  a month plus one working media feature. M7+ is speculative until then.
  Once the platform question is settled, stand up HA Assist for an afternoon
  as an honest latency/behavior benchmark the DIY pipeline has to beat.

## Open questions

- **Platform (decides M5's fate, not just the Matter bulbs) — awaiting Will's call:**
  HAOS in a Hyper-V VM now (review's #1 insisted change: unblocks Matter,
  AirPlay/Music Assistant, all discovery; autostarts headless and survives
  host reboots) vs Docker Desktop interim (fastest WiZ-only M1 today, but a
  guaranteed redo of HA onboarding/entities/token when M5 arrives).
- **Default command-path model — awaiting Will's call:** Opus 5 with default
  thinking on multi-hop commands means multi-second silences and most of the
  budget; review insists on Sonnet 5 (or Haiku 4.5) at low effort for the
  command path, with Opus behind an explicit escalation for open-ended asks.
- Whose voice to clone for TTS, and record the 1–2 min sample (Starter-tier instant clone).
- Wake phrase — stock "Hey Jarvis" until custom training day; pick the real phrase.
- Can pyatv deep-link a specific YouTube video, or only launch the app? (Settles M5 stretch scope.)
