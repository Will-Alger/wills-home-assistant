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
| Wake-word activation | openWakeWord, local (Porcupine free tier is dead) | 3 | $0 | planned |
| Streaming STT | Deepgram Flux (semantic end-of-turn); swappable adapter | 3 | ~0.1¢/command ($200 credit first) | planned |
| Custom-voice TTS | ElevenLabs Flash v2.5 over WebSocket; Starter plan ($6/mo) unlocks instant clone | 4 | $6/mo flat | planned |
| Barge-in (interrupt while it talks) | duck TTS on wake/VAD during playback | 8 | $0 | idea |

## Smart home & media

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| "Party mode"-style reasoning | emergent from M2 tools — the whole point | 2 | — | planned |
| Matter bulbs (Linkind/AiDot) | blocked on Windows Docker; needs HAOS VM (Hyper-V, bridged NIC) or a future Pi host | TBD | $0 | idea |
| Spotify → Apple TV speakers | Music Assistant + AirPlay (stock Spotify integration can't start AirPlay playback); needs Premium + dev app | 5 | $0 | planned |
| YouTube search → play on Apple TV | YouTube Data API (free) for search; deep-link via pyatv/HA app launch — experimental; fallback: launch app + remote | 5 stretch | $0 | idea |

## Assistant intelligence

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Web search ("google it") | Anthropic server-side `web_search` tool — declare it, zero code | 6 | ~$10/1k searches | planned |
| Apple Calendar events | iCloud CalDAV + app-specific password (HA CalDAV or Python `caldav` — verify create-event support when building) | 6 | $0 | planned |
| Long-term memory | local SQLite behind remember/recall tools (Anthropic memory-tool shape) | 6 | pennies | planned |
| Voice ID | local speaker embeddings (SpeechBrain ECAPA), enroll Will once, cosine-match per command. Personalization, NOT security — this project literally clones voices | 7 | $0 | idea |
| Sensitive memories gated by voice ID | memory entries tagged private; recall tool checks speaker match | 7 | $0 | idea |

## Multi-device & desktop integration

| Feature | Approach | Milestone | Cost | Status |
| --- | --- | --- | --- | --- |
| Satellites, closest one responds | client/server split; satellites report wake confidence + mic RMS in a ~1 s window, brain picks winner. Prototype desktop+laptop before buying Pis | 8 | hardware later | idea |
| Custom wake phrase | train openWakeWord model (Colab, ~1–2 h, synthetic speech) | 8 | $0 | idea |
| Voice-dispatch Claude Code | `start_coding_task` tool → headless Claude Code / Agent SDK in a dedicated worktree with constrained permissions; voice-ID-gated | 9 | $/task — biggest spend item, metered | idea |
| Project status by voice | read-only tools: git log, `gh pr status`, running-session transcripts, summarized aloud | 9 | pennies | idea |

## Design notes already locked in

- Audio layer keeps the command audio buffer after STT (voice ID needs it) and
  is a clean frames-in/frames-out boundary (satellites slide a network under it).
- Money is only spent after the wake word — idle listening is local and free.
- No entity IDs or keys in code; everything via `.env` + runtime discovery.
- Budget: ~$20/month soft target (flexible). ElevenLabs sub is the fixed $6;
  the meter (M2) keeps the rest honest.

## Open questions

- Matter bulb path: HAOS VM on the desktop now, or park until Pi hardware? (Affects only those bulbs.)
- Whose voice to clone for TTS, and record the 1–2 min sample (Starter-tier instant clone).
- Wake phrase — stock "Hey Jarvis" until custom training day; pick the real phrase.
- Can pyatv deep-link a specific YouTube video, or only launch the app? (Settles M5 stretch scope.)
