# wills-home-assistant

A wake-word voice assistant with a frontier cloud LLM for a brain. The point —
and the difference from Alexa/Google — is that there are no canned routines:
the model gets the live device list and a set of tools, and *reasons* about
what to do. "Get the room ready for a party" should warm and dim the lights
and start music because the model decided that, not because anyone scripted it.

## Architecture

```
mic ──► wake word (LOCAL, openWakeWord — the only on-device inference)
          │ audio streams only after the wake phrase
          ▼
streaming STT ──── Deepgram Flux (cloud) ────────► transcript
          ▼
LLM + tools ────── Claude (default) or OpenAI ───► tool calls ──► Home Assistant
          ▼                                                        REST API ──► bulbs,
streaming TTS ──── ElevenLabs Flash v2.5 (custom cloned voice)     media, ...
          ▼
speaker
```

Every cloud stage sits behind a small interface (`stt/`, `llm/`, `tts/`) so
providers swap without touching the pipeline. Audio I/O is its own layer
(`audio/`) so the laptop's USB mic and the eventual Raspberry Pi are the same
code path. Hard constraints: **no local LLM**, wake word local, custom voice
via ElevenLabs.

### Why a modular pipeline and not a realtime speech-to-speech API?

Checked against current docs/benchmarks, 2026-08:

|                       | This pipeline                          | OpenAI Realtime (gpt-realtime-2.1)       | Gemini Live |
| --------------------- | -------------------------------------- | ---------------------------------------- | ----------- |
| Custom cloned voice   | ✅ any ElevenLabs voice                | ❌ 10 stock voices; "Custom Voices" is a sales-gated enterprise program | ❌ ~30 stock voices, no cloning |
| Voice-to-voice latency| ~0.7–1.2 s well tuned (~0.6 s floor)   | ~0.5–1.2 s measured time-to-first-audio  | ~0.6–1.0 s |
| Model choice          | any text LLM, per-request              | locked to gpt-realtime models            | locked to Gemini |
| Tool calling          | ✅ full Messages API tool use          | ✅ (incl. MCP)                           | ✅ (sequential) |
| Audio cost            | ~$0.01–0.03/min in components          | ~$0.06–0.11/min measured (with caching)  | ~$0.02/min |

The S2S latency edge has shrunk to a few hundred milliseconds, and the custom
voice requirement disqualifies both S2S APIs outright anyway. Anthropic has no
audio API at all — their own cookbook prescribes exactly this pipeline
(STT → Claude → ElevenLabs).

Two honest alternatives, if DIY ever stops being fun:
**ElevenLabs Agents** is the one hosted product that does cloned voice +
Claude + external tools/MCP end-to-end (~$0.08/min + LLM, ~1.7 s p50 measured
over telephony; you'd still need your own wake word).
**Home Assistant Assist** ships an Anthropic conversation agent with tool
calling and ElevenLabs TTS in core — the 80% shortcut, at the cost of owning
none of the pipeline.

## Milestones

| # | Deliverable | Status |
| - | ----------- | ------ |
| 1 | **Lights, no voice** — HAOS in a Hyper-V VM, bulbs controlled from Python via REST | ⏳ testing |
| 2 | **Brain, no audio** — text REPL → LLM with tools → HA ("get the room ready for a party", typed) | ⏳ testing |
| 3 | **Ears** — mic layer, openWakeWord, streaming STT; wake → transcript | ⏳ testing |
| 4 | **Mouth** — ElevenLabs streaming TTS; full voice loop | |
| 5 | **Music & media** — Spotify / Apple TV tools (likely via Music Assistant + AirPlay); stretch: YouTube search → play on Apple TV | |
| 6 | **Tool belt** — web search (Anthropic server-side tool), Apple Calendar events (iCloud CalDAV), long-term memory (local store behind remember/recall tools) | |
| 7 | **Voice ID** — local speaker embeddings (enroll Will's voiceprint); gates personal memories. Personalization, not security | |
| 8 | **Satellites & Pi** — client/server split (thin mic/speaker/wake satellites → central brain), multi-device wake arbitration (closest responds), barge-in, custom wake phrase, custom voice clone | |
| 9 | **Claude dispatch** — voice-launch Claude Code tasks on the desktop (Agent SDK, headless, own worktree + constrained permissions, voice-ID-gated) and voice status checks on projects/sessions | |

Per-feature detail, hardware plan, and open questions live in the working
backlog: [docs/FEATURES.md](docs/FEATURES.md).

## Setup

Prereqs: [uv](https://docs.astral.sh/uv/), git, Hyper-V (Windows 11 Pro).
uv for a .NET person: `uv sync` ≈ `dotnet restore` (also provisions the right
Python), `uv run x` ≈ `dotnet run` — you never activate a venv or touch the
system Python.

```powershell
uv sync
copy .env.example .env     # then edit .env — see comments in the file
```

Home Assistant runs as **HAOS in a Hyper-V VM** (bridged NIC = real LAN peer,
so mDNS discovery, Matter, and AirPlay all work — Docker Desktop on Windows
can't deliver multicast, which blocks all three; `docker-compose.yml` remains
only as a Linux/Pi-era fallback):

```powershell
# elevated PowerShell:
powershell -ExecutionPolicy Bypass -File scripts\setup_haos_vm.ps1
```

## Milestone 1 runbook: "HA sees and controls my lights"

1. **Onboard HA**: after the VM boots (first boot takes minutes), open
   <http://homeassistant.local:8123>, create the owner account, set
   location/timezone. Put that URL in `.env` as `HA_URL`.
2. **Add the WiZ bulbs**: with the bridged VM they should be auto-discovered
   (*Settings → Devices & services* shows them as "Discovered"). If not, add
   the WiZ integration manually with each bulb's IP (router DHCP list or the
   WiZ app; give them DHCP reservations either way).
3. **Matter bulbs (Linkind/AiDot)**: install the Matter Server add-on
   (*Settings → Add-ons*), then share the bulbs from Apple Home (they stay in
   Apple Home — Matter multi-admin): *Settings → Devices & services → Add
   integration → Matter*, choose "the device is already in use", follow the
   share-code flow from the iPhone. Treat this as a stretch step — WiZ alone
   completes M1.
4. **Name and place everything** (required, not cosmetic): give every light a
   human name and an **area** in the HA UI — the brain reasons in rooms, and
   the API exposes no room data for unassigned entities.
5. **Create the API token**: click your user (bottom-left) → *Security* tab →
   *Long-lived access tokens* → create one, paste into `.env` as `HA_TOKEN`
   (it's shown only once).
6. **Prove it from Python**:

   ```powershell
   uv run scripts/m1_smoke.py                        # health check + list lights
   uv run scripts/m1_smoke.py --entity light.<id> --demo   # red→green→blue→warm
   uv run scripts/m1_smoke.py --entity light.<id> --off
   ```

**Done when** the listing shows your bulbs with areas and `--demo` cycles one.

## Milestone 2 runbook: the brain (no audio)

Typed commands → LLM with tools → lights. Works two ways:

```powershell
uv run scripts/m2_repl.py --fake    # in-memory 5-light apartment; no HA needed
uv run scripts/m2_repl.py           # against your real Home Assistant (after M1)
```

Needs `ANTHROPIC_API_KEY` in `.env` (console.anthropic.com → API keys).
Try: "turn off the hallway", "make the living room cozy", "get the apartment
ready for a party", "is the bedroom light on?". Each reply prints the
end-of-turn intent (`close` / `listen` / `confirm_close` — what the mic will
do once audio exists) and a meter line: hops, latency, tokens, cache
warm/cold, cost for the command and the session (also logged to
`.usage.jsonl`). Notes: replies come through the API's refusal-fallback
routing so a safety decline degrades gracefully instead of dead-ending; run
the live evals with `$env:RUN_EVALS="1"; uv run pytest tests/test_evals.py`
(costs cents).

**Done when** party mode does something sensible to the fake apartment (and
later the real one), and the meter numbers look sane to you.

## Milestone 3 runbook: the ears

1. **Deepgram key**: sign up at <https://console.deepgram.com> (~$200 free
   credit, no card) → `DEEPGRAM_API_KEY` in `.env`.
2. **Audio-reality spike first** (~30 min, the review's must-do):

   ```powershell
   uv run scripts/m3_spike.py --devices      # Blue Snowball should be default
   uv run scripts/m3_spike.py --monitor 10   # quiet room: target ~0 detections
   uv run scripts/m3_spike.py --monitor 10   # again with Spotify/TV playing
   ```

   Then say "hey jarvis" ~10× from normal talking distance and count misses.
   Tune `WAKE_THRESHOLD` in `.env`: raise it for false wakes, lower for misses.
3. **The full loop** (wake → speak → lights → text reply; the voice out is M4):

   ```powershell
   uv run scripts/m3_voice.py --fake     # fake apartment
   uv run scripts/m3_voice.py            # real HA, after milestone 1
   ```

   Say "hey jarvis", wait for the rising beep, then talk. Intents drive the
   mic: a question re-opens it (beep) without the wake phrase; silence or
   "that's all" closes the conversation (falling beep).

**Done when** the wake word fires reliably for you and rarely for the TV, and
"hey jarvis, make the living room cozy" round-trips end to end.

## Layout

```
scripts/setup_haos_vm.ps1   creates the HAOS Hyper-V VM (elevated PowerShell)
docker-compose.yml          fallback: HA container for Linux/Pi hosts
src/assistant/
  config.py                 all settings/secrets from .env — nothing hardcoded
  home/                     HomeApi protocol, real HA client, fake apartment
  llm/                      provider interface + Anthropic adapter
  brain/                    agent loop, tools, system prompt, intents
  meter.py                  per-hop cost + latency, logs to .usage.jsonl
  stt/  tts/  audio/        (arrive with milestones 3–4)
scripts/m1_smoke.py         M1 proof: list + control lights, zero voice
scripts/m2_repl.py          M2 proof: typed commands → LLM tools → lights
tests/                      free stub tests + opt-in live evals (RUN_EVALS=1)
```

## Secrets

All keys live in `.env` (gitignored; `.env.example` is the documented
template). `ha-config/` is HA's own state — it contains auth tokens and never
gets committed. No entity IDs in code; they're discovered at runtime.
