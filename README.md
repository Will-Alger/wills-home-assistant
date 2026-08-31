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
| 1 | **Lights, no voice** — HA running in Docker, bulbs controlled from Python via REST | ⏳ testing |
| 2 | **Brain, no audio** — text REPL → LLM with tools → HA ("get the room ready for a party", typed) | |
| 3 | **Ears** — mic layer, openWakeWord, streaming STT; wake → transcript | |
| 4 | **Mouth** — ElevenLabs streaming TTS; full voice loop | |
| 5 | **Music & media** — Spotify / Apple TV tools (likely via Music Assistant + AirPlay); stretch: YouTube search → play on Apple TV | |
| 6 | **Tool belt** — web search (Anthropic server-side tool), Apple Calendar events (iCloud CalDAV), long-term memory (local store behind remember/recall tools) | |
| 7 | **Voice ID** — local speaker embeddings (enroll Will's voiceprint); gates personal memories. Personalization, not security | |
| 8 | **Satellites & Pi** — client/server split (thin mic/speaker/wake satellites → central brain), multi-device wake arbitration (closest responds), barge-in, custom wake phrase, custom voice clone | |
| 9 | **Claude dispatch** — voice-launch Claude Code tasks on the desktop (Agent SDK, headless, own worktree + constrained permissions, voice-ID-gated) and voice status checks on projects/sessions | |

Per-feature detail, hardware plan, and open questions live in the working
backlog: [docs/FEATURES.md](docs/FEATURES.md).

## Setup

Prereqs: Docker Desktop (running), [uv](https://docs.astral.sh/uv/), git.
uv for a .NET person: `uv sync` ≈ `dotnet restore` (also provisions the right
Python), `uv run x` ≈ `dotnet run` — you never activate a venv or touch the
system Python.

```powershell
uv sync
copy .env.example .env     # then edit .env — see comments in the file
docker compose up -d       # Home Assistant on http://localhost:8123
```

## Milestone 1 runbook: "HA sees and controls my lights"

1. **Onboard HA**: open <http://localhost:8123>, create the owner account, set
   location/timezone.
2. **Add the WiZ bulbs** — auto-discovery does not work in Docker on Windows
   (the container never sees LAN broadcast/mDNS traffic), so add them by IP:
   find each bulb's IP in your router's DHCP client list or the WiZ app
   (device → settings), ideally give them DHCP reservations, then in HA:
   *Settings → Devices & services → Add integration → WiZ* → enter the IP.
3. **The Linkind/AiDot (Matter) bulbs are a known gap on Windows Docker**:
   joining a Matter fabric needs a Matter server with real host networking +
   IPv6 multicast, which Docker Desktop cannot provide. Options: run HAOS in a
   Hyper-V/VirtualBox VM with a bridged NIC now, or wait for the Raspberry Pi
   (where `network_mode: host` makes discovery, Matter, and AirPlay all work).
   Milestone 1 counts with WiZ only.
4. **Create the API token**: click your user (bottom-left) → *Security* tab →
   *Long-lived access tokens* → create one, paste into `.env` as `HA_TOKEN`
   (it's shown only once).
5. **Prove it from Python**:

   ```powershell
   uv run scripts/m1_smoke.py                        # health check + list lights
   uv run scripts/m1_smoke.py --entity light.<id> --demo   # red→green→blue→warm
   uv run scripts/m1_smoke.py --entity light.<id> --off
   ```

**Done when** the listing shows your bulbs and `--demo` visibly cycles one.

## Layout

```
docker-compose.yml       Home Assistant container (config in a named Docker volume)
src/assistant/
  config.py              all settings/secrets, loaded from .env — nothing hardcoded
  home/                  Home Assistant REST client
  llm/  stt/  tts/  audio/   (arrive with their milestones)
scripts/m1_smoke.py      milestone-1 proof: list + control lights, zero voice
```

## Secrets

All keys live in `.env` (gitignored; `.env.example` is the documented
template). `ha-config/` is HA's own state — it contains auth tokens and never
gets committed. No entity IDs in code; they're discovered at runtime.
