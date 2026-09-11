# How Alexa came to be

*Distilled from the two-day development conversation between Will and Claude
(Fable, in Claude Code) that built this project — 2026-08-30 and 31.*

## Day one — the plan and the brain (Aug 30)

Will, a C#/.NET engineer, pitched a DIY voice assistant with one thesis: no
canned routines — a frontier LLM gets the device list and tools and *reasons*
("get the room ready for a party"). Hard rules: no local LLM, local wake
word, milestone-at-a-time with testing between.

Research immediately reshaped the plan: Porcupine's free tier was dead
(→ openWakeWord), and Docker-on-Windows couldn't pass multicast (→ HAOS in a
Hyper-V VM, decided after Will commissioned an **adversarial review** that
tore into the plan — its fingerprints are everywhere: the half-duplex audio
design, honest cost math, batched room-aware tools, the failure contract).

By evening the **brain** worked: typed commands → Claude with tools → a fake
five-light apartment. First live words: "Hallway's off." Party mode chose
per-room colors unprompted and left the bedroom alone.

## Day two — voice, home, and hands (Aug 31)

Morning: the **ears** (openWakeWord on the Blue Snowball + Deepgram) — then a
strategic pivot: Will wanted OpenAI-grade conversational feel, and once the
custom-voice requirement was dropped, the engine became **gpt-realtime** with
our wake-gating and tools. The assistant was renamed Alexa (wake word
"alexa", voice Sol — org-gated, auto-falls-back to Marin until it unlocks).
Conversation was declared **co-equal with commands**: "Alexa isn't
conversational at all — that's the whole point of building this."

Then the real world arrived: a Matter bulb (Plant Light 1) shared from Apple
Home ran a color demo — Milestone 1, visually confirmed. The port-80 mystery
(new HAOS serves 80, not 8123) cost an hour and produced an upstream bug
report (home-assistant/operating-system#4993). Music Assistant brought
Spotify→AirPlay ("Coffee Table Jazz" was the first stream), followed by the
great **Spotify rate-limit siege** — resolved by patience, honesty fixes, and
Will's metric: *a song must play in under ten seconds or it's a failure*
(verdict benched, pending).

Afternoon, the capability cascade: **memory & preferences** ("movie time"),
the **generality upgrade** (search/read/act on anything in Home Assistant —
she answered a weather question with no weather tool), the **Learning Loop**
(session-end reflection turns fumbles into permanent lessons), **Always-On**
(a Windows service; she stopped being a terminal window), **self-awareness**
(she reads her own git log and roadmap), and finally **hands**: dispatch.

Her first self-commission ran in 30 seconds and wrote `docs/BIRTH.md`:
*"requested by the assistant, written by an agent, reviewed by a human.
Small note, big day."*

## The day GPT-Live landed (Sep 11)

OpenAI put `gpt-live-1` in the API on the 10th; Will opened the 11th with
"big day today". A week of fighting a turn-based model behind a microphone
that also hears the speaker — tentative talk-over, dings, fragment guards,
"she's talking to herself" on the Echo Dot — became a different contract in
one morning: the model listens while it speaks and stops when he talks over
it; reasoning and every tool go to a backend it delegates to; the session
bills by the second, so the wake word stays the gate and the engine owns the
endings. A bare probe on the real microphone taught the two things no doc
said — her audio is a continuous stream, silence included, so "she is
talking" is energy, and only commentary makes her speak first — before the
engine was written as a subclass of the one that already knew the house.
Same day she was running on it (`docs/LIVE-2026-09-11.md`); Will's ear
decides the rest.

## Standing truths from the journey

- Will's field reports drove a dozen fixes; testing-in-the-room beats
  planning. The ⚙ tool lines exist so failures explain themselves.
- Every claim gets verified against live docs or the actual wire — training
  memory lied about Porcupine, HA fields, ports, and schemas.
- Costs: OpenAI realtime is the only metered bill (~$15–25/mo); everything
  Claude rides Will's Max subscription; wake-gating keeps idle at $0.
- The permanent boundary: Alexa commissions, humans merge.
