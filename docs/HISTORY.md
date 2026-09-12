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

## Music the way Siri does it (Sep 12)

Will stopped choosing and started grading: "when Tony Stark says 'daddy's
home', music starts in two or three seconds — the technical decisions are
yours, the criteria are mine." The twenty-eight seconds were traced to the
minute in Home Assistant's history and Music Assistant's own log: not the
network, not AirPlay (a second), but MA's Apple Music provider getting audio
out at all. So the route changed to the one Siri uses: Apple's public catalog
resolves the title in 300 ms, the Apple TV opens the music.apple.com page in
its own Music app, one remote command presses the keys (Down × track number,
then select), and only the TV's reported title counts as "playing". Warm,
under three seconds; "play *title* by *artist*" starts the moment the
sentence ends, before the backend has spoken, and the backend's own call
joins it (`docs/MUSIC-NATIVE-2026-09-12.md`).

The same evening the wake word got the same treatment. "Alexa play AC/DC"
in one breath had been answered with "Mm-hm?" and then nothing: her own
"Yes?" came back through the Echo Dot, the frames of that moment were
dropped as her echo, and his command went with them. Now the wake waits a
beat and answers only if he stopped — the microphone keeps a ring of the
room's last seconds and can say whether he kept talking past her name — and
the lines themselves are the two words Jarvis would use ("Yes, sir?", "Sir?",
"Mm-hm?"), rendered without the accent she had been trying on. And the end
tool learned its place: a one-shot command closes the conversation only if he
has nothing more to say — his next sentence, even over her last word,
withdraws the close.

Late that night Will stopped and asked the harder question: *"how do I get
to a point where the conversation feels seamless? I feel like I just keep
churning on little things."* The honest answer (`docs/MEASURABLE-2026-09-12.md`)
was that the churn was real and so was the progress, and that the way out was
to measure instead of remember: the engine's decisions now land on an
always-on timeline, every session keeps its words, anything that felt wrong
becomes a tagged item ("flag that: she cut me off") that groups into a need,
and a dashboard on localhost shows the five numbers that decide whether a
week was better than the last — cut-offs, silences, wake misses, acks, music
start. Next: fewer clocks, judged by those numbers; then the pucks.

## Standing truths from the journey

- Will's field reports drove a dozen fixes; testing-in-the-room beats
  planning. The ⚙ tool lines exist so failures explain themselves.
- Every claim gets verified against live docs or the actual wire — training
  memory lied about Porcupine, HA fields, ports, and schemas.
- Costs: OpenAI realtime is the only metered bill (~$15–25/mo); everything
  Claude rides Will's Max subscription; wake-gating keeps idle at $0.
- The permanent boundary: Alexa commissions, humans merge.
