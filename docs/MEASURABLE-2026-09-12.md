# A measurable assistant

September 12, 2026. Will: *"the system I want is going to be a measurable
system… before making any changes, ask yourself: are the tools we're using
limiting us? are we over-engineering to accommodate something? we'll be
putting multiple units around the apartment with the pucks."*

## The honest answers

**Are the tools limiting us?** In three places, yes.

- **The Settings panel is a Tk window.** Its "live log" is a 300-line ring of
  console text poured into a `tk.Text` with wrapping off, redrawn every
  0.4 s. Tk can wrap text (one flag), but "a real tool to manage logs and flag
  conversations, with tags and ticket statuses" is a data UI, and the only
  kind that also works from a phone or a second unit is a web page. The
  runner now serves one on localhost; the Tk window keeps the device pickers
  and gains a button to open it.
- **The logs were written for eyes, not for questions.** `logs/alexa.log` is a
  terminal transcript hard-wrapped at 78 columns (that is the "long text goes
  off the screen"). `data/sessions.json` kept fifty rows with no transcript.
  `logs/turns.jsonl` has numbers with no words. The engine's decision taps
  (`engine.tap`) only went anywhere while a manual recording ran. A measurable
  system needs one always-on, structured, per-session stream: every decision,
  every line said, every tool with its trace, keyed by session — no audio.
  That is `logs/timeline.jsonl` now, and session rows keep their transcript.
- **The wake path is a desk microphone doing a far-field job.** openWakeWord's
  pretrained "alexa" on a Snowball, a Bluetooth Echo Dot playing her back
  louder than him, no echo cancellation anywhere. The pucks (far-field mic
  arrays, wake word and AEC on the device) are the real fix; the desktop rig
  is the training bench until then.

Not limiting: GPT-Live (its turn-taking is better than our clocks), Home
Assistant as the device layer, the task board as the ticket system (it already
has states, specs and agents — feedback becomes needs, needs become tasks).

**Are we over-engineering to accommodate something?** Yes, two things.

- **The acoustics.** Gated/duplex echo policy, coupling calibration, the
  ack-echo drop, suspect windows, `_SUSPECT_LEVEL` — all of it exists because
  her voice reaches the microphone louder than his. A puck with hardware AEC
  deletes that whole class. It stays minimal until then and is not worth
  another evening of tuning.
- **The endings.** Eight clocks decide when a conversation is over: command
  quick-close, question quick-close, wrap-up grace, farewell cap, continuation
  window, thinking freeze, false-wake watch, idle timeout (plus the session
  cap). Each is an edge where she can be wrong, and tonight's cut-off was one.
  GPT-Live plus the backend's end tool already decide endings. Target: his
  wrap-up or stop; her end tool after her last word, unless he speaks; one idle
  timeout. Keep the two that are about money (session cap, false-wake watch).

**Multiple units.** The boundary is ears-and-mouth (a satellite: microphone,
speaker, wake word, AEC, audio streamed both ways) versus the brain (one
process: the Live session, tools, memory, this dashboard). The engine already
talks to abstract audio sources and speakers; the session rows and the
timeline carry a `unit` from today so nothing has to be re-keyed later. Home
Assistant's Voice pucks speak Wyoming to an Assist pipeline; making her brain
that pipeline's endpoint is its own milestone, and nothing built here gets in
its way.

## What "measurable" means here

Five numbers, per day, from the timeline and the turn log, on the dashboard:

| Number | Definition |
| --- | --- |
| Cut-offs | sessions that ended within 2 s of his last words, unless he said the ending himself |
| Silences | his turns with no voice from her within 3 s |
| Wake misses | near-miss wake scores (0.2 to the threshold) that led nowhere |
| Acks | how many wakes were answered with a clip vs. answered by the reply itself |
| Music | request → confirmed playing, median and worst |

Plus cost per day and sessions per day. Progress is those numbers moving;
a night of test phrases is not.

## Feedback

Anything that felt wrong becomes an item: from the dashboard (select a
session, write it, tag it) or by voice ("Alexa, flag that: she cut me off"
→ the `flag_conversation` tool attaches it to the conversation it was said
in). Items carry tags (`cut-off`, `no-reply`, `too-eager`, `too-slow`,
`wrong-action`, `music`, `wake`, `style`…) and a status (`new` → `triaged` →
`planned` → `fixed` / `wontfix`). Items group into **needs** — "he prefers
shorter replies", "she must wait until he is done before acknowledging" —
which is what actually gets built; a need can be promoted to the task board
with one action, and the reflection job proposes groupings from new items.

## The order of work

1. **Foundation (this commit):** the always-on timeline, transcripts on the
   session rows, the feedback store and needs, the `flag_conversation`
   tool, the dashboard on `http://127.0.0.1:8765` (sessions, one session's
   timeline with wrapped text, flagging with tags, feedback and needs boards,
   the live log, the five numbers), a wrapped Tk feed and an "Open dashboard"
   button.
2. **Subtract clocks** (next): the command and question quick-closes, the
   continuation window and the wrap-up grace go; her end tool, his wrap-up and
   one idle timeout stay. Judged by the cut-off and silence numbers, not by ear.
3. **Satellites** (when the pucks arrive): the Wyoming/Assist boundary, the
   echo stack deleted, the wake word off the desktop.
