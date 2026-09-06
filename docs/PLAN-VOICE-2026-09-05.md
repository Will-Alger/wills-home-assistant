# Plan: dependable activation, interruption, ending and reactivation

## Progress (updated 2026-09-05, 20:15)

| Phase | State | Where |
|---|---|---|
| 0 ship the fixes | done | `3b363ec` `211bdd5` `238ba13` |
| 1 measure | task 12 merged (`e8e9197`); task 19 replay tests merged (`3141673`, and it caught a closing bug) | `logs/turns.jsonl`, `tests/test_replay.py` |
| 2 output ownership | done | `210d83b` — `audio/io.py`, one mic and one speaker across idle and talk |
| 3 receiver, playback, closing | done | `ad460cf` — supervised tasks, "since" note, delivered = played, farewell first |
| 4 tentative interruption | done, tuning by ear | `762927d` — `REALTIME_TENTATIVE_INTERRUPT`, levels in the log |
| 5 detection tuning | far_field live (`e50fa81`); `WAKE_THRESHOLD` 0.5→0.4 from two logged misses; task 14 vocabulary merged (`410f400`); task 10 awaits Will | |
| 6 continuity | task 20 merged (`406a12b`: follow-ups retire on evidence, reads wait for playback); 21 working context merged (`src/assistant/context.py`, `{context}` placeholder); 22 receipts and undo merged (`src/assistant/receipts.py`, `undo_last`, per-bulb outcomes) | `raise_follow_up`, `tests/test_kept_promises.py`, `tests/test_context.py`, `tests/test_receipts.py` |
| 7 conversation quality | "let me think" patience state shipped (`c71f441`); 23 thinker jobs merged (`src/assistant/thoughts.py`, `list_thoughts`, `cancel_thought`); 24 scoped memory merged (`update_memory`, subject/confidence/supersedes, kind `house`); 25 tool contracts merged (`brain/outcome.py` ToolOutcome, `tests/test_outcome.py`) — phase 7 complete apart from Will's by-ear tuning | `is_thinking`, `_THINKING_S`, `tests/test_thoughts.py` |

2026-09-06 10:05: 23, 24, 25 merged; **task 10 rebased and built** (seven conflicts resolved by its agent; push-to-talk opens on AudioIO, a press mid-reply truncates) — "switch to task 10" to try it, "approve task 10" to merge; 13 spoken fallbacks merged (`assets/voice/*.wav`, `audio/fallbacks.py`); 11 working cue merged (`working_cue` task, tick at 2 s then every 6 s); 15 level meter merged (`audio/level.py`, a bar beside Listening in the panel). 16 spoken reply rules merged (endings and reading aloud). Tasks 10 (push to talk, hotkey Ctrl+Alt) and 9 (ping → pong) merged 2026-09-06 11:40 on Will's say-so. Afternoon, from Will's own tests: turns end on silence (`REALTIME_TURN_DETECTION=server_vad`, 800 ms) instead of a guess about the words; a mic still hot after a commit drops her reply; a regurgitated vocabulary prompt is noise, not a turn; her Bluetooth echo tail never counts as him; the turn-over ding waits a beat and only if the commit stood. Task 26 (spoken wake acknowledgment, "Yes?" in her own voice, `WAKE_ACK`) merged 14:50. Late afternoon, after a false wake to a vacuum cleaner (`7e83002`): wake threshold back to 0.5 with openWakeWord's Silero speech gate; two false wakes in three minutes raise the bar for ten; a wake with no real speech in six seconds dies quietly; a local speech gate in front of the socket (the server hears speech or clean silence, never the room); turn detection back to semantic at eagerness auto. **Board: 21 of 26 merged**; 17 and 18 drafted, Will's call; nothing building. Rule adopted: one knob per by-ear session from here.

Board order from here, one build at a time (2026-09-06 morning): 23 (resumed after the usage cap) → 24 → 25 → **10 revised** ("main moved: persistent AudioIO, supervised receiver, tentative talk-over — merge main into the branch, resolve, tests green"; eleven files overlap, so its agent rebases rather than a blind hand merge) → 13 spoken fallbacks → 11 working cue → 15 level meter; task 16 folds into 25; 17, 18 deferred. Will's voice tests remain the acceptance gate for 10.

2026-09-05, 12:46. Inputs reconciled: `wills-home-assistant-audit.md` (ChatGPT Astra, 11
findings + a 7-step delivery order + an 18-item voice acceptance script),
`docs/VOICE-UX-RESEARCH-2026-09-05.md` (my research, 9 ranked changes), her task board
(9 and 10 built, 11–18 drafted), and the uncommitted work in the tree. Plan only — no code.

## Where the two documents agree, and where the audit corrects the research

They agree on the order: activation and audio ownership first, then interruption with
deterministic closing, then detection tuning, then continuity, then conversation quality.
Astra's corrections to my report are right and are adopted below:

- Chimes already follow the saved speaker (`tones.set_output`); the idle-path problem is the
  per-chime stream, not the device choice.
- `far_field` noise reduction cannot help idle wake detection — openWakeWord runs before the
  Realtime connection exists. It is an in-session trial only.
- Attributing every miss to the "hey" prefix is premature. Log detector scores and chime
  playback separately, then compare thresholds against TV, music and ordinary speech.
- Truncation is not a one-event change: `Speaker` knows queue length, not per-item audible
  position. Playback accounting has to exist first.
- Client-side interrupt gating only works if the server's `interrupt_response` and
  `create_response` are turned off while VAD events keep flowing; otherwise the server acts
  before the local gate decides. And rejected echo must not become a user turn.
- The AEC reference must follow the audio actually rendered (the speaker callback), not the
  arrival of cloud audio chunks.
- "Transcript gating is what everyone ships", "Bluetooth AEC is useless", "AirPods need no
  AEC" are hypotheses to test on the Snowball/Echo Dot/AirPods, not facts.

Astra adds four things the research did not cover, all confirmed against the code:

- The receiver blocks while a tool runs (`_handle_response_done` awaits the tool), so a
  correction spoken during a slow search is processed only after the search returns.
- Mid-session notifications are marked delivered and read at `response.done`, before playback
  drains; an interrupted two-part announcement vanishes from unread.
- Every pending conversation follow-up is marked raised at session end if the owner said
  anything at all, whether or not she mentioned it. A one-line lighting command consumes
  "ask me how the demo went".
- Nothing supervises the receiver and mic tasks: a receiver exception leaves the session
  waiting on the idle timer.

## Usage budget is the constraint

Yesterday's research burned two full usage windows. Her dispatched agents draw on the same
Max window as this session. Rules for this plan: at most **one** agent build in flight; the
deep `run_conversation` work (phases 2–4) is done hands-on here with Will present to test
by ear, because a dispatched agent cannot hear the room; self-contained features and tests
go to her board. Phase 0 costs no usage at all.

## Phase 0 — today, no usage: ship what exists and re-check by ear

1. Commit the three uncommitted change sets as three commits, plus the research report and
   Astra's audit: (a) mic/speaker twins and the virtual-default guards; (b) wrap-up close,
   per-response `interrupted` reset, background reflection; (c) the two documents.
2. Drop the stale AirPods speaker pin from `data/panel.json` (the Echo Dot is Windows'
   default now) and restart her. She has been on `c686631` since 00:28.
3. Will runs acceptance items **18** (say "that's all"; separately say "wait" during the
   farewell; then re-activate immediately after a close — no deaf window) and **13** (say
   "Alexa" and "hey Alexa" at two distances) and reports what he heard.

Known limit to state plainly: during her farewell, "wait" is only honoured as a wake-phrase
barge-in today (half-duplex). A plain "wait" becomes possible in phase 4.

## Phase 1 — measure and regress (her board; one build)

- **Promote task 12 "Turn latency log"** to the first build, with its spec widened to Astra's
  activation timeline: wake detected → chime enqueued → chime audible (callback consumed) →
  mic ready → session connected → speech end → first audio → playback end → ended-by, plus
  interruption events and detector scores on every wake. One JSONL row per turn in `logs/`,
  content-light. This is the instrument every later phase is judged with.
- **New task: "Delayed-playback replay tests"** — a `DelayedSpeaker` that plays in real time,
  late user transcripts, slow tools, a receiver failure, and an interrupted two-part
  announcement. Regression-lock the three phase-0 fixes. Tests only; safe for an agent.
- **Task 15 (level meter)** waits for the telemetry it will display.

## Phase 2 — one owner for output, nothing lost after the wake (hands-on)

- One persistent output stream owned by the runner across idle and conversation; chimes are
  mixed into its callback (the docs' recommendation, and what the session path already does);
  the twin/fallback work from phase 0 handles device removal, and a reconnect re-opens the
  stream on the next real device.
- The idle loop listens on the same 24 kHz microphone the session uses, downsampling for the
  wake model (the barge-in path already does this). No 16→24 kHz reopen, no 0.2 s settle.
- An activation capture buffer: the last second before the wake and everything spoken while
  the session connects is appended to the session input, so speaking straight after "Alexa"
  works (acceptance item 14).
- `Speaker` gains per-item playback accounting: which assistant item is audible and how many
  milliseconds of it have been consumed by the callback. Prerequisite for phase 3.
- **Task 11 "Working cue"** ships on top of this stream once it exists; hold it until then.

## Phase 3 — responsive receiver, honest playback, deterministic closing (hands-on)

- Tool calls run as supervised tasks; the receiver keeps reading. Obsolete read-only work is
  cancelled and its speech suppressed; household mutations track not-started / sent /
  confirmed and are never blindly re-issued. Receiver and mic tasks are supervised: a failure
  ends the session through one bounded path to idle.
- On a confirmed barge-in send `conversation.item.truncate` with the item id, content index 0
  and the clamped played milliseconds from phase 2; late audio for that item is dropped.
- Notifications become "delivered" only when their playback completed; an interrupted batch
  keeps its unheard items unread. Follow-ups are marked raised only by the response that
  actually spoke them. (**Board task**, self-contained: "Follow-ups and notifications count
  only when spoken", with the replay tests from phase 1.)
- Closing: `end_conversation` returns a farewell instruction as its tool result (LiveKit's
  pattern) so the goodbye is generated on purpose; the engine waits for the response's audio
  to finish arriving and for local playback to drain, with a bounded fallback; the user
  wrap-up path from phase 0 stays. No assistant phrase is ever authoritative on its own.
- **Task 13 "Spoken fallbacks"** belongs here, next to the supervised-failure path.

## Phase 4 — tentative interruption without the wake word (hands-on spike)

- Stream the mic during playback with `interrupt_response: false` and `create_response:
  false`, so VAD events arrive but the client owns every decision.
- State machine: Speaking → Possible interruption (pause playback locally, keep the resumable
  audio, collect evidence) → False alarm (resume from the paused position) or Confirmed
  (cancel, truncate, let the user turn through) → bounded timeout, never paused forever.
- Evidence: VAD onset plus a transcript within ~2 s; an n-gram overlap with her last words is
  one signal, not a veto; a ~1 s post-onset cooldown; one-word "stop" and "wait" keep their
  instant path. Rejected echo is removed from server history (`conversation.item.delete` on
  the created item) before any response is created — the exact mechanism is the spike's
  first question.
- Trial order: Snowball + Echo Dot (the hard case), then AirPods (predicted easy). Judged
  with phase-1 telemetry: interruption-to-silence, false-alarm rate, echo-as-turn rate.
- AEC (`pywebrtc-audio`, wired speakers) only if this spike disappoints on wired output.

## Phase 5 — detection tuning (board + Will)

- Controlled `wake_threshold` trials using the phase-1 scores: 0.5 vs 0.35 across "Alexa",
  "hey Alexa", TV and music. Adopt a default only from the numbers.
- `noise_reduction: far_field` in-session trial, scored separately from wake recall.
- **Task 10 push-to-talk**: voice-test ("switch to task 10", hold Ctrl+Alt) and approve.
- **Task 14 transcriber vocabulary**: cheap and independent; a filler build any time.
- A custom `hey_alexa` model only if the preferred phrase is still unreliable at the tuned
  threshold (Colab runbook exists).

## Phase 6 — continuity (board, one at a time)

- Working context across sessions: current topic, salient entities, last action, pending
  question and options, temporary overrides, with timestamps and expiry, so "a little dimmer"
  after a close still targets the same lamps.
- Action receipts (targets, before/after, outcome, reversible?) and a limited "undo that";
  partial success reported as such.
- Notification stages queued / selected / playing / played / acknowledged.

## Phase 7 — conversation quality (board)

- Engine-level pacing states: command, conversation, waiting-for-answer, user-thinking ("let
  me think" lengthens the pause allowance) — behaviours, not prompt hints.
- Thinker jobs get an id, status, cancellation and supersession ("assume I wait a year").
- Scoped memory (subject, source, confidence, supersession; atomic preference replacement).
- One capability contract for tools (structured outcomes: success / partial / pending /
  unavailable / needs clarification).
- **Task 16 "Spoken reply rules"** merges with Astra's conversational moves here.

## Deferred on purpose

Tasks **17** (see the screen) and **18** (desktop actions) stay drafted until phases 3–4 are
solid — Astra's point stands that more things to announce amplify the delivery problems.
The Windows Voice Capture DSP and the Win11 AEC endpoint APIs are noted, not planned.

## Board actions, in order

| When | Action |
|---|---|
| Phase 0 | "approve task 9" by voice (a free end-to-end merge drill) |
| Phase 1 | Revise task 12's spec to the activation timeline, then "start task 12" |
| Phase 1 | Draft "Delayed-playback replay tests"; start after 12 merges |
| Phase 3 | Draft "Follow-ups and notifications count only when spoken" |
| Phase 5 | Test and approve task 10; start task 14 as a filler |
| Held | 11 (needs phase 2), 13 (phase 3), 15 (phase 1 data), 16 (phase 7), 17–18 (deferred) |

## What "done" looks like for the next release

Astra's acceptance items 13, 14, 15, 16, 17 and 18, run before and after with the same
devices, plus the phase-1 numbers: interruption-to-silence, premature-close rate, repeated
wake words, deaf-window after close. Judge the release on activation, interruption, ending
and reactivation first; on unfinished business second.
