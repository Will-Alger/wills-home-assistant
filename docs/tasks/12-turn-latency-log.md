# Task 12: Turn latency log

## Goal
Tune her timing from data instead of feel. Today we record nothing about how long a turn takes, and a
missing activation chime cannot be told apart from a missed wake word or a slow connection. See
docs/CLICKY-REVIEW-2026-09-04.md item 3 and docs/PLAN-VOICE-2026-09-05.md phase 1. This log is the
instrument every later phase is judged with.

## Behavior
- One JSON row per turn appended to logs/turns.jsonl (the app writes it at runtime; keep rows small,
  content-light: no transcripts, no audio). Fields, all monotonic seconds relative to the wake or None
  when the step did not happen:
  - activation: wake_score (the detector score that fired), chime_enqueued, chime_audible (the speaker
    callback consumed the chime — the moment it could be heard, not the moment it was queued), mic_ready
    (the session mic is open), connected (session.updated received), first_speech (input speech_started).
  - per user turn: speech_end (speech_stopped), transcript_at (transcription completed), first_audio
    (first output audio delta), playback_end (speaker drained), and a list of tools as [name, seconds].
  - interruptions: how many barge-ins, and for each the seconds between the wake phrase and silence.
  - ended_by (the engine's reason string) and the session id from data/sessions.json.
- The same timings for the last turn are stored compactly on the session row in data/sessions.json
  (sessions.py) so "how fast were you today?" can be answered without re-reading the log.
- The console line printed when a conversation closes includes "first audio 0.9 s" for the last turn.
- A voice tool latency_report {since?} that answers "how fast were you today?" with medians for
  wake-to-chime, speech-end-to-first-audio and the slowest tool, in one or two spoken sentences; it says
  so plainly when there is no data yet.
- Wake-word events while idle that did NOT fire (score above 0.2 but below the threshold) are logged
  too, with the score, so thresholds can be compared later from real household audio.
- Tests: the FakeConnection drives a session with scripted delays and the row carries the timings; the
  idle path is tested with a fake detector whose scores are scripted; the tool renders a sentence; an
  empty log renders the "no data" sentence.

## Voice test
Ask three things, then "Alexa, how fast were you today?" — she names a median and the slowest step.
Then open logs/turns.jsonl and check the last row has chime_audible and first_audio filled in.

## Out of scope
Changing the close windows, thresholds or the chime path themselves; that is phases 2 and 5 once
there is data. Do not touch .env, data/ or logs/ in the worktree; the app writes logs/turns.jsonl at
runtime.
