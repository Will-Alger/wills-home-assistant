# Task 15: Live level meter in the settings panel

## Goal
You always know whether she is hearing you. Clicky's waveform reacts to the live level and hides the instant the key is released. Our panel shows a listening flag only. See docs/CLICKY-REVIEW-2026-09-04.md, item 6.

## Behavior
- Compute an RMS level from the microphone frames (cheap, per 80 ms frame, smoothed with a short decay) and expose it on the status object.
- The settings panel draws a small level bar while listening (updated on its own thread from the snapshot, like the rest of the panel); it reads flat when not listening.
- The listening flag drops on the user's action (turn ended, hotkey released once that exists), never waiting for the API.
- Tests: the level computation on silence and on a synthetic tone; the status snapshot carries it.

## Voice test
"Alexa, open the settings panel." Talk: the bar moves. Stop: it falls flat within a second.

## Out of scope
A full waveform or recording controls.
