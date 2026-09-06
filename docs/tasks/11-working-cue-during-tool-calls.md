# Task 11: Working cue during tool calls

## Goal
Never leave the room in silence while she is doing something. Clicky shows a spinner from key-up to first audio; our silent gap is a tool call (a light command about one second, a web search about four). See docs/CLICKY-REVIEW-2026-09-04.md, item 2.

## Behavior
- A new soft "working" earcon in tones.py (short, quiet, lower than the wake ding; through the session speaker like the other cues).
- Deterministic engine rule: when a response has started and no audio has arrived within 2 seconds while a tool call is running (the engine already tracks a busy flag around tool execution), play the cue once, then again every 6 seconds while still waiting. Never play it while she is speaking, never after the first audio delta of that response, never for tools that return at once.
- The settings panel state shows "working" during that window.
- Tests: an engine test with a slow fake tool asserts one cue at ~2 s and a second at ~8 s and none after audio arrives; a fast tool produces none.

## Voice test
"Alexa, search the web for tonight's Celtics score." Hear the soft tick before the answer. "Alexa, lights off" (fast): no tick.

## Out of scope
Spoken fillers ("one moment"); the model may still say them on its own.
