# Task 21: Working context across sessions

## Goal
After a session closes, "a little dimmer" or "let's do the second option" has nothing to bind to. Keep a
small working context across sessions so references resolve. See wills-home-assistant-audit.md finding 4
and docs/PLAN-VOICE-2026-09-05.md phase 6.

## Behavior
- src/assistant/context.py: WorkingContext stored in data/context.json with timestamps: topic (one line),
  entities touched by the last home actions (entity ids and friendly names), last successful action (tool,
  args, outcome), pending question and its options (when she asked something and got no answer),
  temporary overrides, outstanding jobs (thinker, tasks). Each field expires on its own clock (default 30
  minutes; the pending question 10 minutes).
- The engine updates it from tool results and from the end of each session (sessions.py already has the
  first line and summary); a `{context}` placeholder in the instructions renders it in one short
  paragraph: "Just now: …; 'it' or 'that' most likely means …".
- Resolution rule in the instructions: with one fresh candidate, act; with two plausible candidates, ask
  ONE specific question; with none, ask what they mean.
- Tests: set the living-room lamps, close, then "a little dimmer" in a new session targets the same lamps
  (the instructions carry them); after 31 minutes the context is empty; a pending question renders with
  its options.

## Voice test
"Alexa, make the living room cozy." Wait for the close. "Alexa, a little brighter." It changes the same
lamps without asking which.

## Out of scope
Durable personal memory (memory.py) — this is conversation-scoped and expires.
