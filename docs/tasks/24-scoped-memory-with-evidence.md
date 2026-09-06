# Task 24: Scoped memory with evidence

## Goal
Memory should be evidence with scope, not a growing list injected whole. Today items have only id, kind,
text and date; all preferences are injected; lessons are picked by recency; preference replacement is a
two-step forget-then-remember the model can leave half done. See wills-home-assistant-audit.md finding 8
and docs/PLAN-VOICE-2026-09-05.md phase 7.

## Behavior
- MemoryItem gains subject (a short scope: "lights", "music", "calendar", "house", "will"), source
  (voice | reflection | tool), last_verified, confidence (0–1) and supersedes (an id). Old rows load with
  defaults; nothing is lost.
- Atomic replacement: a new tool update_memory {id, text} (or remember with supersedes=id) writes the new
  item and marks the old one superseded in ONE write; forget-then-remember is no longer the instructed
  pattern. The instructions and the remember/forget tool descriptions change accordingly.
- Retrieval instead of recital: the instructions get the preferences whose subject matches the tools and
  entities in play plus the 5 most recent others, never the whole store once it passes 15 items; lessons
  need a confidence of at least 0.6 to be injected, and reflection assigns confidence from evidence (a
  verified tool outcome raises it, a single inference is 0.4).
- House defaults versus personal: a kind "house" for shared defaults ("make this the house default")
  separate from "preference" ("remember this for me"); the instructions distinguish the two phrases.
- Tests: an interrupted replacement never leaves the store without the old value; a temporary music outage
  reflected once does not become an injected lesson; a large store injects only the matching subset.

## Voice test
"Alexa, remember I like the living room at 3000 kelvin." Then "Alexa, actually make that 2700." "What do
you remember about the living room?" — one answer, 2700, no contradiction.

## Out of scope
Voice identity; guest mode beyond the house/personal split.
