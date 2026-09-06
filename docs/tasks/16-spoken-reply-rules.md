# Task 16: Spoken reply rules

## Goal
Two habits from Clicky's prompt that serve conversation quality. See docs/CLICKY-REVIEW-2026-09-04.md, item 7.

## Behavior
- Add to her instructions: never end a reply with a dead-end yes/no question ("want me to explain more?"); when it fits, end by planting a seed (something worth coming back to) or simply stop. Spell out small numbers and avoid abbreviations that sound wrong aloud; tool results that carry "18:30" or ids are read as times and plain words.
- Keep the brevity rules exactly as they are; this is about endings and reading aloud, not length.
- Tests: the instructions render with the new lines; no other assertions change.

## Voice test
"Alexa, what's HTML?" She answers without "want me to explain more?" and, if she adds anything, it is a seed, not a question. "What's on my schedule at 18:30?" is read back as a spoken time.

## Out of scope
Anything else in the prompt.
