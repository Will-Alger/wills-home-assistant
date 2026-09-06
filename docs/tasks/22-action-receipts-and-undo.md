# Task 22: Action receipts and undo

## Goal
"Done" is only true when it is. Lighting execution applies per-entity commands and answers "Done" even
when one failed, and there is no undo. See wills-home-assistant-audit.md finding 5 and
docs/PLAN-VOICE-2026-09-05.md phase 6.

## Behavior
- Every home mutation produces an ActionReceipt: targets, before-state (from get_entity, cheap), requested
  state, per-entity outcome, reversible (bool), timestamp. Kept in memory for the session and the last 20
  in data/receipts.json.
- Partial success is reported as such in the tool result ("3 of 4 lights changed; the bedroom lamp did not
  respond"), never as "Done".
- A tool undo_last {} restores the before-state of the last reversible receipt (lights only in this task)
  and says plainly when the last action cannot be undone (a media command, a calendar delete).
- Instructions: "undo that", "no, the bedroom", "keep the brightness but change the color" map to
  undo_last or a corrected set_lights using the receipt's targets.
- Tests with FakeHome: fail the second of three light updates — the result names the failure; undo_last
  restores the first light; undo after a non-reversible action is refused with a sentence.

## Voice test
"Alexa, turn the living room red." Then "Alexa, undo that." The lights return to what they were.

## Out of scope
Undo for calendar, schedules, memory — refuse with a sentence for now.
