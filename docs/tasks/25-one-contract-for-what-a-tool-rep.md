# Task 25: One contract for what a tool reports

## Goal
One consistent contract for what a capability reports. Today home actions, calendar writes, scheduling,
memory and background jobs each return free-text strings the model reads as control flow. See
docs/AUDIT-2026-09-05.md finding 9 and docs/PLAN-VOICE-2026-09-05.md phase 7.

## Behavior
- A ToolOutcome dataclass in src/assistant/brain/outcome.py: status (success | partial | pending |
  unavailable | needs_clarification), summary (one sentence she can say), details (dict), reversible
  (bool), follow_up (a short instruction for the model, optional). Every tool path in
  realtime_engine.py:_handle_response_done and brain/tools.py returns one, rendered into the
  function_call_output JSON with those exact keys; the old (text, is_error) pairs are adapted, not
  rewritten, where the tool is untouched otherwise.
- The instructions get one paragraph on the contract: act on status, speak the summary, ask the
  needs_clarification question verbatim, never claim success on partial.
- Tests: every tool name advertised in realtime_tools() produces a valid ToolOutcome for at least one
  call (a parametrised test over the fake home); a partial lighting change renders status=partial.

## Voice test
Fail one of three lights in the fake house (tests) — by voice: "Alexa, turn on everything" with a bulb
unplugged; she says which one did not respond and does not say "done".

## Out of scope
Hiding tools dynamically by state — measure routing accuracy first.
