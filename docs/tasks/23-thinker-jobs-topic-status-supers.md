# Task 23: Thinker jobs: topic, status, supersession

## Goal
The deeper mind should belong to the current topic. Today a `think` question runs in the background with no
id, no status, no cancellation and no freshness check, and its answer comes back as an urgent thought even
if the owner changed the premise meanwhile. See docs/AUDIT-2026-09-05.md finding 7 and
docs/PLAN-VOICE-2026-09-05.md phase 7.

## Behavior
- Every thinker job gets an id, the question, the topic (one line), started/finished times and a status:
  running, done, superseded, cancelled. Kept in data/thoughts.json (last 20) and listed by a tool
  list_thoughts {}; cancel_thought {id} cancels a running one.
- Supersession: a new think on the same topic (word overlap with the running job's question, or the owner
  saying "actually, assume …" while one runs) marks the old job superseded; its answer is never spoken.
  The instructions tell her to start a fresh think with the changed assumption instead of waiting.
- Freshness on delivery: when a thought answer is about to be spoken, a short bridge is required
  ("With the extra year in mind…") when the question was superseded-then-restarted, and the answer is
  dropped with a journal row when it is older than 30 minutes and the topic has moved on.
- Urgency: a thought answer is urgent only while the conversation that asked it is still open; afterwards
  it is a normal announcement (quiet hours and focus apply).
- Tests: two thinks on one topic — only the second answer is spoken; cancel_thought stops delivery; an old
  answer after the topic moved on is journaled, not spoken.

## Voice test
"Alexa, help me think through buying a bike." While she reasons: "Actually, assume I wait a year." Only
the answer to the revised question is spoken, with a short bridge.

## Out of scope
Giving the thinker tools or retrieval — a separate task.
