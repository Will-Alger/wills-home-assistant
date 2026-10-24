# Task 20: Follow-ups and notifications count only when spoken

## Goal
A promise is kept only when it was actually spoken. Today every pending conversation follow-up is marked
raised at session end if the owner said anything at all, and list_notifications marks items read before
the response reading them has played. See docs/AUDIT-2026-09-05.md findings 2 and 3 and
docs/PLAN-VOICE-2026-09-05.md phase 3.

## Behavior
- A follow-up is marked raised only when (a) the assistant's transcript in that session shares at least
  two significant words (4+ letters, not stop words) with the follow-up's text, or (b) the model called a
  new tool raise_follow_up {id} — add it to FOLLOWUP_TOOLS with a one-line instruction: call it when you
  bring a follow-up up. Otherwise it stays pending and a journal row says "follow-up N not raised: not
  mentioned".
- list_notifications and mark_notifications defer the read marking until the response that carries the
  answer has finished playing (the engine's finish_playback), and skip it if that response was cut off by
  a barge-in.
- Tests: a one-line lighting command leaves the demo follow-up pending; asking the question (transcript
  overlap) advances it once; raise_follow_up advances it; an interrupted list_notifications reply leaves the
  items unread.

## Voice test
Say "next time we talk, ask me how the demo went". Close. Say "Alexa, turn off the hallway" and let it
close. Then "Alexa, what are you waiting on me for?" — the demo question is still there.

## Out of scope
Notification stages beyond delivered/read (queued/selected/playing) — a later task.
