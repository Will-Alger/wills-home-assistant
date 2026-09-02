# Overnight run — 2026-09-02 (05:53 → ~06:20)

Scheduled by Will before sleeping. Everything below shipped to `main` with the
suite green after each milestone (106 → 124 tests) and she was restarted onto
it. She is running `main` now; quiet hours are on (23:00–08:00), so nothing
announced itself while you slept.

## What shipped

1. **Phase 5 polish** — `announcement_history` ("what did you tell me this
   morning?"), ranked task search (title hits first), the wake-time status line
   leads with "N awaiting your approval", cloud dispatch documented as opt-in
   watch mode, a README runbook for the loop.
2. **The brain layer** — `think(question)`: she hands hard questions (plans,
   comparisons, advice) to Opus via `claude -p` on your Max subscription, with
   her preferences, facts, lessons, task board, and the conversation as
   context. The tool returns at once — she says she's thinking — and the
   answer arrives as an event she speaks in her own words, mid-conversation or
   at the next idle moment. Live-verified: Opus answered in 7 s.
3. **Event reactivity** — she holds Home Assistant's websocket open (verified
   live on HA 2026.8.3, reconnects with backoff) and evaluates standing watches
   deterministically: entity, to/from state, time windows that wrap overnight,
   days, once/keep, normal/urgent. Hits become announcements. Persisted in
   `data/watches.json`.
4. **Scheduling & routines (new milestone)** —
   - timers, one-shot or recurring alarms with snooze, scheduled reminders, and
     scheduled **actions** (any home tool at a time / after a delay / repeating
     on days). Persisted in `data/schedule.json`, ticked every second in the
     app. Owner-set items are urgent: they speak through quiet hours.
   - **routines**: structured "WHEN tool + conditions THEN defaults/overrides"
     rules applied *deterministically* inside the tool executor, so "after 5pm
     lights come on warm orange" and "TV volume defaults to 65%" never depend
     on the model remembering. Listed in her instructions; tool results report
     which routines applied. Persisted in `data/routines.json`.
5. **Morning briefing** — a scheduled `briefing` kind: today's calendar, tasks
   awaiting approval, what's scheduled today. You schedule it by voice.

## Voice tests (exact phrases)

- "Alexa, what did you tell me this morning?"
- "Alexa, think about whether I should get a HomePod mini or a second Voice
  PE puck." → "thinking it over" → the answer arrives a little later.
- "Alexa, tell me when the living room plant light turns on." → toggle it in
  the Home app → she announces it.
- "Alexa, set a two minute timer." → chime + "your timer is up".
- "Alexa, set a work alarm for 7 AM on weekdays." / "Alexa, snooze."
- "Alexa, remind me at 6 PM to call Mom."
- "Alexa, turn on the hallway light at 6:30 every evening."
- "Alexa, brief me weekdays at 7:30." → the next weekday morning at 7:30.
- "Alexa, from now on when I ask for lights after 5 PM, make them warm
  orange." → later: "turn on the living room lights" → they come on orange.
- "Alexa, the TV volume should default to 65 percent." → "set the volume"
  (no number) → 65.
- "Alexa, what's on my schedule?" / "what routines do I have?" /
  "what are you watching for?"

## Still open

- Phases 2–4 of the loop (task board, switch_build, approve, revise) have not
  been voice-tested by you yet — task 1 (birth certificate) and task 5
  (staging drill) are on the board for exactly that.
- Door/motion watches need sensors (Eve Door & Window / Motion, Thread via
  the Apple TV). Any existing entity works for a first watch.
- Calendar RESCHEDULE (an update tool; today it's delete + create).
- Voice PE satellites, Telegram front door, HA MCP evaluation, voice ID.
- The briefing reads the calendar and the board; weather would be a nice add.

## .env

No values changed tonight. New optional keys are documented in `.env.example`
(`BRAIN_MODEL`, `BRAIN_EFFORT`, `DISPATCH_TIMEOUT_S`, `DISPATCH_RESUME_DELAY_S`,
`WEB_SEARCH_MODEL`, `WEB_SEARCH_CONTEXT`, `UV_EXE`); all have sane defaults.
New data files appear on first use: `data/watches.json`, `data/schedule.json`,
`data/routines.json`.
