# Task 19: Delayed-playback replay tests

## Goal
Tests only. The engine's lifecycle rules are locked in with scripted sessions, but the fake speaker plays
instantly and every transcript arrives on time, so "generated versus heard" bugs cannot show up. Build a
replay suite with real delays. See docs/PLAN-VOICE-2026-09-05.md phase 1 and wills-home-assistant-audit.md
finding 11.

## Behavior
- tests/fake_realtime.py gains a DelayedSpeaker: audio drains in real time at 24 kHz int16 (bytes / 48000
  per second, optionally scaled), supports begin_item / played_ms / clear / wait_idle like the real
  Speaker (src/assistant/audio/speaker.py), and reports played_ms from what has actually drained.
- tests/test_replay.py, each scenario driven through RealtimeEngine.run_conversation with FakeClient:
  1. a late transcript: the user's transcription.completed arrives 1 s after response.done — a wrap-up
     ("that's all") still closes without a listening window.
  2. a slow tool (0.8 s) with a correction spoken meanwhile — the correction is in the tool output's
     "since" field and the session does not idle out under the tool.
  3. a receiver failure mid-response — ended_by starts with "session error" within one second.
  4. a two-part announcement interrupted by the wake phrase after the first part — both items are
     delivered and unread afterwards; a reply to the interruption does not read them.
  5. "that's all" said while a tool is still running — she closes after the tool's reply plays.
  6. a barge-in during her goodbye — the session stays open for what he wants to say.
  7. the end tool without any audio — a goodbye is requested and the session closes after it.
- Every scenario asserts ended_by, what was sent to the server (truncate, cancel, response.create
  counts), and the announcer/notification states where relevant.
- If a scenario exposes a real engine bug, fix it minimally in src and say exactly what changed in
  your summary; otherwise src is untouched.

## Voice test
None — this is a test suite. Run `uv run pytest -q tests/test_replay.py`.

## Out of scope
New engine features. The tentative-interruption state machine (plan phase 4).
