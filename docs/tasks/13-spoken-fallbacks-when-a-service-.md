# Task 13: Spoken fallbacks when a service is down

## Goal
When her voice service or Home Assistant is unreachable she goes silent today (an error tone at best). Clicky's operating-system voice still says "I'm all out of credits". Give her four short pre-rendered lines that need no network. See docs/CLICKY-REVIEW-2026-09-04.md, item 4.

## Behavior
- A script scripts/render_fallbacks.py renders four WAVs once with OpenAI TTS (verify the endpoint and voice against the installed SDK) into assets/voice/: "I can't reach my voice service right now.", "The home isn't answering.", "Something failed. Check the log.", "One moment." Commit the WAVs (they are small); the script is only for re-rendering with another voice.
- The app plays them locally (sounddevice, through the session speaker when one is open): the first when the Realtime connection fails after the wake word; the second when a home command fails because Home Assistant is unreachable; the third when a session dies on an unexpected error; the fourth is available to the engine for later use.
- Never more than one fallback line per failure; the error tone still plays where it does today.
- Tests: the runner's recovery path plays the right file name for a connection error (audio playback stubbed).

## Voice test
Disconnect the network. "Alexa, turn on the lights." Hear "I can't reach my voice service right now." Reconnect; she works again with no restart.

## Out of scope
A general offline mode.
