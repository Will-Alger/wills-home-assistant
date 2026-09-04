# Task 10: Push to talk hotkey

## Goal
A second way to talk to her beside the wake word: hold a keyboard shortcut and speak; releasing it ends the turn. The user's own action defines the utterance, so a pause never cuts him off and she never keeps listening after he is done (the two open complaints). Modeled on Clicky (github.com/farzaa/clicky), see docs/CLICKY-REVIEW-2026-09-04.md, item 1.

## Behavior
- A system-wide hotkey on the Windows desktop, default: hold Ctrl+Alt together (modifier-only, no letter), configurable in .env as PTT_HOTKEY. It must work while she runs windowless from the watchdog and while another app has focus. Verify the library choice against its docs (keyboard, pynput, or a Win32 hook); never require admin rights.
- Press while idle: open a Realtime session at once, play the listening ding, no wake chime, no wake word needed. Press while she is speaking: barge-in (clear the speaker, cancel the response) and start a new turn, like the wake word does today.
- While held: turn detection is OFF for that turn (the installed openai SDK accepts turn_detection null; verify the exact session.update shape), so a mid-sentence pause never ends the turn. Release: commit the audio buffer (input_audio_buffer.commit, verified to exist) and request the response. A press-and-release under 300 ms with no speech does nothing.
- After her reply, a push-to-talk session closes 8 seconds later unless he holds again (each hold = one turn). Wake-word sessions keep today's behavior.
- The settings panel shows the hotkey and reflects listening while held; the listening flag drops the instant the key is released, not when the API confirms.
- Tests with fakes: a fake hotkey source driving press/release into the engine; the FakeConnection asserts the session.update, commit and response.create sequence and the ordering around barge-in.

## Voice test
1. Hold Ctrl+Alt, say "what's the weather like", pause three full seconds mid-sentence, finish, release. She answers once, after the release, never during the pause.
2. Hold it again while she is still talking: she stops at once and takes the new question.
3. Release without speaking: nothing happens, no error tone.
4. "Alexa, …" still works exactly as before.

## Out of scope
A phone button or a physical satellite button (later tasks). Changing the wake word path.
