# Task 7: Voice cues and settings panel

Goal
Provide reliable voice-capture audio cues and a voice-openable desktop Settings panel for status and configuration.

Behavior
- Play a start-listening ding every time listening begins, including follow-up turns.
- Play a distinct end-listening ding when listening ends normally.
- On capture failure, timeout, or error, play a brief error tone and do not indicate listening is active.
- Add a voice-controlled Settings panel on desktop: commands like “open settings panel” and “close settings panel” show or hide it.
- The panel shows current microphone, whether the assistant is currently listening, current voice selection, current wake word, status summary, and a live log feed (best-effort; may be limited to what’s available in the running session).
- The panel includes controls to change voice selection and wake word, and a button to restart the assistant.
- The panel is informational and configuration-focused; it must not directly toggle listening on/off.

Voice test plan
- User performs two conversational turns and confirms hearing start and end tones each time.
- User triggers a failure in a test scenario and hears the error tone; panel reflects an error in status/log.
- User opens the Settings panel by voice, changes voice and wake word, uses the restart control, then closes the panel.

Out of scope
- Changes to wake word detection accuracy, recognition model behavior, or backend logging beyond exposing a live feed.
- Full waveform visualization, recording controls, or exporting logs.
