# Clicky review — what to borrow (2026-09-04)

Source: github.com/farzaa/clicky, MIT, last public commit 2026-04-27 (Farza took
newer work private). Read in full: 25 Swift files (~5,500 lines), the Cloudflare
Worker, AGENTS.md. This compares it with our app as of main `3f3fb15`.

## Verdict

- Clicky is a **desk companion**, not a home assistant: push-to-talk (hold
  Control+Option), transcribe, send the words plus screenshots to Claude, speak the
  reply with ElevenLabs, and fly a blue cursor to whatever it mentions. No wake
  word, no tools, no memory beyond ten exchanges, no async. On everything past the
  microphone we are far ahead.
- Its one big idea for us is about **when listening starts and stops**: the user's
  own action (key down / key up) defines the utterance, and every wait after that
  is bounded (1.4 s grace, 2.8 s hard fallback). Nothing guesses. We should offer
  that as a second activation path beside the wake word.
- Its second lesson is **feedback discipline**: four states (idle, listening,
  processing, responding), each with its own visual, state changes that follow the
  user's action instantly rather than the pipeline, and a live level meter while
  listening. Task 7 gave us the dings; the level meter and a "working" cue are the
  gaps.
- Three of its habits are cheap wins: bounded waits with honest fallbacks (down to
  a local voice saying "I'm out of credits"), a vocabulary list handed to the STT,
  and per-event instrumentation to tune timings with data instead of feel.
- Do not copy the pipeline itself (STT → text LLM → TTS across three vendors),
  the keyboard-bound activation, or the ten-exchange memory. Our speech-native
  engine is faster and our memory is better.

## What Clicky actually is

One interaction, start to finish (from `CompanionManager`, `BuddyDictationManager`,
`AssemblyAIStreamingTranscriptionProvider`, `ClaudeAPI`, `ElevenLabsTTSClient`):

1. A listen-only `CGEvent` tap watches modifier flags system-wide. Control+Option
   down → `pressed`; up → `released`. A quick press-and-release cancels the pending
   start so the waveform never sticks.
2. On press: cancel any in-flight response and stop TTS (a new utterance always
   wins), fetch a 480 s AssemblyAI token from the Worker, open a websocket
   (`u3-rt-pro`, 16 kHz PCM16, `format_turns`, a `keyterms_prompt` list of product
   names), start `AVAudioEngine`, stream buffers, show a 5-bar waveform driven by RMS
   (×10.2 boost, 0.72 decay).
3. On release: stop the mic, send `ForceEndpoint`, wait up to **1.4 s** for a
   formatted end-of-turn; a **2.8 s** fallback delivers the best partial; a socket
   error mid-finalize also delivers the partial. Empty transcripts are dropped.
4. Capture every monitor as JPEG (labelled with pixel dimensions and "primary
   focus" for the cursor's screen), send transcript + images + the last 10
   exchanges to Claude Sonnet 4.6 over SSE through the Worker. The stream is not
   displayed; a spinner runs until audio starts.
5. Parse a trailing `[POINT:x,y:label:screenN]` tag out of the reply, scale the
   coordinate from screenshot pixels to display points, fly the cursor along a
   bezier arc (0.6–1.4 s by distance), show a bubble, hold 3 s, fly back.
6. Download the whole ElevenLabs `eleven_flash_v2_5` clip, then play. State becomes
   "responding" only when audio is actually playing. If the API fails, macOS's own
   `NSSpeechSynthesizer` says "I'm all out of credits".
7. Transient mode: if the cursor is hidden, it fades in for the interaction and
   out 1 s after TTS and pointing finish.

Other things it does: permissions polled every 1.5 s with per-permission tracking;
a provider chain (AssemblyAI → OpenAI upload → Apple Speech); TLS warm-up at launch
and one shared URLSession (a new one per session corrupted the socket pool);
PostHog events for every step; Sparkle auto-update; keys only on the Worker.

## Side by side

| | Clicky | Ours |
| --- | --- | --- |
| Activation | Hold Control+Option (needs a keyboard, accessibility permission) | Wake word, hands-free, anywhere in the room |
| Utterance end | Key up, then a bounded finalize (1.4 s / 2.8 s) | Semantic VAD (eagerness high); quick-close 8 s after a command, 15 s after an answer; 45 s idle |
| Interruption | Press again: cancels response + TTS instantly | Wake word during her speech (half-duplex); stop phrases |
| Pipeline | AssemblyAI → Claude (vision) → ElevenLabs, via a Cloudflare Worker | OpenAI Realtime speech-to-speech, tools in the loop |
| Time to first audio | Roughly 4–8 s (finalize + vision call + full TTS download; not measured by them either) | Usually 1–2 s; not instrumented |
| Context | Screenshots of every monitor | The home: lights, media, calendar, memory, board |
| Feedback | Triangle / waveform with level / spinner / bubble; instant on key-up | Wake, listen-end, close, error chimes (task 7); settings panel with a listening flag |
| Memory | Last 10 exchanges, in RAM | Preferences, facts, lessons, sessions, journal |
| Tools / actions | None (the point tag is the only side channel) | Home, media, calendar, web, scheduler, tasks, phone |
| Failure story | Bounded waits, partial transcripts, local spoken fallback | Error tone; retry backoff; no spoken fallback when the voice API is down |
| Instrumentation | PostHog event per step | `.usage.jsonl`, journal; no per-turn latencies |

## What to borrow, ranked

1. **Push-to-talk as a second activation path.** A global hotkey on the desktop
   (later: a phone card button, a physical button on a satellite). Press: open the
   session at once, listening ding, no wake chime. While held: turn detection off so
   a pause never ends the turn. Release: `input_audio_buffer.commit` then
   `response.create`. Press during her speech: barge-in plus a new utterance, like
   Clicky. Verified in the installed SDK today: `turn_detection` accepts null and
   `InputAudioBufferCommitEvent` exists. Effort: about a day. This directly attacks
   the two complaints we still have ("she cut me off", "she kept listening").
2. **A "working" cue.** Clicky never leaves you staring at nothing: the spinner
   runs from key-up to the first audio. Our silent gap is tool calls (a light
   command ~1 s, web search ~4 s, a merge now in the background). Deterministic
   rule: if a response has started and no audio has arrived within 2 s while a tool
   runs, play a soft tick; repeat every ~6 s. We already track `tool_busy`. Effort:
   an hour.
3. **Instrument every turn.** Clicky tracks each step; we tune eagerness and close
   windows by feel. Add to the session log: wake → connected, speech stopped →
   first audio, each tool's duration, why the session ended. Then set the windows
   from data. Effort: half a day.
4. **A local spoken fallback.** Clicky's "out of credits" line comes from the OS
   voice. Pre-render four short WAVs with OpenAI TTS once ("I can't reach my voice
   right now", "the home isn't answering", "something failed; check the log", "one
   moment") and play them when the Realtime connection or Home Assistant fails.
   This was in the 2026-08-30 review and never built. Effort: two hours.
5. **Vocabulary for the transcriber.** Clicky passes product names as
   `keyterms_prompt`. Our transcription model accepts a `prompt` string
   (`AudioTranscription.prompt`, verified). Feed it area and device names, playlist
   and artist names, "Alexa", task titles. It only improves the transcript (the
   model hears audio directly), but the transcript drives the stop-phrase match,
   the journal, reflection and the phone-visible history. Effort: an hour.
6. **Live level meter and instant state.** Their waveform reacts to RMS at 36 fps
   and the overlay hides the instant the key is released, before the pipeline
   catches up. Our panel shows a listening flag; add a level bar from the mic
   frames, and make sure the flag drops on the user's action, not on the API's
   turn-end event. Effort: two hours.
7. **Spoken-reply prompt rules.** Worth lifting into our instructions: never end
   with a dead-end yes/no ("want me to explain more?"), and when it fits, end by
   planting a seed instead; spell out small numbers and avoid abbreviations that
   sound wrong aloud (tool results still leak "18:30" and ids). This serves the
   north star, conversation quality. Effort: minutes; judge by ear.
8. **Screen awareness, optional.** Clicky's whole value is "I can see your screen".
   Realtime accepts `input_image` on a user item (verified). A `look_at_screen` tool
   that screenshots the desktop and attaches it would let her answer "what does this
   error mean?" at the desk. Off-thesis for a home assistant, cheap to try, easy to
   remove. Effort: half a day.
9. **Keys behind the desktop, not on satellites.** Clicky's Worker exists so keys
   never ship in a binary. When phone or satellite clients arrive, they should talk
   to her, never to the vendors.

## What not to copy

- The three-vendor pipeline. Speech-to-speech with tools is why she feels alive;
  Clicky waits for a whole clip to download before speaking.
- Push-to-talk as the only way in. It needs a keyboard and a permission; the wake
  word is what makes her a presence in the room. Offer both.
- Ten exchanges of in-memory history as "memory". Ours persists and reflects.
- Onboarding music, video and welcome bubble: product polish for a download, not
  for a household.
- The point tag parser as a general side channel. We have real tools.

## API facts verified today (installed openai SDK, `types/realtime`)

- `realtime_audio_config_input.py`: `turn_detection: Optional[...]` — manual mode.
- `input_audio_buffer_commit_event.py`: `InputAudioBufferCommitEvent` — commit on release.
- `audio_transcription.py`: `AudioTranscription.prompt: Optional[str]` — vocabulary.
- `realtime_conversation_item_user_message.py`: content type `input_image` (base64 data URI, `detail`) — screenshots.

## Suggested tasks, one line each, ready for the board

- **Push to talk**: hold a hotkey to talk; release ends the turn; press during her
  reply interrupts; the wake word still works. Voice test: hold, pause mid-sentence,
  release; she waits through the pause and answers once.
- **Working cue**: a soft tick 2 s into a silent tool call, repeating every 6 s.
  Voice test: "search the web for…" and hear the tick before the answer.
- **Turn latency log**: per-turn timings in `data/sessions.json` and a
  `latency_report` tool ("how fast were you today?").
- **Spoken fallbacks**: four pre-rendered lines for voice-API and HA failures.
  Voice test: pull the network, say the wake word, hear "I can't reach my voice".
- **Transcriber vocabulary**: pass names to the transcription prompt. Voice test:
  say a playlist name and check the journal transcript.
- **Level meter**: a live bar in the settings panel while listening.
