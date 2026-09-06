# Task 26: Spoken wake acknowledgment in her own voice

## Goal
When the owner says the wake word she should answer at once, in her own voice — "Yes?" — instead of a
ding, and still be listening. Not by asking the model (the session takes about a second to open); by
playing a short clip rendered ONCE in her real voice, off disk, through the persistent speaker the
instant the wake fires, while the session connects underneath. Owner's words: "real magic… when I say
Alexa she immediately goes 'Yes sir?' but still is listening."

## Behavior
- `scripts/render_acks.py` renders the acknowledgment clips in her REAL voice through the Realtime API,
  the way `scripts/m4_realtime.py --text-probe` already saves a reply as a WAV (reuse that machinery:
  the engine's text_probe path, `settings.realtime_voice` with its automatic fallback). Each clip is
  requested with an instruction to say exactly the phrase and nothing else; verify every clip is under
  0.9 s and re-render one that came out longer or with extra words. Save 24 kHz mono int16 WAVs under
  `assets/voice/ack/<slug>.wav` and commit them. THIS SPEND IS AUTHORIZED: the owner asked for it; it
  is a few cents, once. Nothing at runtime calls the renderer.
- Phrases (short is the whole point — see the catch below): "Yes?", "Yes, Will?", "Go ahead.",
  "Listening.", "Mm-hm?", "I'm here.", plus "Morning." and "Evening." Pick one at random per wake so it
  never sounds canned; before 11:00 weight "Morning." in, after 18:00 "Evening.", never the same clip
  twice in a row.
- `src/assistant/audio/acks.py`: loads the clips at boot (missing clips = fall back to the ding with one
  boot note, never a crash), picks one, plays it through the session `Speaker` (the persistent one the
  runner owns — `Speaker.enqueue` of the PCM, resampled if a clip is not 24 kHz) and returns its
  duration. The runner (`scripts/m4_realtime.py` `one_cycle`) uses it in place of the wake ding for BOTH
  the wake word and a push-to-talk press; `VoiceCues.start` still runs so the listening flag, the panel
  state and the level meter behave exactly as now — give it a `sound=False` option rather than
  duplicating its bookkeeping.
- THE CATCH, and the part that must be right: her "Yes?" comes out of the speaker and back into the
  microphone, and with silence-based turn detection the server would treat it as the owner's speech and
  answer it. The `Microphone` (src/assistant/audio/mic.py) must stamp every frame with
  `time.monotonic()` at capture and expose `ignore_before(deadline)`; `get_frame` drops frames captured
  before the deadline. The runner sets the deadline to clip end + 0.15 s (echo tail). Frames captured
  AFTER the deadline — the owner's command said right after the clip — must still reach the session
  first, exactly as they do today (the persistent-mic property from plan phase 2).
- `WAKE_ACK=voice|ding|off` in config.py and .env.example (default `voice`; `ding` is today's behaviour;
  `off` is silence). Document it.
- The in-conversation cues (listen_end, the listening ding after her replies, the working tick, the
  goodbye chime) are unchanged.
- Tests with fakes: a wake plays exactly one ack clip through the speaker and sets the listening flag;
  two consecutive wakes never pick the same clip; frames captured inside the ignore window are dropped
  and the first frame after it is delivered; WAKE_ACK=ding plays the ding and no clip; missing clip
  files fall back to the ding with a note; the clock weighting; the renderer's duration check rejects a
  long clip (unit-test the check, not the API).

## Voice test
Say "Alexa" — she answers "Yes?" (or one of the others) in her own voice within a blink, then obeys the
command you give next. Say "Alexa, turn off the hallway" in one breath — the command still lands. Hold
Ctrl+Alt: same acknowledgment. Set WAKE_ACK=ding, restart: the old ding is back.

## Out of scope
Voicing the working tick ("One moment." is already rendered under assets/voice for a later task).
Any change to how turns end or to the talk-over. Do not touch .env, data/ or logs/.
