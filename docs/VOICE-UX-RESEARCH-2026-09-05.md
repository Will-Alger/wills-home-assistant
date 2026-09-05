# Making activation, barge-in and ending feel like magic

Research report — 2026-09-05. Commissioned by Will: *"the reliability and consistency of
when it activates, when it stops listening, the ability to quickly ask a follow-up after it
has stopped listening… I want it on par with what GPT-Live would be like."*

## Status of this research — read this first

A five-angle deep-research run (24 sources fetched, 119 claims extracted) completed its
**search and fetch phases in full**, but the adversarial verification phase and the automated
synthesis step both died on Claude usage limits — twice, in two separate runs (2026-09-04
23:56 and 2026-09-05 07:35). Of 25 claims selected for 3-vote verification, 7 got a verdict;
the rest were extracted from their sources but never independently checked.

So this document is a **hand-written synthesis of a machine-assisted search**, not a finished
verified report. Every claim below carries its confidence:

| Tag | Meaning |
|---|---|
| **[SDK-verified]** | I checked it myself against the installed `openai` 3.6.0 source in `.venv` |
| **[confirmed 3-0]** | Three independent verifier agents fetched the primary source and agreed |
| **[REFUTED]** | Verifiers fetched the source and found the claim wrong — the correction is given |
| **[extracted]** | Pulled from a primary source by the fetch agent, never independently verified |
| **[blog]** / **[forum]** | Vendor blog or community post — directionally useful, not authoritative |

The five angles were: AEC on Windows in Python; OpenAI Realtime turn-taking; deterministic
conversation-ending; wake-word reliability and earcons; open-source full-duplex frameworks.

---

## Complaint 1 — "hey alexa" sometimes doesn't chime

### 1a. The wake model was never trained on "hey alexa" **[extracted, primary source]**

openWakeWord's own model card for the pretrained `alexa` model says it was trained on the
phrases *"Alexa"* and *"Alexa &lt;random words&gt;"*, and states verbatim:

> Other similar phrases such as "Hey Alexa" or "Alexa stop" may also work, but likely with
> higher false-reject rates.

Saying **"hey alexa" is out-of-distribution for the model we ship**. The misses are expected
behaviour, not a bug in our code. Three fixes, cheapest first:

1. Say just "Alexa" (the substring already fires reliably).
2. Lower `wake_threshold` from 0.5 to roughly 0.3–0.4. The README notes the 0.5 default is
   nominal and "using a lower or higher threshold in practice may result in significantly
   better performance". We already expose this as a setting.
3. Train a custom `hey_alexa` model — the runbook is in `docs/custom-wake-word.md`.

Two knobs that will **not** help: `vad_threshold` only suppresses false *accepts*, so it
cannot fix misses; and `enable_speex_noise_suppression` (which would reduce both error kinds)
depends on `speexdsp-ns`, listed as Linux x86/arm64 only — unavailable on Windows without a
custom build. We currently set neither.

### 1b. `sd.play()` opens a brand-new stream for every chime **[extracted, primary docs]**

The python-sounddevice docs state that `sd.play()` calls `stop()` on any running playback,
then creates a **new** `OutputStream` on the **default** device on every call, and describe it
as "a convenience function for interactive use and for small scripts". For gapless or
low-latency use the docs say to explicitly create an `OutputStream` yourself.

That is precisely our silent-chime failure mode, and we already solved it *inside* a session
(tones are enqueued into the live session speaker because "a fresh sd.play stream loses the
race against a live PortAudio stream" — the comment is in `cues.py`). **The gap is the idle
path**, which still uses `sd.play`. Recommendation: hold one persistent callback-driven
output stream open for the whole idle loop, feed it silence, and mix pre-rendered chime PCM
into the callback.

### 1c. PortAudio never notices a device change **[extracted, primary source]**

The PortAudio wiki states that released v19.7 — what the sounddevice wheels ship — has **no
hot-plug or default-device-change detection**; `Pa_RefreshDeviceList()` exists only on an
unreleased branch, and a stream whose device disconnects has no defined error state
("Device disconnection may require async error reporting… not currently implemented").

The device list and default output are snapshotted at initialisation, so a Bluetooth
reconnect or a Windows default flip to "Steam Streaming Speakers" is invisible to a running
process. A community report **[forum]** additionally claims `sd._terminate(); sd._initialize()`
is *not* by itself enough to refresh the list on Windows, and that a `dlclose`/`dlopen` of the
PortAudio DLL was needed. That casts doubt on our `devices.refresh()`; pinning an explicit
device by name (which we now do) is the more reliable strategy.

### 1d. Target: chime within 250 ms, before the session connects **[extracted, primary docs]**

Amazon's own Alexa Automotive guidance: *"Play the Start of Request sound immediately after
the wake word is detected"*, *"Play the End of Request sound at the end of speech input"*, and
for push-to-talk Alexa "should be invoked immediately (within 250 ms)". The wake sound is tied
to **local detection, not cloud readiness** — which is the pattern we want: fire the chime
from the detector path through an already-open stream, before the Realtime session exists.

---

## Complaint 2 — having to say "hey alexa" to interrupt her

### The strategic answer: GPT-Live is not in the API **[blog — flagged]**

GPT-Live-1 (full-duplex, launched 2026-07-08) is **ChatGPT-only** — iOS, Android and web. There
is no API endpoint, model ID or pricing. OpenAI's public statement is only that they "plan to
bring them to the API soon", with a notification sign-up form; one secondary write-up cites
"weeks rather than months" but gives no date. `gpt-realtime-2.1` remains turn-based.

Flag: openai.com returned 403 to the fetch, so this rests on blogs quoting OpenAI rather than
on the primary announcement. Treat the direction as reliable and the timing as unknown.
**Conclusion: build the fix on gpt-realtime-2.1 now, don't wait.**

### Why the server can never interrupt for us as currently built **[SDK-verified]**

`interrupt_response` fires only on a **server-side VAD start event**, which requires the client
to keep streaming microphone audio while the assistant speaks. Our half-duplex loop stops
feeding the mic during playback, so the mechanism can never trigger. Feed it *without* echo
cancellation and her own voice coming out of the speakers triggers the cancel instead.

So there are exactly two ways forward: cancel the echo, or gate the interrupt client-side.

### Option A — real acoustic echo cancellation

Two pip-installable paths to WebRTC's AEC3 with **Windows wheels**, no compiler needed:

- **`pywebrtc-audio` v0.2.0** (2026-09-03), Python 3.10–3.14, `win_amd64`. One
  `AudioProcessor(echo_cancellation=…, noise_suppression=…)` and a single `process(near, far)`
  call. Roughly 146× real-time at 16 kHz — about **7 ms of CPU per second of audio**.
  **[confirmed 3-0]**
- **`livekit`** (livekit-rtc) — `livekit.rtc.AudioProcessingModule(echo_cancellation=True,
  noise_suppression=…, high_pass_filter=…, auto_gain_control=…)`, with `process_stream()` for
  the mic and `process_reverse_stream()` for the reference. **[confirmed 3-0]**

**We do not need a WASAPI loopback capture.** We already hold the exact PCM we play (the
Realtime audio deltas), and that is a valid far-end reference. **[confirmed 3-0]**

**Frame size — a stale docstring, corrected. [REFUTED]** LiveKit's Python docstring says
"Audio frames must be exactly 10 ms in duration." A verifier read the shipped Rust and found
that `rust-sdks` PR #843 ("allow apm >=10ms frames", merged 2026-01-21) replaced the strict
equality with `assert!(data.len() % samples_per_10ms == 0)` followed by internal chunking.
Every currently installable release contains it. So **our 80 ms openWakeWord frames can be
passed straight in**; only non-multiples of 10 ms are a problem, and they trip a Rust
`assert!` — a panic across the FFI boundary, not a clean Python exception.

**The Bluetooth problem is the real blocker.** **[blog, corroborated]** AEC3 needs the
playback-to-capture delay known to within a few milliseconds. Bluetooth's loop is 20–200 ms
and *moving* — "radio conditions change, buffers adjust, the link renegotiates, and the delay
shifts mid-call" — so "the estimator locks on, the delay shifts, the lock breaks, echo leaks
while it re-locks". Bluetooth is ranked the hardest real case.

**And `stream_delay_ms` is not the knob for it. [REFUTED]** The wrapper builds a
default-constructed `EchoCanceller3Config`, so `use_external_delay_estimator` stays false and
the hint is consumed only as an *initial seed* inside `Reset()`. Changing it at runtime has no
effect until the next reset; ongoing tracking is done entirely by AEC3's internal
matched-filter estimator. The config fields that actually govern large or variable delay
(`num_filters`, `default_delay`, `delay_headroom_samples`) are exposed by neither wrapper. A
2016 forum thread reports that supplying delay hints helped wired devices and was
counterproductive for wireless ones — the opposite of the intuition.

**Practical consequence: AEC will likely work on wired desk speakers and be flaky or useless
on the Echo Dot over Bluetooth.** One bright spot: sealed AirPods have almost no acoustic echo
path at all, so **with AirPods, full-duplex barge-in works without any AEC** — the microphone
simply doesn't hear the speaker.

**Windows' own AEC — two separate mechanisms, and a correction:**

- The Win11 AEC *framework* (`IApoAcousticEchoCancellation`) is driver-side, implemented as an
  APO hosted in `audiodg.exe`. But the conclusion "a Python app therefore cannot touch it" was
  **[REFUTED]**: `IAcousticEchoCancellationControl` (audioclient.h, Build 22621+) lets an
  application detect AEC support and **choose which render endpoint is used as the reference
  stream**; `IAudioEffectsManager` (Build 22000+) enumerates and toggles per-stream effects;
  and WinRT `AcousticEchoCancellationConfiguration.SetEchoCancellationRenderEndpoint` exists in
  24H2. Microsoft ships an application-level sample. All plain COM/WinRT, reachable from Python
  via `comtypes`. What survives: the AEC *algorithm* must still be an APO on the capture
  endpoint, which a generic USB-audio-class Blue Snowball almost certainly lacks — so this path
  probably yields nothing for us, but for the right reason. Note also that the AEC APO is only
  in the path at all if the app selects `AudioCategory_Communications`.
- The **Voice Capture DSP** (`CLSID_CWMAudioAEC`, a Media Foundation DMO, Vista+) is the one
  built-in AEC an application drives itself. In "source mode" it opens and synchronises the
  capture/render pair itself, so **no aligned reference signal is required**. Limits: output is
  8/11.025/16/22.05 kHz mono 16-bit (a resample to 24 kHz is needed), it is a COM DMO reachable
  from Python only through hand-written `comtypes` definitions, the docs were last updated in
  2021, and Bluetooth behaviour is undocumented. **[extracted]**

### Option B — client-side interrupt gating (no AEC; what everyone actually ships)

This is the copyable state of the art, and it is cheap. Four independent implementations agree
on the shape:

- **Kyutai Unmute** (source read directly): while the bot speaks there are two interrupt paths
  — any non-empty STT transcript interrupts, or VAD interrupts *only if* a semantic pause
  prediction is below 0.4 **and** at least 3 s of bot audio has already played. That second
  guard exists explicitly because "the ASR sometimes hears a bit of the TTS audio".
- **Pipecat** `MinWordsUserTurnStartStrategy(min_words=N)`: the N-word threshold applies **only
  while the bot is speaking**; when it isn't, one word triggers. Exactly the asymmetric gate we
  need. Pipecat also ships `WakePhraseUserTurnStartStrategy(phrases, timeout, single_activation)`
  — wake-phrase gating with a re-arm timeout is a first-class supported pattern, i.e. our
  current design is a normal one, not a workaround.
- **LiveKit**: `min_duration` (0.5 s default), `min_words`, `false_interruption_timeout`
  (2.0 s default) and `resume_false_interruption`. The behaviour worth stealing: if VAD fires
  and the agent stops but the transcript comes back empty, it waits the timeout, emits
  `agent_false_interruption`, and **resumes speaking from where it left off**. Their adaptive
  model (encoder + CNN; 30 ms inference, 216 ms median audio needed, 86% precision / 100%
  recall at 500 ms overlap, 51% fewer false positives than VAD) is LiveKit-Cloud-only, but the
  *design* is copyable: a ~1 s onset cooldown with VAD fallback, then transcript confirmation.
- **Deepgram**: require at least 2 words in an interim result, or a final with at least 1 word;
  and run a "software echo canceller" by comparing the STT output against the text the agent
  just spoke, discarding on match. We already receive her output transcript, so an n-gram
  overlap check costs nothing.
- **Moshi** is the cautionary note: even a true full-duplex speech model recommends the browser's
  echo cancellation, and its CLI clients do none. **Barge-in quality is bounded by the capture
  chain, not by the model.**

**Recommended for us:** keep the mic streaming during playback, but gate acceptance in our own
code — energy/VAD pauses playback, the transcript within ~2 s confirms it (otherwise resume
from the buffered position), plus an n-gram check against what she just said and a ~1 s
post-onset cooldown. That buys talk-over on the Snowball with no AEC and no Bluetooth
dependency, and it degrades gracefully.

### The truncation obligation we are currently ignoring **[SDK-verified]**

Over WebSocket — our transport — the server does not know how much audio we actually played.
On any barge-in the client must stop playback, measure the played milliseconds, and send
`conversation.item.truncate` with `item_id`, `content_index` (must be 0) and `audio_end_ms`.
Only assistant items can be truncated, and if `audio_end_ms` exceeds the real duration the
server returns an error, so the value must be clamped. Truncation deletes the unheard audio
**and its transcript**, so the model's memory matches what he actually heard.

**We do not do this.** After a wake-word barge-in today, she believes she said the whole
sentence. That is a real bug with a cheap fix, independent of everything else in this report.

---

## Complaint 3 — she is bad at ending

Three primary sources describe the same state machine, and we match none of it:

- **LiveKit `EndCallTool`** — `end_instructions` (default "say goodbye to the user") is returned
  **as the tool output**, so the model's next reply *is* the farewell. For realtime models it
  defers shutdown in a background task, waits up to 5 s for that reply's speech to be created,
  awaits the speech handle to finish playing, and only then shuts down. **The agent never takes
  another user turn after the tool fires.**
- **Pipecat** — inside the function call: push the farewell, resolve the tool callback, then push
  the end frame. Ordering is guaranteed because "EndFrame is queued and processes after any
  pending frames (like goodbye messages)". `CancelFrame` is the immediate, discard-everything
  variant.
- **Vapi's call-ended taxonomy** is a useful design checklist: `assistant-ended-call` (a tool),
  `assistant-said-end-call-phrase` (a phrase trigger matched on the **assistant's** speech),
  `assistant-ended-call-after-message-spoken`, `silence-timed-out`, `exceeded-max-duration`.
  Two deterministic strategies combined: an explicit tool **and** a phrase trigger on her own
  closing words as a backstop.

**Pitfall** (livekit/agents issue #5742): do not tear down on `response.done` or the function
call alone. A realtime model can emit the tool call before flushing its final audio chunks,
which cuts the goodbye off mid-syllable. Wait for audio quiescence (~0.5 s with no new frames)
*and* for local playback to drain.

**WebSocket specifics** **[forum — flagged, but directly on point]**:
`output_audio_buffer.stopped` is emitted for WebRTC and SIP only. Over WebSocket, audio is
finished only when you have the `.done` event **and** your local playback queue is exhausted.
`response.output_audio.done` marks the end of the audio *data*, not of playback.

**`idle_timeout_ms` cannot help us here** **[SDK-verified]**: the field exists on `ServerVad`
and is absent from `SemanticVad`, so it is unavailable while we use semantic VAD — and it makes
the model *speak again* to prompt the user, rather than closing the session. It is not a
replacement for our 8/15/45 s engine timers.

---

## Complaint 4 — slow to re-activate after she stops listening

This one's root cause was found in our own code rather than in the literature: after every
conversation the runner **awaited the reflection call** (a Claude CLI subprocess, several
seconds) *before* reopening the wake-word microphone. That is a deaf window at exactly the
moment you would naturally speak again — which matches "it takes several tries to reactivate".

The research adds one free improvement: `noise_reduction` accepts `near_field` or `far_field`
**[SDK-verified: `NoiseReductionType = Literal["near_field", "far_field"]`]**, and the filtering
happens **before** VAD and the model. `far_field` is the documented choice for a desk-mounted
microphone at speaking distance. We currently send none at all.

---

## Already fixed locally (tested, uncommitted as of this writing)

Two root causes found while reading our own engine during the research:

1. **The barge-in flag was never cleared.** `interrupted` was set by a wake-word barge-in and
   stayed set for the rest of the session, and the goodbye path is vetoed when `interrupted` is
   true. So after any interruption, "that's all" could never close the conversation: she would
   say "closing out", then re-arm and listen until the 45 s idle timer. It is now reset per
   response.
2. **Only the model could end a conversation.** The engine now treats a whole-utterance wrap-up
   ("that's all", "thanks, bye", "never mind", "good night") as authoritative: it closes after
   her goodbye drains whether or not the model calls the tool, and a wrap-up that lands with
   nothing playing closes after a 1.5 s grace instead of opening a listening window.
3. **Reflection moved off the critical path**, closing the deaf window in complaint 4.

Still to adopt from this research: return the farewell instruction *as the tool result*
(LiveKit's trick) so her goodbye is deliberately generated rather than hoped for, and add an
assistant-phrase trigger as a backstop.

---

## Ranked recommendations

| # | Change | Fixes | Effort | Risk |
|---|---|---|---|---|
| 1 | Say "Alexa", and lower `wake_threshold` to ~0.35 | 1 | minutes | none |
| 2 | One persistent idle output stream; chime mixed into its callback | 1 | small | low |
| 3 | Send `conversation.item.truncate` on every barge-in | 2 | small | none |
| 4 | Set `noise_reduction: far_field` | 1, 4 | minutes | low |
| 5 | Client-side interrupt gating: stream the mic during playback, confirm by transcript within 2 s, resume if empty, n-gram check against her own words, 1 s onset cooldown | 2 | medium | medium — needs live tuning |
| 6 | Farewell-as-tool-result + wait for audio quiescence before closing | 3 | small | low |
| 7 | Train a custom `hey_alexa` wake model | 1 | hours (Colab) | low |
| 8 | WebRTC AEC3 via `pywebrtc-audio`, wired speakers only | 2 | large | high on Bluetooth |
| 9 | Windows Voice Capture DSP via comtypes | 2 | large | high, poorly documented |

Items 1–4 are unambiguous wins. Item 5 is the one that actually delivers "just talk over her".
Item 8 is the purist's answer and the one most likely to disappoint on the Echo Dot.

---

## Sources

Primary: openWakeWord README and `alexa` model card; python-sounddevice convenience-function
docs; PortAudio HotPlug wiki; Alexa Automotive "Invoking Alexa"; OpenAI Realtime VAD guide,
Realtime conversations guide, client-secrets session reference, and developer blog;
`openai-python` realtime types; `strands-labs/pywebrtc-audio`; `livekit/python-sdks` `apm.py`;
Microsoft Learn on Win11 APO APIs and the Voice Capture DSP; Pipecat pipeline-termination and
user-turn-strategies; LiveKit EndCallTool, turns overview and adaptive interruption handling;
Kyutai `unmute_handler.py`; Moshi README; livekit/agents issue #5742.

Secondary/vendor: Fora Soft on AEC with Bluetooth and AirPods; Deepgram barge-in guide;
PyAudioWPatch (WASAPI loopback); apidog on GPT-Live. Forum: webrtcHacks Realtime guide;
OpenAI community thread on detecting audio completion; python-sounddevice issue #516.

## Still unverified — worth re-running when usage allows

The Voice Capture DSP's source-mode behaviour and format limits; PyAudioWPatch's loopback
device enumeration; whether Windows 11 swaps the AEC reference stream automatically when the
render device changes (directly relevant to the Echo Dot); and the LiveKit APM's
`set_stream_delay_ms` contract. None of these block recommendations 1–7.
