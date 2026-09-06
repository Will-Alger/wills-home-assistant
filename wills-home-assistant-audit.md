# Making wills-home-assistant feel like magic

Audit date: September 5, 2026. Reviewed Desktop working tree at commit `c686631`, including existing uncommitted changes. This is a source and interaction-design audit, not a live listening evaluation. No application code was changed.

Updated September 5 after reviewing the project's `docs/VOICE-UX-RESEARCH-2026-09-05.md`, checking relevant current code and installed SDK types, and consulting the official OpenAI VAD guide. The delivery order below supersedes the initial audit's ordering. The research report's remaining external claims have not all been independently reverified.

**My assessment**

The app already has the ingredients of a compelling personal assistant: speech-native conversation, household tools, persistent memory, background reasoning, an asynchronous task board, reminders, and delivery that considers presence and focus. The biggest improvement would come from making these capabilities behave as one continuous conversation.

“Magic” here means you can speak imprecisely, pause, change your mind, return to a topic, and trust that the assistant knows what happened. The current architecture still exposes some of its machinery through early closing, delayed corrections, lost follow-ups, stale answers, and task-oriented wording. A future model could improve judgment, but these state and execution problems would remain.

The research clarifies the immediate user priority: reliable activation, natural interruption, predictable endings, and quick reactivation. My first release would therefore focus on **activation and audio reliability, responsive interruption, and deterministic closing**. Conversational continuity, honest completion, and repair remain essential and follow those foundations. The finding numbers below are stable references; use the revised delivery table for implementation order.

**Scope and confidence**

I examined the running Realtime engine and runner; home tools and client; memory, reflection, recent-session storage, routines and delivery; background reasoning; relevant test fixtures and tests; and the README, feature inventory, and existing Clicky review. I treated source behavior as stronger evidence than roadmap claims. The README still leads with an older STT → Claude → TTS architecture, while the active runner uses the Realtime engine.

I did not read `.env`, `data/`, or `logs/`, consistent with the repository's `CLAUDE.md`. Consequently, saved lessons, actual device configuration, personal conversation history, and deployed settings were not assessed. Recommendations about timing and personality are hypotheses to evaluate by ear, not measured shortcomings of your current voice.

The initial offline test run could not complete: the project Python launcher was denied execution; a bundled Python fallback reached collection but native NumPy and Pydantic extensions were denied loading. No passing test result is claimed. These were limitations of the audit environment at that time, not evidence that your normal installation is broken. Tests were not rerun for this document-only revision. No paid API evaluations or real household actions were run.

**Research findings incorporated into the audit**

Three fixes described as locally completed in the research are present in the reviewed working tree: the interruption flag resets per response, whole-utterance user wrap-ups can close the session without a model tool call, and reflection runs in a background task so it does not delay reopening the wake microphone. Preserve and regression-test these; they are not new implementation tasks. Their presence in source does not establish live audio reliability.

**Activation and chimes.** The idle chime still uses `sd.play()` in `audio/tones.py`; session output uses `Speaker` and a persistent callback stream. Establish clear output ownership across idle and conversation states, preferably reusing a stream where the device permits it. Include disconnect/reconnect and device-switch recovery. A persistent stream can reduce repeated initialization, but does not by itself solve device removal. The current chime code also selects a saved output before falling back to the default, so it is not accurate to describe this app's chimes as always default-device playback.

Separate wake detection, chime enqueue, actual playback, microphone readiness, and cloud readiness in telemetry. A missing chime does not prove the wake model missed the phrase. Preserve speech immediately following activation through a bounded capture buffer; otherwise a faster chime can invite the user to speak before the conversation path is ready.

The research's Alexa model-card finding supports comparing “Alexa” with “hey Alexa,” but attributing all misses to phrase training is premature. Lowering the threshold to around 0.35 is a tuning experiment, with a false-activation tradeoff. Compare thresholds using intended phrases, TV/music, and ordinary household speech before adopting a default. A custom model is appropriate if the preferred phrase remains unreliable.

**Noise reduction has a specific scope.** Trial `far_field` for the desk microphone during Realtime sessions. It cannot improve idle openWakeWord detection: that local detector processes audio before the Realtime connection. Evaluate its effect on in-session speech and false interruptions separately from wake-word recall.

**Interruption requires a tentative state.** The proposed transcript-confirmed gate is a useful prototype, but it needs the following explicit lifecycle:

| State | Required behavior |
| --- | --- |
| Speaking | Track the current output item and playback position; retain resumable audio |
| Possible interruption | Pause locally, keep the receive loop responsive, and collect bounded evidence without immediately discarding the response |
| False alarm | Resume retained audio from the paused position; prevent rejected echo from becoming a real user turn |
| Confirmed interruption | Cancel remaining generation when applicable, discard the unheard tail, reconcile server history, and process the user's speech |
| Failure or timeout | Use an explicit fallback and bounded recovery; never stay indefinitely paused |

If the client owns the acceptance decision, it must explicitly control automatic server cancellation and response creation. Both `interrupt_response` and `create_response` can be disabled while VAD events continue. Merely streaming the mic with default turn handling lets the server act before the local gate decides. This is supported by the installed SDK and the [official OpenAI VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad).

A local transcript rejection also needs a server-history strategy: microphone audio sent to the main conversation may already have produced an input item. Define how rejected echo is excluded or removed before generating a response. Treat transcript overlap with the assistant as one signal, not an unconditional veto: the user may intentionally repeat its words. Likewise, one-word “stop” and “wait” need a path through a multiword gate. Do not cancel and truncate during a tentative pause if false-alarm recovery depends on resuming the original audio.

**Playback and truncation.** The installed `ConversationItemTruncateEvent` documents synchronizing server history with client playback and deleting the server-side transcript when audio is truncated. On confirmed interruption, send the current assistant item ID, content index, and bounded `audio_end_ms`. Track acknowledgment and late audio belonging to the interrupted response. The current `Speaker` exposes queued duration and an approximate drain wait, not per-item audible position. Implement that accounting before treating truncation as a trivial event-send change; callback consumption and sound reaching the listener are not identical, especially with buffered outputs.

**Closing.** Retain the existing user-wrap-up fixes. For model-requested closing, explicitly return a farewell instruction when another response is needed, then wait for that response's audio completion and local playback drain with a bounded fallback. A short quiescence guard can supplement event ordering; it should not be the only completion signal. Do not make arbitrary assistant phrases authoritative: an incidental “goodnight” must not close an ongoing discussion. Preserve a deliberate “wait…” escape during an ordinary farewell; a hard user stop should remain immediate.

**Evidence limits.** Library availability, SDK schema support, and verifier agreement do not establish acoustic performance on the Snowball/Echo Dot setup. Treat claims that Bluetooth AEC will be useless, headphones always eliminate the need for AEC, or transcript gating is what everyone ships as hypotheses or scoped observations. Keep wired and Bluetooth trials separate. An 80 ms frame being accepted by a wrapper also does not establish that it is the best scheduling interval for responsive interruption. AEC's reference should follow the audio actually rendered, including relevant local sounds and timing, rather than the arrival time of cloud audio chunks. No model-release timing claim is needed to justify these improvements.

**What is already worth preserving**

- Home commands and ordinary conversation share a speech-native engine. Keep that continuity.
- Routines can deterministically apply household preferences, rather than relying entirely on prompt recall.
- Background development and reasoning do not require keeping the original session alive.
- Notifications already distinguish delivered and read, and have presence, focus, and quiet-hour policies. Improve their lifecycle rather than replacing the system.
- Reflection is selective and explicitly discourages learning permanent limitations from transient failures.
- The fake house and scripted Realtime connection provide useful foundations for testing behavior without touching devices.

**Prioritized findings and suggestions**

**1. Keep the conversation receiver responsive while tools run. Priority: highest.**

Evidence: `src/assistant/engines/realtime_engine.py:2250` handles `response.done` by awaiting `_handle_response_done`; that method awaits tool execution inside its output loop (`:1360`). Until a slow tool returns, this receiver cannot consume subsequent speech, transcription, cancellation-related, or connection events. The mic pump is separate, but receiving and interpreting the user's correction is delayed.

What this feels like: “Search for…” followed by “Actually, never mind” can leave the assistant finishing the obsolete request before processing the correction.

Move tool work into supervised tasks and keep the socket receiver reading. Associate each call and result with the originating turn. Cancel obsolete read-only work; suppress obsolete speech. For household mutations, distinguish “not started,” “already sent,” and “confirmed complete”—cancelling a Python task does not undo a light change. Preserve ordering for dependent actions, while allowing independent reads to run together.

Also supervise the receiver and mic tasks. `run_conversation` waits on `ended`, while child-task exceptions are only suppressed during teardown. A failed receiver can leave a session waiting for a timeout, or longer if `response_active` stays true. A connection failure should close the session through one bounded recovery path.

Acceptance: inject a slow search, then a user correction; the correction is processed before the search finishes. Inject a receiver failure while a response is active; the app exits to a recoverable idle state within a defined timeout.

**2. Track what was played, not just what was generated. Priority: highest.**

Evidence: interruption clears the speaker queue and sends cancellation (`realtime_engine.py:2133` onward), but I found no corresponding reconciliation of conversation history with how much audio was actually heard. Complete assistant transcripts are appended on transcript completion (`:2207` onward). Mid-session notifications are marked delivered and read at `response.done` (`:2260` onward), before `finish_playback` has waited for the speaker to drain.

What this feels like: interrupting a long answer can leave the assistant assuming it already explained the part you never heard. News can disappear from unread even when you cut it off.

Track response/item IDs and the played-audio position. Reconcile interrupted assistant history with playback, and keep local reflection from treating an unheard tail as shared knowledge. Represent notification stages separately: queued, selected, playback started, playback completed, acknowledged. Playback completion still does not prove attention; use explicit acknowledgment or an appropriate reply when that distinction matters.

Acceptance: interrupt a two-part announcement after the first part. “What did I miss?” must preserve the unheard news. “Explain that last part” must not assume the unheard explanation was delivered.

**3. Do not retire follow-ups merely because a session happened. Priority: highest; relatively small change.**

Evidence: `_session_config` collects every pending conversation follow-up into `_raised_followups` (`realtime_engine.py:1249`). At session end, any user reply causes all those IDs to be marked raised (`:2362`), without checking that the assistant mentioned them.

What this feels like: “Next time we talk, ask me how the demo went.” Later you say only “Turn off the hallway.” The promise can be consumed without being fulfilled.

Give each follow-up an explicit raised/delivered acknowledgment tied to a response. Keep it pending when the topic was omitted or interrupted. Apply the same principle when `list_notifications` currently marks items read before the spoken response is played (`:1824` onward).

Acceptance: a brief lighting command leaves the demo follow-up pending; actually asking the question and completing playback advances its state once.

**4. Add a compact working context across sessions. Priority: high.**

Evidence: `sessions.py` stores the first user line, a summary, tool names and statistics. The default instruction context includes only three sessions from the last day. It does not explicitly preserve the last referenced device, unresolved question, selected search result, proposed action, or last reversible change.

What this feels like: after a session closes, “Make it a little warmer” or “Let's do the second option” has weak grounding.

Persist a small working-context record: current topic, salient entities, last successful action, pending question and options, temporary overrides, and outstanding jobs. Include timestamps and short expirations; distinguish conversation references from durable personal memories. Resolve “it,” “that,” and “the other one” against this record. Ask one specific question when two candidates are plausible.

Acceptance: set the living-room lamps, allow the session to close, then say “A little dimmer.” It targets the same lamps if the context is still fresh. After intervening media activity, ambiguous “turn it off” triggers a brief clarification.

**5. Make repair and undo real capabilities. Priority: high.**

Evidence: the tool surface provides actions, routines and removals, but I found no general action ledger and undo operation. Lighting execution builds commands and returns “Done” after `home.apply` (`brain/tools.py:537`); the real client applies per-entity commands sequentially (`home/client.py:120`). A failure partway through can leave a partially changed room.

Add an action receipt with target IDs, before-state, requested state, actual outcome, and whether reversal is supported. Report partial success specifically. For compatible lighting changes, group service calls where practical; one model tool call currently does not imply one household request. Verify resulting state when feasible, using an appropriate bounded wait for transitions.

Support “Undo that,” “No, the bedroom,” “Keep the brightness but change the color,” and “Only this time.” Limit undo to operations with a reliable inverse; explain when an external effect cannot be undone. Avoid blindly repeating a mutation after an uncertain timeout.

Acceptance: fail the second of three light updates. The assistant identifies partial completion, offers a useful recovery, and never claims that every light changed.

**6. Give conversation its own pacing policy. Priority: high, validate by ear.**

Evidence: the instructions cap ordinary conversation at a sentence or two, prohibit unsolicited follow-up suggestions, and say “when genuinely unsure ... close” (`realtime_engine.py:81–269`). Defaults use high semantic-VAD eagerness, an eight-second command window, fifteen seconds after an initial answer, and forty-five seconds of idle (`config.py:154` onward). The prompt asks for immediate command closing, so the eight-second backstop does not guarantee a follow-up opportunity.

Keep short command responses, but distinguish command, conversation, waiting-for-answer, and user-thinking states. If the assistant asks a question, it should expect an answer. “Let me think” should lengthen the pause allowance; “that's all” should end promptly. These should be consistent engine behaviors rather than contradictory prompt hints.

Offer push-to-talk as an optional desktop activation path, as the existing Clicky review proposes. For natural spoken interruption through desktop speakers, treat echo handling as part of the work; simply enabling the existing talk-over option is not sufficient evidence of good room performance.

Acceptance: pause mid-explanation, answer a question after a reflective pause, insert a light command during a conversation, and resume the original topic. Measure premature endings and unwanted lingering separately.

**7. Make the deeper mind belong to the current topic. Priority: high.**

Evidence: `thinker.py` receives up to twenty-four transcript lines and returns three to six sentences, with no tools. `_execute_brain_tool` snapshots context, launches a background task, and queues its answer as an urgent thought with a three-hour expiry (`realtime_engine.py:1580`). There is no exposed cancellation or supersession mechanism in this path. Urgent delivery can bypass quiet hours.

The voice-plus-deeper-reasoning design is valuable. Add a topic/job ID, status, cancellation, and a freshness check. “Actually, assume I stay here another year” should update or supersede the old question. On return, reconcile the result with intervening conversation. Deliver a requested thought promptly when still relevant; reserve urgency that overrides household settings for cases that need it.

For questions depending on current facts, provide verified retrieval results to the thinker or give it a controlled retrieval path. A reasoning pass with `tools=[]` cannot independently establish new facts.

Acceptance: change a major assumption while a thought runs. Only the applicable conclusion is spoken, with a short contextual bridge such as “With the extra year in mind…”

**8. Refine memory into evidence with scope. Priority: medium.**

Evidence: `MemoryItem` has only ID, kind, text and creation date; preferences are all injected, lessons are selected by recency, and reflection deduplicates exact strings (`memory.py`, `learning.py`). Lessons are introduced as “house truths.” Preference replacement is a prompt-directed forget-then-remember sequence.

Add subject/scope, source, last verification, confidence and supersession. Make preference updates atomic so an interrupted replacement cannot discard the old preference without saving the new one. Retrieve relevant memories instead of indefinitely growing the prompt. Let verified tool outcomes support operational lessons; a single reflection should not turn an uncertain inference into permanent instruction.

Keep household defaults separate from personal information. Since anyone in the room can talk and voice identity is not implemented, use an explicit guest/shared interaction mode where needed. “Remember this for me” and “Make this the house default” are different intents.

Acceptance: a temporary music outage never becomes “music is unsupported.” A personal preference can be updated, explained, and forgotten without a contradictory duplicate.

**9. Introduce one consistent contract for skills. Priority: medium.**

The existing tools are broad, but their user-facing contracts differ. Home actions, calendar writes, scheduling, memory, background jobs and code approval each have their own handling paths.

Give each capability a descriptor covering required information, defaults, missing-information behavior, expected duration, side effects, retry policy, completion evidence, cancellation, and reversibility. Use structured internal results—success, partial, pending, unavailable, or needs clarification—with separate speech summaries. Reduce reliance on reading human error strings as control flow.

Keep the current curated tools. Gradually organize their instructions by capability and conversational state, so the voice does not have to reread a long development runbook for every ordinary exchange. Measure routing accuracy before hiding tools dynamically.

| Capability | Most useful next improvement | Desired experience |
| --- | --- | --- |
| Lights and media | Working references, action receipts, temporary overrides | “The other lamp. Same brightness.” |
| Calendar and reminders | Explicit pending proposals and corrections | “Actually Thursday, same time.” |
| Routines | Clear defaults versus mandatory overrides; one-use exception | “Cool white just for tonight.” |
| Memory | Scoped, replaceable memories with evidence | “I changed my mind about that.” |
| Background thought | Topic tracking and supersession | A useful answer to the revised question |
| Development tasks | Reference resolution and concrete approval state | “Try the microphone fix” without memorizing task IDs |
| Notifications and follow-ups | Playback-aware delivery, explicit completion, batching | Relevant news once, at an appropriate moment |

For routines specifically, defaults already respect explicit light colors, which is good. Mandatory overrides always win in `routines.py`. Make that difference understandable and provide a deliberate one-time exception for ordinary preferences, without silently bypassing rules intended to be mandatory.

**10. Tune personality through conversational moves. Priority: medium.**

Adding “be warm and natural” to the prompt is unlikely to fix the main experience. Define a few observable behaviors instead:

- Answer the substance first; avoid announcing tool mechanics.
- In a reflective conversation, contribute one useful observation or specific question when it advances the topic.
- Make light commands brief without ending the surrounding discussion.
- Use remembered context sparingly, when it helps, rather than reciting memory.
- Admit a misunderstanding plainly and repair it immediately.
- Avoid a generic “anything else?” at every turn; do not replace it with a compulsory question or anecdote.

Example: “I'm not sure this project is worth the time.” A better response is “Which part is wearing you down—the maintenance, or that it still doesn't feel natural to talk to?” That opens a useful conversation without immediately dispatching a feature or turning the interaction into a project-status report.

An illustrative core rule: “Match the depth of the moment. Be brief for an action and engaged for a conversation. Ask only questions that help. When corrected, adopt the correction and continue. Keep the topic alive across incidental household commands.” Treat this as a candidate for evaluation, not a full replacement prompt.

**11. Measure the experience on the actual engine. Priority: high; begin immediately.**

Evidence: `tests/test_evals.py` contains six opt-in, single-command evaluations against the Anthropic text Agent. They are useful for home behavior, but they do not establish the quality of the running Realtime conversation. Scripted tests cover many lifecycle rules; some currently assert immediate read marking. `InstantSpeaker` completes without real playback delay, so it cannot expose all generated-versus-heard problems.

Create an offline event-replay suite with delayed speakers, late transcripts, slow tools, interruptions and connection failure. Separately maintain a small consented voice evaluation set on the actual engine. Existing mocks cannot judge whether a response feels rushed or whether the model chooses the right tool.

Track speech-end to first audible answer, time to physical action, interruption-to-silence, correction recovery, premature close rate, repeated wake words, unfulfilled follow-ups, and unsupported success claims. Report median and slow-tail latency by task type. Keep telemetry content-light by default; recordings and transcripts are a separate choice.

Candidate release targets, to calibrate against your setup: local cancellation silences output within about 250 ms; ordinary successful actions get only a short acknowledgment; long waits get one useful cue; interrupted or omitted follow-ups are never silently retired in replay tests. These are proposed acceptance targets, not measured results or guarantees.

**Suggested delivery order**

| Sequence | Work package | Why it goes here |
| --- | --- | --- |
| 1 | Separate activation timings; delayed-playback replay tests; regress the three existing fixes | Distinguish missed detection from missing chimes and readiness gaps |
| 2 | Persistent output ownership, reconnect handling, activation capture buffer, and per-item playback accounting | Make the chime dependable and preserve immediate user speech |
| 3 | Responsive receiver and supervised failures; confirmed-interruption truncation; deterministic closing | Make stop, correction, and ending reliable before adding automatic talk-over |
| 4 | Prototype tentative pause, transcript confirmation, rejected-echo handling, and false-alarm resume | Evaluate natural interruption with explicit recovery on the actual hardware |
| 5 | Controlled wake-threshold and far-field trials; optional push-to-talk; custom wake model if needed | Tune separate detection paths against measured false accepts and misses |
| 6 | Follow-up and notification lifecycle fixes; working context, action receipts and limited undo | Preserve promises and make references and changes of mind succeed |
| 7 | Conversation pacing, topic-aware thoughts, scoped memory and skill contracts | Improve sustained discussion and consistency across capabilities |

The Clicky review's working cue, local spoken fallback, transcription vocabulary and optional push-to-talk remain sensible candidates. I would prioritize them according to observed friction, and defer screen awareness, new integrations, and more proactive behavior until interruption and continuity are solid. More things to announce will amplify the current delivery issues.

**A practical voice acceptance script**

Run these before and after changes, with the same devices and room conditions. Record success, awkward moments, unnecessary clarification, wake-word repetitions and latency—not just whether a tool was called.

1. “Make the living room cozy.” After session close: “A little brighter.”
2. “I'm trying to decide whether to move…” Pause mid-thought and continue.
3. During that discussion: “Dim the lights a bit.” Then: “Anyway, the commute is the problem.”
4. Ask for a long answer; interrupt halfway. “Explain the last thing I heard.”
5. Start a slow search. “Actually, don't bother.”
6. “Next time we talk, ask me how the demo went.” End, then issue a one-shot light command. The promise must remain available if omitted.
7. “Help me think through buying this.” While reasoning runs: “Assume I wait a year instead.”
8. Set an ordinary warm-light default. “Cool white, just this once.” Then issue another ordinary command.
9. “Undo that.” Test successful, partially successful and non-reversible actions.
10. Start a multi-item announcement and interrupt it. “What did I miss?”
11. Ask for an ambiguous change with two plausible targets. It asks one specific clarifying question.
12. Simulate a household-service or voice-connection failure with fakes. It reports uncertainty honestly and returns to a usable state.
13. Say “Alexa” and “hey Alexa” at different distances, then repeat with television and music. Log detector scores and chime playback separately; compare misses and false activations for each threshold.
14. Speak a command immediately after the wake phrase. The beginning of the command is preserved while the cloud connection opens.
15. Cough or make a brief noise during an answer. If playback pauses, it resumes without repeating or skipping the answer and without adding echo as user speech.
16. Interrupt with “stop,” “wait,” and a phrase that repeats the assistant's last words. Real interruptions are accepted despite word-count or transcript-overlap heuristics.
17. Disconnect and reconnect the selected speaker, then activate again. The chime and response use a functioning output and state reporting reflects the actual device.
18. Say “that's all,” then deliberately say “wait” during the farewell. Check ordinary closing recovery separately from an immediate hard stop. Repeat rapid activation after closing to verify reflection causes no deaf window.

Judge the next release first on dependable activation, interruption, ending, and reactivation; then on how well it preserves the conversation's unfinished business.
