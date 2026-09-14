"""GPT-Live: the full-duplex engine.

One WebSocket per conversation, opened after the local wake word, on
`gpt-live-1`: the model listens while it speaks, decides itself when to talk,
backchannels, and stops when he talks over it. Reasoning and every tool go
to a backend model the live model delegates to (Responses delegation); the
engine feeds clean audio up, runs the tools the backend asks for, and owns
the endings and the money — an open session bills by the second.

What the first probe runs taught, and what this file is built on:

- Output audio is a continuous real-time stream once she starts, silence
  included. "She is talking" is the energy of the last deltas, never the
  fact that deltas arrive; "she stopped" is a beat without her voice.
- `session.commentary.append` makes her say something; a "speak first"
  instruction does not. Openers and mid-session events go through
  commentary, behaviour nudges through instructions.
- Transcripts come word by word, about 150 ms ahead of the audio, on a
  200 ms grid. The SDK's grouping policy turns them into turns.
- The server does not cancel echo. Her own voice at the microphone is
  learned during her first reply: small (headphones, wired speakers) and
  every frame goes up — talk over her freely; large (a Bluetooth speaker
  beside the mic) and the engine feeds silence while she plays, and the
  wake word cuts in. Same contract as the realtime engine has today.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from assistant.audio.acks import normalise
from assistant.audio.level import LevelMeter
from assistant.audio.level import rms as _rms
from assistant.engines.realtime_engine import (
    _INJECT_MIN_AGE_S,
    _INJECT_QUIET_S,
    _INSTRUCTIONS,
    _NO_SPEECH_S,
    _PLAYED_WINDOW_S,
    _PLAYING,
    _PTT_MIN_HOLD_S,
    _PTT_POLL_S,
    _SILENCE_FRAME,
    _SUSPECT_LEVEL,
    _THINKING_S,
    _WORKING_CUE_AFTER_S,
    _WORKING_CUE_EVERY_S,
    COMMAND_TOOLS,
    REALTIME_RATE,
    RealtimeEngine,
    SessionStats,
    _her_own_words,
    _Levels,
    _SpeechGate,
    downsample_24k_to_16k,
    is_stop_command,
    is_thinking,
    is_wrapup,
)
from assistant.engines.transcripts import Segment, Segmenter
from assistant.latency import TurnTrace
from assistant.music import PLAY_WORD as _PLAY_WORD
from assistant.music import parse_play_request

LIVE_PRICE_PER_MIN = 0.05  # gpt-live-1, billed per second
# per 1M tokens: input, cached input, output — the backend's share
BACKEND_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
}
LIVE_VOICES = frozenset(
    ["alloy", "ash", "ballad", "beacon", "bossa", "cedar", "cinder", "coral", "delta", "echo", "gleam", "marin", "meridian", "quartz", "ripple", "sage", "shimmer", "stone", "tempo", "verse", "vesper", "willow"]
)
DEFAULT_LIVE_VOICE = "marin"  # also the voice her cue clips are rendered in

# Her voice in the stream: a delta louder than this (int16 RMS) is her
# speaking; the silence the stream carries between words sits at 10–30.
_SPEECH_RMS = 200.0
# This long without her voice in the stream and the reply is over — the
# stream itself never stops. Her pauses between sentences are shorter.
_OUTPUT_QUIET_S = 0.7
# A wake-word barge-in: the mic goes up ungated for this long so the model
# hears him and yields on its own, and whatever she is still saying is not
# played for this long (the instruction to stop takes her a moment).
_BARGE_UNGATE_S = 3.0
_BARGE_MUTE_S = 1.5
# The end tool fired: she gets this long to say her goodbye before she is
# asked for one, and that long again before the engine closes anyway.
_FAREWELL_MAX_S = 4.0
# "That's all": she gets this long to start her closing word before the
# engine closes without one.
_WRAPUP_GRACE_S = 2.5
# His words arrive one at a time: a stop, a wrap-up or "let me think" counts
# once the turn stopped growing for this long (or it closed).
_SETTLE_S = 0.8
# A backend delegation that never reports completion stops counting as
# "busy" after this — nothing may hold the idle clock forever on Live.
_DELEGATION_MAX_S = 90.0
_START_TIMEOUT_S = 8.0
_CLOSE_TIMEOUT_S = 2.0
_MUSIC_ACK_S = 1.2
_TICK_S = 0.05

_STOP_LINE = (
    "The speaker just interrupted you: stop talking now, do not finish the sentence, and listen."
)
_NO_SEARCH_EFFORTS = ("none", "minimal")  # the backend's web_search refuses these
_GOODBYE_LINE = "A goodbye of two or three words, then stop."

LIVE_INSTRUCTIONS = """You are {name}, the voice of {owner}'s home. Calm, warm and natural, at an unhurried \
pace, in short sentences. You are a person in the room, not a product: no "how can I help you \
today", no reading of lists, no announcing what you are about to do.

Speaking aloud: everything you say is heard, never read. Say times, numbers and names the way \
a person does ("half past six", "seventy percent"). Never say the word "{wake_phrase}" yourself.

Backchannels: a brief "mm-hm" only while {owner} tells a longer story. Never over a command, \
never while he is mid-word, never while he is thinking.

Interruptions: the moment he talks over you, stop mid-word and listen. Never resume the \
sentence unless he asks. "Wait", "stop" and "hang on" are instant.

Silence: keep listening while he pauses to think — "let me think" means wait in silence. A \
cough, music, a television or a nearby conversation is not a request.

What you know and what you do: you know nothing about the home, the lights, the music, the \
calendar, timers, reminders, his memory, his projects or the web, and you act on none of it \
yourself. Anything that needs a fact, a lookup or an action goes to your backend AT ONCE, \
before you speak — at most one short word while it works. When his command came in the same \
breath as your name, a brief "Sure thing" or "On it" is the whole acknowledgment. The \
conversation opens on your name: if the first thing you hear is only your name, answer with a \
single "Yes?" or "Mm-hm?" and wait — no remark on how it was said, no question about what \
happened, nothing more until he speaks. If you did not hear your name at all, or only a stray \
word or a noise, say nothing and wait. Say its answer once, in one breath, \
in your own words, never the mechanics ("done", not "I've called the tool"). Never claim \
something is done, set or found before the backend says so; if it says pending, say it is \
underway and carry on. A bit of chat, or a question the conversation itself answers, needs \
no backend.

The backend can: control lights and media, set timers, alarms, reminders and scheduled \
actions, read and write his calendar, remember and recall things, search the web, run deeper \
thinking, watch the house for events, manage his task board and settings, and end the \
conversation.

Endings: after a one-shot order, confirm in a few words and stop — the house closes the \
conversation. When he wraps up ("that's all", "thanks, bye"), one closing word and stop. A \
goodbye is never followed by a question. Otherwise, when a reply is done, stop and listen; \
no "anything else?".
{extra}"""

BACKEND_PREAMBLE = """You are the reasoning and tool backend of {name}, a voice assistant in {owner}'s home. \
A live voice model talks to him and hands you, as text, whatever needs a fact, a lookup or an \
action. Transcripts can contain mistakes, unfinished phrases and later corrections: use the \
latest context. Act with your tools, then return ONLY what the voice should say — spoken \
style, one or two short sentences, the answer first. Never invent a finished action; if \
something is pending or failed, say so plainly. Everything below is what you know and how \
you work.

"""


def voice_for_live(voice: str) -> str:
    """The voice Live will accept: hers if it has it, marin otherwise."""
    voice = (voice or "").strip().lower()
    return voice if voice in LIVE_VOICES else DEFAULT_LIVE_VOICE


def backend_cost(usage: dict[str, Any] | None, model: str) -> float:
    """One backend response's cost from its Responses usage block."""
    if not usage:
        return 0.0
    prices = next((p for name, p in BACKEND_PRICES.items() if model.startswith(name)), None)
    if prices is None:
        return 0.0
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    fresh = int(usage.get("input_tokens") or 0) - cached
    out = int(usage.get("output_tokens") or 0)
    return (max(fresh, 0) * prices[0] + cached * prices[1] + out * prices[2]) / 1_000_000


class LiveStartError(RuntimeError):
    def __init__(self, message: str, *, code: str = "", param: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.param = param


@dataclass
class _Delegation:
    """One piece of work the live model handed to the backend."""

    id: str
    target: str
    created_at: float
    turn_at_start: int  # how often he had spoken when it began (late notes)
    response_id: str | None = None
    calls: list[tuple[str, str, str]] = field(default_factory=list)  # (call_id, name, arguments)
    ran: list[str] = field(default_factory=list)
    text: str = ""
    done: bool = False
    running: bool = False


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class _LiveSession:
    """One wake-to-close conversation on a Live socket: its state and tasks."""

    def __init__(
        self,
        engine: LiveEngine,
        conn: Any,
        mic: Any,
        speaker: Any,
        wake: Any,
        ui: Any,
        stats: SessionStats,
        *,
        announce: bool,
        ptt: Any | None,
        ptt_session: bool,
    ) -> None:
        self.engine = engine
        self.conn = conn
        self.mic = mic
        self.speaker = speaker
        self.wake = wake
        self.ui = ui
        self.stats = stats
        self.announce = announce
        self.ptt = ptt
        self.ptt_session = ptt_session
        self.ended = asyncio.Event()
        self.closed = asyncio.Event()
        self.started = asyncio.Event()
        self.start_error: LiveStartError | None = None
        self.pending: list[asyncio.Task] = []
        now = time.monotonic()
        self.started_at = now
        self.last_activity = now  # his words, her voice, backend work — never the silent stream
        # her voice in the stream
        self.speaking_out = False
        self.last_loud_at = 0.0
        self.reply_run = 0
        self.output_muted_until = 0.0
        self.interrupted = False
        # his words
        self.segments = Segmenter()
        self.latest_fragment_at = 0.0
        self.user_seg_id = ""
        self.user_seg_text = ""
        self.user_grew_at = 0.0
        self.user_judged: set[str] = set()
        self.prewoke = False  # the TV was woken on a mid-sentence "play"
        self.warm = False  # on a socket held open while idle: the wake word itself is replayed to the model
        self.speech_segments = 0
        self.user_turns = 0
        self.heard_speech = False
        self.thinking_until = 0.0
        # the backend
        self.delegations: dict[str, _Delegation] = {}
        self.tools_since_reply: list[str] = []
        self.quiet_since: float | None = None
        self.working_ticks = 0
        # endings
        self.closing = False
        self.closing_ms = 0  # the session clock when the end tool ran: words after it are his
        self.farewell_deadline = 0.0
        self.farewell_pending = False
        self.wrapup_heard = False
        self.command_pending = False
        self.quick_close_armed = False
        self.quick_close_window = engine._command_close_s
        self.quick_close_reason = "command complete"
        # announcements
        self.announcing: list[int] = []
        self.announcing_opener = False
        self.waits_for_reply = False
        self.opener_ids: list[int] = []
        # the room
        self.levels = _Levels()
        self.echo_mode = engine._echo_policy if engine._echo_policy in ("duplex", "gated") else "auto"
        self.ungated_until = 0.0
        self.gate = _SpeechGate()
        self.meter = LevelMeter()
        self.frames_sent = 0
        # push to talk
        self.ptt_active = False
        self.ptt_started = 0.0
        self.ptt_seen = getattr(ptt, "presses", 0) - (1 if ptt_session else 0)

    # ── plumbing ───────────────────────────────────────────────────────────

    def _note(self, text: str) -> None:
        note = getattr(self.ui, "note", None)
        if note is not None:
            with contextlib.suppress(Exception):
                note(text)

    def _tap(self, kind: str, **fields: Any) -> None:
        self.engine._tap(kind, **fields)

    def _end(self, why: str) -> None:
        if not self.ended.is_set():
            self.stats.ended_by = why
            self.ended.set()

    def _supervise(self, task: asyncio.Task) -> None:
        if task.cancelled() or self.ended.is_set():
            return
        err = task.exception()
        if err is not None:
            self.stats.ended_by = f"session error: {err}"[:160]
            with contextlib.suppress(Exception):
                self.ui.error(str(err))
            self.engine._fallbacks.say("failed", self.speaker)
            self.ended.set()

    def _spawn(self, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        task.add_done_callback(self._supervise)
        self.pending.append(task)
        return task

    def _suspect(self, seconds: float) -> None:
        flag = getattr(self.mic, "suspect_before", None)
        if flag is not None:
            with contextlib.suppress(Exception):
                flag(time.monotonic() + seconds + 0.5)

    @property
    def his_turn_live(self) -> bool:
        """His words are still arriving: an open user segment that grew
        within the settle window. (An open segment that stopped growing is
        a turn nobody has closed yet, not a sentence in progress.)"""
        return bool(self.user_seg_id) and time.monotonic() - self.user_grew_at < _SETTLE_S

    def session_ms(self) -> float:
        """Where the session timeline is now, from the last fragment's stamp
        plus the time since it arrived (the timeline starts at session.started)."""
        now = time.monotonic()
        if self.latest_fragment_at:
            return self.segments.latest_ms + (now - self.latest_fragment_at) * 1000
        return (now - self.started_at) * 1000

    @property
    def tool_busy(self) -> bool:
        now = time.monotonic()
        return any(not d.done and now - d.created_at < _DELEGATION_MAX_S for d in self.delegations.values())

    async def _send(self, event: dict[str, Any]) -> None:
        await self.conn.send(event)

    async def _send_audio(self, frame: bytes) -> None:
        self.frames_sent += 1
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(frame).decode("ascii")})

    # ── announcements ──────────────────────────────────────────────────────

    async def deliver(self, items: list[Any], *, opener: bool) -> None:
        """Queued announcements: the lead is a behaviour nudge, the news is
        commentary she paraphrases aloud. Never fake user speech."""
        owner = self.engine._owner
        items = sorted(items, key=lambda i: 0 if getattr(i, "kind", "") == "presence" else 1)
        said = " ".join(i.text for i in items)
        kinds = {getattr(i, "kind", "") for i in items}
        if kinds == {"thought"}:
            context = self.engine._context
            if context is not None:
                context.clear_job("think")
                for item in items:
                    ref = str(getattr(item, "ref", ""))
                    if ref.startswith("thought:"):
                        context.clear_job(f"think:{ref.split(':', 1)[1]}")
            lead = (
                f"Your deeper reasoning finished the question {owner} asked earlier: give him the "
                "answer now, in your own words, as if you had just worked it out."
            )
        elif "presence" in kinds:
            lead = (
                f"{owner} just walked in; nobody has spoken. Welcome him in a few words, then tell him "
                "what happened while he was out, most important first. If he replies, carry on; if not, stop."
            )
        elif "question" in kinds:
            lead = (
                f"Something needs {owner}'s decision; nobody has spoken. Say the question in your own "
                "words, ask him, and wait for his answer; hand his answer to the backend. If he says later, stop."
            )
        elif opener:
            lead = "Nobody has spoken; you are opening this conversation. Say what follows, briefly, then listen."
        else:
            lead = "Something just came in mid-conversation: mention it briefly at a natural moment, then continue."
        self.announcing = [i.id for i in items]
        self.announcing_opener = opener
        self.waits_for_reply = bool(kinds & {"presence", "question"})
        if opener:
            self.opener_ids.extend(self.announcing)
        await self._send({"type": "session.instructions.append", "delegation_id": None, "content": lead})
        await self._send({"type": "session.commentary.append", "delegation_id": None, "content": said[:1500]})
        self.last_activity = time.monotonic()
        self.stats.transcript.append(("event", said))
        self._tap("announce", ids=list(self.announcing), opener=opener)
        self._note(f"announcing: {said[:160]}")

    # ── the microphone ─────────────────────────────────────────────────────

    def _decide_mode(self) -> None:
        coupling = self.levels.coupling
        self.echo_mode = "duplex" if coupling <= self.engine._duplex_max_coupling else "gated"
        self._tap("echo_mode", mode=self.echo_mode, coupling=round(coupling, 2))
        if self.echo_mode == "duplex":
            self._note(f"full duplex: her echo at the mic is small (coupling {coupling:.2f}) — just talk over her")
        else:
            self._note(
                f"gated: the speaker is louder at the mic than you are (coupling {coupling:.2f}) — "
                "the wake word cuts in while she talks"
            )

    async def pump_mic(self) -> None:
        engine, mic, speaker, wake = self.engine, self.mic, self.speaker, self.wake
        played_level = getattr(speaker, "played_level", None)
        while True:
            frame = await mic.get_frame()
            heard_at = time.monotonic()
            level = _rms(frame)
            if getattr(mic, "last_suspect", False) and level > _SUSPECT_LEVEL:
                # her "Yes?" coming back in through a loudspeaker: not him
                self._tap("ack_echo_dropped", level=round(level))
                frame, level = _SILENCE_FRAME, 0.0
            played = played_level(_PLAYED_WINDOW_S) if played_level is not None else None
            audible = self.speaking_out or (played is not None and played > _PLAYING)
            if engine._cues is not None:
                blocked = audible and self.echo_mode != "duplex"
                engine._cues.level(self.meter.idle() if blocked else self.meter.push(level))
            if audible:
                if wake is not None and wake.detect(downsample_24k_to_16k(frame)):
                    await self.barge_in(heard_at, "wake phrase")
                if self.echo_mode == "auto":
                    if self.levels.calibrating(heard_at, played):
                        # her first moments teach the coupling; she hears nothing meanwhile
                        self.levels.echo(level, played)
                        await self._send_audio(_SILENCE_FRAME)
                        continue
                    self._decide_mode()
                if self.echo_mode == "gated" and heard_at > self.ungated_until:
                    await self._send_audio(_SILENCE_FRAME)
                    continue
                await self._send_audio(frame)  # duplex: the model's own ear owns interruption
                continue
            self.levels.quiet(level)
            if self.ptt_session and not self.ptt_active:
                await self._send_audio(_SILENCE_FRAME)
                continue
            if engine._speech_gate and not self.ptt_active:
                speech = (
                    wake.speech_probability(downsample_24k_to_16k(frame))
                    if wake is not None and getattr(wake, "has_vad", False)
                    else None
                )
                was_open = self.gate.open
                passed = self.gate.update(level, heard_at, frame, speech)
                if self.gate.open != was_open:
                    self._tap(
                        "gate", open=self.gate.open, level=round(level), floor=round(self.gate.floor),
                        speech=None if speech is None else round(speech, 2),
                    )
                    if self.gate.open:
                        self.heard_speech = True
                        self.stats.heard_speech = True
                if passed:
                    for held in self.gate.take_preroll():
                        await self._send_audio(held)
                    await self._send_audio(frame)
                else:
                    await self._send_audio(_SILENCE_FRAME)
                continue
            await self._send_audio(frame)

    async def barge_in(self, heard_at: float, how: str) -> None:
        """He cut in with the wake phrase or the push-to-talk key: local
        silence at once, the mic goes up so the model hears him, and the
        model is told to stop. Backend work in flight is not cancelled — the
        API cannot — its answer rides out with the late note."""
        now = time.monotonic()
        self.interrupted = True
        item = getattr(self.speaker, "current_item", "") or ""
        heard_ms = self.speaker.played_ms(item) if item and hasattr(self.speaker, "played_ms") else 0
        self.speaker.clear()
        self.output_muted_until = now + _BARGE_MUTE_S
        self.ungated_until = now + _BARGE_UNGATE_S
        self.quiet_since = None
        await self._send({"type": "session.instructions.append", "delegation_id": None, "content": _STOP_LINE})
        self.engine._trace.interrupted(now - heard_at)
        self._tap("barge_in", how=how, heard_ms=heard_ms, item=item, mode=self.echo_mode)
        self.ui.interrupted()
        if how != "wake phrase":
            self._note(f"interrupted by {how}")
        if self.speech_segments == 0:
            # he cut into an announcement: the question window, then close
            self.quick_close_armed = True
            self.quick_close_window = self.engine._info_close_s
            self.quick_close_reason = "interrupted announcement"
        self.last_activity = now

    # ── push to talk ───────────────────────────────────────────────────────

    async def ptt_press(self) -> None:
        if self.ptt_active:
            return
        if self.speaking_out:
            await self.barge_in(time.monotonic(), "push to talk")
        self.ptt_active = True
        self.ptt_started = getattr(self.ptt, "pressed_at", 0.0) or time.monotonic()
        self.quick_close_armed = False
        self.last_activity = time.monotonic()
        if self.engine._cues is not None:
            self.engine._cues.start(self.speaker)
        self.ui.listening()

    async def ptt_release(self) -> None:
        if not self.ptt_active:
            return
        let_go = getattr(self.ptt, "released_at", 0.0)
        held_s = (let_go if let_go > self.ptt_started else time.monotonic()) - self.ptt_started
        long_enough = held_s >= _PTT_MIN_HOLD_S
        self.ptt_active = False
        if self.engine._cues is not None:
            if long_enough:
                self.engine._cues.end(self.speaker)
            else:
                self.engine._cues.idle()
        if self.ptt_session and not long_enough:
            self.quick_close_armed = True
            self.quick_close_window = self.engine._command_close_s
            self.quick_close_reason = "nothing said"
        self.last_activity = time.monotonic()

    async def ptt_watch(self) -> None:
        ptt = self.ptt
        while True:
            if self.ptt_active and (not ptt.held or ptt.presses > self.ptt_seen):
                await self.ptt_release()
            if not self.ptt_active and ptt.presses > self.ptt_seen:
                self.ptt_seen = ptt.presses
                await self.ptt_press()
            await asyncio.sleep(_PTT_POLL_S)

    # ── what she says ──────────────────────────────────────────────────────

    def _output_audio(self, delta: str) -> None:
        pcm = base64.b64decode(delta)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        level = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        now = time.monotonic()
        if level > _SPEECH_RMS:
            self.last_loud_at = now
            self.last_activity = now
            if not self.speaking_out:
                self.speaking_out = True
                self.reply_run += 1
                self.stats.responses = self.reply_run  # what the close line calls replies: times she spoke
                self.interrupted = False  # a barge-in belongs to the reply it cut
                self.quiet_since = None
                self.levels.new_playback(now)
                begin = getattr(self.speaker, "begin_item", None)
                if begin is not None:
                    begin(f"reply_{self.reply_run}")
                self.engine._trace.audio_delta()
                self._tap("audio_first", run=self.reply_run)
        if now < self.output_muted_until:
            return  # the tail of what he interrupted
        self.speaker.enqueue(pcm)

    async def _reply_played(self) -> None:
        """A beat without her voice: the reply he just heard is over. The
        decisions the realtime engine makes at response.done live here."""
        engine, stats = self.engine, self.stats
        engine._trace.playback_done()
        self._tap("reply_played", run=self.reply_run, interrupted=self.interrupted)
        ran, self.tools_since_reply = self.tools_since_reply, []
        announced, self.announcing = self.announcing, []
        opener_batch, self.announcing_opener = self.announcing_opener, False
        if announced and engine._announcer is not None:
            # Delivered means PLAYED. Cut off: spoken-but-unread.
            engine._announcer.mark_delivered(announced)
            stats.announced.extend(announced)
            if self.interrupted:
                for nid in announced:
                    if nid in self.opener_ids:
                        self.opener_ids.remove(nid)
                self._note("announcement cut short — kept unread")
            elif not opener_batch:
                engine._announcer.mark_read(announced)
        if engine._deferred_reads and not self.tool_busy:
            engine._apply_deferred_reads(interrupted=self.interrupted)
            if self.interrupted:
                self._note("notifications kept unread — she was cut off")
        if any(t in COMMAND_TOOLS for t in ran):
            self.command_pending = True
        if self.ptt_session:
            # each hold is one turn: she answered it, and closes unless he holds again
            self.quick_close_armed = True
            self.quick_close_window = engine._command_close_s
            self.quick_close_reason = "push to talk turn done"
        elif self.speech_segments <= 1:
            # one utterance, answered (the tools it needed ran before this
            # reply): a command closes fast, a question gets the longer window
            self.quick_close_armed = True
            if self.command_pending:
                self.quick_close_window = engine._command_close_s
                self.quick_close_reason = "command complete"
            else:
                self.quick_close_window = engine._info_close_s
                self.quick_close_reason = "question answered"
        if announced and self.announce and self.speech_segments == 0 and not self.interrupted:
            if self.waits_for_reply:
                # she asked him something (or welcomed him home): the question window, then close
                self.quick_close_armed = True
                self.quick_close_window = engine._info_close_s
                self.quick_close_reason = "no reply"
            else:
                self._end("announcement delivered")  # she said her piece; nobody replied
                return
        if (self.closing or self.wrapup_heard) and not self.interrupted and not self.tool_busy and not self.his_turn_live:
            # never on the beat after her last word while his next sentence is still arriving
            self._end("end_conversation" if self.closing else "wrap-up")
            return
        if not self.ptt_session:
            self.ui.listening()

    # ── what he says ───────────────────────────────────────────────────────

    def _fragment(self, speaker: str, delta: str, start_ms: int, end_ms: int) -> None:
        self.latest_fragment_at = time.monotonic()
        self._on_segments(self.segments.fragment(speaker, delta, start_ms, end_ms))  # type: ignore[arg-type]

    def _on_segments(self, segs: list[Segment]) -> None:
        for seg in segs:
            if seg.speaker == "user":
                self._user_segment(seg)
            else:
                self._assistant_segment(seg)

    def _user_segment(self, seg: Segment) -> None:
        now = time.monotonic()
        if seg.id != self.user_seg_id:
            self.user_seg_id = seg.id
            self.user_judged = set()
            self.speech_segments += 1
            self.engine._trace.speech_started()
            self._tap("speech_started", segment=self.speech_segments, speaking=self.speaking_out)
            if self.speech_segments > 1:
                self.command_pending = False  # a conversation now
                self.quick_close_armed = False
            if self.wrapup_heard and self.speaking_out:
                self.wrapup_heard = False  # "wait —" during the goodbye
                self._tap("wrapup_withdrawn")
            if self.closing and seg.start_ms > self.closing_ms:
                self._withdraw_close()  # the backend closed a one-shot command; he has more
            with contextlib.suppress(Exception):
                self.ui.user_speaking()
        self.last_activity = now
        if seg.text != self.user_seg_text:
            # it grew: whatever the settled judgement was, it is stale — "make
            # the living room warm… okay that's all" in one breath is a wrap-up
            self.user_judged.discard("settled")
        self.user_seg_text = seg.text
        self.user_grew_at = now
        if not seg.closed:
            with contextlib.suppress(Exception):
                self.ui.user_partial(seg.text)
            self._maybe_prewake(seg.text)
            return
        self.user_seg_id = ""
        self._finish_user_turn(seg.text)

    def _withdraw_close(self) -> None:
        """The end tool ran on a one-shot command, and then he started talking
        again ("…also make the living room…"): the close is his to overrule.
        Nothing ends while his words are still arriving, and the model is told
        to stay rather than say goodbye. (Will's 18:48 session closed on the
        beat after "Volume's at one hundred percent" with his next sentence
        half-transcribed — talking over her never set `interrupted`.)"""
        self.closing = False
        self.farewell_deadline = 0.0
        self.farewell_pending = False
        self.quick_close_armed = False
        self._tap("close_withdrawn")
        self._note("he has more — the close is off")
        self._spawn(self._send({
            "type": "session.instructions.append", "delegation_id": None,
            "content": "He has more to say: the conversation is NOT over. Answer him; do not say goodbye "
                       "or wrap up unless he does.",
        }))

    def _maybe_prewake(self, partial: str) -> None:
        """"…play…" heard mid-sentence: wake the TV now, while he finishes.
        The Apple TV needs about five seconds; the rest of his sentence and
        the backend's decision cover most of it."""
        if self.prewoke or not _PLAY_WORD.search(partial):
            return
        self.prewoke = True

        async def wake() -> None:
            with contextlib.suppress(Exception):
                if await self.engine._executor.music.prewake():
                    self._tap("prewake")
                    self._note("woke the TV on “play”")

        self._spawn(wake())

    def _maybe_fast_start(self, text: str) -> None:
        """"Play <title> by <artist>" is unmistakable: start it the moment his
        sentence ends, before the backend has decided anything. The backend's
        own play_music call for the same title joins the start already under
        way (MusicCoordinator.begin), so nothing plays twice."""
        intent = parse_play_request(text)
        if intent is None or self.closing:
            return
        self._tap("fast_start", title=intent.title, artist=intent.artist, destination=intent.destination)

        async def start() -> None:
            try:
                result = await self.engine._executor.music.fast_start(intent)
            except Exception as err:  # noqa: BLE001 — the backend's own call is still coming
                self._tap("fast_start_failed", error=str(err)[:120])
                return
            if result is not None:
                self._tap("fast_started", title=result.get("title"), verified=result.get("verified"))
                self._note(f"started {result.get('title')} before the backend answered")

        self._spawn(start())

    def _judge_open_user(self) -> None:
        """His turn stopped growing for a beat: the phrases that need acting
        on before she answers (a stop, a wrap-up, "let me think")."""
        if self.user_seg_id and "settled" not in self.user_judged:
            self.user_judged.add("settled")
            self._judge(self.user_seg_text)

    def _finish_user_turn(self, text: str) -> None:
        engine, stats = self.engine, self.stats
        if normalise(text) in engine._own_lines:
            self._tap("own_echo", text=text)
            self._note(f"heard her own “{text.strip()}” come back — ignored")
            return
        if _her_own_words(text, stats.transcript):
            self._tap("own_echo", text=text, how="her words")
            self._note(f"that was her own voice coming back (“{text.strip()[:60]}”) — ignored")
            return
        engine._trace.speech_stopped()
        engine._trace.transcribed()
        stats.transcript.append(("you", text))
        stats.replied = True
        self.heard_speech = True
        stats.heard_speech = True
        self.user_turns += 1
        self._tap("you_said", text=text)
        with contextlib.suppress(Exception):
            self.ui.user_said(text)
        if "settled" not in self.user_judged:
            self.user_judged.add("settled")
            self._judge(text)
        self._maybe_fast_start(text)

    def _judge(self, text: str) -> None:
        now = time.monotonic()
        if normalise(text) in {"cancel the music", "cancel music", "stop the music", "never mind", "nevermind"}:
            self.engine._executor.music.cancel_pending()
        if is_thinking(text):
            self.thinking_until = now + _THINKING_S
            self._tap("thinking", text=text, seconds=_THINKING_S)
            self._note(f"taking his time — patient for {_THINKING_S:.0f} s")
        elif self.thinking_until:
            self.thinking_until = 0.0
        if is_wrapup(text):
            self.wrapup_heard = True
            self._tap("wrapup", text=text)
        if is_stop_command(text):
            self.engine._executor.music.cancel_pending()
            self._tap("stop_command", text=text)
            self.speaker.clear()
            self._end("stop command")

    def _assistant_segment(self, seg: Segment) -> None:
        if not seg.closed:
            return
        self.stats.transcript.append(("alexa", seg.text))
        self._tap("alexa_said", text=seg.text)
        with contextlib.suppress(Exception):
            self.ui.assistant_said(seg.text)

    # ── the backend ────────────────────────────────────────────────────────

    def _delegation(self, delegation_id: str | None, target: str = "responses") -> _Delegation:
        key = delegation_id or "session"
        d = self.delegations.get(key)
        if d is None:
            d = _Delegation(key, target, time.monotonic(), self.user_turns)
            self.delegations[key] = d
            if self.quiet_since is None and not self.speaking_out:
                self.quiet_since = time.monotonic()
                self.working_ticks = 0
        return d

    def _backend_event(self, event: Any) -> None:
        inner = _field(event, "event") or {}
        if not isinstance(inner, dict):
            inner = dict(inner) if hasattr(inner, "keys") else {}
        kind = str(inner.get("type") or "")
        d = self._delegation(_field(event, "delegation_id"))
        self.last_activity = time.monotonic()
        if kind == "response.created":
            d.response_id = (inner.get("response") or {}).get("id")
        elif kind == "response.output_text.delta":
            d.text += str(inner.get("delta") or "")
        elif kind == "response.output_item.done":
            item = inner.get("item") or {}
            if item.get("type") == "function_call":
                self.engine._trace.first_call()
                d.calls.append((str(item.get("call_id") or ""), str(item.get("name") or ""), str(item.get("arguments") or "")))
        elif kind == "response.completed":
            usage = (inner.get("response") or {}).get("usage") or {}
            cost = backend_cost(usage, self.engine._backend_model)
            self.stats.backend_cost_usd += cost
            self.stats.cost_usd = self.stats.seconds / 60 * LIVE_PRICE_PER_MIN + self.stats.backend_cost_usd
            self.engine._log_live_usage(
                kind="backend", model=self.engine._backend_model, cost_usd=round(cost, 6),
                input_tokens=usage.get("input_tokens", 0),
                cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
            self._tap("backend", what="completed", delegation=d.id, calls=[c[1] for c in d.calls], cost=round(cost, 5))
            if d.calls:
                self._spawn(self._run_calls(d))
            else:
                self._delegation_done(d)
        elif kind in ("response.failed", "response.incomplete"):
            message = str((inner.get("response") or {}).get("error") or kind)
            self._tap("backend", what=kind, delegation=d.id, message=message[:200])
            with contextlib.suppress(Exception):
                self.ui.error(f"backend {kind}: {message[:160]}")
            self._delegation_done(d)

    def _delegation_done(self, d: _Delegation) -> None:
        d.done = True
        self.last_activity = time.monotonic()
        if self.closing and not self.farewell_deadline:
            # the end tool ran; she may already have said goodbye, or be about to
            self.farewell_deadline = time.monotonic() + _FAREWELL_MAX_S

    async def _music_ack(self, d: _Delegation) -> None:
        """One acknowledgment for a slow music tool, never a claim of playback."""
        await asyncio.sleep(_MUSIC_ACK_S)
        request = self.engine._executor.music.active
        if (request is None or request.cancelled.is_set() or self.user_turns != d.turn_at_start
                or self.speaking_out or self.closing or self.wrapup_heard):
            return
        await self._send({
            "type": "session.commentary.append", "delegation_id": None,
            "content": "Music is still preparing. If you have not acknowledged this request, say only 'One moment.' Do not claim it is playing.",
        })

    async def _run_calls(self, d: _Delegation) -> None:
        """The backend asked for tools: run them, hand back every output,
        then let it continue — as its own task, so the receiver keeps reading."""
        engine, stats = self.engine, self.stats
        d.running = True
        turn_at_start = d.turn_at_start

        def late_note() -> str:
            if self.user_turns <= turn_at_start:
                return ""
            latest = next((t for r, t in reversed(stats.transcript) if r == "you"), "")
            return (
                f"while this ran {engine._owner} said: '{latest}' — if that changes or "
                "cancels the request, follow it and skip what is now moot"
            )

        try:
            calls, d.calls = d.calls, []
            engine.last_response_tools = []
            for call_id, name, arguments in calls:
                stats.tool_calls.append(name)
                d.ran.append(name)
                self.tools_since_reply.append(name)
                engine.last_response_tools.append(name)
                if name == "end_conversation":
                    self.closing = True
                    self.closing_ms = self.session_ms()
                    self._tap("end_tool")
                    payload: dict[str, Any] = {
                        "status": "success",
                        "summary": "closing after your last words",
                        "details": {},
                        "reversible": False,
                        "follow_up": "Return a goodbye of a few words, nothing more; the conversation closes on its own.",
                    }
                else:
                    progress = asyncio.create_task(self._music_ack(d)) if name == "play_music" else None
                    try:
                        payload = await engine._run_call(name, arguments, stats, late_note)
                    finally:
                        if progress is not None:
                            progress.cancel()
                            await asyncio.gather(progress, return_exceptions=True)
                await self._send(
                    {
                        "type": "response.item.create",
                        "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(payload)},
                    }
                )
            if engine._instructions_stale:
                # a preference changed: the backend prompt is the only one Live lets us refresh
                engine._instructions_stale = False
                await self._send(
                    {
                        "type": "session.update",
                        "session": {
                            "delegation": {
                                "type": "responses",
                                "responses": {
                                    "instructions": await engine._render_backend_instructions(),
                                    "tools": engine._backend_tools(),
                                },
                            }
                        },
                    }
                )
            await self._send({"type": "response.create"})
        finally:
            d.running = False
            d.turn_at_start = self.user_turns
            self.last_activity = time.monotonic()

    # ── the socket ─────────────────────────────────────────────────────────

    async def receive(self) -> None:
        engine = self.engine
        async for event in self.conn:
            kind = str(_field(event, "type") or "")
            if kind == "session.output_audio.delta":
                self._output_audio(_field(event, "delta") or "")
            elif kind == "session.input_transcript.delta":
                self._fragment("user", _field(event, "delta") or "", int(_field(event, "start_ms") or 0), int(_field(event, "end_ms") or 0))
            elif kind == "session.output_transcript.delta":
                self._fragment("assistant", _field(event, "delta") or "", int(_field(event, "start_ms") or 0), int(_field(event, "end_ms") or 0))
            elif kind == "session.started":
                session = _field(event, "session")
                self.started_at = time.monotonic()
                self.last_activity = self.started_at
                engine._trace.connected()
                self._tap("live_started", session=_field(session, "id", ""), expires_at=_field(session, "expires_at"))
                self.started.set()
            elif kind == "session.delegation.created":
                info = _field(event, "delegation")
                d = self._delegation(_field(info, "id"), str(_field(info, "target") or "responses"))
                self.last_activity = time.monotonic()
                self._tap("delegation", id=d.id, target=d.target, offset_ms=_field(event, "offset_ms"))
            elif kind == "response.event":
                self._backend_event(event)
            elif kind == "session.usage.updated":
                usage = _field(event, "usage")
                self.stats.seconds = float(_field(usage, "seconds", 0.0) or 0.0)
                self.stats.cost_usd = self.stats.seconds / 60 * LIVE_PRICE_PER_MIN + self.stats.backend_cost_usd
                window = _field(event, "context_window")
                ratio = _field(window, "usage_ratio") if window is not None else None
                self._tap("usage", seconds=self.stats.seconds, context=ratio)
                if ratio is not None and ratio > 0.8:
                    self._note(f"context {ratio:.0%} full — the session will compact itself")
            elif kind == "error":
                err = _field(event, "error")
                message = str(_field(err, "message", err) or err)
                code = str(_field(err, "code", "") or "")
                param = str(_field(err, "param", "") or "")
                self._tap("error", message=message[:200], code=code, param=param)
                if not self.started.is_set():
                    self.start_error = LiveStartError(message, code=code, param=param)
                    self.started.set()
                    continue
                if await self._heal_backend(message):
                    self._note("the backend refused that request — settings adjusted, say it again")
                if engine._cues is not None and engine._cues.listening and not self.speaking_out:
                    spoken = engine._cues.error(message, self.speaker)
                    if spoken:
                        self._suspect(spoken)
                with contextlib.suppress(Exception):
                    self.ui.error(message)
            elif kind == "session.closed":
                usage = _field(event, "usage")
                if usage is not None:
                    self.stats.seconds = float(_field(usage, "seconds", self.stats.seconds) or 0.0)
                    self.stats.cost_usd = self.stats.seconds / 60 * LIVE_PRICE_PER_MIN + self.stats.backend_cost_usd
                reason = str(_field(event, "reason") or "")
                self._tap("session_closed", reason=reason, seconds=self.stats.seconds)
                self.closed.set()
                if not self.ended.is_set():
                    self._end(
                        {
                            "expired": "session expired",
                            "content": "closed by policy",
                            "remote_hangup": "remote hangup",
                            "connection_lost": "session error: connection lost",
                        }.get(reason, reason or "closed")
                    )
                return
            elif kind in (
                "session.updated", "session.instructions.appended", "session.commentary.appended",
                "session.thinking.appended", "session.input_audio.muted", "session.input_audio.unmuted", "info",
            ):
                self._tap(kind)

    async def _heal_backend(self, message: str) -> bool:
        """The server refused a delegation because of the backend settings:
        change them for the rest of this session (delegation settings are
        the one thing Live lets us update) so the next request goes through.
        The request that failed is lost — he is told to say it again."""
        engine = self.engine
        text = message.lower()
        if "reasoning" not in text and "web_search" not in text:
            return False
        responses: dict[str, Any] = {}
        if "web_search" in text and engine._backend_web_search and engine._backend_reasoning in _NO_SEARCH_EFFORTS:
            engine._backend_reasoning = "low"
            responses["reasoning"] = {"effort": "low"}
        elif "web_search" in text and engine._backend_web_search:
            engine._backend_web_search = False  # the backend's search is out: ours is back in the list
            responses["tools"] = engine._backend_tools()
        elif "reasoning" in text and engine._backend_reasoning:
            engine._backend_reasoning = ""
            responses["reasoning"] = {"effort": None}
        else:
            return False
        self._tap("backend_healed", **{k: v for k, v in responses.items() if k != "tools"})
        await self._send({"type": "session.update", "session": {"delegation": {"type": "responses", "responses": responses}}})
        return True

    # ── the clocks ─────────────────────────────────────────────────────────

    async def watch(self) -> None:
        engine = self.engine
        tick = 0
        while True:
            await asyncio.sleep(_TICK_S)
            tick += 1
            now = time.monotonic()
            if self.speaking_out and now - self.last_loud_at > _OUTPUT_QUIET_S:
                self.speaking_out = False
                await self._reply_played()
                if self.ended.is_set():
                    return
            segs = self.segments.advance(self.session_ms())
            if segs:
                self._on_segments(segs)
            if self.user_seg_id and now - self.user_grew_at > _SETTLE_S:
                self._judge_open_user()
            if self.ended.is_set():
                return
            if tick % 5:
                continue
            if self.ptt_active:
                continue
            busy = self.tool_busy
            quiet = now - self.last_activity
            if (
                engine._announcer is not None
                and not self.announcing
                and not self.speaking_out
                and not busy
                and not self.closing
                and quiet > _INJECT_QUIET_S
                and now - self.started_at > _INJECT_MIN_AGE_S
                and engine._announcer.due()
            ):
                items = engine._announcer.take_due()
                if items:
                    await self.deliver(items, opener=False)
                    continue
            if now - self.started_at > engine._max_session_s:
                self._end("session cap")
                return
            if now < self.thinking_until:
                continue
            if self.speaking_out or busy or self.his_turn_live:
                continue  # her voice, a tool, or his sentence still arriving: no clock closes anything
            if self.closing:
                if self.farewell_deadline and now > self.farewell_deadline:
                    if not self.farewell_pending:
                        # the end tool ran and she never said a word: ask for the goodbye
                        self.farewell_pending = True
                        self.farewell_deadline = now + _FAREWELL_MAX_S
                        self._tap("farewell_requested")
                        await self._send(
                            {"type": "session.commentary.append", "delegation_id": None, "content": _GOODBYE_LINE}
                        )
                        continue
                    self._end("end_conversation")
                    return
                continue
            if self.wrapup_heard and quiet > _WRAPUP_GRACE_S:
                self._end("wrap-up")
                return
            if self.quick_close_armed and quiet > self.quick_close_window:
                self._end(self.quick_close_reason)
                return
            if quiet > engine._live_idle_timeout_s:
                self._end("idle timeout")
                return

    async def working_cue(self) -> None:
        """Never a silent room while the backend works: her "One moment" two
        seconds in, then every six, unless she is already saying something."""
        engine = self.engine
        played_level = getattr(self.speaker, "played_level", None)
        while True:
            await asyncio.sleep(_TICK_S)
            if engine._cues is None or self.quiet_since is None or not self.tool_busy or self.speaking_out:
                continue
            if played_level is not None and played_level(0.3) > _PLAYING:
                continue
            due = self.quiet_since + _WORKING_CUE_AFTER_S + self.working_ticks * _WORKING_CUE_EVERY_S
            if time.monotonic() >= due:
                self.working_ticks += 1
                spoken = engine._cues.working(self.speaker)
                if spoken:
                    self._suspect(spoken)

    async def false_wake_watch(self) -> None:
        await asyncio.sleep(_NO_SPEECH_S)
        if self.stats.replied or self.ended.is_set() or self.speech_segments > 0 or self.heard_speech:
            return
        self._tap("nobody_spoke", segments=self.speech_segments)
        self.speaker.clear()
        self._end("nobody spoke")

    # ── the conversation ───────────────────────────────────────────────────

    async def run(self, items: list[Any]) -> None:
        engine = self.engine
        await self._send({"type": "session.start", "session": await engine._live_session_config()})
        receiver = asyncio.create_task(self.receive())
        receiver.add_done_callback(self._supervise)
        try:
            try:
                await asyncio.wait_for(self.started.wait(), timeout=_START_TIMEOUT_S)
            except TimeoutError as exc:
                raise LiveStartError("session.started never came") from exc
            if self.start_error is not None:
                raise self.start_error
            if self.warm:
                # The idle loop consumed the frames that carried his "Alexa";
                # the model hears them first, so a name and a pause get its own
                # "Yes?" in the tone he used, and a command in one breath is
                # heard whole.
                replay = getattr(self.mic, "replay_since", None)
                if replay is not None:
                    frames = replay(engine._trace.t0 - _WAKE_BACKLOG_S)
                    for frame in frames:
                        await self._send_audio(frame)
                    self._tap("wake_replayed", frames=len(frames))
            if engine.voice_note:
                self._note(engine.voice_note)
                engine.voice_note = None
            if engine.backend_note:
                self._note(engine.backend_note)
                engine.backend_note = None
            if engine._cues is not None:
                engine._cues.start(self.speaker, sound=False)  # always listening: the flag stays up
            if items:
                await self.deliver(items, opener=True)
            if self.ptt is not None and self.ptt_session and self.ptt.presses > self.ptt_seen:
                self.ptt_seen = self.ptt.presses
                await self.ptt_press()
            tasks = [
                self._spawn(self.pump_mic()),
                self._spawn(self.watch()),
                self._spawn(self.working_cue()),
            ]
            if self.ptt is not None:
                tasks.append(self._spawn(self.ptt_watch()))
            if not self.announce and not self.ptt_session:
                self._spawn(self.false_wake_watch())
            await self.ended.wait()
            # Whatever either side was still saying counts: his open turn is
            # his reply (an opener he answered on its last word is read).
            self._on_segments(self.segments.close(self.session_ms()))
            if self.stats.ended_by == "unknown":
                self.stats.ended_by = "end_conversation"
            self._tap(
                "session_over", by=self.stats.ended_by, replied=self.stats.replied,
                segments=self.speech_segments, seconds=self.stats.seconds,
            )
            if self.stats.replied and self.opener_ids and engine._announcer is not None:
                engine._announcer.mark_read(self.opener_ids)  # she opened with news and he answered
            engine._settle_followups(self.stats)
        finally:
            for task in self.pending:
                if task is not receiver:
                    task.cancel()
            for task in self.pending:
                if task is not receiver:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if self.started.is_set() and self.start_error is None and not self.closed.is_set():
                with contextlib.suppress(Exception):
                    await self._send({"type": "session.close"})
                with contextlib.suppress(TimeoutError, Exception):
                    await asyncio.wait_for(self.closed.wait(), timeout=_CLOSE_TIMEOUT_S)
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver
            self._on_segments(self.segments.close(self.session_ms()))  # a start that failed: nothing
            engine._log_live_usage(
                kind="live", model=engine._model, seconds=self.stats.seconds,
                cost_usd=round(self.stats.seconds / 60 * LIVE_PRICE_PER_MIN, 6), ended_by=self.stats.ended_by,
            )


_WARM_MAX_AGE_S = 240.0  # a held socket is replaced after this long
_WARM_RETRY_S = 5.0  # after a failed connect
_WAKE_BACKLOG_S = 0.9  # how far before the wake the replayed microphone reaches: "Alexa" itself, little before it
# A wake the detector was less sure of than this gets the clip, not the model,
# and nothing is replayed: a false trigger then costs one "Yes?" and a quiet
# close, never a reply to whatever the room was saying (the first warm session
# answered "oh shit", replayed from a 0.56 wake).
CONFIDENT_WAKE = 0.6


class WarmSocket:
    """A Live socket connected while she is idle, so the wake word reaches the
    model at once and the model answers its own name — in the tone of how it
    was said — instead of a clip off the disk.

    Measured 2026-09-13: a connected, unstarted socket is not billed (26 s and
    five minutes idle, 0 s of usage), `session.start` is acknowledged in
    0.28 s and her first word follows 0.7 s later, against 1.5–2.5 s to
    connect cold. The holder never reads the socket while it waits (one reader
    at a time), so a socket the server dropped quietly is only found out at
    the wake — the engine then connects cold, as it always did."""

    def __init__(
        self,
        client: Any,
        *,
        max_age_s: float = _WARM_MAX_AGE_S,
        retry_s: float = _WARM_RETRY_S,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._max_age_s = max_age_s
        self._retry_s = retry_s
        self._log = log
        self._conn: Any | None = None
        self._taken = False
        self._release = asyncio.Event()
        self.connects = 0
        self.uses = 0

    @property
    def ready(self) -> bool:
        return self._conn is not None and not self._taken

    def take(self) -> tuple[Any, Callable[[], None]] | None:
        """The held connection and the call that gives it back once the
        conversation on it is over (the holder then connects the next one)."""
        if not self.ready:
            return None
        self._taken = True
        self.uses += 1
        return self._conn, self._release.set

    async def run(self) -> None:
        while True:
            try:
                async with self._client.live.connect() as conn:
                    self.connects += 1
                    self._conn, self._taken = conn, False
                    self._release = asyncio.Event()
                    try:
                        await asyncio.wait_for(self._release.wait(), timeout=self._max_age_s)
                    except TimeoutError:
                        if self._taken:
                            await self._release.wait()  # in use past its age: the conversation finishes first
                    finally:
                        self._conn = None
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 — the network; she still wakes cold
                self._conn = None
                if self._log is not None:
                    with contextlib.suppress(Exception):
                        self._log(f"warm socket: {err} — retrying in {self._retry_s:.0f}s")
                await asyncio.sleep(self._retry_s)


class LiveEngine(RealtimeEngine):
    """The realtime engine's tools, prompts, memory and accounting on a
    GPT-Live session. Everything conversational is `_LiveSession`."""

    @property
    def warm_ready(self) -> bool:
        """A socket is held open: the model, not a clip, answers the wake."""
        warm = getattr(self, "_warm", None)
        return warm is not None and warm.ready

    def answers_wake(self, score: float | None) -> bool:
        """Whether the model answers this wake itself: a held socket, and a
        detector sure enough of the word for its audio to be worth replaying."""
        return self.warm_ready and (score is None or float(score) >= CONFIDENT_WAKE)

    # The runner answers a wake only after a beat of quiet: on full duplex a
    # command in the same breath as her name needs no "Yes?" (m4_realtime.acknowledge_later).
    defers_ack = True

    def __init__(
        self,
        *,
        backend_model: str = "gpt-5.6-luna",
        backend_reasoning: str = "minimal",
        backend_web_search: bool = True,
        echo_policy: str = "auto",
        duplex_max_coupling: float = 0.4,
        max_session_s: float = 600.0,
        live_idle_timeout_s: float | None = None,
        store: bool = False,
        live_voice: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        wanted = (live_voice or self._voice or "").strip().lower()
        self._voice = voice_for_live(wanted)
        if self._voice != wanted:
            self.voice_note = f"'{wanted}' is not a Live voice — using {self._voice}"
        self._backend_model = backend_model
        self._backend_reasoning = (backend_reasoning or "").strip().lower()
        self._backend_web_search = backend_web_search
        self.backend_note: str | None = None
        if backend_web_search and self._backend_reasoning in _NO_SEARCH_EFFORTS:
            # the backend refuses its native web search at these efforts —
            # every tool request failed with it on day one
            self.backend_note = (
                f"backend reasoning '{self._backend_reasoning}' cannot run the native web search — using low"
            )
            self._backend_reasoning = "low"
        self._echo_policy = (echo_policy or "auto").strip().lower()
        self._duplex_max_coupling = float(duplex_max_coupling)
        self._max_session_s = float(max_session_s)
        self._live_idle_timeout_s = float(live_idle_timeout_s if live_idle_timeout_s is not None else self._idle_timeout_s)
        self._store = bool(store)

    # ── prompts and config ─────────────────────────────────────────────────

    def _live_instructions(self) -> str:
        extra = f"\n{self._extra_instructions}\n" if self._extra_instructions else ""
        return LIVE_INSTRUCTIONS.format(
            name=self._name, owner=self._owner, wake_phrase=self._wake_phrase, extra=extra
        )

    async def _render_backend_instructions(self) -> str:
        text, _lights, _players = await self._render_instructions(_INSTRUCTIONS)
        return BACKEND_PREAMBLE.format(name=self._name, owner=self._owner) + text

    def _backend_tools(self) -> list[dict[str, Any]]:
        tools = self._tools()
        if self._backend_web_search:
            tools = [t for t in tools if t.get("name") != "web_search"]
            tools.append({"type": "web_search"})
        return tools

    async def _live_session_config(self) -> dict[str, Any]:
        responses: dict[str, Any] = {
            "model": self._backend_model,
            "instructions": await self._render_backend_instructions(),
            "tools": self._backend_tools(),
            "tool_choice": "auto",
        }
        if self._backend_reasoning:
            responses["reasoning"] = {"effort": self._backend_reasoning}
        return {
            "model": self._model,
            "instructions": self._live_instructions(),
            "audio": {
                "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                "output": {"voice": self._voice},
            },
            "delegation": {"type": "responses", "responses": responses},
            "store": self._store,
        }

    def _log_live_usage(self, **entry: Any) -> None:
        if self._usage_log is None:
            return
        row = {"ts": time.time(), "engine": "live", **entry}
        with contextlib.suppress(Exception), self._usage_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    # ── the conversation ───────────────────────────────────────────────────

    async def run_conversation(
        self,
        mic: Any,
        speaker: Any,
        wake: Any,
        ui: Any,
        *,
        announce: bool = False,
        trace: TurnTrace | None = None,
        ptt: Any | None = None,
        ptt_session: bool = False,
    ) -> SessionStats:
        """Same contract as the realtime engine's: one wake-to-close
        conversation on the runner's open streams."""
        stats = SessionStats()
        self._trace = trace if trace is not None else TurnTrace()
        self._speaker = speaker
        self._session_opened = time.time()
        self._conversations += 1
        self._live_conversation = self._conversations
        self._raised_followups = []
        self._deferred_reads = []
        self._ui_tool_hook = getattr(ui, "tool", None)
        self._ui_note_hook = getattr(ui, "note", None)  # the music trace line, per request
        items: list[Any] = []
        if announce:
            items = self._announcer.take_due() if self._announcer is not None else []
            if not items:
                stats.ended_by = "nothing to announce"
                return stats
        elif self._presence is not None:
            with contextlib.suppress(Exception):
                self._presence.observe("home", source="voice")
        try:
            warm = getattr(self, "_warm", None)
            held = warm.take() if warm is not None else None
            if held is not None:
                # The socket was connected while she was idle: start it now, and
                # only if it turns out dead (the server drops idle ones quietly)
                # connect cold below — never re-run a conversation that began.
                conn, release = held
                session = _LiveSession(
                    self, conn, mic, speaker, wake, ui, stats,
                    announce=announce, ptt=ptt, ptt_session=ptt_session,
                )
                score = getattr(self._trace, "wake_score", None)
                session.warm = not announce and (score is None or float(score) >= CONFIDENT_WAKE)
                try:
                    await session.run(items)
                    return stats
                except Exception as err:  # a dead held socket is the one case we go around
                    if session.started.is_set() or not isinstance(err, (OSError, ConnectionError, LiveStartError)):
                        raise  # a conversation that began, or a failure that is not the socket's
                    note = getattr(ui, "note", None)
                    if note is not None:
                        with contextlib.suppress(Exception):
                            note(f"the held socket was dead ({type(err).__name__}) — connecting cold")
                    stats = SessionStats()
                finally:
                    release()
            for attempt in (1, 2):
                try:
                    async with self._client.live.connect() as conn:
                        session = _LiveSession(
                            self, conn, mic, speaker, wake, ui, stats,
                            announce=announce, ptt=ptt, ptt_session=ptt_session,
                        )
                        await session.run(items)
                    break
                except LiveStartError as err:
                    if attempt == 1 and self._backend_reasoning and "reasoning" in (err.param or ""):
                        # this backend does not take a reasoning setting: go without
                        note = getattr(ui, "note", None)
                        if note is not None:
                            note(f"backend rejected reasoning '{self._backend_reasoning}' — retrying without")
                        self._backend_reasoning = ""
                        continue
                    raise
        finally:
            self._live_conversation = 0
            if self._cues is not None:
                self._cues.reset()
        return stats
