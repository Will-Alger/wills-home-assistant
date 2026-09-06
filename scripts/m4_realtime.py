"""Milestone 4: the OpenAI Realtime voice engine — it talks back.

    uv run scripts/m4_realtime.py --fake                 # fake apartment, full voice
    uv run scripts/m4_realtime.py                        # real Home Assistant
    uv run scripts/m4_realtime.py --fake --text-probe "turn off the hallway"
                                                         # no mic: prove the engine,
                                                         # saves the reply as reply.wav

Say "hey jarvis" to open a conversation; it speaks, listens, runs your
lights, and closes itself when you wrap up ("that's all") or after silence.
Say the wake phrase mid-reply to cut it off. Or hold the push-to-talk hotkey
(PTT_HOTKEY, default Ctrl+Alt) and speak: your hold IS the turn, so a pause
never ends it and she stops listening the moment you let go. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
import wave
from pathlib import Path

from rich.console import Console
from rich.markup import render

from assistant.announce import Announcer
from assistant.app import RescanClock, record_session, reflect_session, wait_for_trigger
from assistant.audio import devices, tones
from assistant.audio.acks import ECHO_TAIL_S, WakeAcks
from assistant.audio.cues import VoiceCues
from assistant.audio.fallbacks import SpokenFallbacks
from assistant.audio.io import AudioIO
from assistant.audio.mic import describe_device
from assistant.brain.thinker import Thinker
from assistant.briefing import compose_briefing
from assistant.config import code_root, home_dir, load_settings
from assistant.context import WorkingContext
from assistant.delivery import Courier, DeliveryPolicy, DeliverySettings
from assistant.dispatch import Dispatcher, load_extra_routines, migrate_cloud_routines
from assistant.engines.realtime_engine import (
    FRAME_SAMPLES_24K,
    REALTIME_RATE,
    RealtimeEngine,
    downsample_24k_to_16k,
)
from assistant.events import EventWatcher, WatchStore
from assistant.followups import FollowUpStore
from assistant.home import HomeAssistantClient
from assistant.home.fake import FakeHome
from assistant.hotkey import HotkeyError, build_push_to_talk
from assistant.journal import Journal
from assistant.latency import LatencyLog
from assistant.learning import Reflector
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.memory import MemoryStore
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.presence import Presence
from assistant.push import PhoneActions, PhonePusher
from assistant.receipts import ReceiptBook
from assistant.recording import Recorder
from assistant.routines import RoutineStore
from assistant.scheduler import Scheduler
from assistant.sessions import SessionLog
from assistant.status import AssistantStatus
from assistant.tasks import TaskBoard
from assistant.thoughts import ThoughtBook
from assistant.wake.detector import WakeDetector
from assistant.web import WebSearch

console = Console()
RESCAN_S = 30.0  # on a fallback microphone: how often the idle loop looks for the real one again
MIC_STALL_S = 5.0  # idle with no mic frame for this long: the device is gone, reopen


def recover(err: BaseException, failures: int, cues, fallbacks, speaker) -> float:
    """One cycle died: what she does about it, and how long she then waits.

    On the FIRST failure of a streak the error tone plays where it always
    has, and then ONE pre-rendered line says what broke — "I can't reach my
    voice service right now" when nothing could be reached (the wake word
    with the network out), "Something failed. Check the log." otherwise. A
    streak repeats neither: it backs off instead of spinning, because a
    microphone that will not open would beep five times a second. Returns
    the seconds to wait before trying again (5 s doubling to 60 s)."""
    if failures == 1:
        if cues is not None:
            cues.error(f"{err}")  # never a listening ding on a failure
        if fallbacks is not None:
            fallbacks.after_failure(err, speaker)
    return min(5.0 * 2 ** (failures - 1), 60.0)


def acknowledge(cues, acks, speaker, mic, trace) -> float:
    """Answer the wake, before there is a session to answer with.

    WAKE_ACK=voice says one of the short lines rendered in her own voice
    (audio/acks.py) through the stream already open — no race, no wait — and
    the ding stands down; `ding`, or a clip that would not load, rings as it
    always has; `off` does neither. The listening flag, the panel state and
    the level meter are the cues' bookkeeping either way.

    Her voice then comes straight back in through the microphone, and a server
    that ends turns on silence would answer it as if HE had spoken, so
    everything captured until the clip has died away is dropped — and nothing
    after it, because the command he gives the instant she stops is the whole
    point. Returns the seconds she spoke for (0.0 when she didn't).

    The log wants both moments — queued, and when it could be heard, which
    the speaker reports the instant its callback pulls that byte."""
    at = speaker.enqueued  # where the answer starts in the byte stream
    spoken = acks.acknowledge(speaker) if acks is not None else 0.0
    silent = acks is not None and acks.silent
    cues.start(speaker, sound=not spoken and not silent)
    trace.stamp("chime_enqueued")
    if speaker.enqueued > at:
        speaker.notify_when_played(at, trace.audible)
    if spoken:
        mic.ignore_before(time.monotonic() + spoken + ECHO_TAIL_S)
    return spoken


def say(status, text: str, style: str = "") -> None:
    """Print a line AND put it in the panel's live feed (plain, no markup)."""
    console.print(f"[{style}]{text}[/{style}]" if style else text, highlight=False)
    if status is not None:
        with contextlib.suppress(Exception):
            status.note(text)


class ConsoleUi:
    def __init__(self, name: str, status=None) -> None:
        self._name = name.lower()
        self._last_status = ""
        self._status = status  # the panel's live feed reads what the console shows

    def _say(self, text: str) -> None:
        self._last_status = ""
        console.print(text, highlight=False)
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.note(render(text).plain)

    def listening(self) -> None:
        if self._last_status != "listening":  # multi-step replies fire this repeatedly
            self._say("[green]● listening[/green]")
            self._last_status = "listening"

    def user_speaking(self) -> None:
        pass  # semantic VAD handles it; printing here is just noise

    def user_partial(self, heard_so_far: str) -> None:
        # One line, overwritten in place: show the TAIL, padded so shorter
        # updates fully cover longer ones (wrapping broke the old version).
        width = max(20, console.width - 6)
        tail = heard_so_far.strip().replace("\n", " ")[-width:]
        console.print(f"[dim]… {tail:<{width}}[/dim]", end="\r", highlight=False)
        self._last_status = "partial"

    def user_said(self, transcript: str) -> None:
        if transcript.strip():
            if self._last_status == "partial":
                console.print(" " * max(20, console.width - 2), end="\r")  # clear the tail line
            self._say(f"[bold]you>[/bold] {transcript.strip()}")

    def assistant_said(self, transcript: str) -> None:
        if transcript.strip():
            self._say(f"[bold cyan]{self._name}>[/bold cyan] {transcript.strip()}")

    def interrupted(self) -> None:
        self._say("[yellow]— interrupted —[/yellow]")

    def tool(self, name: str, result: str, is_error: bool) -> None:
        color = "red" if is_error else "dim"
        self._say(f"[{color}]⚙ {name} → {result[:120]}[/{color}]")

    def note(self, message: str) -> None:
        self._say(f"[yellow]{message}[/yellow]")

    def error(self, message: str) -> None:
        self._say(f"[red]{message}[/red]")
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.error(message)


def build_calendar(settings, fake: bool):
    """Fake apartment gets a fake calendar; otherwise iCloud, if configured."""
    if fake:
        from assistant.calendar.fake import FakeCalendar

        return FakeCalendar()
    if not (settings.icloud_username and settings.icloud_app_password):
        return None  # unconfigured: the calendar tools stay hidden
    from assistant.calendar.apple import AppleCalendar

    return AppleCalendar(
        settings.icloud_username,
        settings.icloud_app_password,
        url=settings.icloud_caldav_url,
        default_calendar=settings.icloud_calendar_name,
    )


def build_engine(fake: bool):
    settings = load_settings()
    root = home_dir()  # .env, data/, logs/: the MAIN repo even when staged
    overrides = PanelOverrides(root / "data" / "panel.json")
    recorder = Recorder(root / "data" / "recordings")  # opt-in session recordings, for debugging
    applied = overrides.apply(settings)  # voice / wake word chosen in the panel
    settings.require("openai_api_key")
    home = FakeHome() if fake else HomeAssistantClient(settings.ha_url, settings.ha_token)
    if not fake:
        settings.require("ha_url", "ha_token")
    staged = os.environ.get("ALEXA_STAGED_TASK", "").strip()
    memory = MemoryStore(root / "data" / "memory.json")
    journal = Journal(root / "data" / "journal", keep_days=settings.journal_keep_days)
    sessions = SessionLog(root / "data" / "sessions.json")
    context = WorkingContext(root / "data" / "context.json")
    announcer = Announcer(
        root / "data" / "announcements.json",
        quiet_hours=settings.announce_quiet_hours,
        max_attempts=settings.announce_max_attempts,
    )
    announcer.subscribe(
        lambda item, event: (
            journal.write(
                "notification", f"{event}: {item.text[:120]}", source=item.kind, data={"id": item.id}
            )
            if event != "grouped"
            else None
        )
    )
    calendar = build_calendar(settings, fake)
    presence = None
    if settings.presence_entity and not fake:
        presence = Presence(
            root / "data" / "presence.json",
            settings.presence_entity,
            owner=settings.owner_name,
            arrive_after_s=settings.presence_arrive_s,
            leave_after_s=settings.presence_leave_s,
            settle_s=settings.presence_settle_s,
            journal=journal,
        )
    delivery = DeliverySettings(root / "data" / "delivery.json")
    latency = LatencyLog(root / "logs" / "turns.jsonl")  # one row per turn, timings only
    announcer.policy = DeliveryPolicy(
        quiet=announcer.is_quiet,
        presence=presence,
        push_kinds=frozenset(k.strip() for k in settings.push_while_away_kinds.split(",") if k.strip()),
        settings=delivery,
    ).decide
    reflector = None
    if settings.use_claude_subscription:
        from assistant.llm.claude_cli import ClaudeCli

        reflector = Reflector(ClaudeCli(), memory)  # billed to Max, not API
    elif settings.anthropic_api_key:
        reflector = Reflector(
            AnthropicProvider(
                model=settings.llm_model,
                effort="low",
                api_key=settings.anthropic_api_key,
                workspace_id=settings.anthropic_workspace_id or None,
            ),
            memory,
        )
    moved = migrate_cloud_routines(root)  # pre-M11 layout: cloud creds in routines.json
    if moved:
        console.print(f"[dim]{moved}[/dim]")
    board = (
        TaskBoard(
            root,
            runner=Dispatcher(
                root,
                routine_id=settings.claude_routine_id,
                routine_token=settings.claude_routine_token,
                extra_routines=load_extra_routines(root),
                model=settings.dispatch_model,
                effort=settings.dispatch_effort,
                timeout_s=settings.dispatch_timeout_s,
            ),
            announcer=announcer,
            staged_task_id=int(staged) if staged.isdigit() else None,
            uv_exe=settings.uv_exe,
            resume_delay_s=settings.dispatch_resume_delay_s,
            journal=journal,
        )
        if settings.use_claude_subscription
        else None
    )
    watches = WatchStore(root / "data" / "watches.json")
    routines = RoutineStore(root / "data" / "routines.json")
    receipts = ReceiptBook(root / "data" / "receipts.json")
    thoughts = ThoughtBook(root / "data" / "thoughts.json")
    scheduler = Scheduler(root / "data" / "schedule.json", announcer=announcer, journal=journal)
    followups = FollowUpStore(root / "data" / "followups.json", announcer=announcer, journal=journal)
    thinker = None
    if settings.use_claude_subscription:
        from assistant.llm.claude_cli import ClaudeCli

        thinker = Thinker(
            ClaudeCli(model=settings.brain_model, effort=settings.brain_effort),
            memory=memory,
            board=board,
            name=settings.assistant_name,
            owner=settings.owner_name,
        )
    tones.set_output(settings.audio_output_device)  # chimes follow the chosen speaker
    hotkey = None
    hotkey_note = ""
    try:
        hotkey = build_push_to_talk(settings.ptt_hotkey)
    except HotkeyError as err:  # a typo must never cost her the wake word
        hotkey_note = f"push to talk is off: {err}"
    status = AssistantStatus(
        mic=describe_device(settings.audio_input_device),
        speaker=devices.describe(settings.audio_output_device, "output"),
        voice=settings.realtime_voice,
        wake_word=settings.wake_phrase,
        home="fake apartment" if fake else settings.ha_url,
        hotkey=hotkey.label if hotkey is not None else "",
    )
    for change in applied:
        status.note(f"settings panel: {change}")
    if hotkey_note:
        status.note(hotkey_note)
    cues = VoiceCues(rate=REALTIME_RATE, status=status)
    # One player for the runner and the engine, so a collapse both of them
    # see is spoken once (audio/fallbacks.py).
    fallbacks = SpokenFallbacks()
    # Her answer to the wake word, read off the disk now so the wake itself
    # only has to queue it (audio/acks.py).
    acks = WakeAcks(mode=settings.wake_ack, rate=REALTIME_RATE)

    def request_restart() -> None:
        # the runner exits after this cycle; the watchdog brings her back
        engine.restart_requested = True

    if board is not None:
        board.on_restart = request_restart  # a background merge landed: restart when idle

    def apply_audio(microphone: str, speaker: str) -> None:
        # saved by the panel or by voice: the idle mic reopens on the new
        # device at the next cycle, the next session's speaker follows too
        settings.audio_input_device = microphone
        settings.audio_output_device = speaker
        tones.set_output(speaker)
        engine.audio_reconfigure = True

    def rescan_audio() -> None:
        engine.audio_reconfigure = True  # PortAudio re-enumerates between idle cycles

    panel = SettingsPanel(
        status,
        overrides,
        restart=request_restart,
        models_dir=code_root() / "models",
        log=lambda m: console.print(f"[dim]{m}[/dim]"),
        on_audio_change=apply_audio,
        on_refresh_devices=rescan_audio,
        recorder=recorder,
    )
    engine = RealtimeEngine(
        api_key=settings.openai_api_key,
        model=settings.realtime_model,
        voice=settings.realtime_voice,
        home=home,
        owner=settings.owner_name,
        name=settings.assistant_name,
        wake_phrase=settings.wake_phrase,
        idle_timeout_s=settings.realtime_idle_timeout_s,
        command_close_s=settings.realtime_command_close_s,
        info_close_s=settings.realtime_info_close_s,
        talk_over=settings.realtime_talk_over,
        tentative_interrupt=settings.realtime_tentative_interrupt,
        noise_reduction=settings.realtime_noise_reduction,
        turn_detection=settings.realtime_turn_detection,
        silence_ms=settings.realtime_silence_ms,
        speech_gate=settings.realtime_speech_gate,
        eagerness=settings.realtime_eagerness,
        extra_instructions=settings.assistant_extra_instructions,
        memory=memory,
        calendar=calendar,
        task_board=board,
        usage_log=root / ".usage.jsonl",
        announcer=announcer,
        thinker=thinker,
        thoughts=thoughts,
        watches=watches,
        scheduler=scheduler,
        routines=routines,
        receipts=receipts,
        journal=journal,
        sessions=sessions,
        context=context,
        presence=presence,
        followups=followups,
        delivery=delivery,
        web=WebSearch(
            settings.openai_api_key,
            model=settings.web_search_model,
            context_size=settings.web_search_context,
        ),
        cues=cues,
        panel=panel,
        latency=latency,
        fallbacks=fallbacks,
    )
    scheduler._executor = engine._executor  # scheduled actions run through her tools
    scheduler.briefing = compose_briefing(calendar, board, scheduler, announcer, settings.owner_name)
    engine.scheduler = scheduler
    engine.journal = journal
    engine.sessions = sessions
    engine.context = context
    engine.presence = presence
    engine.home = home
    engine.status = status  # what the settings panel shows
    engine.hotkey = hotkey  # push to talk: held right now, presses so far
    engine.hotkey_note = hotkey_note
    engine.cues = cues
    engine.fallbacks = fallbacks  # the runner's recovery path speaks too
    engine.acks = acks  # "Yes?" in her own voice, the instant the wake fires
    engine.panel = panel
    engine.recorder = recorder  # "record this session": the tool opens one mid-conversation
    engine.latency = latency
    engine.overrides = overrides
    engine.audio_reconfigure = False  # set when the mic/speaker choice changes
    pusher = None
    if settings.phone_notify_service and not fake:
        pusher = PhonePusher(home, settings.phone_notify_service, name=settings.assistant_name, journal=journal)
        # a card he has read or that is moot comes off his phone
        announcer.subscribe(
            lambda item, event: (
                pusher.clear_later(item)
                if event in ("read", "resolved", "cancelled", "journaled") and item.pushed is not None
                else None
            )
        )

    engine.phone_actions = PhoneActions(
        announcer,
        board=board,
        scheduler=scheduler,
        pusher=pusher,
        journal=journal,
        log=lambda m: say(status, m, "dim"),
        request_restart=request_restart,
    )
    engine.courier = Courier(
        announcer,
        presence=presence,
        pusher=pusher,
        followups=followups,
        journal=journal,
        settings=delivery,
        calendar=calendar,
        board=board,
        owner=settings.owner_name,
        escalate_after_s=settings.escalate_after_h * 3600,
        unread_expire_s=settings.unread_expire_days * 86400,
        nudge_after_s=settings.nudge_after_days * 86400,
        focus_from_calendar=settings.focus_from_calendar,
        log=lambda m: say(status, m, "dim"),
    )

    def on_state(entity_id: str, old: str | None, new: str | None) -> None:
        if presence is not None and entity_id == settings.presence_entity:
            presence.observe(new)

    engine.event_watcher = (
        None
        if fake
        else EventWatcher(
            settings.ha_url, settings.ha_token, watches, announcer,
            log=lambda m: say(status, m, "dim"),
            journal=journal,
            on_state=on_state,
            on_event=(
                {"mobile_app_notification_action": engine.phone_actions.handle_event}
                if pusher is not None
                else None
            ),
            on_connect=(lambda: presence.sync(home)) if presence is not None else None,
        )
    )
    return settings, home, engine, reflector


async def text_probe(fake: bool, text: str) -> int:
    settings, home, engine, _reflector = build_engine(fake)
    console.print(f"[dim]probing {settings.realtime_model} · voice {settings.realtime_voice}[/dim]")
    transcript, audio, stats = await engine.text_probe(text)
    if engine.voice_note:
        console.print(f"[yellow]{engine.voice_note}[/yellow]")
    console.print(f"[bold]you (typed)>[/bold] {text}")
    console.print(
        f"[bold cyan]{settings.assistant_name.lower()}>[/bold cyan] "
        f"{transcript or '(no transcript)'}"
    )
    console.print(
        f"[dim]tools: {stats.tool_calls or 'none'} · {len(audio) / 2 / REALTIME_RATE:.1f}s "
        f"of speech · ${stats.cost_usd:.4f} · ended by: {stats.ended_by}[/dim]"
    )
    if isinstance(home, FakeHome):
        console.print(f"[dim]entities touched: {sorted(home.entities_touched()) or 'none'}[/dim]")
    if audio:
        out = Path("reply.wav")
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(REALTIME_RATE)
            wav.writeframes(audio)
        console.print(f"[green]✓[/green] spoken reply saved to {out} — play it!")
    return 0


def load_wake_detectors(settings, overrides) -> tuple[WakeDetector, WakeDetector]:
    """The idle detector and the barge-in one. A wake word chosen in the
    panel that will not load must never brick the boot: drop the override,
    fall back to the .env value, and say so."""

    def build() -> tuple[WakeDetector, WakeDetector]:
        return (
            WakeDetector(
                settings.wake_model,
                threshold=settings.wake_threshold,
                vad_threshold=getattr(settings, "wake_vad_threshold", 0.0),
            ),
            WakeDetector(
                settings.wake_model,
                threshold=settings.wake_threshold,
                vad_threshold=getattr(settings, "wake_vad_threshold", 0.0),
            ),
        )

    try:
        return build()
    except Exception as err:  # a missing model is not a crash
        if overrides is None or not overrides.wake_model:
            raise
        overrides.clear("wake_model")
        settings.wake_model = load_settings().wake_model
        console.print(
            f"[yellow]wake word from the settings panel would not load ({err}) — "
            f"back to '{settings.wake_phrase}'[/yellow]"
        )
        return build()


async def voice(fake: bool) -> int:
    settings, _home, engine, reflector = build_engine(fake)
    status = getattr(engine, "status", None)
    cues = getattr(engine, "cues", None)
    board = getattr(engine, "_board", None)
    if board is not None:
        with contextlib.suppress(Exception):  # housekeeping must never block boot
            await board.startup_maintenance()
    staged = os.environ.get("ALEXA_STAGED_TASK", "").strip()
    if staged:
        say(status, f"◈ running the STAGED build of task {staged}", "magenta")
    spoken = getattr(engine, "fallbacks", None)
    if spoken is not None and (gap := spoken.note()):
        say(status, gap, "yellow")  # she would have nothing to say when a service dies
    acks = getattr(engine, "acks", None)
    if acks is not None and (gap := acks.note()):
        say(status, gap, "yellow")  # said once at boot, never again per wake
    presence = getattr(engine, "presence", None)
    if presence is not None:
        with contextlib.suppress(Exception):  # HA down at boot: keep what we knew
            await presence.sync(engine.home, boot=True)
        say(status, f"presence: {presence.describe()}", "dim")
    watcher = getattr(engine, "event_watcher", None)
    watcher_task = asyncio.create_task(watcher.run()) if watcher is not None else None
    scheduler = getattr(engine, "scheduler", None)
    scheduler_task = asyncio.create_task(scheduler.run()) if scheduler is not None else None
    courier = getattr(engine, "courier", None)
    courier_task = asyncio.create_task(courier.run()) if courier is not None else None
    hotkey = getattr(engine, "hotkey", None)
    hotkey_task = asyncio.create_task(hotkey.run()) if hotkey is not None else None
    if getattr(engine, "hotkey_note", ""):
        say(status, engine.hotkey_note, "yellow")
    # Read the music library once now, so the FIRST wake after a restart can
    # already spell the owner's playlists and artists in its transcript.
    music_task = engine.music_names.refresh_soon()
    console.print("Loading wake model...")
    wake, session_wake = load_wake_detectors(settings, getattr(engine, "overrides", None))
    total_cost = 0.0
    mic_name = describe_device(settings.audio_input_device)
    speaker_name = devices.describe(settings.audio_output_device, "output")
    if status is not None:
        status.configure(
            mic=mic_name, speaker=speaker_name,
            voice=settings.realtime_voice, wake_word=settings.wake_phrase,
        )
        status.set_state("idle")
    hold = f" · or hold {hotkey.label} and speak" if hotkey is not None else ""
    say(
        status,
        f"Voice online. “{settings.wake_phrase}” to talk to {settings.assistant_name}"
        f"{hold} · voice: {settings.realtime_voice} · mic: {mic_name} · "
        f"speaker: {speaker_name} · "
        f"home: {'fake apartment' if fake else settings.ha_url} · Ctrl+C quits.",
    )
    failures = 0
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                total_cost = await one_cycle(
                    settings, engine, wake, session_wake, total_cost, reflector, hotkey
                )
                failures = 0
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as err:  # noqa: BLE001 — the app must never die on its own
                # One error tone and one spoken line per streak, then a
                # growing wait instead of a spin (see recover()).
                failures += 1
                open_audio = engine.__dict__.get("_audio")
                delay = recover(
                    err, failures, cues,
                    getattr(engine, "fallbacks", None),
                    getattr(open_audio, "speaker", None),
                )
                say(status, f"recovered from: {err!r} — retrying in {delay:.0f}s", "red")
                await asyncio.sleep(delay)
                continue
            if engine.restart_requested:
                say(status, "self-restart requested — exiting for the watchdog", "yellow")
                actions = getattr(engine, "phone_actions", None)
                if actions is not None:
                    await actions.drain()  # never exit mid-merge from a phone tap
                return 0  # the always-on service relaunches us in seconds
    for background in (watcher_task, scheduler_task, courier_task, music_task, hotkey_task):
        if background is not None:
            background.cancel()
    console.print(f"\n[dim]total: ${total_cost:.4f}[/dim]")
    return 0


async def one_cycle(
    settings, engine, wake, session_wake, total_cost: float, reflector, hotkey=None
) -> float:
    """One idle→wake→conversation cycle; returns the updated running cost.

    One microphone and one speaker stay open across cycles (AudioIO): the
    wake chime goes through the same stream her voice does, and what he
    says right after the wake phrase queues on the same mic and reaches the
    session first — push to talk rides those same streams, so a hold that
    starts while she is idle is captured from the press. They are reopened
    only when a device choice changes, a stream stalls, or the boot cycle
    finds nothing open."""
    if engine.restart_requested:
        return total_cost  # a phone approve while idle: the runner exits for the watchdog
    status = getattr(engine, "status", None)
    cues = getattr(engine, "cues", None)
    announcer = getattr(engine, "announcer", None)
    audio: AudioIO | None = engine.__dict__.get("_audio")
    if audio is None:
        audio = engine._audio = AudioIO(settings, rate=REALTIME_RATE, frame_samples=FRAME_SAMPLES_24K)
    latency = engine.latency  # the turn log; its stopwatch starts at the wake
    trace = None
    wake_score = None
    quiet = False  # a periodic re-scan while on a fallback mic: repeat nothing unless it changed
    if getattr(engine, "audio_reconfigure", False) or not audio.is_open:
        engine.audio_reconfigure = False
        quiet = getattr(engine, "_rescan", False)
        engine._rescan = False
        await audio.reopen()  # PortAudio looks at the machine again; both streams reopen, then stay
        tones.set_output(settings.audio_output_device)
        line = f"audio: mic {audio.mic_in_use} · speaker {audio.speaker_in_use}"
        if not quiet or line != getattr(engine, "_audio_line", ""):
            say(status, line, "dim")
        engine._audio_line = line
        notes = audio.notes()
        known = getattr(engine, "_audio_notes", ())
        for note in notes:
            if note not in known:
                say(status, note, "yellow")  # say it once per fallback, not per cycle
        if known and not notes:
            say(status, f"devices back: mic {audio.mic_in_use} · speaker {audio.speaker_in_use}", "green")
        engine._audio_notes = tuple(notes)
        if status is not None:
            status.configure(mic=audio.mic_in_use, speaker=audio.speaker_in_use)  # what she is really on
    mic, speaker = audio.mic, audio.speaker
    assert mic is not None and speaker is not None
    recorder = getattr(engine, "recorder", None)
    if recorder is not None:
        # Always tapped, rarely writing: idle frames feed a three-second
        # ring only while armed, and nothing at all reaches disk until a
        # recording opens.
        mic.tap = recorder.mic
        speaker.tap = recorder.spoke
    # Something to say already? Skip the wait and speak. Otherwise IDLE:
    # wake-gate on the open mic (local, free, private) while watching the
    # announcement queue, the restart flag, the audio-change flag and stalls.
    trigger = "announce" if announcer is not None and announcer.due() else ""
    if not trigger:
        if status is not None:
            status.set_state("idle")
        if not quiet:
            waiting = "say the wake phrase"
            if hotkey is not None:
                waiting += f", or hold {hotkey.label}"
            console.print(f"[dim]○ idle — {waiting}[/dim]")
        rescan = RescanClock(audio.fallback, every_s=RESCAN_S)

        def rescan_due() -> bool:
            if getattr(engine, "audio_reconfigure", False):
                return True
            if rescan.due():  # on a fallback mic: look again for the real one
                engine.audio_reconfigure = True
                engine._rescan = True
                return True
            return False

        def on_score(score: float, fired: bool) -> None:
            # The wake starts the stopwatch; the near misses (loud
            # enough to be someone trying, too quiet to fire) are
            # kept so the threshold can be argued from real audio.
            nonlocal trace, wake_score
            if fired:
                trace = latency.wake(score)
                wake_score = score
            else:
                latency.near_miss(score, settings.wake_threshold)

        mic.drain()  # whatever the mic caught since the last conversation ended
        trigger = await wait_for_trigger(
            mic, wake, announcer,
            restart=lambda: engine.restart_requested,
            reconfigure=rescan_due,
            convert=downsample_24k_to_16k,
            stall_s=MIC_STALL_S,
            on_score=on_score,
            ptt=hotkey,
        )
    if trigger == "stalled":
        # an unplugged microphone gives no error, only silence: reopen
        say(status, "the microphone went silent — reopening the audio devices", "yellow")
        engine.audio_reconfigure = True
        return total_cost
    if trigger in ("restart", "reconfigure"):
        return total_cost
    announcing = trigger == "announce"
    push = trigger == "ptt"
    if trace is None:
        trace = latency.wake(None)  # she opened this one: no wake word, no score
    recording = ""
    if recorder is not None:
        # Armed from the panel or by voice: the timeline opens before the
        # acknowledgment, with the three seconds of room before the wake.
        recording = recorder.begin("announce" if announcing else "ptt" if push else "wake") or ""
        if recording:
            engine.tap = recorder.event
            if trigger == "wake":
                recorder.event(
                    "wake", score=wake_score, threshold=settings.wake_threshold,
                    effective=getattr(wake, "effective_threshold", None),
                )
    if not announcing and cues is not None:
        # "Yes?" in her own voice (or the ding) through the stream already
        # open. Push to talk gets it the moment he presses, and the same
        # microphone keeps running, so the words he says while the socket is
        # still connecting are already queued for this session.
        spoken = acknowledge(cues, getattr(engine, "acks", None), speaker, mic, trace)
        if recording:
            recorder.event("acknowledged", seconds=round(spoken, 2))
    trace.stamp("mic_ready")  # both streams were open before the wake word
    session_wake.reset()
    if announcing:
        say(status, "◆ announcing", "cyan")
        with contextlib.suppress(Exception):
            speaker.enqueue(tones.pcm("announce", REALTIME_RATE))
    elif push:
        say(status, "● connected — keep holding, let go when you're done", "green")
    else:
        say(status, "● connected — talk", "green")
    sessions = getattr(engine, "sessions", None)
    row = None
    if sessions is not None:
        with contextlib.suppress(Exception):
            row = sessions.start("announce" if announcing else "ptt" if push else "wake")
    stats = await engine.run_conversation(
        mic, speaker, session_wake,
        ConsoleUi(settings.assistant_name, status), announce=announcing,
        trace=trace, ptt=hotkey, ptt_session=push,
    )
    with contextlib.suppress(Exception):
        if cues is not None:
            cues.session_end(speaker)  # the goodbye chime, same stream
        await asyncio.wait_for(speaker.wait_idle(), timeout=3.0)
    if speaker.stalled:
        engine.audio_reconfigure = True  # the speaker stopped taking audio: reopen next cycle
    if recorder is not None and recorder.active:
        # (opened above, or mid-conversation by "record this session")
        engine.tap = None
        with contextlib.suppress(Exception):
            summary = recorder.end(ended_by=stats.ended_by, transcript=stats.transcript, replied=stats.replied)
            if summary:
                say(status, f"recorded → data/recordings/{summary['name']} ({summary['duration_s']:.0f} s)", "dim")
    if trigger == "wake":
        # A wake nobody followed up counts against the detector; two in three
        # minutes (the vacuum cleaner) raise its bar for ten. A real one clears.
        if stats.replied:
            wake.backoff.real_wake()
        elif stats.ended_by == "nobody spoke" and wake.backoff.false_wake():
            say(
                status,
                f"two false wakes in three minutes — the wake word now needs "
                f"{wake.effective_threshold:.2f} for the next {wake.backoff.seconds_left / 60:.0f} minutes",
                "yellow",
            )
    total_cost += stats.cost_usd
    if engine.voice_note:
        say(status, engine.voice_note, "yellow")
        engine.voice_note = None
        if status is not None:
            status.configure(voice=engine.voice)  # the panel shows what she really speaks with
    reason = {
        "idle timeout": "quiet too long — closed to stop the meter; say the wake word anytime",
        "end_conversation": "she wrapped up",
        "wrap-up": "you wrapped up — closed after her goodbye",
        "nobody spoke": "false wake — closed quietly, nothing answered",
        "question answered": "question answered — closed after quiet",
        "announcement delivered": "announced, back to sleep",
        "nothing to announce": "announcement was already handled",
        "no reply": "asked, no reply — back to sleep",
        "interrupted announcement": "you cut in — closed after quiet",
        "push to talk turn done": "answered — hold the key again to carry on",
        "nothing said": "nothing was said — back to sleep",
    }.get(stats.ended_by, stats.ended_by)
    with contextlib.suppress(Exception):  # the log is an instrument, never a blocker
        trace.finish(stats.ended_by, session=getattr(row, "id", None))
    timed = trace.console_note()  # 'first audio 0.9 s' for the last turn
    say(
        status,
        f"conversation closed ({reason}) · {stats.responses} replies · "
        f"tools: {stats.tool_calls or 'none'} · "
        + (f"{timed} · " if timed else "")
        + f"${stats.cost_usd:.4f} (${total_cost:.4f} session)",
    )
    await record_session(
        sessions, getattr(engine, "journal", None), stats, row, None,
        timings=trace.compact(), context=getattr(engine, "context", None),
    )
    if reflector is not None and stats.transcript:
        # Reflection is a Claude CLI call (seconds). It used to run here, in
        # line, with the wake-word mic closed: "hey alexa" right after a
        # conversation went unheard. Now the mic reopens at once.
        async def reflect_later() -> None:
            reflection = await reflect_session(
                reflector, sessions, row, stats, getattr(engine, "context", None)
            )
            if reflection is not None:
                for lesson in reflection.lessons:
                    console.print(f"[magenta]✎ learned:[/magenta] {lesson}")
                for obs in reflection.observations:
                    console.print(f"[magenta]✎ noticed (will ask):[/magenta] {obs}")

        keep = engine.__dict__.setdefault("_reflections", set())
        task = asyncio.create_task(reflect_later())
        keep.add(task)
        task.add_done_callback(keep.discard)
    return total_cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="use the in-memory fake apartment")
    parser.add_argument("--text-probe", metavar="TEXT", help="no-mic engine test; saves reply.wav")
    args = parser.parse_args()
    try:
        if args.text_probe:
            return asyncio.run(text_probe(args.fake, args.text_probe))
        return asyncio.run(voice(args.fake))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
