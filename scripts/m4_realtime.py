"""Milestone 4: the OpenAI Realtime voice engine — it talks back.

    uv run scripts/m4_realtime.py --fake                 # fake apartment, full voice
    uv run scripts/m4_realtime.py                        # real Home Assistant
    uv run scripts/m4_realtime.py --fake --text-probe "turn off the hallway"
                                                         # no mic: prove the engine,
                                                         # saves the reply as reply.wav

Say "hey jarvis" to open a conversation; it speaks, listens, runs your
lights, and closes itself when you wrap up ("that's all") or after silence.
Say the wake phrase mid-reply to cut it off. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import wave
from pathlib import Path

from rich.console import Console
from rich.markup import render

from assistant.announce import Announcer
from assistant.app import record_session, wait_for_trigger
from assistant.audio import tones
from assistant.audio.cues import VoiceCues
from assistant.audio.mic import Microphone, describe_device
from assistant.audio.speaker import Speaker
from assistant.brain.thinker import Thinker
from assistant.briefing import compose_briefing
from assistant.config import code_root, home_dir, load_settings
from assistant.delivery import Courier, DeliveryPolicy, DeliverySettings
from assistant.dispatch import Dispatcher, load_extra_routines, migrate_cloud_routines
from assistant.engines.realtime_engine import (
    FRAME_SAMPLES_24K,
    REALTIME_RATE,
    RealtimeEngine,
)
from assistant.events import EventWatcher, WatchStore
from assistant.followups import FollowUpStore
from assistant.home import HomeAssistantClient
from assistant.home.fake import FakeHome
from assistant.journal import Journal
from assistant.learning import Reflector
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.memory import MemoryStore
from assistant.panel import PanelOverrides, SettingsPanel
from assistant.presence import Presence
from assistant.push import PhoneActions, PhonePusher
from assistant.routines import RoutineStore
from assistant.scheduler import Scheduler
from assistant.sessions import SessionLog
from assistant.status import AssistantStatus
from assistant.tasks import TaskBoard
from assistant.wake.detector import WakeDetector
from assistant.web import WebSearch

console = Console()


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
    applied = overrides.apply(settings)  # voice / wake word chosen in the panel
    settings.require("openai_api_key")
    home = FakeHome() if fake else HomeAssistantClient(settings.ha_url, settings.ha_token)
    if not fake:
        settings.require("ha_url", "ha_token")
    staged = os.environ.get("ALEXA_STAGED_TASK", "").strip()
    memory = MemoryStore(root / "data" / "memory.json")
    journal = Journal(root / "data" / "journal", keep_days=settings.journal_keep_days)
    sessions = SessionLog(root / "data" / "sessions.json")
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
    status = AssistantStatus(
        mic=describe_device(settings.audio_input_device),
        voice=settings.realtime_voice,
        wake_word=settings.wake_phrase,
        home="fake apartment" if fake else settings.ha_url,
    )
    for change in applied:
        status.note(f"settings panel: {change}")
    cues = VoiceCues(rate=REALTIME_RATE, status=status)

    def request_restart() -> None:
        # the runner exits after this cycle; the watchdog brings her back
        engine.restart_requested = True

    if board is not None:
        board.on_restart = request_restart  # a background merge landed: restart when idle

    panel = SettingsPanel(
        status,
        overrides,
        restart=request_restart,
        models_dir=code_root() / "models",
        log=lambda m: console.print(f"[dim]{m}[/dim]"),
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
        eagerness=settings.realtime_eagerness,
        extra_instructions=settings.assistant_extra_instructions,
        memory=memory,
        calendar=calendar,
        task_board=board,
        usage_log=root / ".usage.jsonl",
        announcer=announcer,
        thinker=thinker,
        watches=watches,
        scheduler=scheduler,
        routines=routines,
        journal=journal,
        sessions=sessions,
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
    )
    scheduler._executor = engine._executor  # scheduled actions run through her tools
    scheduler.briefing = compose_briefing(calendar, board, scheduler, announcer, settings.owner_name)
    engine.scheduler = scheduler
    engine.journal = journal
    engine.sessions = sessions
    engine.presence = presence
    engine.home = home
    engine.status = status  # what the settings panel shows
    engine.cues = cues
    engine.panel = panel
    engine.overrides = overrides
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
            WakeDetector(settings.wake_model, threshold=settings.wake_threshold),
            WakeDetector(settings.wake_model, threshold=settings.wake_threshold),
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
    console.print("Loading wake model...")
    wake, session_wake = load_wake_detectors(settings, getattr(engine, "overrides", None))
    total_cost = 0.0
    mic_name = describe_device(settings.audio_input_device)
    if status is not None:
        status.configure(mic=mic_name, voice=settings.realtime_voice, wake_word=settings.wake_phrase)
        status.set_state("idle")
    say(
        status,
        f"Voice online. “{settings.wake_phrase}” to talk to {settings.assistant_name} · "
        f"voice: {settings.realtime_voice} · mic: {mic_name} · "
        f"home: {'fake apartment' if fake else settings.ha_url} · Ctrl+C quits.",
    )
    failures = 0
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                total_cost = await one_cycle(
                    settings, engine, wake, session_wake, total_cost, reflector
                )
                failures = 0
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as err:  # noqa: BLE001 — the app must never die on its own
                # Back off instead of spinning: a mic that won't open would
                # otherwise retry (and beep) five times a second. One error
                # tone per streak, then 5 s → 60 s between attempts.
                failures += 1
                if failures == 1 and cues is not None:
                    cues.error(f"{err}")  # never a listening ding on a failure
                delay = min(5.0 * 2 ** (failures - 1), 60.0)
                say(status, f"recovered from: {err!r} — retrying in {delay:.0f}s", "red")
                await asyncio.sleep(delay)
                continue
            if engine.restart_requested:
                say(status, "self-restart requested — exiting for the watchdog", "yellow")
                actions = getattr(engine, "phone_actions", None)
                if actions is not None:
                    await actions.drain()  # never exit mid-merge from a phone tap
                return 0  # the always-on service relaunches us in seconds
    for background in (watcher_task, scheduler_task, courier_task):
        if background is not None:
            background.cancel()
    console.print(f"\n[dim]total: ${total_cost:.4f}[/dim]")
    return 0


async def one_cycle(settings, engine, wake, session_wake, total_cost: float, reflector) -> float:
    """One idle→wake→conversation cycle; returns the updated running cost."""
    await asyncio.sleep(0.2)  # let PortAudio settle between 24k/16k stream switches
    if engine.restart_requested:
        return total_cost  # a phone approve while idle: the runner exits for the watchdog
    status = getattr(engine, "status", None)
    cues = getattr(engine, "cues", None)
    announcer = getattr(engine, "announcer", None)
    # Something to say already? Skip the mic and speak. Otherwise IDLE:
    # wake-gate on a 16 kHz mic (local, free, private) while watching the
    # announcement queue and the restart flag.
    trigger = "announce" if announcer is not None and announcer.due() else ""
    if not trigger:
        async with Microphone(settings.audio_input_device) as mic16:
            if mic16.device_note and mic16.device_note != getattr(engine, "_mic_note", None):
                engine._mic_note = mic16.device_note  # say it once per fallback, not per cycle
                say(status, mic16.device_note, "yellow")
            if status is not None:
                status.configure(mic=mic16.device_in_use)  # the mic she is really on
                status.set_state("idle")
            console.print("[dim]○ idle — say the wake phrase[/dim]")
            trigger = await wait_for_trigger(
                mic16, wake, announcer, restart=lambda: engine.restart_requested
            )
    if trigger == "restart":
        return total_cost
    announcing = trigger == "announce"
    if not announcing and cues is not None:
        cues.start()  # the wake ding, through a fresh stream: no session yet
    # Session: 24 kHz mic + speaker, wake detector kept for barge-in
    session_wake.reset()
    async with (
        Microphone(
            settings.audio_input_device,
            samplerate=REALTIME_RATE,
            frame_samples=FRAME_SAMPLES_24K,
        ) as mic24,
        Speaker(REALTIME_RATE) as speaker,
    ):
        if announcing:
            say(status, "◆ announcing", "cyan")
            with contextlib.suppress(Exception):
                speaker.enqueue(tones.pcm("announce", REALTIME_RATE))
        else:
            say(status, "● connected — talk", "green")
        sessions = getattr(engine, "sessions", None)
        row = None
        if sessions is not None:
            with contextlib.suppress(Exception):
                row = sessions.start("announce" if announcing else "wake")
        stats = await engine.run_conversation(
            mic24, speaker, session_wake,
            ConsoleUi(settings.assistant_name, status), announce=announcing,
        )
        # Goodbye chime through the SESSION speaker: a fresh sd.play stream
        # right after this one closes silently loses the race on Windows.
        with contextlib.suppress(Exception):
            if cues is not None:
                cues.session_end(speaker)
            await asyncio.wait_for(speaker.wait_idle(), timeout=3.0)
    total_cost += stats.cost_usd
    if engine.voice_note:
        say(status, engine.voice_note, "yellow")
        engine.voice_note = None
        if status is not None:
            status.configure(voice=engine.voice)  # the panel shows what she really speaks with
    reason = {
        "idle timeout": "quiet too long — closed to stop the meter; say the wake word anytime",
        "end_conversation": "she wrapped up",
        "question answered": "question answered — closed after quiet",
        "announcement delivered": "announced, back to sleep",
        "nothing to announce": "announcement was already handled",
        "no reply": "asked, no reply — back to sleep",
        "interrupted announcement": "you cut in — closed after quiet",
    }.get(stats.ended_by, stats.ended_by)
    say(
        status,
        f"conversation closed ({reason}) · {stats.responses} replies · "
        f"tools: {stats.tool_calls or 'none'} · ${stats.cost_usd:.4f} "
        f"(${total_cost:.4f} session)",
    )
    reflection = await record_session(
        sessions, getattr(engine, "journal", None), stats, row, reflector
    )
    if reflection is not None:
        for lesson in reflection.lessons:
            console.print(f"[magenta]✎ learned:[/magenta] {lesson}")
        for obs in reflection.observations:
            console.print(f"[magenta]✎ noticed (will ask):[/magenta] {obs}")
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
