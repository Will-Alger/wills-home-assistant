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

from assistant.announce import Announcer
from assistant.app import wait_for_trigger
from assistant.audio import tones
from assistant.audio.mic import Microphone, describe_device
from assistant.audio.speaker import Speaker
from assistant.config import home_dir, load_settings
from assistant.dispatch import Dispatcher, load_extra_routines
from assistant.engines.realtime_engine import (
    FRAME_SAMPLES_24K,
    REALTIME_RATE,
    RealtimeEngine,
)
from assistant.home import HomeAssistantClient
from assistant.home.fake import FakeHome
from assistant.learning import Reflector
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.memory import MemoryStore
from assistant.tasks import TaskBoard
from assistant.wake.detector import WakeDetector
from assistant.web import WebSearch

console = Console()


class ConsoleUi:
    def __init__(self, name: str) -> None:
        self._name = name.lower()
        self._last_status = ""

    def _say(self, text: str) -> None:
        self._last_status = ""
        console.print(text, highlight=False)

    def listening(self) -> None:
        if self._last_status != "listening":  # multi-step replies fire this repeatedly
            self._last_status = "listening"
            console.print("[green]● listening[/green]")

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
    settings.require("openai_api_key")
    home = FakeHome() if fake else HomeAssistantClient(settings.ha_url, settings.ha_token)
    if not fake:
        settings.require("ha_url", "ha_token")
    root = home_dir()  # .env, data/, logs/: the MAIN repo even when staged
    staged = os.environ.get("ALEXA_STAGED_TASK", "").strip()
    memory = MemoryStore(root / "data" / "memory.json")
    announcer = Announcer(
        root / "data" / "announcements.json",
        quiet_hours=settings.announce_quiet_hours,
        max_attempts=settings.announce_max_attempts,
    )
    calendar = build_calendar(settings, fake)
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
        task_board=(
            TaskBoard(
                root,
                runner=Dispatcher(
                    root,
                    routine_id=settings.claude_routine_id,
                    routine_token=settings.claude_routine_token,
                    extra_routines=load_extra_routines(root),
                    model=settings.dispatch_model,
                    effort=settings.dispatch_effort,
                ),
                announcer=announcer,
                staged_task_id=int(staged) if staged.isdigit() else None,
                uv_exe=settings.uv_exe,
            )
            if settings.use_claude_subscription
            else None
        ),
        usage_log=root / ".usage.jsonl",
        announcer=announcer,
        web=WebSearch(
            settings.openai_api_key,
            model=settings.web_search_model,
            context_size=settings.web_search_context,
        ),
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


async def voice(fake: bool) -> int:
    settings, _home, engine, reflector = build_engine(fake)
    board = getattr(engine, "_board", None)
    if board is not None:
        with contextlib.suppress(Exception):  # housekeeping must never block boot
            await board.startup_maintenance()
    staged = os.environ.get("ALEXA_STAGED_TASK", "").strip()
    if staged:
        console.print(f"[magenta]◈ running the STAGED build of task {staged}[/magenta]")
    console.print("Loading wake model...")
    wake = WakeDetector(settings.wake_model, threshold=settings.wake_threshold)
    session_wake = WakeDetector(settings.wake_model, threshold=settings.wake_threshold)
    total_cost = 0.0
    console.print(
        f"[bold]Voice online.[/bold] “{settings.wake_phrase}” to talk to "
        f"{settings.assistant_name} · voice: {settings.realtime_voice} · "
        f"mic: {describe_device(settings.audio_input_device)} · "
        f"home: {'fake apartment' if fake else settings.ha_url} · Ctrl+C quits."
    )
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                total_cost = await one_cycle(
                    settings, engine, wake, session_wake, total_cost, reflector
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as err:  # noqa: BLE001 — the app must never die on its own
                tones.play("error")
                console.print(f"[red]recovered from: {err!r} — back to idle[/red]")
            if engine.restart_requested:
                console.print("[yellow]self-restart requested — exiting for the watchdog[/yellow]")
                return 0  # the always-on service relaunches us in seconds
    console.print(f"\n[dim]total: ${total_cost:.4f}[/dim]")
    return 0


async def one_cycle(settings, engine, wake, session_wake, total_cost: float, reflector) -> float:
    """One idle→wake→conversation cycle; returns the updated running cost."""
    await asyncio.sleep(0.2)  # let PortAudio settle between 24k/16k stream switches
    announcer = getattr(engine, "announcer", None)
    # Something to say already? Skip the mic and speak. Otherwise IDLE:
    # wake-gate on a 16 kHz mic (local, free, private) while watching the
    # announcement queue.
    trigger = "announce" if announcer is not None and announcer.due() else ""
    if not trigger:
        async with Microphone(settings.audio_input_device) as mic16:
            console.print("[dim]○ idle — say the wake phrase[/dim]")
            trigger = await wait_for_trigger(mic16, wake, announcer)
    announcing = trigger == "announce"
    if not announcing:
        tones.play("wake")
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
            console.print("[cyan]◆ announcing[/cyan]")
            with contextlib.suppress(Exception):
                speaker.enqueue(tones.pcm("announce", REALTIME_RATE))
        else:
            console.print("[green]● connected — talk[/green]")
        stats = await engine.run_conversation(
            mic24, speaker, session_wake, ConsoleUi(settings.assistant_name), announce=announcing
        )
        # Goodbye chime through the SESSION speaker: a fresh sd.play stream
        # right after this one closes silently loses the race on Windows.
        with contextlib.suppress(Exception):
            speaker.enqueue(tones.pcm("close", REALTIME_RATE))
            await asyncio.wait_for(speaker.wait_idle(), timeout=3.0)
    total_cost += stats.cost_usd
    if engine.voice_note:
        console.print(f"[yellow]{engine.voice_note}[/yellow]")
        engine.voice_note = None
    reason = {
        "idle timeout": "quiet too long — closed to stop the meter; say the wake word anytime",
        "end_conversation": "she wrapped up",
        "question answered": "question answered — closed after quiet",
        "announcement delivered": "announced, back to sleep",
        "nothing to announce": "announcement was already handled",
    }.get(stats.ended_by, stats.ended_by)
    console.print(
        f"[bold]conversation closed[/bold] ({reason}) · {stats.responses} replies · "
        f"tools: {stats.tool_calls or 'none'} · ${stats.cost_usd:.4f} "
        f"(${total_cost:.4f} session)"
    )
    if reflector is not None and stats.transcript:
        with contextlib.suppress(Exception):  # learning must never break the loop
            reflection = await reflector.reflect(stats.transcript)
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
