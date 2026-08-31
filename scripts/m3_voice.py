"""Milestone 3: the ears. Say "hey jarvis, ..." — replies are text (M4 speaks).

    uv run scripts/m3_voice.py --fake    # fake apartment, real ears
    uv run scripts/m3_voice.py           # real Home Assistant

Needs ANTHROPIC_API_KEY and DEEPGRAM_API_KEY in .env. Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from pathlib import Path

from rich.console import Console

from assistant.audio import tones
from assistant.audio.mic import Microphone
from assistant.brain import Agent, AgentReply
from assistant.config import load_settings
from assistant.home import HomeAssistantClient
from assistant.home.fake import FakeHome
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.meter import Meter
from assistant.stt.deepgram_flux import DeepgramFlux
from assistant.voice import VoiceLoop
from assistant.wake.detector import WakeDetector

console = Console()


class ConsoleUi:
    def __init__(self, meter: Meter) -> None:
        self._meter = meter

    def wake(self) -> None:
        tones.play("wake")
        console.print("\n[bold green]● listening[/bold green]")

    def partial(self, transcript: str) -> None:
        console.print(f"[dim]… {transcript}[/dim]", end="\r")

    def transcript(self, transcript: str) -> None:
        console.print(f"[bold]you>[/bold] {transcript}")

    def reply(self, reply: AgentReply) -> None:
        console.print(f"[bold cyan]jarvis>[/bold cyan] {reply.speech}")
        console.print(f"[dim]intent: {reply.intent} · {self._meter.command_line()}[/dim]")

    def follow_up(self, intent: str) -> None:
        tones.play("wake")
        prompt = "listening for your answer" if intent == "listen" else "anything else?"
        console.print(f"[green]● {prompt}[/green]")

    def idle(self) -> None:
        tones.play("close")
        console.print("[dim]○ idle — say the wake phrase[/dim]")

    def error(self, message: str) -> None:
        tones.play("error")
        console.print(f"[red]{message}[/red]")


async def run(fake: bool) -> int:
    settings = load_settings()
    settings.require("anthropic_api_key", "deepgram_api_key")
    home = FakeHome() if fake else HomeAssistantClient(settings.ha_url, settings.ha_token)
    if not fake:
        settings.require("ha_url", "ha_token")

    meter = Meter(log_path=Path(__file__).resolve().parents[1] / ".usage.jsonl")
    agent = Agent(
        home,
        AnthropicProvider(
            model=settings.llm_model,
            effort=settings.llm_effort,
            api_key=settings.anthropic_api_key or None,
            workspace_id=settings.anthropic_workspace_id or None,
        ),
        meter,
        owner=settings.owner_name,
    )
    await agent.start_session()

    console.print("Loading wake model (first run downloads it)...")
    wake = WakeDetector(settings.wake_model, threshold=settings.wake_threshold)
    stt = DeepgramFlux(settings.deepgram_api_key)

    from assistant.audio.mic import describe_device

    mode = "fake apartment" if fake else settings.ha_url
    console.print(
        f"[bold]Ears online.[/bold] Wake phrase: “{settings.wake_model.replace('_', ' ')}” · "
        f"mic: {describe_device(settings.audio_input_device)} · home: {mode} · Ctrl+C quits."
    )
    async with Microphone(settings.audio_input_device) as mic:
        loop = VoiceLoop(mic, wake, stt, agent, ConsoleUi(meter))
        # Ctrl+C arrives as CancelledError inside asyncio.run on Windows
        with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
            await loop.run()
    if not fake:
        with contextlib.suppress(Exception):
            await home.close()  # type: ignore[union-attr]
    console.print(
        f"\n[dim]session: ${meter.session_cost_usd:.4f} across {meter.session_hops} hops[/dim]"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="use the in-memory fake apartment")
    args = parser.parse_args()
    try:
        return asyncio.run(run(fake=args.fake))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
