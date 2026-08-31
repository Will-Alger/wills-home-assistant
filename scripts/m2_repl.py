"""Milestone 2: talk to the brain by keyboard — no audio, real tool use.

    uv run scripts/m2_repl.py --fake     # fake 5-light apartment, no HA needed
    uv run scripts/m2_repl.py            # against your real Home Assistant

Needs an Anthropic API key: put ANTHROPIC_API_KEY in .env (or have `ant auth
login` credentials). Each reply shows the end-of-turn intent and a meter line
(hops, latency, tokens, cost) — this data decides the model/effort question.

REPL commands: /reset (new conversation), /quit
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from pathlib import Path

import anthropic
from rich.console import Console

from assistant.brain import Agent
from assistant.config import load_settings
from assistant.home import HomeAssistantClient
from assistant.home.fake import FakeHome
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.meter import Meter

console = Console()

INTENT_HINTS = {
    "close": "conversation closed",
    "listen": "mic would stay open — it asked you something",
    "confirm_close": "mic would open briefly for an 'anything else?'",
}


async def run(fake: bool) -> int:
    settings = load_settings()
    if settings.llm_provider != "anthropic":
        console.print(f"[red]LLM_PROVIDER={settings.llm_provider} not implemented yet — only 'anthropic'.[/red]")
        return 1

    home = FakeHome() if fake else HomeAssistantClient(settings.ha_url, settings.ha_token)
    if not fake:
        settings.require("ha_url", "ha_token")
        if not await home.api_alive():  # type: ignore[union-attr]
            console.print(
                f"[red]Can't reach Home Assistant at {settings.ha_url}.[/red] "
                "Try --fake to play with the brain without HA."
            )
            return 1

    llm = AnthropicProvider(
        model=settings.llm_model,
        effort=settings.llm_effort,
        api_key=settings.anthropic_api_key or None,
        workspace_id=settings.anthropic_workspace_id or None,
    )
    meter = Meter(log_path=Path(__file__).resolve().parents[1] / ".usage.jsonl")
    agent = Agent(home, llm, meter)
    await agent.start_session()

    mode = "fake apartment" if fake else settings.ha_url
    console.print(f"[bold]{settings.llm_model}[/bold] · effort={settings.llm_effort} · home: {mode}")
    console.print("Type a command ('turn off the hallway'), /reset, or /quit.\n")

    while True:
        try:
            user_text = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            agent.reset()
            console.print("[dim]conversation cleared[/dim]\n")
            continue

        try:
            reply = await agent.handle(user_text)
        except anthropic.AuthenticationError:
            console.print(
                "[red]No valid Anthropic credentials.[/red] Put ANTHROPIC_API_KEY "
                "in .env (console.anthropic.com → API keys)."
            )
            return 1
        except anthropic.BadRequestError as err:
            if "workspace" in str(err).lower():
                console.print(
                    "[red]Your API key is identity-linked and needs a workspace id.[/red] "
                    "Console → Settings → Workspaces → copy the wrkspc_... id into .env "
                    "as ANTHROPIC_WORKSPACE_ID, then restart."
                )
                return 1
            console.print(f"[red]API rejected the request:[/red] {err.message}")
            continue
        except anthropic.APIConnectionError:
            console.print("[red]Network error talking to the Anthropic API.[/red]")
            continue

        console.print(f"[bold cyan]{settings.llm_model.split('-')[1]}>[/bold cyan] {reply.speech}")
        console.print(
            f"[dim]intent: {reply.intent} ({INTENT_HINTS.get(reply.intent, '?')}) · "
            f"{meter.command_line()}[/dim]\n"
        )

    if not fake:
        with contextlib.suppress(Exception):
            await home.close()  # type: ignore[union-attr]
    console.print(f"[dim]session total: ${meter.session_cost_usd:.4f} across {meter.session_hops} hops[/dim]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="use the in-memory fake apartment")
    args = parser.parse_args()
    return asyncio.run(run(fake=args.fake))


if __name__ == "__main__":
    raise SystemExit(main())
