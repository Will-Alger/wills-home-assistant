"""Phase 0 of the GPT-Live plan: a bare gpt-live-1 session on the real
microphone and speaker, before any engine code exists.

    uv run scripts/live_probe.py                               # 25 s, marin, no backend
    uv run scripts/live_probe.py --seconds 40 --backend gpt-5.6-luna --web
    uv run scripts/live_probe.py --gate                        # silence while she plays (Echo Dot)
    uv run scripts/live_probe.py --speak none                  # she does not speak first

She greets you and counts slowly so you can practise talking over her. Every
server event is printed with its time since connect (audio deltas are only
counted), transcripts arrive as fragments with their session-timeline
milliseconds, and the end prints connect→started, first-audio latency, live
seconds and the cost. Both audio sides and the event timeline land in
data/recordings/<stamp>/ — the panel's Recordings… window lists them, and
scratch profile_recording.py reads them. Ctrl+C ends it early.

Runs alone: the assistant must be stopped (data/stop.flag), because a
kernel-streaming speaker cannot be opened by two processes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import os
import sys
import time
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from assistant.audio.io import AudioIO
from assistant.config import home_dir, load_settings
from assistant.recording import Recorder

RATE = 24_000
FRAME = 1920  # 80 ms, the frame the wake path already uses
SILENCE = b"\x00\x00" * FRAME
LIVE_PRICE_PER_MIN = 0.05
BACKEND_PRICES = {  # per 1M tokens: input, cached input, output
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
}

INSTRUCTIONS = """You are Alexa, a calm, friendly voice assistant in Will's home. Speak warmly
and naturally at an unhurried pace, in one or two short sentences.
Backchannels: a brief "mm-hm" only while Will tells a longer story; never over a
command, never while he is mid-word.
Interruptions: the moment Will talks over you, stop mid-word and listen. Never
resume the sentence unless he asks. "Wait", "stop" and "hang on" are instant.
Silence: keep listening while he pauses to think. A cough, music or a nearby
conversation is not a request.
Never say the word "alexa" yourself."""

BACKEND = """You are the reasoning and tool backend of Alexa, a voice assistant in Will's
home. A live voice model talks to him and hands you what needs facts or a
lookup, as text. Transcripts can contain mistakes. Return only the sentence or
two the voice should say, spoken style, short. Never invent a completed action."""

SPEAK_FIRST = (
    "Speak first: say hello to Will in one short sentence, then count slowly from "
    "one to twenty, one number per second, so he can practise talking over you. "
    "The moment he speaks, stop counting and answer him. If he asks you to stop, stop."
)
GREETING = (
    "Hello Will. I'm going to count slowly from one to twenty so you can practise "
    "talking over me — cut in whenever you like."
)
SPEECH_RMS = 200.0  # a delta above this is her voice, below it the silence a full-duplex model streams anyway


def stamp(t0: float) -> str:
    return f"{time.monotonic() - t0:6.2f}s"


def backend_cost(usage: dict[str, Any], model: str) -> float:
    prices = next((p for name, p in BACKEND_PRICES.items() if model.startswith(name)), None)
    if prices is None:
        return 0.0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    inp = (usage.get("input_tokens", 0) or 0) - cached
    out = usage.get("output_tokens", 0) or 0
    return (inp * prices[0] + cached * prices[1] + out * prices[2]) / 1_000_000


async def probe(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is empty in .env")
        return 2
    audio = AudioIO(settings, rate=RATE, frame_samples=FRAME)
    await audio.open()
    mic, speaker = audio.mic, audio.speaker
    assert mic is not None and speaker is not None
    print(f"audio: mic {audio.mic_in_use} · speaker {audio.speaker_in_use}", flush=True)
    for note in audio.notes():
        print(f"  note: {note}", flush=True)
    recorder = Recorder(home_dir() / "data" / "recordings", rate=RATE)
    mic.tap = recorder.mic
    speaker.tap = recorder.spoke
    name = recorder.start()
    recorder.event("probe", model=args.model, backend=args.backend, voice=args.voice, gate=args.gate)
    print(f"recording → data/recordings/{name}", flush=True)

    cfg: dict[str, Any] = {
        "model": args.model,
        "instructions": INSTRUCTIONS,
        "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": args.voice}},
        "store": False,
    }
    if args.backend:
        responses: dict[str, Any] = {
            "model": args.backend,
            "instructions": BACKEND,
            "tools": [{"type": "web_search"}] if args.web else [],
            "reasoning": {"effort": args.effort},
        }
        cfg["delegation"] = {"type": "responses", "responses": responses}

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    t0 = time.monotonic()
    started = asyncio.Event()
    closed = asyncio.Event()
    facts: dict[str, Any] = {
        "deltas": 0, "loud_deltas": 0, "speech_on": False, "seconds": 0.0, "backend_usd": 0.0, "ratio": None,
    }
    lines: dict[str, str] = {"you": "", "alexa": ""}

    def log(msg: str) -> None:
        print(f"[{stamp(t0)}] {msg}", flush=True)

    async with client.live.connect() as conn:
        log("socket open — session.start")
        await conn.session.start(session=cfg)

        async def receive() -> None:
            async for event in conn:
                kind = getattr(event, "type", "?")
                if kind == "session.output_audio.delta":
                    pcm = base64.b64decode(event.delta)
                    if facts["deltas"] == 0:
                        facts["first_audio_s"] = round(time.monotonic() - t0, 3)
                        log(f"first audio delta ({len(pcm) // 2 * 1000 // RATE} ms)")
                        recorder.event("audio_first")
                    facts["deltas"] += 1
                    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
                    level = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
                    loud = level > SPEECH_RMS
                    if loud:
                        facts["loud_deltas"] += 1
                        if "first_speech_s" not in facts:
                            facts["first_speech_s"] = round(time.monotonic() - t0, 3)
                    if loud != facts["speech_on"]:
                        facts["speech_on"] = loud
                        log(f"her audio {'ON' if loud else 'off'} (rms {level:.0f}, queue {speaker.pending_seconds:.2f} s)")
                        recorder.event("her_audio", on=loud, rms=round(level))
                    speaker.enqueue(pcm)
                    continue
                if kind == "session.started":
                    facts["started_s"] = round(time.monotonic() - t0, 3)
                    session = event.session
                    log(f"session.started id={session.id} expires_at={getattr(session, 'expires_at', None)}")
                    recorder.event("live_started", session=session.id)
                    started.set()
                elif kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
                    who = "you" if kind.startswith("session.input") else "alexa"
                    lines[who] += event.delta
                    log(f"{who:5} {event.start_ms:>6}-{event.end_ms:<6} {event.delta!r}")
                    recorder.event("transcript", who=who, text=event.delta, start_ms=event.start_ms, end_ms=event.end_ms)
                elif kind == "session.delegation.created":
                    d = event.delegation
                    log(f"delegation.created id={d.id} target={d.target} offset_ms={event.offset_ms}")
                    recorder.event("delegation", id=d.id, target=d.target)
                elif kind == "response.event":
                    inner = event.event if isinstance(event.event, dict) else {}
                    itype = inner.get("type")
                    if itype == "response.completed":
                        usage = (inner.get("response") or {}).get("usage") or {}
                        cost = backend_cost(usage, args.backend or "")
                        facts["backend_usd"] += cost
                        log(f"backend response.completed usage={usage} ${cost:.4f}")
                    elif itype == "response.output_item.done":
                        item = inner.get("item") or {}
                        log(f"backend output_item.done {item.get('type')} {item.get('name', '')}")
                    elif itype and not itype.endswith(".delta"):
                        log(f"backend {itype}")
                elif kind == "session.usage.updated":
                    facts["seconds"] = event.usage.seconds
                    facts["ratio"] = event.context_window.usage_ratio if event.context_window else None
                    log(f"usage seconds={event.usage.seconds} context={facts['ratio']}")
                elif kind == "error":
                    err = event.error
                    log(f"ERROR {err.type} {err.code}: {err.message} (param={err.param})")
                    recorder.event("error", message=err.message, code=err.code)
                elif kind == "session.closed":
                    facts["seconds"] = event.usage.seconds
                    log(f"session.closed reason={event.reason} seconds={event.usage.seconds}")
                    recorder.event("session_over", reason=event.reason, seconds=event.usage.seconds)
                    closed.set()
                    return
                else:
                    log(kind)

        async def pump() -> None:
            await started.wait()
            while True:
                frame = await mic.get_frame()
                if args.gate and (speaker.pending_seconds > 0 or speaker.played_level(1.2) > 100):
                    frame = SILENCE
                await conn.session.input_audio.append(audio=base64.b64encode(frame).decode("ascii"))

        receiver = asyncio.create_task(receive())
        pumper = asyncio.create_task(pump())
        try:
            await asyncio.wait_for(started.wait(), timeout=10.0)
            if args.speak == "instructions":
                await conn.session.instructions.append(content=SPEAK_FIRST, delegation_id=None)
                log("instructions.append: speak first")
            elif args.speak == "commentary":
                await conn.session.commentary.append(content=GREETING, delegation_id=None)
                log("commentary.append: greeting")
            print(f"\n--- talk now; ends in {args.seconds:.0f} s (Ctrl+C sooner) ---\n", flush=True)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(closed.wait(), timeout=args.seconds)
        except TimeoutError:
            log("session.started never came")
        except KeyboardInterrupt:
            log("Ctrl+C")
        finally:
            pumper.cancel()
            if not closed.is_set():
                log("session.close")
                with contextlib.suppress(Exception):
                    await conn.session.close()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(closed.wait(), timeout=5.0)
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver

    with contextlib.suppress(Exception):
        await asyncio.wait_for(speaker.wait_idle(), timeout=3.0)
    summary = recorder.stop(reason="probe ended")
    await audio.close()
    live_usd = facts["seconds"] / 60 * LIVE_PRICE_PER_MIN
    print("\n--- probe ---", flush=True)
    print(
        f"connect→started {facts.get('started_s', '?')} s · first audio {facts.get('first_audio_s', '?')} s"
        f" · first speech {facts.get('first_speech_s', 'never')} s",
        flush=True,
    )
    print(
        f"audio deltas {facts['deltas']} ({facts['loud_deltas']} with her voice) · live seconds {facts['seconds']}"
        f" · context {facts['ratio']}",
        flush=True,
    )
    print(f"cost ${live_usd + facts['backend_usd']:.4f} (live ${live_usd:.4f} + backend ${facts['backend_usd']:.4f})", flush=True)
    print(f"you>   {lines['you'].strip()}", flush=True)
    print(f"alexa> {lines['alexa'].strip()}", flush=True)
    if summary:
        print(f"recording: data/recordings/{summary.get('name', name)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--model", default="gpt-live-1")
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--backend", default="", help="Responses backend model, e.g. gpt-5.6-luna ('' = none)")
    parser.add_argument("--effort", default="minimal", help="backend reasoning effort")
    parser.add_argument("--web", action="store_true", help="give the backend the native web_search tool")
    parser.add_argument("--gate", action="store_true", help="send silence while she plays (loud speaker)")
    parser.add_argument(
        "--speak", choices=("commentary", "instructions", "none"), default="commentary",
        help="how she is asked to speak first: a commentary line to say, a speak-first instruction, or not at all",
    )
    args = parser.parse_args()
    try:
        code = asyncio.run(probe(args))
    except KeyboardInterrupt:
        code = 130
    sys.stdout.flush()
    os._exit(code)  # PortAudio streams and the Tk-less process still hang the interpreter on Windows


if __name__ == "__main__":
    main()
