"""Audio-reality spike: measure the wake word on YOUR mic in YOUR room.

Run these before trusting Milestone 3 (the review's insisted change #2):

    uv run scripts/m3_spike.py --devices            # list input devices
    uv run scripts/m3_spike.py --monitor 10         # 10-min live score monitor
    uv run scripts/m3_spike.py --record test.wav 5  # capture 5s for inspection

Monitor protocol: run it (a) in a quiet room, (b) with TV/Spotify playing,
(c) while having a normal conversation — each for ~10+ minutes WITHOUT saying
the wake phrase, and note detections/hour (false accepts). Then say
"hey jarvis" from across the room ~10 times and count misses. Threshold 0.5
is the tuning knob: raise for fewer false wakes, lower for fewer misses.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import wave

from rich.console import Console

from assistant.audio.base import FRAME_MS, SAMPLE_RATE
from assistant.audio.mic import Microphone
from assistant.config import load_settings

console = Console()


def list_devices() -> int:
    import sounddevice as sd

    console.print(sd.query_devices())
    default_in = sd.query_devices(kind="input")
    console.print(f"\n[bold]Default input:[/bold] {default_in['name']}")
    return 0


async def monitor(minutes: float) -> int:
    import numpy as np

    from assistant.audio.mic import describe_device
    from assistant.wake.detector import WakeDetector

    settings = load_settings()
    console.print(f"Loading wake model '{settings.wake_model}' (first run downloads it)...")
    detector = WakeDetector(settings.wake_model, threshold=settings.wake_threshold)
    detections = 0
    started = time.monotonic()
    second_max = 0.0
    peak_level = 0.0
    last_tick = started

    console.print(f"[bold]Mic:[/bold] {describe_device(settings.audio_input_device)}")
    console.print(
        f"[green]Monitoring[/green] for {minutes:g} min (threshold "
        f"{settings.wake_threshold}). Talk or snap your fingers — the level "
        "readout should jump. Ctrl+C to stop early.\n"
    )
    try:  # Ctrl+C arrives as CancelledError inside asyncio.run on Windows
        async with Microphone(settings.audio_input_device) as mic:
            while time.monotonic() - started < minutes * 60:
                frame = await mic.get_frame()
                score = detector.score(frame)
                second_max = max(second_max, score)
                level = float(np.abs(np.frombuffer(frame, np.int16)).max()) / 32768
                peak_level = max(peak_level, level)
                now = time.monotonic()
                if score >= settings.wake_threshold:
                    detections += 1
                    stamp = time.strftime("%H:%M:%S")
                    console.print(f"[bold red]DETECTION[/bold red] {stamp} score={score:.2f}")
                    detector.reset()
                elif score >= 0.3:
                    console.print(f"[yellow]near miss[/yellow] score={score:.2f}")
                if now - last_tick >= 30:
                    elapsed = (now - started) / 60
                    console.print(
                        f"[dim]{elapsed:.1f} min · {detections} detection(s) · "
                        f"top score last 30s: {second_max:.2f} · "
                        f"peak mic level: {peak_level:.0%}[/dim]"
                    )
                    second_max = 0.0
                    peak_level = 0.0
                    last_tick = now
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    elapsed_min = (time.monotonic() - started) / 60
    per_hour = detections / elapsed_min * 60 if elapsed_min else 0.0
    console.print(
        f"\n[bold]Stats:[/bold] {elapsed_min:.1f} min · {detections} detection(s) "
        f"· {per_hour:.1f}/hour"
    )
    console.print(
        "Quiet-room target: ~0/hour. With TV/music: a few/hour may need a higher "
        "WAKE_THRESHOLD in .env."
    )
    return 0


async def record(path: str, seconds: float) -> int:
    settings = load_settings()
    frames_needed = int(seconds * 1000 / FRAME_MS)
    console.print(f"Recording {seconds:g}s to {path}...")
    chunks: list[bytes] = []
    async with Microphone(settings.audio_input_device) as mic:
        for _ in range(frames_needed):
            chunks.append(await mic.get_frame())
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(chunks))
    console.print(f"[green]✓[/green] wrote {path} — play it back and check you sound clear")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--devices", action="store_true")
    group.add_argument("--monitor", type=float, metavar="MINUTES")
    group.add_argument("--record", nargs=2, metavar=("PATH", "SECONDS"))
    args = parser.parse_args()
    try:
        if args.devices:
            return list_devices()
        if args.monitor is not None:
            return asyncio.run(monitor(args.monitor))
        return asyncio.run(record(args.record[0], float(args.record[1])))
    except KeyboardInterrupt:
        return 130  # quiet exit; stats were already printed by monitor()


if __name__ == "__main__":
    raise SystemExit(main())
