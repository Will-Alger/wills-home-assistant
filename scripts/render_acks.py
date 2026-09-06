"""Render the wake acknowledgments — the eight ways she answers her name.

    uv run scripts/render_acks.py                    # rewrite assets/voice/ack/*.wav
    uv run scripts/render_acks.py --only yes mm-hm   # re-take just those two
    uv run scripts/render_acks.py --list             # what would be spoken

Rendered through the REALTIME API, not the TTS one. This is the one place
where what she says off the disk has to be indistinguishable from what she
says live a second later, and that means her actual Realtime voice
(`REALTIME_VOICE` — `sol`, with the engine's own fallback to `marin` while it
is org-gated). The machinery is the engine's `text_probe`, the same path
`m4_realtime.py --text-probe` uses to save a reply as a WAV; the clips are
saved 24 kHz mono int16, what the session speaker takes.

The WAVs are committed; this script exists only to re-render them, and it is
one of the two things in the repo that spend API money (the other is
`render_fallbacks.py`) — a few cents for eight two-word clips, once. Nothing
at runtime calls it.

Every take is trimmed of the silence the model leaves around two words
(`trim`) and then checked before it is kept (`problem`): under `MAX_S` and the
words that were asked for, nothing else. A wake acknowledgment that runs long
stops being an acknowledgment and becomes a reply he has to sit through, and
every extra syllable is another moment the microphone has to be held shut for
(audio/acks.py). A take that fails is asked for again.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import numpy as np

from assistant.audio.acks import PHRASES, RATE, ack_dir, phrase
from assistant.audio.fallbacks import write_wav
from assistant.config import load_settings
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.home.fake import FakeHome

MAX_S = 0.9  # longer than this is not an acknowledgment any more
TRIES = 3
SILENCE = 200  # int16 RMS below this, over 10 ms, is nothing
PAD_S = 0.04  # kept either side of the words, so no consonant is clipped

# She is answering her name from across the room, not opening a conversation.
DELIVERY = (
    "Warm, quiet and unhurried, at an ordinary indoor speaking pace, over in "
    "half a second. You have just heard your name and you are answering it. "
    "No brightness, no performance, no trailing words."
)


def ask(line: str) -> str:
    """The one thing said to the model for a clip."""
    return (
        f'Say exactly this, and nothing else: "{line}" — not one word more, '
        f"no tools, and do not treat it as a question to answer. {DELIVERY}"
    )


def trim(pcm: bytes, rate: int = RATE) -> bytes:
    """Cut the silence the model leaves around a two-word answer.

    A third of a second of nothing after "Yes?" is a third of a second longer
    the microphone stays shut for a sound nobody hears (audio/acks.py).
    Measured in 10 ms windows, with a pad kept either side."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    win = max(1, int(rate * 0.01))
    blocks = len(samples) // win
    loud = [i for i in range(blocks) if np.sqrt(np.mean(samples[i * win : (i + 1) * win] ** 2)) > SILENCE]
    if not loud:
        return pcm  # all quiet: not ours to judge, `problem` will say so
    pad = int(rate * PAD_S)
    start = max(0, loud[0] * win - pad)
    end = min(len(samples), (loud[-1] + 1) * win + pad)
    return samples[start:end].astype("<i2").tobytes()


def words(text: str) -> list[str]:
    """The spoken words of a line, for holding a take against what was asked."""
    return [word for word in re.split(r"[^a-z0-9']+", text.lower()) if word]


def problem(line: str, transcript: str, pcm: bytes, rate: int = RATE) -> str:
    """Why this take cannot be kept — "" when it can.

    Length is the hard rule and is judged from the audio itself; the
    transcript catches the take where she said the line AND something else.
    """
    if not pcm:
        return "no audio came back"
    seconds = len(pcm) / 2 / rate
    if seconds > MAX_S:
        return f"{seconds:.2f}s, over the {MAX_S:.1f}s ceiling"
    if transcript.strip() and words(transcript) != words(line):
        return f"she said “{transcript.strip()}”"
    return ""


async def render(out: Path, slugs: list[str], tries: int) -> int:
    settings = load_settings()
    settings.require("openai_api_key")
    engine = RealtimeEngine(
        api_key=settings.openai_api_key,
        model=settings.realtime_model,
        voice=settings.realtime_voice,
        home=FakeHome(),
        owner=settings.owner_name,
        name=settings.assistant_name,
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(slugs)} clips · {settings.realtime_model} · voice {settings.realtime_voice}")
    spent = 0.0
    failed: list[str] = []
    for slug in slugs:
        line = phrase(slug, settings.owner_name)
        keep, why = b"", "nothing was rendered"
        for _attempt in range(tries):
            transcript, audio, stats = await engine.text_probe(ask(line))
            spent += stats.cost_usd
            audio = trim(audio, RATE)
            if engine.voice_note:
                print(f"  {engine.voice_note}")
                engine.voice_note = None
            why = problem(line, transcript, audio, RATE)
            if not why:
                keep = audio
                break
            print(f"  retaking {slug}: {why}")
            if not keep and audio and len(audio) / 2 / RATE <= MAX_S:
                keep = audio  # the right length, only the words looked wrong
        if not keep:
            failed.append(slug)
            print(f"  ✗ {slug}: {why} — not written")
            continue
        path = out / f"{slug}.wav"
        write_wav(path, keep, RATE)
        print(f"  {path.name}  {len(keep) / 2 / RATE:.2f}s  “{line}”")
    print(f"\n{len(slugs) - len(failed)} of {len(slugs)} clips in {out} · ${spent:.4f}")
    if failed:
        print(f"still missing: {', '.join(failed)} — run again for those")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="SLUG", help=f"a subset of {', '.join(PHRASES)}")
    parser.add_argument("--out", type=Path, default=None, help="where to write the WAVs")
    parser.add_argument("--tries", type=int, default=TRIES, help=f"takes per clip (default {TRIES})")
    parser.add_argument("--list", action="store_true", help="print the lines and stop")
    args = parser.parse_args()
    owner = load_settings().owner_name
    if args.list:
        for slug in PHRASES:
            print(f"{slug:>10}  “{phrase(slug, owner)}”")
        return 0
    slugs = args.only or list(PHRASES)
    unknown = [slug for slug in slugs if slug not in PHRASES]
    if unknown:
        print(f"unknown clip(s): {', '.join(unknown)} — one of {', '.join(PHRASES)}")
        return 2
    return asyncio.run(render(args.out or ack_dir(), slugs, max(1, args.tries)))


if __name__ == "__main__":
    raise SystemExit(main())
