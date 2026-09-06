"""Render the four spoken fallbacks — the only lines she can say offline.

    uv run scripts/render_fallbacks.py                  # rewrite assets/voice/*.wav
    uv run scripts/render_fallbacks.py --voice cedar    # ...in another voice
    uv run scripts/render_fallbacks.py --list           # what would be spoken

The WAVs are committed; this script exists only to re-render them, and it
is the one thing in the repo that spends API money — a few tenths of a
cent for four short sentences. Nothing at runtime calls it.

Shapes verified against the installed OpenAI SDK (`openai/types/audio/
speech_create_params.py`, `openai/resources/audio/speech.py`):
`client.audio.speech.create` takes `model`, `voice`, `input`, an optional
`response_format` ("wav" among them) and `instructions` (ignored by tts-1),
and returns a binary response whose `.content` is the file's bytes. Her
Realtime voice, `sol`, is not one of the speech voices (alloy, ash,
ballad, coral, echo, sage, shimmer, verse, marin, cedar), so the default
here is `marin` — the same voice the engine falls back to when sol is
gated for the account.
"""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

from openai import OpenAI

from assistant.audio.fallbacks import FILES, LINES, RATE, to_pcm, voice_dir, write_wav
from assistant.config import load_settings

MODEL = "gpt-4o-mini-tts"
VOICE = "marin"

# The lines land in a house where something has just broken. Steady and
# short beats apologetic: he needs the fact, not a performance.
INSTRUCTIONS = (
    "Calm, warm and matter-of-fact, at an unhurried indoor speaking pace. "
    "You are stating a fact about something that is down, not apologising "
    "for it. No brightness, no drama, no rising question at the end."
)


def render(voice: str, model: str, out: Path) -> int:
    settings = load_settings()
    settings.require("openai_api_key")
    client = OpenAI(api_key=settings.openai_api_key)
    out.mkdir(parents=True, exist_ok=True)
    for kind, line in LINES.items():
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=line,
            response_format="wav",
            instructions=INSTRUCTIONS,
        )
        # Whatever rate the API chose, the committed asset is the one the
        # session speaker takes: 24 kHz, mono, 16-bit.
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            pcm = to_pcm(
                wav.readframes(wav.getnframes()),
                channels=wav.getnchannels(),
                source_rate=wav.getframerate(),
                rate=RATE,
            )
        path = out / FILES[kind]
        write_wav(path, pcm, RATE)
        print(f"  {path.name}  {len(pcm) / 2 / RATE:.1f}s  “{line}”")
    print(f"\n{len(LINES)} lines in {voice} → {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=VOICE, help=f"speech voice (default {VOICE})")
    parser.add_argument("--model", default=MODEL, help=f"TTS model (default {MODEL})")
    parser.add_argument("--out", type=Path, default=None, help="where to write the WAVs")
    parser.add_argument("--list", action="store_true", help="print the lines and stop")
    args = parser.parse_args()
    if args.list:
        for kind, line in LINES.items():
            print(f"{kind:>14}  “{line}”")
        return 0
    return render(args.voice, args.model, args.out or voice_dir())


if __name__ == "__main__":
    raise SystemExit(main())
