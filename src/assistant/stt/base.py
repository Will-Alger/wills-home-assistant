"""Provider-agnostic streaming STT interface.

A stream lasts one listening turn: audio frames go in, transcript events come
out, and a `final` event means the provider decided the speaker is done
(end-of-turn) — the pipeline's cue to hand the transcript to the brain.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol


class SttError(RuntimeError):
    pass


@dataclass(frozen=True)
class SttEvent:
    kind: Literal["partial", "final"]
    transcript: str


class SttStream(Protocol):
    async def send_audio(self, pcm: bytes) -> None: ...

    def events(self) -> AsyncIterator[SttEvent]: ...


class SttProvider(Protocol):
    def stream(self) -> AbstractAsyncContextManager[SttStream]: ...
