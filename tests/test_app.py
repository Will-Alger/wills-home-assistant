"""The idle trigger loop: wake word or a due announcement starts a conversation."""

from __future__ import annotations

from assistant.app import wait_for_trigger

WAKE = b"WAKE"
NOISE = b"noise"


class Source:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    async def get_frame(self) -> bytes:
        return self._frames.pop(0) if self._frames else NOISE


class Wake:
    def detect(self, frame: bytes) -> bool:
        return frame == WAKE


class Announcer:
    def __init__(self, due_after_frames: int) -> None:
        self.calls = 0
        self._due_after = due_after_frames

    def due(self) -> bool:
        self.calls += 1
        return self.calls >= self._due_after


async def test_wake_word_wins() -> None:
    assert await wait_for_trigger(Source([NOISE, NOISE, WAKE]), Wake(), None) == "wake"


async def test_due_announcement_triggers_without_wake() -> None:
    announcer = Announcer(due_after_frames=3)
    assert await wait_for_trigger(Source([NOISE] * 10), Wake(), announcer) == "announce"
    assert announcer.calls == 3  # checked once per frame, no busy loop


async def test_wake_beats_announcement_on_the_same_frame() -> None:
    announcer = Announcer(due_after_frames=1)
    assert await wait_for_trigger(Source([WAKE]), Wake(), announcer) == "wake"


async def test_restart_request_ends_the_idle_wait() -> None:
    """A phone approve while idle: the runner must notice without a wake word."""
    assert await wait_for_trigger(Source([NOISE] * 5), Wake(), None, restart=lambda: True) == "restart"
    flips = iter([False, False, True])
    assert await wait_for_trigger(Source([NOISE] * 5), Wake(), None, restart=lambda: next(flips)) == "restart"
