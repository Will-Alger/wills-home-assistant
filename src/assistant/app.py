"""The idle loop, extracted so it can be tested without a microphone.

While idle, the app watches every 80 ms mic frame for the wake word AND
checks whether an announcement is due — the two things that can start a
conversation. `wait_for_trigger` returns which one happened.
"""

from __future__ import annotations

from typing import Any


async def wait_for_trigger(source: Any, wake: Any, announcer: Any = None) -> str:
    """Block until the wake phrase is heard ("wake") or an announcement is
    ready to be spoken ("announce")."""
    while True:
        frame = await source.get_frame()
        if wake.detect(frame):
            return "wake"
        if announcer is not None and announcer.due():
            return "announce"
