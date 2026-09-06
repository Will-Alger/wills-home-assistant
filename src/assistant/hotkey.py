"""Push to talk: hold a system-wide hotkey and speak.

A second way in beside the wake word. HIS action defines the utterance, so a
mid-sentence pause never ends his turn and she never keeps listening once he
has let go — the two complaints semantic VAD keeps earning.

Windows, no admin, no window: `GetAsyncKeyState` polled fifty times a second
from the event loop. It reports the global key state whatever has focus, and
its most significant bit means "down right now" (verified against the Win32
reference, 2026-09-04). We deliberately do NOT install a keyboard hook — a
hook would let us swallow the keystroke, but it also needs the elevation this
app must never ask for. The price is that the keys still reach whatever app
is focused, which is exactly why the default is modifier-only (Ctrl+Alt on
its own means nothing to most programs). The one blind spot is documented by
Microsoft: while an ELEVATED window has focus the call returns zero, so a
press there simply isn't seen.

`PushToTalk` is the whole contract the rest of the app sees: `held` right
now, and `presses` counting up. Level, not events — a session that opens
mid-hold reads the same state the idle loop did, and no edge can be lost
between the two. Tests drive `press()` / `release()` directly instead of a
keyboard.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Callable, Sequence

# Virtual-key codes (winuser.h). The modifiers are the un-sided constants:
# either Ctrl, either Alt — he shouldn't have to care which hand he used.
_MODIFIERS: dict[str, int] = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "menu": 0x12, "option": 0x12,
    "shift": 0x10,
    "win": 0x5B, "cmd": 0x5B, "super": 0x5B,
    "rwin": 0x5C,
}
_NAMED: dict[str, int] = {"space": 0x20, "tab": 0x09, "capslock": 0x14, **_MODIFIERS}
_LABELS: dict[str, str] = {
    "ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "menu": "Alt", "option": "Alt",
    "shift": "Shift", "win": "Win", "cmd": "Win", "super": "Win", "rwin": "Win",
    "space": "Space", "tab": "Tab", "capslock": "CapsLock",
}

# 50 Hz: a press is felt within 20 ms and the poll costs nothing measurable.
POLL_S = 0.02


class HotkeyError(ValueError):
    """The configured hotkey can't be used here — always readable aloud."""


def _key_code(part: str) -> int | None:
    if part in _NAMED:
        return _NAMED[part]
    if len(part) == 1 and (part.isalpha() or part.isdigit()):
        return ord(part.upper())  # A-Z and 0-9 are their own VK codes
    if part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
        return 0x70 + int(part[1:]) - 1
    return None


def _parts(spec: str) -> list[str]:
    return [p.strip().lower() for p in spec.replace("-", "+").split("+") if p.strip()]


def parse_hotkey(spec: str) -> list[int]:
    """'ctrl+alt' -> the virtual-key codes that must all be down at once."""
    parts = _parts(spec)
    if not parts:
        raise HotkeyError("the push-to-talk hotkey is empty")
    codes = []
    for part in parts:
        code = _key_code(part)
        if code is None:
            raise HotkeyError(
                f"'{part}' isn't a key I know — use ctrl, alt, shift, win, "
                "space, a letter, a digit or f1 to f24"
            )
        codes.append(code)
    return codes


def describe_hotkey(spec: str) -> str:
    """The spoken/written form: 'ctrl+alt' -> 'Ctrl+Alt'."""
    return "+".join(_LABELS.get(p, p.upper()) for p in _parts(spec))


def windows_key_poll(codes: Sequence[int]) -> Callable[[], bool]:
    """A callable answering 'are all these keys down right now?' on Windows."""
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    keys = list(codes)

    def held() -> bool:
        # The high bit means down. A zero return means the call could not see
        # the keyboard (an elevated window has focus) — read that as not held
        # rather than pretending, so she never opens a session on a guess.
        return all(user32.GetAsyncKeyState(key) & 0x8000 for key in keys)

    return held


class PushToTalk:
    """Whether the hotkey is held, and how many times it has been pressed.

    `run()` polls the keyboard; a test calls `press()` / `release()` instead.
    """

    def __init__(
        self,
        poll: Callable[[], bool] | None = None,
        *,
        label: str = "",
        interval_s: float = POLL_S,
    ) -> None:
        self._poll = poll
        self._interval = interval_s
        self.label = label
        self.held = False
        self.presses = 0
        self.pressed_at = 0.0
        self.released_at = 0.0

    def press(self) -> None:
        if self.held:
            return
        self.held = True
        self.presses += 1
        self.pressed_at = time.monotonic()

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        self.released_at = time.monotonic()

    async def run(self) -> None:
        """Watch the keyboard forever. Cancelled with the app."""
        if self._poll is None:
            return
        while True:
            down = False
            with contextlib.suppress(Exception):  # a poll that fails is 'not held'
                down = self._poll()
            if down:
                self.press()
            else:
                self.release()
            await asyncio.sleep(self._interval)


def build_push_to_talk(spec: str) -> PushToTalk | None:
    """The hotkey source for this machine; None when it is switched off.

    Raises HotkeyError with a sentence she can read aloud when the spec is a
    typo or the platform has no way to watch the keyboard.
    """
    spec = (spec or "").strip()
    if not spec or spec.lower() in ("off", "none"):
        return None
    codes = parse_hotkey(spec)
    if sys.platform != "win32":
        raise HotkeyError("push to talk needs Windows — the wake word still works")
    return PushToTalk(windows_key_poll(codes), label=describe_hotkey(spec))
