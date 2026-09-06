"""What the running session knows about itself, right now.

The Settings panel reads this from its own (Tk) thread while the app writes
it from the event loop, so every field lives behind one lock and comes out as
a plain dict. Deliberately shallow: a live mirror plus a small ring of recent
lines — the journal and the session log stay the durable record.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class AssistantStatus:
    def __init__(
        self,
        *,
        mic: str = "",
        voice: str = "",
        wake_word: str = "",
        home: str = "",
        speaker: str = "",
        keep_lines: int = 300,
    ) -> None:
        self._lock = threading.Lock()
        self._fields = {
            "mic": mic, "voice": voice, "wake_word": wake_word, "home": home, "speaker": speaker,
        }
        self._state = "starting"
        self._listening = False
        self._level = 0.0  # live mic level, 0..1, only while she is listening
        self._error = ""
        self._log: deque[tuple[float, str]] = deque(maxlen=keep_lines)
        self._started = time.time()

    # ── writes (event loop) ───────────────────────────────────────────────

    def configure(self, **fields: str) -> None:
        """Set the descriptive fields (mic, speaker, voice, wake_word, home)."""
        with self._lock:
            for key, value in fields.items():
                if key in self._fields and value:
                    self._fields[key] = value

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state
            if state != "error":
                self._error = ""

    def set_listening(self, on: bool) -> None:
        with self._lock:
            self._listening = bool(on)
            if not self._listening:
                self._level = 0.0  # the bar goes flat with the flag, not after it

    def set_level(self, level: float) -> None:
        """How loud the room is right now, 0 (flat) to 1 (full bar). Ignored
        unless a listening window is open: a moving bar must never claim she
        is hearing you when she isn't."""
        with self._lock:
            self._level = min(1.0, max(0.0, float(level))) if self._listening else 0.0

    def note(self, text: str) -> None:
        """One line for the live feed. Empty lines and repeats are dropped."""
        text = " ".join(str(text).split())
        if not text:
            return
        with self._lock:
            if self._log and self._log[-1][1] == text:
                return
            self._log.append((time.time(), text))

    def error(self, message: str) -> None:
        message = " ".join(str(message).split()) or "something went wrong"
        with self._lock:
            self._listening = False  # a failure must never look like listening
            self._level = 0.0
            self._state = "error"
            self._error = message
        self.note(f"error: {message}")

    # ── reads (any thread) ────────────────────────────────────────────────

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    @property
    def level(self) -> float:
        """The bar's height. Read many times a second by the panel, so it
        stays a plain float behind the same lock — no snapshot needed."""
        with self._lock:
            return self._level

    def summary(self) -> str:
        with self._lock:
            state, error, since = self._state, self._error, self._started
        minutes = int((time.time() - since) // 60)
        uptime = f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"
        return f"{state}{f' — {error}' if error else ''} · up {uptime}"

    def lines(self, limit: int = 200) -> list[str]:
        with self._lock:
            recent = list(self._log)[-limit:]
        return [f"{time.strftime('%H:%M:%S', time.localtime(at))}  {text}" for at, text in recent]

    def snapshot(self, log_lines: int = 200) -> dict[str, object]:
        with self._lock:
            fields = dict(self._fields)
            fields["state"] = self._state
            fields["error"] = self._error
            fields["listening"] = self._listening
            fields["level"] = self._level
        fields["summary"] = self.summary()
        fields["log"] = self.lines(log_lines)
        return fields
