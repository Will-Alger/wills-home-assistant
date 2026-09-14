"""A tray icon at the bottom right: is she running, and one click from the dashboard.

Will: "not sure if she's running. It would be great if I could have a little
icon to click at the bottom right with a caret to open the dashboard." The
icon is a dot in her state's colour — grey idle, green while she listens,
amber while a tool works, red on an error — with the status line as its
tooltip; a left click opens the dashboard, the menu has the Settings panel,
Restart and Quit. pystray draws it on its own thread; the menu's callbacks
are handed in by the runner already made safe to call from there. No tray
(a headless run, pystray missing) is a note, never a failure.
"""

from __future__ import annotations

import contextlib
import threading
import time
import webbrowser
from collections.abc import Callable
from typing import Any

COLORS: dict[str, tuple[int, int, int]] = {
    "starting": (128, 128, 200),
    "idle": (150, 150, 150),
    "listening": (52, 168, 83),
    "working": (240, 170, 40),
    "error": (200, 60, 50),
}
_SIZE = 64


def color_for(state: str, listening: bool) -> tuple[int, int, int]:
    """The dot's colour: listening wins, then the state, grey for the unknown."""
    if listening:
        return COLORS["listening"]
    return COLORS.get(str(state), COLORS["idle"])


def draw_icon(color: tuple[int, int, int], size: int = _SIZE) -> Any:
    """A filled circle with a small dark centre, on a transparent square."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    pad = size // 8
    draw.ellipse((pad, pad, size - pad, size - pad), fill=(*color, 255))
    centre = size // 2
    r = size // 9
    draw.ellipse((centre - r, centre - r, centre + r, centre + r), fill=(30, 30, 30, 255))
    return image


class TrayIcon:
    def __init__(
        self,
        status: Any,
        *,
        name: str = "Alexa",
        dashboard_url: str = "",
        open_settings: Callable[[], Any] | None = None,
        restart: Callable[[], Any] | None = None,
        quit: Callable[[], Any] | None = None,
        backend: Any | None = None,
        poll_s: float = 1.0,
    ) -> None:
        self._status = status
        self._name = name
        self.dashboard_url = dashboard_url
        self._open_settings = open_settings
        self._restart = restart
        self._quit = quit
        self._backend = backend  # the pystray module, or a stand-in in tests
        self._poll_s = poll_s
        self._icon: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.shown: tuple[str, bool] | None = None  # the state the icon was last drawn for
        self.note = ""  # why there is no icon, for the boot log

    # ── what the menu offers ──────────────────────────────────────────────

    def open_dashboard(self) -> None:
        if self.dashboard_url:
            with contextlib.suppress(Exception):
                webbrowser.open(self.dashboard_url)

    def open_settings(self) -> None:
        """The dashboard's Settings tab (the Tk panel is being retired)."""
        if self._open_settings is not None:
            with contextlib.suppress(Exception):
                self._open_settings()
        elif self.dashboard_url:
            with contextlib.suppress(Exception):
                webbrowser.open(self.dashboard_url + "#settings")

    def menu_items(self) -> list[tuple[str, Callable[[], Any] | None]]:
        """(label, action) in menu order; None is a separator. Pure, so the
        menu can be checked without a tray."""
        items: list[tuple[str, Callable[[], Any] | None]] = []
        if self.dashboard_url:
            items.append(("Open dashboard", self.open_dashboard))
            items.append(("Settings", self.open_settings))
        elif self._open_settings is not None:
            items.append(("Settings", self.open_settings))
        if items:
            items.append(("", None))
        if self._restart is not None:
            items.append(("Restart", self._restart))
        if self._quit is not None:
            items.append(("Quit", self._quit))
        return items

    # ── the icon itself ───────────────────────────────────────────────────

    def start(self) -> bool:
        backend = self._backend
        if backend is None:
            try:
                import pystray as backend  # type: ignore[no-redef]
            except Exception as err:  # noqa: BLE001 — no tray is a note, never a failure
                self.note = f"no tray icon: {err}"
                return False
        try:
            entries = []
            for index, (label, action) in enumerate(self.menu_items()):
                if action is None:
                    entries.append(backend.Menu.SEPARATOR)
                    continue
                entries.append(backend.MenuItem(label, self._guarded(action), default=(index == 0)))
            state, listening = self._current()
            self._icon = backend.Icon(self._name, draw_icon(color_for(state, listening)), self._title(), backend.Menu(*entries))
            self._icon.run_detached()
        except Exception as err:  # noqa: BLE001
            self.note = f"no tray icon: {err}"
            self._icon = None
            return False
        self.shown = (state, listening)
        self._thread = threading.Thread(target=self._poll, name="tray", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._icon is not None:
            with contextlib.suppress(Exception):
                self._icon.stop()
            self._icon = None

    @staticmethod
    def _guarded(action: Callable[[], Any]) -> Callable[..., None]:
        def run(*_args: Any) -> None:  # pystray passes (icon, item)
            with contextlib.suppress(Exception):
                action()

        return run

    def _current(self) -> tuple[str, bool]:
        try:
            snap = self._status.snapshot(log_lines=0)
            return str(snap.get("state", "idle")), bool(snap.get("listening"))
        except Exception:  # noqa: BLE001
            return "idle", False

    def _title(self) -> str:
        try:
            return f"{self._name} · {self._status.summary()}"[:127]  # Windows caps the tooltip
        except Exception:  # noqa: BLE001
            return self._name

    def refresh(self) -> bool:
        """Redraw if her state changed since the last drawing; True when it did."""
        state, listening = self._current()
        if self._icon is not None:
            with contextlib.suppress(Exception):
                self._icon.title = self._title()
        if (state, listening) == self.shown:
            return False
        self.shown = (state, listening)
        if self._icon is not None:
            with contextlib.suppress(Exception):
                self._icon.icon = draw_icon(color_for(state, listening))
        return True

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            time.sleep(self._poll_s)
