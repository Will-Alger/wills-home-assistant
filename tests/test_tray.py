"""The tray icon: her state as a colour, the menu, and a stand-in for pystray."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import ClassVar

from assistant.status import AssistantStatus
from assistant.tray import COLORS, TrayIcon, color_for, draw_icon


def test_the_colour_follows_her_state_and_listening_wins() -> None:
    assert color_for("idle", False) == COLORS["idle"]
    assert color_for("error", False) == COLORS["error"]
    assert color_for("error", True) == COLORS["listening"]
    assert color_for("nonsense", False) == COLORS["idle"]
    image = draw_icon(COLORS["listening"], size=32)
    assert image.size == (32, 32) and image.mode == "RGBA"
    assert image.getpixel((16, 4))[:3] == COLORS["listening"] and image.getpixel((0, 0))[3] == 0


def test_the_menu_is_the_dashboard_first_then_the_panel_then_the_two_ways_out() -> None:
    opened: list[str] = []
    tray = TrayIcon(AssistantStatus(), dashboard_url="http://127.0.0.1:8765/",
                    open_settings=lambda: opened.append("settings"), restart=lambda: opened.append("restart"),
                    quit=lambda: opened.append("quit"))
    labels = [label for label, _ in tray.menu_items()]
    assert labels == ["Open dashboard", "Settings", "", "Restart", "Quit"]
    for label, action in tray.menu_items():
        if action is not None and label != "Open dashboard":
            action()
    assert opened == ["settings", "restart", "quit"]
    bare = TrayIcon(AssistantStatus())
    assert bare.menu_items() == []


class FakePystray:
    """Just enough of pystray to see what the tray does with it."""

    class Menu:
        SEPARATOR = "---"

        def __init__(self, *entries) -> None:
            self.entries = entries

    class MenuItem:
        def __init__(self, text, action, default=False) -> None:
            self.text, self.action, self.default = text, action, default

    class Icon:
        created: ClassVar[list] = []

        def __init__(self, name, icon, title, menu) -> None:
            self.name, self.icon, self.title, self.menu = name, icon, title, menu
            self.detached = False
            self.stopped = False
            FakePystray.Icon.created.append(self)

        def run_detached(self) -> None:
            self.detached = True

        def stop(self) -> None:
            self.stopped = True


def test_the_icon_is_drawn_detached_and_redrawn_when_her_state_changes() -> None:
    status = AssistantStatus(mic="Snowball")
    status.set_state("idle")
    tray = TrayIcon(status, name="Alexa", dashboard_url="http://x/", restart=lambda: None, backend=FakePystray, poll_s=0.02)
    assert tray.start() is True
    icon = FakePystray.Icon.created[-1]
    assert icon.detached and icon.name == "Alexa" and icon.title.startswith("Alexa · idle")
    assert [e.text for e in icon.menu.entries if e != "---"] == ["Open dashboard", "Settings", "Restart"]
    assert icon.menu.entries[0].default is True  # a left click opens the dashboard
    before = icon.icon
    status.set_listening(True)
    time.sleep(0.1)
    assert icon.icon is not before and tray.shown == ("idle", True)
    tray.stop()
    assert icon.stopped


def test_no_pystray_is_a_note_not_a_failure() -> None:
    broken = SimpleNamespace(Icon=None, Menu=None, MenuItem=None)
    tray = TrayIcon(AssistantStatus(), dashboard_url="http://x/", backend=broken)
    assert tray.start() is False and tray.note.startswith("no tray icon")
