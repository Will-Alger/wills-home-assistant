"""The Settings panel — a small desktop window she opens and closes by voice.

It shows what the running session knows about itself (microphone, whether she
is listening, voice, wake word, a status summary and a live log feed) and
holds the two settings worth changing without a keyboard — her voice and her
wake word — plus a restart button, because both only take effect on a fresh
session. It is deliberately NOT a microphone switch: nothing in the panel
turns listening on or off.

The window is Tk (standard library, no new dependency) on its own thread, and
every Tk call happens on that thread: the app posts intent through flags the
refresh tick reads. A machine with no Tk gets an honest spoken sentence
instead of a crash.

Changes are written beside her other state (`data/panel.json`), never into
`.env` — that file stays the owner's. `PanelOverrides.apply` folds them over
the loaded settings at startup.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant.config import wake_phrase

# The Realtime voices the API accepts (verified against the installed openai
# SDK: RealtimeAudioConfigOutput), plus "sol" — Will's pick, still org-gated,
# and the engine falls back to marin on its own when it is refused.
VOICES: tuple[str, ...] = (
    "sol", "marin", "cedar", "alloy", "ash", "ballad",
    "coral", "echo", "sage", "shimmer", "verse",
)

# openWakeWord's pretrained wake phrases (verified against the installed
# openwakeword package); custom .onnx models in models/ are offered too.
WAKE_MODELS: tuple[str, ...] = ("alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy")

_REFRESH_S = 0.4


class PanelUnavailable(RuntimeError):
    """No desktop window is possible here (no Tk, no display, no session)."""


class PanelOverrides:
    """Voice and wake word chosen from the panel. A tiny JSON file so a
    restart picks them up; unreadable or half-written, it simply doesn't
    apply — the .env value stands."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: str(v) for k, v in data.items() if isinstance(data, dict) and v}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def voice(self) -> str:
        return self._read().get("realtime_voice", "")

    @property
    def wake_model(self) -> str:
        return self._read().get("wake_model", "")

    def set(self, *, voice: str = "", wake_model: str = "") -> dict[str, str]:
        data = self._read()
        if voice:
            data["realtime_voice"] = voice
        if wake_model:
            data["wake_model"] = wake_model
        self._write(data)
        return data

    def clear(self, field: str) -> None:
        data = self._read()
        if data.pop(field, None) is not None:
            self._write(data)

    def apply(self, settings: Any) -> list[str]:
        """Fold the overrides over freshly loaded settings; returns what
        actually changed, for the boot log."""
        changed = []
        for field, attr in (("realtime_voice", "realtime_voice"), ("wake_model", "wake_model")):
            value = self._read().get(field, "")
            if value and value != getattr(settings, attr, value):
                setattr(settings, attr, value)
                changed.append(f"{attr}={value}")
        return changed


class SettingsPanel:
    """Everything the panel shows and does. The Tk window is a view over
    this; tests drive it with a fake one."""

    def __init__(
        self,
        status: Any,
        overrides: PanelOverrides,
        *,
        restart: Callable[[], None] | None = None,
        models_dir: Path | None = None,
        view_factory: Callable[[SettingsPanel], Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._status = status
        self._overrides = overrides
        self._restart = restart
        self._models_dir = Path(models_dir) if models_dir else None
        self._view_factory = view_factory or _tk_view
        self._log = log
        self._view: Any | None = None

    # ── opened and closed by voice ────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._view is not None

    def open(self) -> str:
        if self.is_open:
            return "the settings panel is already open"
        try:
            view = self._view_factory(self)
            view.start()
        except PanelUnavailable:
            raise
        except Exception as err:  # a window that won't open is a sentence, not a crash
            raise PanelUnavailable(str(err)) from err
        self._view = view
        self.note("settings panel opened")
        return "the settings panel is on the desktop now"

    def close(self) -> str:
        if not self.is_open:
            return "the settings panel isn't open"
        view = self._view
        self._view = None
        with contextlib.suppress(Exception):
            view.stop()
        self.note("settings panel closed")
        return "settings panel closed"

    def view_closed(self) -> None:
        """The window itself went away (he clicked the X)."""
        if self._view is not None:
            self._view = None
            self.note("settings panel closed")

    # ── what the window renders ───────────────────────────────────────────

    def voice_choices(self) -> list[str]:
        return list(VOICES)

    def wake_choices(self) -> dict[str, str]:
        """Spoken label -> wake model name/path, pretrained plus any custom
        model sitting in models/."""
        choices = {wake_phrase(name): name for name in WAKE_MODELS}
        if self._models_dir is not None:
            with contextlib.suppress(OSError):
                for model in sorted(self._models_dir.glob("*.onnx")):
                    choices[f"{wake_phrase(model.name)} (custom)"] = str(model)
        return choices

    def snapshot(self) -> dict[str, object]:
        data = self._status.snapshot() if self._status is not None else {}
        data["saved_voice"] = self._overrides.voice
        data["saved_wake_word"] = self._overrides.wake_model
        return data

    def note(self, text: str) -> None:
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.note(text)
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log(text)

    # ── the controls ──────────────────────────────────────────────────────

    def save(self, *, voice: str = "", wake_word: str = "") -> str:
        """Store a new voice and/or wake word. Both apply on the next start,
        which is what the restart button is for."""
        voice = voice.strip()
        if voice and voice not in VOICES:
            return f"'{voice}' isn't one of her voices"
        model = ""
        label = wake_word.strip()
        if label:
            choices = self.wake_choices()
            model = choices.get(label) or (label if label in choices.values() else "")
            if not model:
                return f"'{label}' isn't a wake word she can load"
        if not voice and not model:
            return "nothing to save"
        self._overrides.set(voice=voice, wake_model=model)
        saved = ", ".join(
            part for part in (f"voice {voice}" if voice else "", f"wake word {wake_phrase(model)}" if model else "")
            if part
        )
        self.note(f"saved {saved} — restart to apply")
        return f"saved {saved} — restart to apply"

    def restart(self) -> str:
        if self._restart is None:
            return "restarting isn't wired up in this session"
        self.note("restart requested from the settings panel")
        self._restart()
        return "restarting — she's back in about fifteen seconds"


# ── the Tk window ─────────────────────────────────────────────────────────


def _tk_view(panel: SettingsPanel) -> Any:
    return _TkPanel(panel)


class _TkPanel:
    """The window. Created, refreshed and destroyed on its own thread —
    the only thread that ever touches Tk."""

    def __init__(self, panel: SettingsPanel) -> None:
        self._panel = panel
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._error: str = ""
        self._shown: list[str] = []

    def start(self) -> None:
        try:
            import tkinter  # noqa: F401 — presence check before spawning a thread
        except Exception as err:
            raise PanelUnavailable(f"this machine has no Tk ({err})") from err
        self._thread = threading.Thread(target=self._run, name="settings-panel", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._closing.set()
            raise PanelUnavailable("the window did not come up")
        if self._error:
            raise PanelUnavailable(self._error)

    def stop(self) -> None:
        self._closing.set()  # the refresh tick tears the window down, on its thread

    def _run(self) -> None:
        try:
            self._build()
        except Exception as err:  # noqa: BLE001 — report, never crash the app
            self._error = str(err)
            self._ready.set()
            return
        self._ready.set()
        with contextlib.suppress(Exception):
            self._root.mainloop()
        self._panel.view_closed()

    def _build(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("Alexa — Settings")
        self._root.geometry("560x520")
        self._root.protocol("WM_DELETE_WINDOW", self._on_x)

        frame = ttk.Frame(self._root, padding=12)
        frame.pack(fill="both", expand=True)

        self._values: dict[str, Any] = {}
        for row, (key, label) in enumerate(
            (("mic", "Microphone"), ("listening", "Listening"), ("summary", "Status"))
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            value = ttk.Label(frame, text="…", wraplength=380, justify="left")
            value.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
            self._values[key] = value

        snapshot = self._panel.snapshot()
        # show what is SAVED when it differs from what is running: he changed
        # it and hasn't restarted yet
        self._voice = tk.StringVar(
            value=str(snapshot.get("saved_voice") or snapshot.get("voice", ""))
        )
        ttk.Label(frame, text="Voice").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame, textvariable=self._voice, values=self._panel.voice_choices(),
            state="readonly", width=18,
        ).grid(row=3, column=1, sticky="w", pady=2)

        self._wake_choices = self._panel.wake_choices()
        current = str(snapshot.get("saved_wake_word") or snapshot.get("wake_word", ""))
        self._wake = tk.StringVar(
            value=next((k for k, v in self._wake_choices.items() if v == current), current)
        )
        ttk.Label(frame, text="Wake word").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame, textvariable=self._wake, values=list(self._wake_choices),
            state="readonly", width=18,
        ).grid(row=4, column=1, sticky="w", pady=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Button(buttons, text="Save", command=self._on_save).pack(side="left")
        ttk.Button(buttons, text="Restart assistant", command=self._on_restart).pack(
            side="left", padx=6
        )
        ttk.Button(buttons, text="Close", command=self._on_x).pack(side="left")

        self._message = ttk.Label(
            frame, text="Voice and wake word take effect on the next start.",
            wraplength=520, justify="left",
        )
        self._message.grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Live log").grid(row=7, column=0, sticky="w")
        self._feed = tk.Text(frame, height=14, width=64, wrap="none", state="disabled")
        self._feed.grid(row=8, column=0, columnspan=3, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self._feed.yview)
        scroll.grid(row=8, column=3, sticky="ns")
        self._feed.configure(yscrollcommand=scroll.set)
        frame.rowconfigure(8, weight=1)
        frame.columnconfigure(2, weight=1)

        self._refresh()

    # callbacks (Tk thread)

    def _on_save(self) -> None:
        self._message.configure(text=self._panel.save(voice=self._voice.get(), wake_word=self._wake.get()))

    def _on_restart(self) -> None:
        self._message.configure(text=self._panel.restart())

    def _on_x(self) -> None:
        self._closing.set()

    def _refresh(self) -> None:
        if self._closing.is_set():
            with contextlib.suppress(Exception):
                self._root.destroy()
            return
        snapshot = self._panel.snapshot()
        self._values["mic"].configure(text=str(snapshot.get("mic", "unknown")))
        self._values["listening"].configure(
            text="● yes — she is hearing you" if snapshot.get("listening") else "○ no"
        )
        self._values["summary"].configure(text=str(snapshot.get("summary", "")))
        lines = list(snapshot.get("log", []))  # type: ignore[arg-type]
        if lines != self._shown:
            self._shown = lines
            self._feed.configure(state="normal")
            self._feed.delete("1.0", "end")
            self._feed.insert("end", "\n".join(lines))
            self._feed.see("end")
            self._feed.configure(state="disabled")
        self._root.after(int(_REFRESH_S * 1000), self._refresh)
