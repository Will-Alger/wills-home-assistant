"""The Settings panel — a small desktop window she opens and closes by voice.

It shows what the running session knows about itself (microphone, whether she
is listening and how loud the room is while she does, the push-to-talk hotkey,
voice, wake word, a status summary and a live log feed) and holds the two
settings worth changing without a keyboard — her voice and her wake word —
plus a restart button, because both only take effect on a fresh session. It is
deliberately NOT a microphone switch: nothing in the panel turns listening on
or off, and the hotkey is shown, not edited (it lives in `.env`, beside the
wake word's threshold).

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
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from assistant.config import wake_phrase
from assistant.recording import SCRIPT, describe

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
_LEVEL_REFRESH_S = 0.1  # the level bar has its own tick: a meter, not a status line
_METER_W, _METER_H = 120, 10  # pixels
_METER_FILL = "#3f9a52"  # she is hearing you
_METER_TROUGH = "#e9e9e9"


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

    # panel keyword -> settings field it overrides
    FIELDS: ClassVar[dict[str, str]] = {
        "voice": "realtime_voice",
        "wake_model": "wake_model",
        "microphone": "audio_input_device",
        "speaker": "audio_output_device",
    }

    @property
    def voice(self) -> str:
        return self._read().get("realtime_voice", "")

    @property
    def wake_model(self) -> str:
        return self._read().get("wake_model", "")

    @property
    def microphone(self) -> str:
        return self._read().get("audio_input_device", "")

    @property
    def speaker(self) -> str:
        return self._read().get("audio_output_device", "")

    def set(
        self, *, voice: str = "", wake_model: str = "", microphone: str = "", speaker: str = ""
    ) -> dict[str, str]:
        data = self._read()
        for key, value in (
            ("voice", voice), ("wake_model", wake_model), ("microphone", microphone), ("speaker", speaker)
        ):
            if value:
                data[self.FIELDS[key]] = value
        self._write(data)
        return data

    def clear(self, field: str) -> None:
        data = self._read()
        if data.pop(self.FIELDS.get(field, field), None) is not None:
            self._write(data)

    def apply(self, settings: Any) -> list[str]:
        """Fold the overrides over freshly loaded settings; returns what
        actually changed, for the boot log."""
        changed = []
        saved = self._read()
        for attr in self.FIELDS.values():
            value = saved.get(attr, "")
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
        on_audio_change: Callable[[str, str], None] | None = None,
        on_refresh_devices: Callable[[], None] | None = None,
        devices: Any | None = None,
        recorder: Any | None = None,
        player: Callable[[bytes, int], None] | None = None,
        stopper: Callable[[], None] | None = None,
    ) -> None:
        self._status = status
        self._overrides = overrides
        self._restart = restart
        self._models_dir = Path(models_dir) if models_dir else None
        self._view_factory = view_factory or _tk_view
        self._log = log
        self._on_audio_change = on_audio_change  # the app applies it to the next conversation
        self._on_refresh_devices = on_refresh_devices  # the app re-scans between idle cycles
        if devices is None:
            from assistant.audio import devices as audio_devices

            devices = audio_devices
        self._devices = devices
        self._recorder = recorder  # session recordings (recording.py), opt-in
        self._player = player or _play_pcm
        self._stopper = stopper or _stop_pcm
        self._script_pos = 0  # the read-aloud test script: steps reached so far
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

    def microphone_choices(self) -> list[str]:
        """'System default' plus every distinct input device plugged in now."""
        try:
            return list(self._devices.names("input"))
        except Exception:  # noqa: BLE001
            return [self._devices.DEFAULT]

    def speaker_choices(self) -> list[str]:
        try:
            return list(self._devices.names("output"))
        except Exception:  # noqa: BLE001
            return [self._devices.DEFAULT]

    def snapshot(self) -> dict[str, object]:
        data = self._status.snapshot() if self._status is not None else {}
        data["saved_voice"] = self._overrides.voice
        data["saved_wake_word"] = self._overrides.wake_model
        data["saved_microphone"] = self._overrides.microphone
        data["saved_speaker"] = self._overrides.speaker
        data["recording"] = self.recording
        return data

    def meter(self) -> tuple[bool, float]:
        """(is she listening, how loud the room is — 0 flat to 1 full). The
        two numbers the level bar redraws from, read many times a second and
        so kept out of the full snapshot."""
        if self._status is None:
            return False, 0.0
        try:
            return bool(self._status.listening), float(self._status.level)
        except Exception:  # noqa: BLE001 — a bar is never worth a crash
            return False, 0.0

    def note(self, text: str) -> None:
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.note(text)
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log(text)

    # ── the controls ──────────────────────────────────────────────────────

    def save(
        self, *, voice: str = "", wake_word: str = "", microphone: str = "", speaker: str = ""
    ) -> str:
        """Store a new voice and/or wake word (both apply on the next start,
        which is what the restart button is for) and/or the microphone and
        speaker (saved, and used from the next conversation on — no restart)."""
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
        audio = self._resolve_audio(microphone, speaker)
        if isinstance(audio, str):
            return audio
        mic_spec, speaker_spec = audio
        if not voice and not model and mic_spec is None and speaker_spec is None:
            return "nothing to save"
        parts: list[str] = []
        if voice or model:
            self._overrides.set(voice=voice, wake_model=model)
            parts.append(
                ", ".join(
                    p for p in (f"voice {voice}" if voice else "", f"wake word {wake_phrase(model)}" if model else "")
                    if p
                )
                + " — restart to apply"
            )
        if mic_spec is not None or speaker_spec is not None:
            for key, spec in (("microphone", mic_spec), ("speaker", speaker_spec)):
                if spec is None:
                    continue
                if self._devices.is_default(spec):
                    self._overrides.clear(key)
                else:
                    self._overrides.set(**{key: spec})
            if self._on_audio_change is not None:
                with contextlib.suppress(Exception):
                    self._on_audio_change(self._overrides.microphone, self._overrides.speaker)
            parts.append(
                ", ".join(
                    p
                    for p in (
                        f"microphone {mic_spec}" if mic_spec is not None else "",
                        f"speaker {speaker_spec}" if speaker_spec is not None else "",
                    )
                    if p
                )
                + " — used from the next conversation"
            )
        message = "saved " + "; ".join(parts)
        self.note(message)
        return message

    def _resolve_audio(self, microphone: str, speaker: str) -> tuple[str | None, str | None] | str:
        """Turn what he typed or said into saved specs: an exact picker entry,
        'System default', or a name fragment that matches a device now."""
        out: list[str | None] = []
        for kind, wanted in (("input", microphone.strip()), ("output", speaker.strip())):
            if not wanted:
                out.append(None)
                continue
            if self._devices.is_default(wanted):
                out.append(self._devices.DEFAULT)
                continue
            if self._devices.find(wanted, kind) is None:
                available = ", ".join(self._devices.names(kind)[1:]) or "none"
                what = "microphone" if kind == "input" else "speaker"
                return f"no {what} matches '{wanted}' — plugged in right now: {available}"
            out.append(wanted)
        return out[0], out[1]

    def refresh_devices(self) -> str:
        """A headset just paired: ask the app to re-scan between idle cycles."""
        if self._on_refresh_devices is None:
            return "re-scanning isn't wired up in this session"
        with contextlib.suppress(Exception):
            self._on_refresh_devices()
        self.note("re-scanning audio devices")
        return "re-scanning audio devices — the lists update in a few seconds"

    def restart(self) -> str:
        if self._restart is None:
            return "restarting isn't wired up in this session"
        self.note("restart requested from the settings panel")
        self._restart()
        return "restarting — she's back in about fifteen seconds"

    # ── session recordings ────────────────────────────────────────────────

    @property
    def recording(self) -> bool:
        """Armed: every conversation from the next one is recorded."""
        if self._recorder is None:
            return False
        try:
            return bool(self._recorder.armed)
        except Exception:  # noqa: BLE001
            return False

    def set_recording(self, on: bool) -> str:
        if self._recorder is None:
            return "recording isn't wired up in this session"
        self._recorder.arm(bool(on))
        if on:
            text = (
                "recording sessions — both sides of the audio and every turn-taking "
                "decision, in data/recordings, from the next conversation"
            )
        else:
            text = "not recording sessions"
            self._script_pos = 0
        self.note(text)
        return text

    def recordings(self) -> list[dict[str, Any]]:
        """Finished recordings, newest first — summary rows (see recording.py)."""
        if self._recorder is None:
            return []
        try:
            return list(self._recorder.list())
        except Exception:  # noqa: BLE001 — a listing is never worth a crash
            return []

    def recording_note(self, name: str, text: str) -> str:
        if self._recorder is None or not self._recorder.note(name, text):
            return f"no recording called {name!r}"
        return f"note saved on {name}"

    def delete_recording(self, name: str) -> str:
        if self._recorder is None or not self._recorder.delete(name):
            return f"couldn't delete {name!r}"
        self.note(f"deleted recording {name}")
        return f"deleted {name}"

    def open_recording(self, name: str) -> str:
        folder = self._recorder.folder(name) if self._recorder is not None else None
        if folder is None:
            return f"no recording called {name!r}"
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]  # Explorer, on Windows
        except (OSError, AttributeError) as err:
            return f"couldn't open the folder: {err}"
        return f"opened {folder}"

    def play_recording(self, name: str, side: str = "mic") -> str:
        """Play one side through the default output. She can hear it too:
        her own name in the mic track will wake her unless he uses headphones."""
        audio = self._recorder.audio(name, side) if self._recorder is not None else None
        if audio is None:
            return f"no {side} audio for {name!r}"
        pcm, rate = audio
        try:
            self._player(pcm, rate)
        except Exception as err:  # noqa: BLE001
            return f"couldn't play it: {err}"
        return f"playing the {side} side of {name} ({len(pcm) / (2 * rate):.0f} s) — headphones, or she hears it too"

    def stop_playback(self) -> str:
        with contextlib.suppress(Exception):
            self._stopper()
        return "stopped"

    # ── the read-aloud test script ────────────────────────────────────────

    def script(self) -> tuple[tuple[str, str], ...]:
        return SCRIPT

    def script_step(self) -> int:
        """Steps reached so far (0 before the first click)."""
        return self._script_pos

    def next_step(self) -> tuple[int, str, str] | None:
        """The owner reached the next step: stamp it into the recording
        (arming the recorder if it was off) and return (n, say, expect);
        None past the end."""
        if self._script_pos >= len(SCRIPT):
            return None
        if self._recorder is not None and not self.recording:
            self._recorder.arm(True)
            self.note("recording sessions — the test script turned it on")
        self._script_pos += 1
        say, expect = SCRIPT[self._script_pos - 1]
        if self._recorder is not None:
            with contextlib.suppress(Exception):
                self._recorder.step(self._script_pos, say)
        self.note(f"test script, step {self._script_pos}: {say}")
        return self._script_pos, say, expect

    def restart_script(self) -> None:
        self._script_pos = 0


def _play_pcm(pcm: bytes, rate: int) -> None:
    """The panel's play button: its own PortAudio stream, on the default
    output, so it never touches the session's speaker."""
    import numpy as np
    import sounddevice as sd

    sd.stop()
    sd.play(np.frombuffer(pcm, dtype=np.int16), rate)


def _stop_pcm() -> None:
    import sounddevice as sd

    sd.stop()


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
            (
                ("mic", "Microphone in use"), ("speaker", "Speaker in use"),
                ("listening", "Listening"), ("hotkey", "Push to talk"),
                ("summary", "Status"),
            )
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            if key == "listening":
                # the flag, and beside it the live level: the bar moves while
                # she is hearing you and reads flat the moment she isn't
                cell = ttk.Frame(frame)
                cell.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
                value = ttk.Label(cell, text="…")
                value.pack(side="left")
                self._meter = tk.Canvas(
                    cell, width=_METER_W, height=_METER_H, bg=_METER_TROUGH,
                    highlightthickness=1, highlightbackground="#b0b0b0",
                )
                self._meter.pack(side="left", padx=8)
                self._meter_fill = self._meter.create_rectangle(
                    0, 0, 0, _METER_H, fill=_METER_FILL, width=0
                )
            else:
                value = ttk.Label(frame, text="…", wraplength=380, justify="left")
                value.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
            self._values[key] = value

        snapshot = self._panel.snapshot()
        # show what is SAVED when it differs from what is running: he changed
        # it and hasn't restarted yet
        self._voice = tk.StringVar(
            value=str(snapshot.get("saved_voice") or snapshot.get("voice", ""))
        )
        ttk.Label(frame, text="Voice").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame, textvariable=self._voice, values=self._panel.voice_choices(),
            state="readonly", width=18,
        ).grid(row=5, column=1, sticky="w", pady=2)

        self._wake_choices = self._panel.wake_choices()
        current = str(snapshot.get("saved_wake_word") or snapshot.get("wake_word", ""))
        self._wake = tk.StringVar(
            value=next((k for k, v in self._wake_choices.items() if v == current), current)
        )
        ttk.Label(frame, text="Wake word").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame, textvariable=self._wake, values=list(self._wake_choices),
            state="readonly", width=18,
        ).grid(row=6, column=1, sticky="w", pady=2)

        # the audio devices: saved by name, used from the next conversation
        default = self._panel._devices.DEFAULT
        self._mic_choice = tk.StringVar(value=str(snapshot.get("saved_microphone") or default))
        ttk.Label(frame, text="Microphone").grid(row=7, column=0, sticky="w", pady=2)
        self._mic_box = ttk.Combobox(
            frame, textvariable=self._mic_choice, values=self._panel.microphone_choices(),
            state="readonly", width=42,
        )
        self._mic_box.grid(row=7, column=1, columnspan=2, sticky="w", pady=2)
        self._speaker_choice = tk.StringVar(value=str(snapshot.get("saved_speaker") or default))
        ttk.Label(frame, text="Speaker").grid(row=8, column=0, sticky="w", pady=2)
        self._speaker_box = ttk.Combobox(
            frame, textvariable=self._speaker_choice, values=self._panel.speaker_choices(),
            state="readonly", width=42,
        )
        self._speaker_box.grid(row=8, column=1, columnspan=2, sticky="w", pady=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Button(buttons, text="Save", command=self._on_save).pack(side="left")
        ttk.Button(buttons, text="Refresh devices", command=self._on_refresh_devices).pack(
            side="left", padx=6
        )
        ttk.Button(buttons, text="Restart assistant", command=self._on_restart).pack(side="left")
        ttk.Button(buttons, text="Close", command=self._on_x).pack(side="left", padx=6)

        # debugging her turn-taking: record sessions, browse them, read the script
        debug = ttk.Frame(frame)
        debug.grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._record = tk.BooleanVar(value=self._panel.recording)
        ttk.Checkbutton(
            debug, text="Record sessions", variable=self._record, command=self._on_record
        ).pack(side="left")
        ttk.Button(debug, text="Recordings…", command=self._open_recordings).pack(side="left", padx=6)
        ttk.Button(debug, text="Test script…", command=self._open_script).pack(side="left")
        self._rec_win: Any = None
        self._script_win: Any = None

        self._message = ttk.Label(
            frame,
            text="Voice and wake word take effect on the next start; microphone and "
            "speaker on the next conversation. Just paired something? Refresh devices.",
            wraplength=520, justify="left",
        )
        self._message.grid(row=11, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Live log").grid(row=12, column=0, sticky="w")
        self._feed = tk.Text(frame, height=14, width=64, wrap="none", state="disabled")
        self._feed.grid(row=13, column=0, columnspan=3, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self._feed.yview)
        scroll.grid(row=13, column=3, sticky="ns")
        self._feed.configure(yscrollcommand=scroll.set)
        frame.rowconfigure(13, weight=1)
        frame.columnconfigure(2, weight=1)

        self._refresh()
        self._tick_level()

    # recordings (Tk thread)

    def _on_record(self) -> None:
        self._message.configure(text=self._panel.set_recording(bool(self._record.get())))

    def _window_alive(self, win: Any) -> bool:
        try:
            return win is not None and bool(win.winfo_exists())
        except Exception:  # noqa: BLE001
            return False

    def _open_recordings(self) -> None:
        tk = self._tk
        from tkinter import ttk

        if self._window_alive(self._rec_win):
            self._rec_win.lift()
            self._reload_recordings()
            return
        win = tk.Toplevel(self._root)
        win.title("Alexa — Recordings")
        win.geometry("760x380")
        self._rec_win = win
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="Each recording: mic.wav (what she heard, from 3 s before the wake), "
            "speaker.wav (what she played), events.jsonl (every decision, with the numbers) "
            "and summary.json. Playback goes through the default speaker — she hears it too.",
            wraplength=720, justify="left",
        ).pack(anchor="w")
        rows = ttk.Frame(body)
        rows.pack(fill="both", expand=True, pady=6)
        self._rec_list = tk.Listbox(rows, height=10, activestyle="dotbox")
        self._rec_list.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(rows, orient="vertical", command=self._rec_list.yview)
        bar.pack(side="right", fill="y")
        self._rec_list.configure(yscrollcommand=bar.set)
        buttons = ttk.Frame(body)
        buttons.pack(anchor="w")
        for label, fn in (
            ("Open folder", lambda: self._rec_do(self._panel.open_recording)),
            ("Play mic", lambda: self._rec_do(lambda n: self._panel.play_recording(n, "mic"))),
            ("Play speaker", lambda: self._rec_do(lambda n: self._panel.play_recording(n, "speaker"))),
            ("Stop", lambda: self._rec_say(self._panel.stop_playback())),
            ("Note…", self._rec_note),
            ("Delete", self._rec_delete),
            ("Refresh", self._reload_recordings),
        ):
            ttk.Button(buttons, text=label, command=fn).pack(side="left", padx=(0, 6))
        self._rec_message = ttk.Label(body, text="", wraplength=720, justify="left")
        self._rec_message.pack(anchor="w", pady=(6, 0))
        self._reload_recordings()

    def _reload_recordings(self) -> None:
        if not self._window_alive(self._rec_win):
            return
        self._rec_rows = self._panel.recordings()
        self._rec_list.delete(0, "end")
        for row in self._rec_rows:
            self._rec_list.insert("end", describe(row))
        if not self._rec_rows:
            self._rec_list.insert("end", "(no recordings yet — tick Record sessions, then talk to her)")

    def _rec_selected(self) -> str:
        try:
            index = self._rec_list.curselection()[0]
            return str(self._rec_rows[index].get("name", ""))
        except (IndexError, AttributeError):
            return ""

    def _rec_say(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self._rec_message.configure(text=text)

    def _rec_do(self, fn: Callable[[str], str]) -> None:
        name = self._rec_selected()
        self._rec_say(fn(name) if name else "pick a recording first")

    def _rec_note(self) -> None:
        from tkinter import simpledialog

        name = self._rec_selected()
        if not name:
            self._rec_say("pick a recording first")
            return
        current = next((str(r.get("note", "")) for r in self._rec_rows if r.get("name") == name), "")
        text = simpledialog.askstring(
            "Note", "What happened? (e.g. 'cut me off after \"repaint the\"')",
            parent=self._rec_win, initialvalue=current,
        )
        if text is None:
            return
        self._rec_say(self._panel.recording_note(name, text))
        self._reload_recordings()

    def _rec_delete(self) -> None:
        from tkinter import messagebox

        name = self._rec_selected()
        if not name:
            self._rec_say("pick a recording first")
            return
        if messagebox.askyesno("Delete", f"Delete recording {name}?", parent=self._rec_win):
            self._rec_say(self._panel.delete_recording(name))
            self._reload_recordings()

    # the read-aloud test script (Tk thread)

    def _open_script(self) -> None:
        tk = self._tk
        from tkinter import ttk

        if self._window_alive(self._script_win):
            self._script_win.lift()
            return
        win = tk.Toplevel(self._root)
        win.title("Alexa — Test script")
        win.geometry("640x340")
        self._script_win = win
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        self._script_head = ttk.Label(body, text="", font=("TkDefaultFont", 11, "bold"))
        self._script_head.pack(anchor="w")
        ttk.Label(body, text="Say").pack(anchor="w", pady=(8, 0))
        self._script_say = tk.Text(body, height=4, wrap="word", state="disabled")
        self._script_say.pack(fill="x")
        ttk.Label(body, text="What should happen").pack(anchor="w", pady=(8, 0))
        self._script_expect = tk.Text(body, height=4, wrap="word", state="disabled")
        self._script_expect.pack(fill="x")
        buttons = ttk.Frame(body)
        buttons.pack(anchor="w", pady=(10, 0))
        ttk.Button(buttons, text="Next step", command=self._script_next).pack(side="left")
        ttk.Button(buttons, text="Start over", command=self._script_restart).pack(side="left", padx=6)
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left")
        self._render_step()

    def _script_set(self, widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _render_step(self, done: bool = False) -> None:
        if not self._window_alive(self._script_win):
            return
        n = self._panel.script_step()
        total = len(self._panel.script())
        if done:
            self._script_head.configure(text=f"Done — all {total} steps are stamped into the recording.")
            self._script_set(self._script_say, "Untick Record sessions when you are finished, or leave it on.")
            self._script_set(self._script_expect, "Open Recordings… to add a note about what went wrong, and play back the mic side.")
            return
        if n == 0:
            self._script_head.configure(text=f"{total} steps. Click Next step as you reach each one.")
            self._script_set(
                self._script_say,
                "Every click is stamped into the recording's timeline, so the events can be read "
                "against the script. Recording turns itself on at the first click.",
            )
            self._script_set(self._script_expect, "Read the step, do it, watch what she does, click Next step.")
            return
        say, expect = self._panel.script()[n - 1]
        self._script_head.configure(text=f"Step {n} of {total}")
        self._script_set(self._script_say, say)
        self._script_set(self._script_expect, expect)

    def _script_next(self) -> None:
        result = self._panel.next_step()
        with contextlib.suppress(Exception):
            self._record.set(self._panel.recording)
        self._render_step(done=result is None)

    def _script_restart(self) -> None:
        self._panel.restart_script()
        self._render_step()

    # callbacks (Tk thread)

    def _on_save(self) -> None:
        self._message.configure(
            text=self._panel.save(
                voice=self._voice.get(), wake_word=self._wake.get(),
                microphone=self._mic_choice.get(), speaker=self._speaker_choice.get(),
            )
        )

    def _on_refresh_devices(self) -> None:
        self._message.configure(text=self._panel.refresh_devices())
        # the app re-scans between idle cycles; re-read the lists once it has
        self._root.after(3000, self._reload_device_lists)

    def _reload_device_lists(self) -> None:
        with contextlib.suppress(Exception):
            self._mic_box.configure(values=self._panel.microphone_choices())
            self._speaker_box.configure(values=self._panel.speaker_choices())

    def _on_restart(self) -> None:
        self._message.configure(text=self._panel.restart())

    def _on_x(self) -> None:
        self._closing.set()

    def _tick_level(self) -> None:
        """The bar's own tick, ten times a second — the frames arrive at
        twelve and a half, and a meter that moved at the panel's 0.4 s pace
        would not be a meter. Two numbers in, one rectangle out."""
        if self._closing.is_set():
            return  # _refresh owns the teardown
        listening, level = self._panel.meter()
        with contextlib.suppress(Exception):
            self._meter.coords(
                self._meter_fill, 0, 0, int(_METER_W * level) if listening else 0, _METER_H
            )
            self._root.after(int(_LEVEL_REFRESH_S * 1000), self._tick_level)

    def _refresh(self) -> None:
        if self._closing.is_set():
            with contextlib.suppress(Exception):
                self._root.destroy()
            return
        snapshot = self._panel.snapshot()
        self._values["mic"].configure(text=str(snapshot.get("mic", "unknown")))
        self._values["speaker"].configure(text=str(snapshot.get("speaker") or "system default"))
        self._values["listening"].configure(
            text="● yes — she is hearing you" if snapshot.get("listening") else "○ no"
        )
        hotkey = str(snapshot.get("hotkey") or "")
        self._values["hotkey"].configure(text=f"hold {hotkey}" if hotkey else "off")
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
