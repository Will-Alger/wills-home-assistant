"""Which speaker Windows calls the default right now — asked of Windows,
not of PortAudio.

PortAudio takes its picture of the machine once, at initialisation, so
`sd.query_devices(kind="output")` keeps answering with the speaker that was
the default when she booted. When he switches Windows from the headphones to
the Echo Dot, the open stream stays on the headphones and nothing in
PortAudio can say so short of tearing it down and asking again. The Core
Audio endpoint enumerator can: `IMMDeviceEnumerator::GetDefaultAudioEndpoint`
answers instantly, and the endpoint ID it returns changes exactly when the
default does. Pure ctypes over ole32 — no COM package to install, and every
failure is an empty string, never an exception.

`DefaultOutputWatch` is the runner's use of it: poll a few times a second at
most, remember the ID the streams were opened against, and say when it has
moved (only meaningful while the speaker setting is "System default").
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import time
from collections.abc import Callable
from ctypes import POINTER, byref, c_int, c_void_p, c_wchar_p, cast
from typing import Any

_E_RENDER = 0  # EDataFlow.eRender
_E_MULTIMEDIA = 1  # ERole.eMultimedia: what an app playing audio gets by default
_CLSCTX_ALL = 23


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def of(cls, text: str) -> _GUID:
        guid = cls()
        ctypes.oledll.ole32.CLSIDFromString(text, byref(guid))
        return guid


def _method(obj: c_void_p, index: int, *argtypes: Any) -> Any:
    """Slot `index` of a COM object's vtable as a callable taking `this`."""
    vtable = cast(cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtable[index])


def default_output_id() -> str:
    """The Core Audio endpoint ID of Windows' current default speaker, or ""
    (not Windows, no audio service, no speaker at all)."""
    if sys.platform != "win32":
        return ""
    try:
        ole32 = ctypes.oledll.ole32
    except AttributeError:
        return ""
    with contextlib.suppress(Exception):
        ole32.CoInitializeEx(None, 0)  # multithreaded; a repeat call is harmless
    enumerator = c_void_p()
    device = c_void_p()
    text = c_wchar_p()
    try:
        ole32.CoCreateInstance(
            byref(_GUID.of("{BCDE0395-E52F-467C-8E3D-C4579291692E}")),  # MMDeviceEnumerator
            None,
            _CLSCTX_ALL,
            byref(_GUID.of("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),  # IMMDeviceEnumerator
            byref(enumerator),
        )
        # IMMDeviceEnumerator: 0-2 IUnknown, 3 EnumAudioEndpoints, 4 GetDefaultAudioEndpoint
        _method(enumerator, 4, c_int, c_int, POINTER(c_void_p))(enumerator, _E_RENDER, _E_MULTIMEDIA, byref(device))
        # IMMDevice: 0-2 IUnknown, 3 Activate, 4 OpenPropertyStore, 5 GetId, 6 GetState
        _method(device, 5, POINTER(c_wchar_p))(device, byref(text))
        return text.value or ""
    except Exception:  # noqa: BLE001 — an answer of "" is the whole contract
        return ""
    finally:
        with contextlib.suppress(Exception):
            if text.value is not None:
                ole32.CoTaskMemFree(text)
        for obj in (device, enumerator):
            with contextlib.suppress(Exception):
                if obj.value:
                    _method(obj, 2)(obj)  # IUnknown::Release


class DefaultOutputWatch:
    """Has Windows' default speaker moved since the streams were opened?"""

    def __init__(
        self,
        *,
        reader: Callable[[], str] = default_output_id,
        every_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read = reader
        self._every = every_s
        self._clock = clock
        self._opened_on = ""
        self._last_poll = -1e9

    def opened(self) -> None:
        """The streams were just (re)opened: this is the default they follow."""
        self._opened_on = self._read()
        self._last_poll = self._clock()

    def changed(self) -> bool:
        """True once, when the default is no longer the one the streams were
        opened against (polled at most every `every_s`)."""
        now = self._clock()
        if now - self._last_poll < self._every:
            return False
        self._last_poll = now
        current = self._read()
        if not current or not self._opened_on or current == self._opened_on:
            return False
        self._opened_on = current  # said once; the reopen re-reads it anyway
        return True


if __name__ == "__main__":  # pragma: no cover — a quick look at the machine
    print(repr(default_output_id()))
