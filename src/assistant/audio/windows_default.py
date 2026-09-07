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


STATES = {1: "active", 2: "disabled", 4: "not present", 8: "unplugged"}  # DEVICE_STATE_*


def _default_output() -> tuple[str, str]:
    """(endpoint ID, state word) of Windows' current default speaker; ("", "")
    when there is none (not Windows, no audio service, no speaker at all)."""
    if sys.platform != "win32":
        return "", ""
    try:
        ole32 = ctypes.oledll.ole32
    except AttributeError:
        return "", ""
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
        state = ctypes.c_uint32()
        _method(device, 6, POINTER(ctypes.c_uint32))(device, byref(state))
        return text.value or "", STATES.get(int(state.value), f"state {state.value}")
    except Exception:  # noqa: BLE001 — an answer of "" is the whole contract
        return "", ""
    finally:
        with contextlib.suppress(Exception):
            if text.value is not None:
                ole32.CoTaskMemFree(text)
        for obj in (device, enumerator):
            with contextlib.suppress(Exception):
                if obj.value:
                    _method(obj, 2)(obj)  # IUnknown::Release


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("r1", ctypes.c_ushort),
        ("r2", ctypes.c_ushort),
        ("r3", ctypes.c_ushort),
        ("pwszVal", c_void_p),
        ("pad", c_void_p),
    ]


_VT_LPWSTR = 31
_STATEMASK_ALL = 0xF


def render_endpoints() -> list[tuple[str, str, str]]:
    """Every speaker Windows knows, as (friendly name, state word, endpoint
    ID) — the unplugged Bluetooth ones included, which is the point: a
    stream opened on one of those plays into nothing while PortAudio, whose
    picture is frozen at boot, still calls it a device. [] when unknown."""
    if sys.platform != "win32":
        return []
    try:
        ole32 = ctypes.oledll.ole32
    except AttributeError:
        return []
    with contextlib.suppress(Exception):
        ole32.CoInitializeEx(None, 0)
    out: list[tuple[str, str, str]] = []
    enumerator = c_void_p()
    collection = c_void_p()
    key = _PROPERTYKEY(_GUID.of("{A45C254E-DF1C-4EFD-8020-67D146A850E0}"), 14)  # PKEY_Device_FriendlyName
    try:
        ole32.CoCreateInstance(
            byref(_GUID.of("{BCDE0395-E52F-467C-8E3D-C4579291692E}")),
            None,
            _CLSCTX_ALL,
            byref(_GUID.of("{A95664D2-9614-4F35-A746-DE8DB63617E6}")),
            byref(enumerator),
        )
        # IMMDeviceEnumerator::EnumAudioEndpoints(eRender, every state, &collection)
        _method(enumerator, 3, c_int, ctypes.c_uint32, POINTER(c_void_p))(
            enumerator, _E_RENDER, _STATEMASK_ALL, byref(collection)
        )
        count = ctypes.c_uint32()
        _method(collection, 3, POINTER(ctypes.c_uint32))(collection, byref(count))  # GetCount
        for i in range(count.value):
            device = c_void_p()
            text = c_wchar_p()
            store = c_void_p()
            try:
                with contextlib.suppress(Exception):  # one bad endpoint never hides the rest
                    _method(collection, 4, ctypes.c_uint32, POINTER(c_void_p))(collection, i, byref(device))  # Item
                    _method(device, 5, POINTER(c_wchar_p))(device, byref(text))  # GetId
                    state = ctypes.c_uint32()
                    _method(device, 6, POINTER(ctypes.c_uint32))(device, byref(state))  # GetState
                    name = ""
                    with contextlib.suppress(Exception):
                        _method(device, 4, ctypes.c_uint32, POINTER(c_void_p))(device, 0, byref(store))  # OpenPropertyStore(STGM_READ)
                        value = _PROPVARIANT()
                        _method(store, 5, POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))(store, byref(key), byref(value))  # GetValue
                        if value.vt == _VT_LPWSTR and value.pwszVal:
                            name = ctypes.wstring_at(value.pwszVal)
                            ole32.CoTaskMemFree(c_void_p(value.pwszVal))
                    out.append((name, STATES.get(int(state.value), f"state {state.value}"), text.value or ""))
            finally:
                with contextlib.suppress(Exception):
                    if text.value is not None:
                        ole32.CoTaskMemFree(text)
                for obj in (store, device):
                    with contextlib.suppress(Exception):
                        if obj.value:
                            _method(obj, 2)(obj)
    except Exception:  # noqa: BLE001
        return out
    finally:
        for obj in (collection, enumerator):
            with contextlib.suppress(Exception):
                if obj.value:
                    _method(obj, 2)(obj)
    return out


def endpoint_state(name: str) -> str:
    """The state of the speaker Windows calls `name` ("active", "unplugged",
    …), or "" when no such speaker is known — a Bluetooth speaker that walked
    away is "unplugged" while a stream on it keeps playing into nothing."""
    wanted = " ".join((name or "").split()).lower()
    if not wanted:
        return ""
    states = [state for friendly, state, _id in render_endpoints() if " ".join(friendly.split()).lower() == wanted]
    if not states:
        return ""
    return "active" if "active" in states else states[0]


def default_output_id() -> str:
    """The Core Audio endpoint ID of Windows' current default speaker, or ""."""
    return _default_output()[0]


def default_output_state() -> str:
    """"active", "disabled", "not present", "unplugged" — or "" when unknown.
    A Bluetooth speaker that walked away is "unplugged" or "not present"
    while its endpoint still exists and a stream on it plays into nothing."""
    return _default_output()[1]


class SpeakerHealth:
    """Is the speaker the stream was opened on still there? Polled a few
    times a minute while idle: a Bluetooth speaker (the Echo Dot) drops its
    link on its own, Windows marks the endpoint "unplugged", and the open
    stream plays into nothing — she answered a whole conversation that way
    once, and the only tell was that the microphone stopped hearing her."""

    def __init__(
        self,
        *,
        state_of: Callable[[str], str] = endpoint_state,
        every_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state_of = state_of
        self._every = every_s
        self._clock = clock
        self._last_poll = -1e9

    def gone(self, name: str) -> str:
        """The state word when the speaker called `name` is no longer active
        ("unplugged", "not present", …); "" while it is, while nothing is
        known about it, and between polls."""
        now = self._clock()
        if now - self._last_poll < self._every or not name:
            return ""
        self._last_poll = now
        state = self._state_of(name)
        return "" if state in ("", "active") else state


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
