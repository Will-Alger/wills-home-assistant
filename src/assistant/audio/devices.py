"""Which microphone and speaker she uses — by name, saved, switchable live.

Windows lists every endpoint once per host API (MME, DirectSound, WASAPI,
WDM-KS), so "AirPods" shows up four times; `entries()` collapses that to one
row per name and keeps the best API's index (WASAPI first). A saved choice is
a case-insensitive name fragment ("airpods", "Snowball") or an index; the
empty string and "System default" mean whatever Windows currently prefers.

`refresh()` makes PortAudio enumerate again so a headset paired after she
started actually appears. It must only run while no stream is open — the
runner calls it between idle cycles.
"""

from __future__ import annotations

import contextlib
from typing import Any

import sounddevice as sd

DEFAULT = "System default"
_API_RANK = ("wasapi", "mme", "directsound")


def _query() -> tuple[list[dict[str, Any]], list[str]]:
    try:
        devices = [dict(d) for d in sd.query_devices()]
        apis = [str(a.get("name", "")) for a in sd.query_hostapis()]
    except Exception:  # noqa: BLE001 — no PortAudio here: nothing to offer
        return [], []
    return devices, apis


def _rank(dev: dict[str, Any], apis: list[str]) -> int:
    api = int(dev.get("hostapi", -1) or 0)
    name = apis[api].lower() if 0 <= api < len(apis) else ""
    for rank, word in enumerate(_API_RANK):
        if word in name:
            return rank
    return len(_API_RANK)


def entries(kind: str) -> list[dict[str, Any]]:
    """One row per distinct device name for `kind` ("input" | "output"),
    each with the index of its best host API."""
    devices, apis = _query()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    best: dict[str, tuple[int, int]] = {}
    for index, dev in enumerate(devices):
        if int(dev.get(key, 0) or 0) <= 0:
            continue
        name = " ".join(str(dev.get("name", "")).split())
        if not name:
            continue
        rank = _rank(dev, apis)
        if name not in best or rank < best[name][0]:
            best[name] = (rank, index)
    rows = sorted(best.items(), key=lambda kv: (kv[1][0], kv[0].lower()))
    return [{"name": name, "index": index} for name, (_rank_, index) in rows]


def names(kind: str) -> list[str]:
    """What a picker shows: the default first, then every distinct device."""
    return [DEFAULT, *(row["name"] for row in entries(kind))]


def is_default(spec: str) -> bool:
    return not (spec or "").strip() or spec.strip().lower() == DEFAULT.lower()


def find(spec: str, kind: str) -> int | None:
    """The device index a saved spec means right now: None for the default
    (or for a name that is not plugged in — callers say so and use the default)."""
    if is_default(spec):
        return None
    spec = spec.strip()
    if spec.isdigit():
        return int(spec)
    needle = spec.lower()
    for row in entries(kind):
        if needle in str(row["name"]).lower():
            return int(row["index"])
    return None


def describe(spec: str, kind: str) -> str:
    """Human name for what `spec` resolves to, honest when it is missing."""
    index = find(spec, kind)
    with contextlib.suppress(Exception):
        if index is None:
            if not is_default(spec):
                return f"{spec.strip()} (not found — using the default)"
            return str(sd.query_devices(kind=kind)["name"])
        return str(sd.query_devices(index)["name"])
    return spec.strip() or DEFAULT


def refresh() -> bool:
    """Re-enumerate the devices (a headset just paired). Only while no
    stream is open."""
    try:
        sd._terminate()  # sounddevice has no public re-scan
        sd._initialize()
    except Exception:  # noqa: BLE001
        return False
    return True
