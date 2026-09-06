"""Which microphone and speaker she uses — by name, saved, switchable live.

Windows lists every endpoint once per host API (MME, DirectSound, WASAPI,
WDM-KS), MME truncates names to 31 characters, and Bluetooth hands-free
endpoints are named after their driver. `entries()` collapses all of that to
one readable row per device and keeps the best API's index (WASAPI first). A
saved choice is a case-insensitive name fragment ("airpods", "Snowball") or
an index; the empty string and "System default" mean whatever Windows
currently prefers.

`refresh()` makes PortAudio enumerate again so a headset paired after she
started actually appears. It must only run while no stream is open — the
runner calls it between idle cycles.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import sounddevice as sd

DEFAULT = "System default"
_API_RANK = ("wasapi", "mme", "directsound")
# MME/DirectSound aliases for "whatever the default is": not devices
_ALIASES = {
    "microsoft sound mapper - input", "microsoft sound mapper - output",
    "primary sound capture driver", "primary sound driver",
}
_MME_NAME_LIMIT = 31
# Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0 ;(OpenMove by AfterShokz))
_BLUETOOTH = re.compile(r"^(?P<kind>\w[\w ]*?) \(@.*?;\((?P<device>[^()]+)\)\)$")


def pretty(name: str) -> str:
    """A readable device name: 'OpenMove by AfterShokz (hands-free)' for the
    driver-named Bluetooth endpoints, 'Headphones (unnamed)' for empty ones."""
    name = " ".join(str(name).split())
    m = _BLUETOOTH.match(name)
    if m:
        return f"{m.group('device').strip()} ({'hands-free' if 'hands-free' in name.lower() else m.group('kind').lower()})"
    if name.endswith("()"):
        return name[:-2].strip() + " (unnamed)"
    return name


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
    """One row per distinct device for `kind` ("input" | "output"): the
    readable `name`, the `raw` name, and the index of its best host API."""
    devices, apis = _query()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    best: dict[str, tuple[int, int]] = {}  # raw name -> (rank, index)
    truncated: set[str] = set()  # names MME cut at 31 characters (before whitespace cleanup)
    for index, dev in enumerate(devices):
        if int(dev.get(key, 0) or 0) <= 0:
            continue
        original = str(dev.get("name", ""))
        raw = " ".join(original.split())
        if not raw or raw.lower() in _ALIASES:
            continue
        if len(original) == _MME_NAME_LIMIT:
            truncated.add(raw)
        rank = _rank(dev, apis)
        if raw not in best or rank < best[raw][0]:
            best[raw] = (rank, index)
    # an MME-truncated name is the same device as the longer name it prefixes
    names = list(best)
    for raw in names:
        if raw in truncated and any(other != raw and other.startswith(raw) for other in names):
            del best[raw]
    rows = sorted(best.items(), key=lambda kv: (kv[1][0], pretty(kv[0]).lower()))
    return [{"name": pretty(raw), "raw": raw, "index": index} for raw, (_rank_, index) in rows]


def _same_device(a: str, b: str) -> bool:
    """Equal names, or one is MME's 31-character cut of the other."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) == _MME_NAME_LIMIT and long_.startswith(short)


def twins(index: int, kind: str) -> list[int]:
    """The same device through the other host APIs, best API first, `index`
    excluded. WASAPI's shared mode refuses a sample rate the device's mix
    format lacks ("Invalid sample rate"); MME resamples — so a twin is the
    same device, not a fallback."""
    found, apis = _query()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if not 0 <= index < len(found):
        return []
    mine = str(found[index].get("name", ""))
    rows = [
        (_rank(dev, apis), i)
        for i, dev in enumerate(found)
        if i != index and int(dev.get(key, 0) or 0) > 0 and _same_device(str(dev.get("name", "")), mine)
    ]
    return [i for _, i in sorted(rows)]


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
    rows = entries(kind)
    for row in rows:  # an exact picker entry first, then any fragment
        if needle in (str(row["name"]).lower(), str(row["raw"]).lower()):
            return int(row["index"])
    for row in rows:
        if needle in str(row["name"]).lower() or needle in str(row["raw"]).lower():
            return int(row["index"])
    return None


def describe(spec: str, kind: str) -> str:
    """Human name for what `spec` resolves to, honest when it is missing."""
    index = find(spec, kind)
    with contextlib.suppress(Exception):
        if index is None:
            if not is_default(spec):
                return f"{spec.strip()} (not found — using the default)"
            raw = " ".join(str(sd.query_devices(kind=kind)["name"]).split())
            # the default is reported through MME, whose name may be truncated
            full = next((row for row in entries(kind) if str(row["raw"]).startswith(raw)), None)
            return str(full["name"]) if full else pretty(raw)
        return pretty(str(sd.query_devices(index)["name"]))
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
