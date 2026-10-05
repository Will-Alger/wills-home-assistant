"""Where did the hub go? Home Assistant runs in a Hyper-V VM on this machine,
and a reboot can hand the VM a new DHCP lease: on 2026-09-14 it came back at
a new address while `.env` still had the old one, and for four days every wake found "the home
isn't answering". The VM's MAC does not change — Hyper-V's start with
00-15-5d — so the ARP table names the candidates, and the one whose `/api/`
accepts our token is the hub. Nothing is written back; the app uses the found
address for the rest of the run and says so, and `.env` is the owner's to fix
(or a DHCP reservation, which is the real fix).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from collections.abc import Callable, Coroutine
from typing import Any

import httpx

HYPERV_MAC_PREFIX = "00-15-5d"
_ARP_ROW = re.compile(r"^\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f]{2}(?:-[0-9a-f]{2}){5})\s", re.IGNORECASE)


def arp_table() -> str:
    """`arp -a`, with no console window (every subprocess here is silent)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            ["arp", "-a"], capture_output=True, text=True, timeout=5, creationflags=flags, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def vm_addresses(arp_text: str, prefix: str = HYPERV_MAC_PREFIX) -> list[str]:
    """IPv4 addresses in an `arp -a` listing whose MAC starts with `prefix`."""
    out: list[str] = []
    for line in arp_text.splitlines():
        match = _ARP_ROW.match(line)
        if match and match.group(2).lower().startswith(prefix.lower()) and match.group(1) not in out:
            out.append(match.group(1))
    return out


async def is_hub(url: str, token: str, *, timeout_s: float = 3.0) -> bool:
    """Home Assistant, and ours: `/api/` answers 200 to our token."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/", headers={"Authorization": f"Bearer {token}"})
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def find_moved_hub(
    current_url: str,
    token: str,
    *,
    arp: Callable[[], str] = arp_table,
    probe: Callable[[str, str], Coroutine[Any, Any, bool]] = is_hub,
    ports: tuple[int, ...] = (80, 8123),
) -> str | None:
    """The hub's new base URL when it answers somewhere other than
    `current_url`, else None. Only VM addresses are tried, on the port the
    current URL uses first."""
    candidates = vm_addresses(arp())
    if not candidates:
        return None
    current = current_url.rstrip("/")
    current_port = int(current.rsplit(":", 1)[1]) if current.count(":") == 2 else 80
    ordered_ports = (current_port, *[p for p in ports if p != current_port])
    urls = [
        f"http://{ip}" if port == 80 else f"http://{ip}:{port}"
        for ip in candidates for port in ordered_ports
    ]
    urls = [u for u in urls if u != current]
    results = await asyncio.gather(*(probe(u, token) for u in urls), return_exceptions=True)
    for url, ok in zip(urls, results, strict=True):
        if ok is True:
            return url
    return None
