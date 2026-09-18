"""Finding a hub whose VM came back on a new address."""

from __future__ import annotations

from assistant.home.discover import find_moved_hub, vm_addresses

ARP = """
Interface: 192.168.1.157 --- 0x12
  Internet Address      Physical Address      Type
  192.168.1.1           a4-2b-8c-11-22-33     dynamic
  192.168.1.116         00-15-5d-01-9d-02     dynamic
  192.168.1.120         00-15-5D-01-9d-09     static
  192.168.1.255         ff-ff-ff-ff-ff-ff     static
  224.0.0.22            01-00-5e-00-00-16     static
"""


def test_only_hyper_v_addresses_are_candidates() -> None:
    assert vm_addresses(ARP) == ["192.168.1.116", "192.168.1.120"]
    assert vm_addresses("") == []


async def test_the_hub_is_the_vm_that_takes_our_token() -> None:
    asked: list[str] = []

    async def probe(url: str, token: str) -> bool:
        asked.append(url)
        return url == "http://192.168.1.116" and token == "t"

    found = await find_moved_hub("http://192.168.1.114", "t", arp=lambda: ARP, probe=probe)
    assert found == "http://192.168.1.116"
    assert "http://192.168.1.114" not in asked and asked[0] == "http://192.168.1.116"  # the current port first
    assert await find_moved_hub("http://192.168.1.116", "t", arp=lambda: ARP, probe=probe) is None  # already there
    assert await find_moved_hub("http://192.168.1.114", "t", arp=lambda: "", probe=probe) is None


async def test_the_client_and_the_watcher_move_with_the_hub(tmp_path) -> None:
    from assistant.announce import Announcer
    from assistant.events import EventWatcher, WatchStore
    from assistant.home.client import HomeAssistantClient

    client = HomeAssistantClient("http://192.168.1.114", "t")
    try:
        assert client.base_url == "http://192.168.1.114"
        client.rebase("http://192.168.1.116/")
        assert client.base_url == "http://192.168.1.116" and client._http.headers["Authorization"] == "Bearer t"
    finally:
        await client.close()
    watcher = EventWatcher("http://192.168.1.114", "t", WatchStore(tmp_path / "w.json"), Announcer(tmp_path / "a.json"))
    watcher.rebase("http://192.168.1.116")
    assert watcher._url == "ws://192.168.1.116"


async def test_the_current_port_is_kept_when_the_hub_moves() -> None:
    async def probe(url: str, token: str) -> bool:
        return url == "http://192.168.1.120:8123"

    assert await find_moved_hub("http://192.168.1.114:8123", "t", arp=lambda: ARP, probe=probe) == "http://192.168.1.120:8123"
