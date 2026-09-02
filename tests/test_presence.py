"""Presence: debounced arrivals and departures, voice as proof, boot and
reconnect syncs — all on a fake clock and a fake home."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from assistant.home.fake import FakeHome
from assistant.presence import Presence

LOCAL = datetime.now().astimezone().tzinfo


class Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def at(hour: int, minute: int = 0) -> float:
    return datetime(2026, 9, 2, hour, minute, tzinfo=LOCAL).timestamp()


def test_debounce_flapping_and_transitions(tmp_path: Path) -> None:
    clock = Clock(at(8))
    p = Presence(tmp_path / "presence.json", "person.will", owner="Will", now=clock)
    seen: list = []
    p.on_transition.append(seen.append)
    assert p.state == "unknown" and p.settled()  # unknown counts as home for delivery
    assert "unknown" in p.describe()

    p.observe("home")  # first knowledge, still debounced
    assert p.tick() is None and p.state == "unknown"
    clock.at += 61
    assert p.tick() is None and p.state == "home"  # no story to tell yet
    p.observe("not_home")
    clock.at += 120
    assert p.tick() is None  # two minutes away is not "left"
    p.observe("home")  # GPS flapped back: forget it
    clock.at += 300
    assert p.tick() is None and p.state == "home"
    p.observe("Work")  # any zone that isn't home is away
    clock.at += 301
    left = p.tick()
    assert left is not None and left.kind == "left" and p.state == "away" and p.away_since == clock.at
    assert "away (since" in p.describe()

    clock.at += 3600
    p.observe("home")
    assert p.tick() is None
    clock.at += 61
    arrived = p.tick()
    assert arrived is not None and arrived.kind == "arrived" and abs(arrived.away_for_s - 3661) < 1
    assert not p.settled()  # still parking
    clock.at += 90
    assert p.settled()
    assert [t.kind for t in seen] == ["left", "arrived"]
    assert p.observe("unavailable") is None and p.tick() is None  # a phone that stopped reporting

    again = Presence(tmp_path / "presence.json", "person.will", now=clock)
    assert again.state == "home" and again.since == p.since  # survives a restart


def test_voice_is_proof_of_home(tmp_path: Path) -> None:
    clock = Clock(at(9))
    p = Presence(tmp_path / "presence.json", "person.will", now=clock)
    p.observe("not_home")
    clock.at += 301
    p.tick()
    assert p.state == "away"
    arrived = p.observe("home", source="voice")  # he said the wake word: he is here
    assert arrived is not None and arrived.kind == "arrived" and p.state == "home"
    p.observe("not_home")  # HA still thinks he is out; only a sustained report flips it
    assert p.tick() is None
    clock.at += 301
    assert p.tick().kind == "left"


async def test_sync_reads_the_entity_and_tolerates_missing(tmp_path: Path) -> None:
    clock = Clock(at(8))
    home = FakeHome()
    home.extra_entities.append(
        {
            "entity_id": "person.will", "name": "Will", "state": "not_home", "domain": "person",
            "attributes": {}, "last_changed": "2026-09-02T07:00:00-04:00",
        }
    )
    p = Presence(tmp_path / "p.json", "person.will", now=clock)
    await p.sync(home, boot=True)
    assert p.state == "away"
    assert p.since == datetime.fromisoformat("2026-09-02T07:00:00-04:00").timestamp()
    assert p.tick() is None  # boot: no transition, nothing to announce

    home.extra_entities[-1]["state"] = "home"
    await p.sync(home)  # a reconnect: the change goes through the normal debounce
    assert p.state == "away" and p.tick() is None
    clock.at += 61
    assert p.tick().kind == "arrived"

    nobody = Presence(tmp_path / "q.json", "person.nobody", now=clock)
    await nobody.sync(home, boot=True)
    assert nobody.state == "unknown"  # unknown entity: keep what we had
