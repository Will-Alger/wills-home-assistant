"""Event reactivity: watches fire deterministically on Home Assistant state
changes and become announcements — with a scripted websocket, no network."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pytest

from assistant.announce import Announcer
from assistant.engines.realtime_engine import RealtimeEngine
from assistant.events import EventWatcher, Watch, WatchStore, in_window, matches
from assistant.home.fake import FakeHome

LOCAL = datetime.now().astimezone().tzinfo


def at(hour: int, minute: int = 0, weekday_date: tuple[int, int, int] = (2026, 9, 2)) -> float:
    return datetime(*weekday_date, hour, minute, tzinfo=LOCAL).timestamp()  # 2026-09-02 is a Wednesday


def event(entity: str, old: str | None, new: str | None) -> str:
    return json.dumps(
        {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": entity,
                    "old_state": {"state": old} if old is not None else None,
                    "new_state": {"state": new} if new is not None else None,
                },
            },
        }
    )


def test_matching_rules() -> None:
    w = Watch(id=1, entity_id="binary_sensor.front_door", message="the {entity} is {state}", to_state="on")
    assert matches(w, "binary_sensor.front_door", "off", "on", at(12))
    assert not matches(w, "binary_sensor.front_door", "on", "on", at(12))  # attribute-only update
    assert not matches(w, "binary_sensor.front_door", "on", "off", at(12))  # wrong direction
    assert not matches(w, "binary_sensor.back_door", "off", "on", at(12))
    fragment = Watch(id=2, entity_id="front_door", message="door", once=False)
    assert matches(fragment, "binary_sensor.front_door", "off", "on", at(12))  # any change

    night = Watch(id=3, entity_id="front_door", message="late", to_state="on", after="23:00", before="06:00")
    assert matches(night, "binary_sensor.front_door", "off", "on", at(23, 30))
    assert matches(night, "binary_sensor.front_door", "off", "on", at(2))
    assert not matches(night, "binary_sensor.front_door", "off", "on", at(12))
    assert in_window("", "", datetime.now().astimezone())

    weekdays = Watch(id=4, entity_id="lamp", message="x", days=["mon", "tue", "wed", "thu", "fri"])
    assert matches(weekdays, "light.lamp", "off", "on", at(9))  # Wednesday
    assert not matches(weekdays, "light.lamp", "off", "on", at(9, 0, (2026, 9, 5)))  # Saturday


def test_store_persists_fires_once_and_lists(tmp_path: Path) -> None:
    store = WatchStore(tmp_path / "watches.json", now=lambda: at(12))
    w = store.add(entity_id="binary_sensor.front_door", message="the {entity} just went {state}", to_state="on")
    assert store.describe()[0]["when"] == "turns on"
    with pytest.raises(ValueError, match="23:00"):
        store.add(entity_id="x", message="y", after="nope")

    hits = store.evaluate("binary_sensor.front_door", "off", "on")
    assert [text for _, text in hits] == ["the front door just went on"]
    assert not store.active()  # once → retired
    assert store.evaluate("binary_sensor.front_door", "off", "on") == []

    again = WatchStore(tmp_path / "watches.json")
    assert again.all()[0].fired == 1 and not again.all()[0].active
    keep = again.add(entity_id="light.lamp", message="lamp", once=False)
    assert again.cancel(keep.id) is not None and again.cancel(keep.id) is None
    assert w.id == 1 and keep.id == 2


async def test_watcher_authenticates_subscribes_and_announces(tmp_path: Path) -> None:
    sent: list[dict] = []
    incoming: asyncio.Queue = asyncio.Queue()

    class FakeWs:
        async def recv(self):
            return await incoming.get()

        async def send(self, raw):
            sent.append(json.loads(raw))
            if sent[-1]["type"] == "auth":
                incoming.put_nowait(json.dumps({"type": "auth_ok", "ha_version": "2026.8.3"}))
            if sent[-1]["type"] == "subscribe_events":
                incoming.put_nowait(json.dumps({"id": 1, "type": "result", "success": True}))
                incoming.put_nowait(event("binary_sensor.front_door", "off", "on"))
                incoming.put_nowait(event("light.kitchen", "off", "on"))

    @asynccontextmanager
    async def connector(url: str):
        assert url == "ws://192.168.1.50/api/websocket"
        incoming.put_nowait(json.dumps({"type": "auth_required"}))
        yield FakeWs()

    store = WatchStore(tmp_path / "watches.json")
    store.add(entity_id="front_door", message="Heads up: the {entity} is {state}.", to_state="on", priority="urgent")
    announcer = Announcer(tmp_path / "a.json", quiet_hours="00:00-23:59")
    watcher = EventWatcher("http://192.168.1.50", "tok", store, announcer, connector=connector)
    task = asyncio.create_task(watcher.run())
    async with asyncio.timeout(5):
        while not announcer.pending():
            await asyncio.sleep(0.05)
    watcher.stop()
    task.cancel()
    assert sent[0] == {"type": "auth", "access_token": "tok"}
    assert sent[1]["type"] == "subscribe_events" and sent[1]["event_type"] == "state_changed"
    (item,) = announcer.pending()
    assert item.text == "Heads up: the front door is on." and item.kind == "watch"
    assert announcer.due()  # urgent: spoken even in quiet hours
    assert watcher.events_seen == 2


async def test_watcher_feeds_hooks_and_subscribes_custom_events(tmp_path: Path) -> None:
    sent: list[dict] = []
    incoming: asyncio.Queue = asyncio.Queue()
    states: list[tuple] = []
    taps: list[dict] = []
    connects: list[int] = []

    class FakeWs:
        async def recv(self):
            return await incoming.get()

        async def send(self, raw):
            sent.append(json.loads(raw))
            msg = sent[-1]
            if msg["type"] == "auth":
                incoming.put_nowait(json.dumps({"type": "auth_ok"}))
            if msg["type"] == "subscribe_events":
                incoming.put_nowait(json.dumps({"id": msg["id"], "type": "result", "success": True}))
                if msg["id"] == 1:
                    incoming.put_nowait(event("person.owner", "not_home", "home"))
                else:
                    incoming.put_nowait(
                        json.dumps(
                            {
                                "type": "event",
                                "event": {
                                    "event_type": msg["event_type"],
                                    "data": {"action": "alexa:read:1", "reply_text": ""},
                                },
                            }
                        )
                    )

    @asynccontextmanager
    async def connector(url: str):
        incoming.put_nowait(json.dumps({"type": "auth_required"}))
        yield FakeWs()

    async def on_connect() -> None:
        connects.append(1)

    watcher = EventWatcher(
        "http://192.168.1.50", "tok", WatchStore(tmp_path / "w.json"), Announcer(tmp_path / "a.json"),
        connector=connector,
        on_state=lambda entity, old, new: states.append((entity, old, new)),
        on_event={"mobile_app_notification_action": taps.append},
        on_connect=on_connect,
    )
    task = asyncio.create_task(watcher.run())
    async with asyncio.timeout(5):
        while not (states and taps):
            await asyncio.sleep(0.05)
    watcher.stop()
    task.cancel()
    assert sent[1]["event_type"] == "state_changed"
    assert sent[2] == {"id": 2, "type": "subscribe_events", "event_type": "mobile_app_notification_action"}
    assert states == [("person.owner", "not_home", "home")]
    assert taps[0]["action"] == "alexa:read:1" and connects == [1]
    assert watcher.events_seen == 2


def test_watch_tools_by_voice(tmp_path: Path) -> None:
    store = WatchStore(tmp_path / "watches.json")
    engine = RealtimeEngine(api_key="k", model="m", voice="v", home=FakeHome(), owner="Will", watches=store)
    text, is_error = engine._execute_watch_tool(
        "watch_for",
        {"entity_id": "binary_sensor.front_door", "message": "the door opened", "to_state": "on", "after": "23:00", "before": "06:00"},
    )
    assert not is_error and "watch 1 set" in text and "after 23:00" in text
    text, _ = engine._execute_watch_tool("list_watches", {})
    assert json.loads(text)[0]["when"] == "turns on after 23:00 before 06:00"
    text, is_error = engine._execute_watch_tool("cancel_watch", {"id": 1})
    assert not is_error and "cancelled" in text
    text, is_error = engine._execute_watch_tool("cancel_watch", {"id": 1})
    assert is_error
    text, is_error = engine._execute_watch_tool("watch_for", {"entity_id": "", "message": ""})
    assert is_error and "needs an entity" in text
