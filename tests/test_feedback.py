"""Feedback items, their tags and statuses, and the needs they add up to."""

from __future__ import annotations

import json

from assistant.feedback import FeedbackStore


def test_an_item_is_kept_with_its_tags_sessions_and_status(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.json", now=lambda: 100.0)
    item = store.add("she cut me off after the volume", tags=["Cut off", "cut-off", "too_eager"], sessions=[240, "240", "x"],
                     source="voice", unit="desktop")
    assert item is not None
    assert item.tags == ["cut-off", "too-eager"] and item.sessions == [240] and item.status == "new"
    assert store.add("   ") is None
    again = FeedbackStore(tmp_path / "feedback.json")
    assert [i.text for i in again.items()] == ["she cut me off after the volume"]
    assert again.items(session=240) and not again.items(session=1)
    data = json.loads((tmp_path / "feedback.json").read_text(encoding="utf-8"))
    assert data["next_item"] == 2


def test_statuses_and_tags_change_and_bad_values_are_ignored(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "f.json")
    item = store.add("too long an answer", tags=["style"])
    assert item is not None
    updated = store.update(item.id, status="fixed", tags=["style", "too-slow"])
    assert updated is not None and updated.status == "fixed" and updated.tags == ["style", "too-slow"]
    assert store.update(item.id, status="bogus").status == "fixed"
    assert store.update(999, status="new") is None
    assert store.tag_counts() == {}  # fixed items drop out of the open counts
    assert store.tag_counts(open_only=False) == {"style": 1, "too-slow": 1}


def test_needs_group_items_and_promote_to_a_task(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "f.json")
    a = store.add("wait until I'm done before acknowledging", tags=["too-eager"])
    b = store.add("she talked over me again", tags=["too-eager"])
    c = store.add("play music faster", tags=["music"])
    assert a and b and c
    need = store.add_need("Wait until he is done before acknowledging", tags=["too-eager"], items=[a.id, b.id])
    assert need is not None and need.status == "open"
    assert [i.id for i in store.items(need=need.id)] == [b.id, a.id]
    assert all(i.status == "triaged" for i in store.items(need=need.id))
    assert store.items()[0].id == c.id and store.items()[0].need is None
    promoted = store.update_need(need.id, status="planned", task=42)
    assert promoted is not None and promoted.task == 42 and promoted.status == "planned"
    assert store.open_needs_text() == "Wait until he is done before acknowledging (planned)"
    snap = store.snapshot()
    assert len(snap["items"]) == 3 and len(snap["needs"]) == 1 and "cut-off" in snap["tags"]
