"""MemoryStore tests — local file, no network."""

from __future__ import annotations

import json

import pytest

from assistant.memory import (
    INJECT_WHOLE_BELOW,
    LESSON_MIN_CONFIDENCE,
    RECENT_OTHERS,
    MemoryStore,
    infer_subject,
    subjects_in_play,
)


def test_roundtrip_and_persistence(tmp_path) -> None:
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    pref = store.add("preference", "movie time means warm dim living room")
    fact = store.add("fact", "the guest wifi password is on the fridge")

    reloaded = MemoryStore(path)
    assert [i.text for i in reloaded.items("preference")] == [pref.text]
    assert [i.text for i in reloaded.items("fact")] == [fact.text]
    assert pref.text in reloaded.preferences_text()
    assert fact.text not in reloaded.preferences_text()  # facts stay out of the prompt


def test_forget_and_ids_stay_unique(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    a = store.add("fact", "first")
    b = store.add("fact", "second")
    assert store.forget(a.id) is True
    assert store.forget(a.id) is False  # already gone
    c = store.add("fact", "third")
    assert c.id != b.id and c.id != a.id  # ids never reused


def test_rejects_bad_input(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    with pytest.raises(ValueError):
        store.add("vibe", "not a valid kind")
    with pytest.raises(ValueError):
        store.add("fact", "   ")


def test_empty_store_renders_placeholder(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    assert store.preferences_text() == "(none stored yet)"


# ── evidence and scope ────────────────────────────────────────────────────


def test_old_rows_load_with_defaults(tmp_path) -> None:
    """A store written before subjects and evidence existed still opens, and
    nothing in it is lost."""
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "next_id": 3,
                "items": [
                    {"id": 1, "kind": "preference", "text": "movie time means the lamps go warm and dim", "created": "2026-08-01"},
                    {"id": 2, "kind": "fact", "text": "wifi is on the fridge", "created": "2026-08-02"},
                ],
            }
        ),
        encoding="utf-8",
    )
    store = MemoryStore(path)
    assert [i.text for i in store.items()] == ["movie time means the lamps go warm and dim", "wifi is on the fridge"]
    old = store.get(1)
    assert old.subject == "" and old.source == "voice" and old.confidence == 1.0
    assert old.supersedes is None
    assert old.scope == "lights"  # read off the text, so it still gets retrieved
    assert store.add("fact", "third").id == 3  # ids carry on where they left off


def test_replacement_is_one_write_and_keeps_the_old_version(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    old = store.add("preference", "the living room at 3000 kelvin", subject="lights")

    saves = []
    original = store._save
    store._save = lambda: (saves.append(1), original())[1]  # type: ignore[method-assign]
    new = store.replace(old.id, "the living room at 2700 kelvin")
    del store._save

    assert len(saves) == 1  # atomic: the new value and the retirement together
    assert new.supersedes == old.id and new.subject == "lights"
    # One answer, no contradiction — and the old wording is still on disk.
    reloaded = MemoryStore(tmp_path / "memory.json")
    assert [i.text for i in reloaded.items("preference")] == ["the living room at 2700 kelvin"]
    assert reloaded.get(old.id) is None
    assert len(reloaded.items(include_superseded=True)) == 2
    assert "3000" not in reloaded.preferences_text()


def test_interrupted_replacement_never_loses_the_old_value(tmp_path) -> None:
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    old = store.add("preference", "the living room at 3000 kelvin", subject="lights")

    def die() -> None:
        raise OSError("disk went away mid-write")

    store._save = die  # type: ignore[method-assign]
    with pytest.raises(OSError):
        store.replace(old.id, "the living room at 2700 kelvin")
    del store._save

    # In memory and on disk, the store still holds exactly the old value —
    # never a preference that has been forgotten but not yet re-stored.
    assert [i.text for i in store.items("preference")] == [old.text]
    assert [i.text for i in MemoryStore(path).items("preference")] == [old.text]


def test_forgetting_a_correction_does_not_resurrect_the_old_wording(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    old = store.add("preference", "the living room at 3000 kelvin", subject="lights")
    new = store.replace(old.id, "the living room at 2700 kelvin")

    assert store.forget(new.id) is True
    assert store.items(include_superseded=True) == []  # the whole chain goes
    assert store.forget(new.id) is False


def test_supersede_needs_a_real_id(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    with pytest.raises(ValueError):
        store.add("preference", "warmer", subject="lights", supersedes=99)
    with pytest.raises(ValueError):
        store.replace(99, "warmer")


def test_house_defaults_are_separate_from_personal(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    store.add("preference", "I like the living room at 2700", subject="lights")
    store.add("house", "lights go off at midnight", subject="house")
    assert "2700" in store.preferences_text() and "2700" not in store.house_text()
    assert "midnight" in store.house_text() and "midnight" not in store.preferences_text()


def test_a_large_store_injects_only_what_is_in_play(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    for i in range(INJECT_WHOLE_BELOW + 6):
        store.add("preference", f"calendar rule {i}", subject="calendar")
    wanted = store.add("preference", "the living room at 2700 kelvin", subject="lights")

    rendered = store.preferences_text(subjects_in_play(tools=["set_lights"]))
    assert "2700" in rendered  # the subject in play always comes along
    assert len(rendered.splitlines()) == 1 + RECENT_OTHERS  # plus the newest few
    assert "calendar rule 0" not in rendered

    # Ask about the subject directly and the whole of it is there.
    assert [i.id for i in store.items("preference", subject="lights")] == [wanted.id]
    # A small store is still injected entire.
    small = MemoryStore(tmp_path / "small.json")
    for i in range(3):
        small.add("preference", f"rule {i}", subject="music")
    assert len(small.preferences_text(["lights"]).splitlines()) == 3


def test_lessons_below_the_bar_are_not_injected(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    store.add("lesson", "a hunch", source="reflection", confidence=0.4)
    solid = store.add("lesson", "the player entity is X", source="reflection", confidence=0.7)
    assert "a hunch" not in store.lessons_text()
    assert solid.text in store.lessons_text()

    raised = store.reinforce(store.items("lesson")[0].id)
    assert raised.confidence >= LESSON_MIN_CONFIDENCE and raised.last_verified
    assert "a hunch" in store.lessons_text()


def test_subjects_come_from_the_tools_and_entities_in_play() -> None:
    assert subjects_in_play(tools=["set_lights"]) == {"will", "lights"}
    assert "music" in subjects_in_play(entities=["media_player.living_room"])
    assert subjects_in_play() == {"will"}  # the speaker is always in play
    assert infer_subject("keep the volume down after ten") == "music"
    assert infer_subject("the spare key is under the mat") == ""
