"""MemoryStore tests — local file, no network."""

from __future__ import annotations

import pytest

from assistant.memory import MemoryStore


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
