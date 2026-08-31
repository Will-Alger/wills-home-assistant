"""Long-term memory: a small local JSON store behind remember/forget tools.

Two kinds of items:
- "preference": standing instructions ("from now on when I say movie time...")
  — rendered into the assistant's instructions every session, applied without
  being asked.
- "fact": things to recall on demand ("remember the wifi guest password is…").

Privacy posture (docs/FEATURES.md): owner-directed, deliberately stored items
only — never ambient conversation. The file is local and gitignored.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# preference/fact: stored via the voice tools (deliberate, user-directed).
# lesson/observation/episode: written by the session-end reflection pass —
# lessons are operational recipes injected into future instructions;
# observations await the user's consent before becoming preferences;
# episodes are the searchable conversation journal.
KINDS = ("preference", "fact", "lesson", "observation", "episode")


@dataclass(frozen=True)
class MemoryItem:
    id: int
    kind: str
    text: str
    created: str


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._next_id = 1
        self._items: list[MemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._next_id = data.get("next_id", 1)
        self._items = [MemoryItem(**item) for item in data.get("items", [])]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "items": [asdict(item) for item in self._items]}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, kind: str, text: str) -> MemoryItem:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        text = text.strip()
        if not text:
            raise ValueError("empty memory text")
        item = MemoryItem(
            id=self._next_id,
            kind=kind,
            text=text,
            created=time.strftime("%Y-%m-%d"),
        )
        self._next_id += 1
        self._items.append(item)
        self._save()
        return item

    def items(self, kind: str | None = None) -> list[MemoryItem]:
        return [item for item in self._items if kind is None or item.kind == kind]

    def forget(self, item_id: int) -> bool:
        before = len(self._items)
        self._items = [item for item in self._items if item.id != item_id]
        if len(self._items) != before:
            self._save()
            return True
        return False

    def preferences_text(self) -> str:
        prefs = self.items("preference")
        return "\n".join(f"- [id {p.id}] {p.text}" for p in prefs) or "(none stored yet)"

    def lessons_text(self, limit: int = 15) -> str:
        lessons = self.items("lesson")[-limit:]
        return "\n".join(f"- [id {x.id}] {x.text}" for x in lessons) or "(none yet)"

    def observations_text(self, limit: int = 5) -> str:
        obs = self.items("observation")[-limit:]
        return "\n".join(f"- [id {x.id}] {x.text}" for x in obs) or "(none)"
