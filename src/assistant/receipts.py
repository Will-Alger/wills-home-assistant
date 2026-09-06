"""Action receipts: what she actually did, and what it would take to put it back.

"Done" is a claim, and it was being made even when a bulb never answered.
Every home mutation now leaves a receipt — the entities it aimed at, what
each one looked like BEFORE, what was asked of it, and whether it answered —
so partial success can be said out loud as partial success ("2 of 3 lights
changed; the Bedroom Lamp did not respond") and so "undo that" has something
concrete to restore.

Only some actions have a reliable inverse. Lights do: their before-state is
a full description of the bulb. A media command, a calendar delete or a
thermostat nudge do not, and a receipt that says so lets her refuse in one
honest sentence instead of pretending. Undo reaches only the LAST action —
"undo that" means the thing she just did, not the last thing that happened
to be reversible — and only while it is fresh, because a before-state from
an hour ago describes a room that has moved on. Receipts stay in memory for
the session; the last 20 are kept in data/receipts.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KEEP = 20  # what data/receipts.json holds; the session keeps more in memory
SESSION_KEEP = 200
UNDO_TTL_S = 1800.0  # the same half hour the working context trusts a reference for


@dataclass
class EntityOutcome:
    """One entity's share of an action: where it was, what it was asked for,
    and whether it answered."""

    entity_id: str
    name: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    requested: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


@dataclass
class ActionReceipt:
    tool: str
    # The action in past tense ("paused the Living Room Speakers"), so a
    # refusal reads as a sentence: "the last thing I did was <note>".
    note: str = ""
    outcomes: list[EntityOutcome] = field(default_factory=list)
    reversible: bool = False
    at: float = 0.0  # stamped by the book, so tests can drive their own clock

    @property
    def targets(self) -> list[str]:
        return [o.entity_id for o in self.outcomes]

    @property
    def changed(self) -> list[EntityOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[EntityOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def restorable(self) -> list[EntityOutcome]:
        """What an undo could actually put back: it changed, and we know what
        it looked like first."""
        return [o for o in self.changed if o.before]

    def summary(self, noun: str = "device") -> str:
        """Said the way it happened — "Done" only when all of it worked."""
        failed = self.failed
        names = ", ".join(o.name or o.entity_id for o in failed)
        if not failed:
            return f"Done: {len(self.outcomes)} {noun}(s) updated."
        if not self.changed:
            return f"Nothing changed — {names} did not respond."
        return (
            f"{len(self.changed)} of {len(self.outcomes)} {noun}s changed; "
            f"{names} did not respond."
        )


class ReceiptBook:
    """The ledger. `path=None` keeps it in memory only (tests, the REPL)."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        now: Callable[[], float] = time.time,
        undo_ttl_s: float = UNDO_TTL_S,
    ) -> None:
        self._path = path
        self._now = now
        self._ttl = undo_ttl_s
        self._receipts: list[ActionReceipt] = []
        self._load()

    def record(self, receipt: ActionReceipt) -> ActionReceipt:
        if not receipt.at:
            receipt.at = self._now()
        self._receipts.append(receipt)
        del self._receipts[:-SESSION_KEEP]
        self._save()
        return receipt

    def last(self) -> ActionReceipt | None:
        return self._receipts[-1] if self._receipts else None

    def recent(self, limit: int = KEEP) -> list[ActionReceipt]:
        return self._receipts[-limit:]

    def for_undo(self) -> tuple[ActionReceipt | None, str]:
        """The receipt "undo that" acts on, or None and a plain sentence
        saying why it stands. Always the LAST action: undoing something older
        than the thing she just did would surprise the room."""
        receipt = self.last()
        if receipt is None:
            return None, "There is nothing recent to undo."
        if not receipt.reversible:
            what = receipt.note or f"a {receipt.tool} call"
            return None, (
                f"The last thing I did was {what}, and that can't be undone. "
                "Undo covers lights for now."
            )
        if not receipt.restorable():
            return None, "Nothing actually changed there, so there is nothing to put back."
        if self._now() - receipt.at > self._ttl:
            return None, (
                "That light change was more than half an hour ago — too long to put "
                "back blind. Say what you'd like them set to instead."
            )
        return receipt, ""

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            rows = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        known = set(ActionReceipt.__dataclass_fields__)
        outcome_fields = set(EntityOutcome.__dataclass_fields__)
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not row.get("tool"):
                continue
            values = {k: v for k, v in row.items() if k in known}
            values["outcomes"] = [
                EntityOutcome(**{k: v for k, v in o.items() if k in outcome_fields})
                for o in row.get("outcomes", [])
                if isinstance(o, dict) and o.get("entity_id")
            ]
            self._receipts.append(ActionReceipt(**values))

    def _save(self) -> None:
        if self._path is None:
            return
        with contextlib.suppress(OSError):  # a ledger must never break a command
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([asdict(r) for r in self._receipts[-KEEP:]], indent=1),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
