"""One contract for what a tool reports.

Every capability — a lighting change, a calendar write, a timer, a memory, a
background job — used to answer with a free-text sentence, and the model had
to read control flow out of prose nobody had promised it ("Done", "nothing is
playing", "not deleted: restate..."). A ToolOutcome says the same five things
every time, so the model acts on `status` and only SPEAKS the words:

    status   what happened, from a fixed vocabulary (STATUSES)
    summary  one sentence she can say out loud
    details  the structured facts behind it (names, ids, rows)
    reversible  whether undo_last could put it back
    follow_up   a short instruction for the model — never spoken

`is_error` is deliberately NOT one of them. It is the old flag, kept on the
dataclass for the callers that still speak in (text, is_error) pairs — the
batch agent, the scheduler, the journal — because "that can't be undone" is
an answer she says plainly, not a tool failure, and the five-field contract
has no room for that distinction. It is never rendered to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# success            it happened, all of it
# partial            some of it happened; the rest did not
# pending            it is underway and will finish after this call returns
# unavailable        it could not be done (a dead bulb, a missing key, a refusal)
# needs_clarification  it cannot be done until the owner answers something
STATUSES = ("success", "partial", "pending", "unavailable", "needs_clarification")


class NeedsClarification(Exception):
    """A tool cannot act until the owner answers. The message is the question
    to ask him, word for word — never a paraphrase of a failure."""


@dataclass(slots=True)
class ToolOutcome:
    status: str = "success"
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    reversible: bool = False
    follow_up: str = ""
    is_error: bool = False  # the legacy pair's flag; see the module docstring

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown tool status {self.status!r}; expected one of {STATUSES}")

    def payload(self) -> dict[str, Any]:
        """What the model sees, always these five keys in this order."""
        return {
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
            "reversible": self.reversible,
            "follow_up": self.follow_up,
        }

    def as_pair(self) -> tuple[str, bool]:
        """The old (result_text, is_error) shape, for callers not yet moved."""
        return self.summary, self.is_error


def ok(summary: str, **kw: Any) -> ToolOutcome:
    return ToolOutcome("success", summary, **kw)


def unavailable(summary: str, **kw: Any) -> ToolOutcome:
    """Could not be done. `is_error=False` for the refusals she simply says."""
    kw.setdefault("is_error", True)
    return ToolOutcome("unavailable", summary, **kw)


def adapt(text: str, is_error: bool, *, status: str = "", **kw: Any) -> ToolOutcome:
    """Wrap a tool that still answers in free text. The words are untouched —
    they become the summary — and the flag picks the status unless the caller
    knows better (a background job is `pending`, not `success`)."""
    if not status:
        status = "unavailable" if is_error else "success"
    return ToolOutcome(status, text, is_error=is_error, **kw)
