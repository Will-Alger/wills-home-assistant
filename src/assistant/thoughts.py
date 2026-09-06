"""Thinker jobs: the deeper mind belongs to the current topic.

A `think` question used to be fire-and-forget — no id, no status, no way to
stop it, and no check that the world still matched the question by the time
Opus answered. "Help me think through buying a bike" followed a minute later
by "actually, assume I wait a year" produced TWO answers, the stale one
first, both urgent enough to bypass quiet hours.

So every think is a job with an id, a one-line topic, a status and two
timestamps. A new think on the same topic SUPERSEDES the running one: the
old answer is never spoken, and she is told to restart with the changed
assumption rather than wait. The job that replaced it remembers what
changed, so when its answer lands she opens with a short bridge ("With the
extra year in mind…"). An answer whose question is older than half an hour
and whose topic the room has left is dropped to the journal instead.

The last 20 jobs live in data/thoughts.json; `list_thoughts` reads them back
and `cancel_thought` stops one mid-flight.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

KEEP = 20  # what data/thoughts.json holds
STALE_S = 1800.0  # half an hour: past that a premise describes a room that moved on
STATUSES = ("running", "done", "superseded", "cancelled")

# Framing words carry no topic: "help me think through buying a bike" and
# "should I buy a bike" are one subject wearing different clothes.
_STOPWORDS = frozenset(
    {
        "about", "actually", "all", "also", "and", "any", "are", "assume", "assuming",
        "been", "being", "but", "can", "could", "did", "does", "doing", "done", "for",
        "from", "get", "give", "going", "had", "has", "have", "help", "her", "here",
        "him", "his", "how", "instead", "into", "its", "just", "know", "let", "lets",
        "like", "make", "mean", "might", "more", "much", "need", "not", "now", "one",
        "only", "our", "out", "over", "please", "really", "say", "see", "should", "some",
        "suppose", "sure", "take", "tell", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "thing", "things", "think", "thinking", "this", "those",
        "through", "too", "use", "want", "was", "well", "what", "when", "where", "whether",
        "which", "while", "who", "why", "will", "with", "wonder", "would", "you", "your",
    }
)
_WORD = re.compile(r"[a-z0-9']+")
# The lead-in she would strip to hear the actual subject.
_TOPIC_LEAD = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    r"(?:help\s+me\s+)?(?:think(?:ing)?\s+(?:through|about|over)\s+)?"
    r"(?:i\s+)?(?:want\s+to\s+|need\s+to\s+|wonder(?:ing)?\s+(?:if|whether)\s+)?"
    r"(?:should\s+i\s+|do\s+you\s+think\s+)?",
    re.IGNORECASE,
)
# "Actually, assume I wait a year" — the premise moved under a running job.
_ASSUMPTION = re.compile(
    r"\b(?:actually|instead|okay|ok|alright|but|hmm|wait)\b[,.:\s]*"
    r"(?:(?:let's|lets|now|then)\s+)?(?:assume|suppose|say|imagine|pretend)\b"
    r"|^[,.:\s]*(?:(?:let's|lets|now)\s+)?(?:assume|suppose)\b"
    r"|\bwhat if (?:i|we|it|they|he|she)\b",
    re.IGNORECASE,
)
_LEAD_MARKER = re.compile(
    r"^(?:actually|instead|okay|ok|alright|but|hmm|wait)\b[,.:\s]*", re.IGNORECASE
)
_OVERLAP = 0.4  # share of the shorter question's subject words that must match
_MIN_SHARED = 2


def _stem(word: str) -> str:
    """Crude on purpose: "buying" and "buy", "bikes" and "bike" are one word
    to a topic check, and nothing here deserves a real stemmer. Plain "s"
    only — stripping "es" would turn "bikes" into "bik" and match nothing."""
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def subject_words(text: str) -> set[str]:
    """What a question is actually about, once the framing is thrown away."""
    words = _WORD.findall(str(text).lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}


def same_topic(a: str, b: str) -> bool:
    """Two questions about one subject — strict, because getting this wrong
    supersedes a job he still wants. Measured against the SHORTER of the two,
    so "a bike" still matches the long question that contains it, but a
    single shared word is never enough: "which thermostat should I buy" and
    "should I buy a bike" share only the buying."""
    left, right = subject_words(a), subject_words(b)
    if not left or not right:
        return False
    shared = left & right
    if len(shared) < min(_MIN_SHARED, len(left), len(right)):
        return False
    return len(shared) / min(len(left), len(right)) >= _OVERLAP


def moved_on(current: str, *about: str) -> bool:
    """Has the room left this subject entirely? Deliberately looser than
    `same_topic` and pointing the other way: dropping an answer he asked for
    is the worse mistake, so one word still in common keeps it."""
    subject: set[str] = set()
    for text in about:
        subject |= subject_words(text)
    here = subject_words(current)
    if not subject or not here:
        return False
    return not (subject & here)


def assumption_change(said: str) -> str:
    """The clause that moved the premise, or "" — what the bridge names."""
    text = " ".join(str(said).split())
    match = _ASSUMPTION.search(text)
    if not match:
        return ""
    clause = _LEAD_MARKER.sub("", text[match.start() :].strip(" ,.:;"))
    return clause.strip(" ,.:;")[:120]


def topic_of(question: str) -> str:
    """One line: what this think is about — enough to compare and to say."""
    text = " ".join(str(question).split()).strip(" ?.!")
    stripped = _TOPIC_LEAD.sub("", text, count=1).strip()
    return " ".join((stripped or text).split()[:8])[:80]


@dataclass
class Thought:
    id: int
    question: str
    topic: str = ""
    status: str = "running"
    started: float = 0.0
    finished: float = 0.0
    answer: str = ""
    error: str = ""
    # Which conversation asked. An answer is urgent only while that one is
    # still open; afterwards it is an ordinary announcement.
    conversation: int = 0
    restarted_from: int = 0  # it replaced that job: its answer needs a bridge
    superseded_by: int = 0
    changed: str = ""  # the assumption that moved, named in the bridge

    @property
    def running(self) -> bool:
        return self.status == "running"

    def spoken(self, now: float) -> str:
        age = max(0.0, now - self.started)
        when = "just now" if age < 90 else f"{round(age / 60)} minutes ago"
        line = f"{self.id}: '{self.topic or self.question[:60]}' — {self.status}, asked {when}"
        if self.superseded_by:
            line += f" (replaced by thought {self.superseded_by})"
        return line


def announcement(thought: Thought, owner: str = "the owner") -> str:
    """The EVENT text she says in her own words. After a restart the bridge
    is not decoration — it is how he knows which question was answered."""
    question = thought.question[:80]
    if thought.error:
        return f"I couldn't finish thinking about '{question}': {thought.error}"
    if thought.restarted_from:
        changed = thought.changed or "the assumption he changed"
        return (
            f"Your deeper reasoning on '{question}' — you started this one over after "
            f"{owner} changed the premise ({changed}), so OPEN WITH A SHORT BRIDGE naming "
            f'the change ("With the extra year in mind…") and then give the answer: '
            f"{thought.answer}"
        )
    return f"Your deeper reasoning on '{question}': {thought.answer}"


class ThoughtBook:
    """The ledger of think jobs. `path=None` keeps it in memory (tests)."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        now: Callable[[], float] = time.time,
        stale_s: float = STALE_S,
    ) -> None:
        self._path = path
        self._now = now
        self._stale = stale_s
        self._thoughts: list[Thought] = []
        self._next_id = 1
        self._load()

    # ── the job's life ─────────────────────────────────────────────────────

    def start(
        self,
        question: str,
        *,
        topic: str = "",
        conversation: int = 0,
        said: str = "",
    ) -> tuple[Thought, list[Thought]]:
        """Open a job, superseding any running one on the same subject.

        `said` is the owner's last words: "actually, assume I wait a year"
        makes this a revision even when the new wording shares little — but
        only when ONE job is running, since that is the only case where the
        phrase can only mean this one.
        """
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("a thought needs a question")
        topic = " ".join(str(topic).split())[:80] or topic_of(question)
        changed = assumption_change(said)
        running = self.running()
        revision = changed if len(running) == 1 else ""
        replaced: list[Thought] = []
        for job in running:
            if not (revision or same_topic(job.question, question) or same_topic(job.topic, topic)):
                continue
            job.status = "superseded"
            job.finished = self._now()
            job.superseded_by = self._next_id
            replaced.append(job)
        thought = Thought(
            id=self._next_id,
            question=question,
            topic=topic,
            started=self._now(),
            conversation=int(conversation),
            restarted_from=replaced[-1].id if replaced else 0,
            changed=changed if replaced else "",
        )
        self._next_id += 1
        self._thoughts.append(thought)
        del self._thoughts[:-KEEP]
        self._save()
        return thought, replaced

    def finish(self, thought_id: int, answer: str) -> Thought | None:
        """Record the answer. None means the job stopped being ours while it
        ran — superseded or cancelled — and its answer is never spoken."""
        thought = self.get(thought_id)
        if thought is None or not thought.running:
            return None
        thought.answer = " ".join(str(answer).split())
        thought.status = "done"
        thought.finished = self._now()
        self._save()
        return thought

    def fail(self, thought_id: int, error: str) -> Thought | None:
        """The reasoning broke. He still gets told — that is an answer too."""
        thought = self.get(thought_id)
        if thought is None or not thought.running:
            return None
        thought.error = " ".join(str(error).split())[:160]
        thought.status = "done"
        thought.finished = self._now()
        self._save()
        return thought

    def cancel(self, thought_id: Any) -> Thought | None:
        thought = self.get(thought_id)
        if thought is None or not thought.running:
            return None
        thought.status = "cancelled"
        thought.finished = self._now()
        self._save()
        return thought

    # ── reading back ───────────────────────────────────────────────────────

    def get(self, thought_id: Any) -> Thought | None:
        try:
            wanted = int(thought_id)
        except (TypeError, ValueError):
            return None
        return next((t for t in self._thoughts if t.id == wanted), None)

    def running(self) -> list[Thought]:
        return [t for t in self._thoughts if t.running]

    def recent(self, limit: int = KEEP) -> list[Thought]:
        return self._thoughts[-limit:]

    def describe(self, limit: int = 8) -> str:
        rows = self.recent(limit)
        if not rows:
            return "you have not thought anything over yet"
        now = self._now()
        return "\n".join(t.spoken(now) for t in reversed(rows))

    # ── delivery ───────────────────────────────────────────────────────────

    def deliverable(self, thought: Thought, *, current_topic: str = "") -> tuple[bool, str]:
        """Should this answer be spoken? (speak, why-not). A premise older
        than half an hour is dropped only once the room has actually moved
        on — a slow answer to a question still on the table is still wanted."""
        if thought.status != "done":
            return False, f"the job is {thought.status}"
        age = max(0.0, self._now() - thought.started)
        if age <= self._stale:
            return True, ""
        topic = " ".join(str(current_topic).split())
        if not topic or not moved_on(topic, thought.topic, thought.question):
            return True, ""
        return False, (
            f"asked {round(age / 60)} minutes ago about '{thought.topic}'; "
            f"the conversation has moved on to '{topic[:60]}'"
        )

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        known = set(Thought.__dataclass_fields__)
        for row in data.get("thoughts", []) if isinstance(data, dict) else []:
            if not isinstance(row, dict) or not row.get("question"):
                continue
            thought = Thought(**{k: v for k, v in row.items() if k in known})
            # A job that was running when she was last shut down is not
            # running now: nothing is coming back for it.
            if thought.running:
                thought.status = "cancelled"
            self._thoughts.append(thought)
        self._next_id = max(
            int(data.get("next_id", 1)) if isinstance(data, dict) else 1,
            max((t.id for t in self._thoughts), default=0) + 1,
        )

    def _save(self) -> None:
        if self._path is None:
            return
        with contextlib.suppress(OSError):  # bookkeeping must never break a think
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            payload = {
                "next_id": self._next_id,
                "thoughts": [asdict(t) for t in self._thoughts[-KEEP:]],
            }
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, self._path)
