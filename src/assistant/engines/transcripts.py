"""Live transcript fragments become turns here, without a clock of their own.

GPT-Live transcribes both sides as word-level fragments on a 200 ms grid
(`session.input_transcript.delta`, `session.output_transcript.delta`), about
150 ms ahead of the audio. The engine wants turns: "he said X" once he is
done, "she said Y" once she is. The SDK ships the grouping policy — a speaker
change ends a turn, a fragment within 500 ms of the last continues it, two
seconds of her silence ends hers, a sub-second "mm-hm" from her while he
talks is a backchannel and is dropped — but wraps it in real timers, which
the engine's fake clocks cannot drive. Its clock-free core is wrapped here
and advanced on the session timeline by whoever knows the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The policy lives in a private SDK module on purpose: it is a direct port of
# their TypeScript grouper and the public classes only add timers. Pinned by
# the openai version in pyproject; tests/test_transcripts.py breaks first if
# it moves.
from openai.lib.live._transcript_grouping import TranscriptGrouping
from openai.lib.live._types import (
    TranscriptFragment,
    TranscriptGrouperOptions,
    TranscriptSegmentClosedEvent,
)

Speaker = Literal["user", "assistant"]


@dataclass(frozen=True)
class Segment:
    """One side's turn as the grouper sees it right now. `reason` is empty
    while the turn is still growing and names why it closed once it has:
    speaker_change, inactivity, session_closed, manual, timestamp_reset."""

    speaker: Speaker
    text: str
    start_ms: int
    end_ms: int
    reason: str = ""
    id: str = ""

    @property
    def closed(self) -> bool:
        return bool(self.reason)


class Segmenter:
    """Feed it fragments as they arrive and the session time as it passes;
    it hands back the segments that changed, closed ones last."""

    def __init__(
        self,
        *,
        min_turn_separation_ms: float = 500,
        assistant_silence_ms: float = 2000,
        backchannel_max_duration_ms: float = 1000,
        backchannel_isolation_ms: float = 2000,
    ) -> None:
        options = TranscriptGrouperOptions(
            min_turn_separation_ms=min_turn_separation_ms,
            assistant_silence_ms=assistant_silence_ms,
            backchannel_max_duration_ms=backchannel_max_duration_ms,
            backchannel_isolation_ms=backchannel_isolation_ms,
        )
        self._grouping = TranscriptGrouping(options, "seg")
        self.latest_ms = 0  # the furthest point on the session timeline any fragment reached

    @property
    def speaker(self) -> str | None:
        """Whose turn is open right now, if anyone's."""
        return self._grouping.speaker

    def fragment(self, speaker: Speaker, text: str, start_ms: int, end_ms: int) -> list[Segment]:
        self.latest_ms = max(self.latest_ms, end_ms)
        piece = TranscriptFragment(speaker=speaker, text=text, start_ms=start_ms, end_ms=end_ms)
        return _convert(self._grouping.process([piece]))

    def advance(self, now_ms: float) -> list[Segment]:
        """Time passed with nothing new: close whatever the policy's deadline
        says is over by `now_ms` (session milliseconds)."""
        events: list = []
        deadline = self._grouping.deadline()
        while deadline is not None and deadline <= now_ms:
            events.extend(self._grouping.advance(deadline))
            following = self._grouping.deadline()
            if following is None or following == deadline:
                break
            deadline = following
        return _convert(events)

    def close(self, now_ms: float, reason: str = "session_closed") -> list[Segment]:
        """The session ended: everything still open closes, with its words."""
        return _convert(self._grouping.close(now_ms, reason))  # type: ignore[arg-type]


def _convert(events: list) -> list[Segment]:
    out: list[Segment] = []
    for event in events:
        if isinstance(event, TranscriptSegmentClosedEvent):
            seg = event.segment
            out.append(Segment(seg.speaker, seg.text, seg.start_ms, seg.end_ms, event.reason, seg.id))
        else:
            out.append(Segment(event.speaker, event.text, event.start_ms, event.end_ms, "", event.id))
    return out
