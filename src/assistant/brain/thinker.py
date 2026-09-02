"""The brain layer: slow, careful reasoning behind the fast voice.

The realtime model is the voice — quick, warm, and a merely-decent reasoner.
For questions that deserve real thought (plans, comparisons, tradeoffs, "help
me think through…"), the voice hands the question to a frontier text model
running on the owner's Claude subscription (`claude -p`, Opus) with her memory,
her task board, and the conversation so far as context, then keeps talking.
The answer comes back as an EVENT the voice speaks in its own words — through
the same injection path announcements use — so nothing ever blocks the voice.

(true-jarvis pattern: voice ↔ brain ↔ hands, one persona.)
"""

from __future__ import annotations

from typing import Any

_SYSTEM = """\
You are the deeper reasoning behind a voice home assistant named {name}. The \
owner, {owner}, asked her something worth real thought; she is chatting with \
him while you work. Answer FOR SPEECH: plain sentences a voice can read aloud \
— no markdown, no lists, no headings, no code. Lead with the conclusion, then \
the two or three reasons that matter, in 3–6 sentences total. Be concrete \
(numbers, names, tradeoffs) and honest about uncertainty. Use the context \
below only where it actually bears on the question.
"""

_MAX_TRANSCRIPT_LINES = 24


class Thinker:
    def __init__(
        self,
        llm: Any,
        *,
        memory: Any | None = None,
        board: Any | None = None,
        name: str = "Alexa",
        owner: str = "Will",
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._board = board
        self._name = name
        self._owner = owner

    def context(self, transcript: list[tuple[str, str]] | None) -> str:
        parts: list[str] = []
        if self._memory is not None:
            prefs = self._memory.preferences_text()
            if prefs and not prefs.startswith("("):
                parts.append(f"Standing preferences:\n{prefs}")
            facts = [i.text for i in self._memory.items("fact")]
            if facts:
                parts.append("Things the owner asked her to remember:\n- " + "\n- ".join(facts[-15:]))
            lessons = self._memory.lessons_text(limit=8)
            if lessons and not lessons.startswith("("):
                parts.append(f"House lessons:\n{lessons}")
        if self._board is not None:
            parts.append(f"Development tasks: {self._board.status_line()}")
        if transcript:
            lines = [
                f"{role}: {text}" for role, text in transcript[-_MAX_TRANSCRIPT_LINES:] if text
            ]
            if lines:
                parts.append("The conversation so far:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    async def think(self, question: str, transcript: list[tuple[str, str]] | None = None) -> str:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("think needs a question")
        context = self.context(transcript)
        user = (f"{context}\n\n" if context else "") + f"Question: {question}"
        result = await self._llm.turn(
            system=_SYSTEM.format(name=self._name, owner=self._owner),
            tools=[],
            messages=[{"role": "user", "content": user}],
        )
        answer = " ".join(str(getattr(result, "text", "")).split())
        return answer or "I thought about it but came up empty — ask me again with more detail."
