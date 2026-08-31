"""The Learning Loop: session-end reflection that makes Alexa smarter daily.

After each voice conversation, a text-model pass (our Anthropic provider —
cheap, one call) reads the transcript and distills:
- an episode summary  -> journal ("what did we figure out yesterday?")
- lessons             -> operational recipes injected into future instructions
                         (every trial-and-error becomes permanent competence)
- observations        -> noticed patterns that need Will's consent before
                         becoming standing preferences

Weights never change; memory is the learning mechanism.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from assistant.llm.base import LLMProvider
from assistant.memory import MemoryStore

_REFLECT_SYSTEM = """\
You are the reflection process for a voice home assistant. You read one
conversation transcript (user, assistant, and tool outcomes) and extract only
what is durably worth remembering. Be extremely selective — most sessions
teach nothing and empty lists are the normal result.

- summary: one sentence of what happened (always present; for the journal).
- lessons (max 2): operational recipes discovered THIS session, especially
  things resolved through errors or retries — device quirks, correct
  targets, timing realities. Written as directives a future session can act
  on ("The music player entity is X", "Wake the TV before AirPlay").
  Never restate general knowledge or things already in instructions.
  NEVER record a missing tool or capability as a lesson ("there is no way
  to X", "deleting is not supported") — the assistant is actively developed
  and its toolset grows between sessions, so such lessons rot into false
  limitations that make it deny abilities it has gained. The same goes for
  transient state (which jobs are running, what is currently broken).
- observations (max 1): a behavioral pattern of the user worth ASKING about
  before making it a standing preference ("has asked for lower volume at
  night twice"). Only patterns, never one-offs.

Never include secrets, other people's utterances, or ambient chatter.
"""

_REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        # note: maxItems is unsupported in structured-output schemas — the
        # prompt asks for the caps and code clamps below.
        "lessons": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "lessons", "observations"],
    "additionalProperties": False,
}


@dataclass
class Reflection:
    summary: str = ""
    lessons: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


class Reflector:
    def __init__(self, llm: LLMProvider, memory: MemoryStore) -> None:
        self._llm = llm
        self._memory = memory

    async def reflect(self, transcript: list[tuple[str, str]]) -> Reflection:
        """Distill one session; stores results and returns them for display."""
        if len([1 for role, _ in transcript if role == "you"]) == 0:
            return Reflection()
        lines = "\n".join(f"{role}: {text}" for role, text in transcript[-120:])
        result = await self._llm.turn(
            system=_REFLECT_SYSTEM,
            tools=[],
            messages=[{"role": "user", "content": f"Transcript:\n{lines}"}],
            response_schema=_REFLECT_SCHEMA,
        )
        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            return Reflection()
        reflection = Reflection(
            summary=str(data.get("summary", "")).strip(),
            lessons=[str(x).strip() for x in data.get("lessons", []) if str(x).strip()][:2],
            observations=[str(x).strip() for x in data.get("observations", []) if str(x).strip()][:1],
        )
        if reflection.summary:
            self._memory.add("episode", reflection.summary)
        existing = {item.text for item in self._memory.items()}
        for lesson in reflection.lessons:
            if lesson not in existing:
                self._memory.add("lesson", lesson)
        for obs in reflection.observations:
            if obs not in existing:
                self._memory.add("observation", obs)
        return reflection
