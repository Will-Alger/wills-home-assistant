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
from assistant.memory import INFERENCE_CONFIDENCE, VERIFIED_CONFIDENCE, MemoryStore

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
  transient state (which development tasks are building, built, or staged;
  what is currently broken).
  Each lesson also carries its scope and its evidence:
    subject — the one word it belongs to: lights, music, climate, tv,
      calendar, house, will (another short word is fine). It is how the
      lesson finds its way back into a future conversation about that thing.
    evidence — "tool" ONLY when a tool result in THIS transcript demonstrates
      it (an error that named the right entity, a retry that finally worked);
      "inference" when you concluded it from what was said. A one-off outage
      or a slow night is an inference, not evidence. Say "tool" only if you
      could point at the line.
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
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "subject": {"type": "string"},
                    "evidence": {"type": "string", "enum": ["tool", "inference"]},
                },
                "required": ["text", "subject", "evidence"],
                "additionalProperties": False,
            },
        },
        "observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "lessons", "observations"],
    "additionalProperties": False,
}


@dataclass
class Lesson:
    """One recipe with the evidence behind it — the confidence it is stored
    at decides whether future sessions ever see it."""

    text: str
    subject: str = ""
    evidence: str = "inference"

    @property
    def verified(self) -> bool:
        return self.evidence == "tool"


@dataclass
class Reflection:
    summary: str = ""
    lessons: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


def _parse_lessons(raw: object) -> list[Lesson]:
    """The model returns objects; older payloads (and stubs) return bare
    strings. A bare string has no evidence, which is exactly what it means."""
    lessons: list[Lesson] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
            subject = str(entry.get("subject", "") or "").strip()
            evidence = str(entry.get("evidence", "") or "").strip().lower()
        else:
            text, subject, evidence = str(entry).strip(), "", "inference"
        if text:
            lessons.append(
                Lesson(text, subject, "tool" if evidence == "tool" else "inference")
            )
    return lessons


def _tool_outcome_seen(transcript: list[tuple[str, str]]) -> bool:
    """Did anything in this session actually work? Nothing can be evidence in
    a conversation where no tool ever returned — a claim of "tool" there is
    the model believing itself, and drops back to a plain inference."""
    return any(
        str(role).startswith("tool ") and not str(text).startswith("ERROR:")
        for role, text in transcript
    )


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
        lessons = _parse_lessons(data.get("lessons"))[:2]
        reflection = Reflection(
            summary=str(data.get("summary", "")).strip(),
            lessons=[lesson.text for lesson in lessons],
            observations=[str(x).strip() for x in data.get("observations", []) if str(x).strip()][:1],
        )
        if reflection.summary:
            self._memory.add(
                "episode", reflection.summary, source="reflection", confidence=1.0
            )
        # Evidence decides what a lesson is worth. A tool outcome that proved
        # it out beats a single inference, which stays below the bar for
        # injection until a later session runs into the same thing again.
        verified_session = _tool_outcome_seen(transcript)
        known = {item.text: item for item in self._memory.items(include_superseded=True)}
        for lesson in lessons:
            confidence = (
                VERIFIED_CONFIDENCE
                if lesson.verified and verified_session
                else INFERENCE_CONFIDENCE
            )
            seen = known.get(lesson.text)
            if seen is not None:
                # Learned twice is its own evidence: raise what it is held at
                # rather than storing the same sentence again.
                self._memory.reinforce(
                    seen.id, confidence=max(confidence, seen.confidence + 0.2)
                )
                continue
            self._memory.add(
                "lesson",
                lesson.text,
                subject=lesson.subject,
                source="reflection",
                confidence=confidence,
            )
        for obs in reflection.observations:
            if obs not in known:
                self._memory.add("observation", obs, source="reflection", confidence=1.0)
        return reflection
