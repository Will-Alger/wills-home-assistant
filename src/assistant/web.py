"""Web search for the voice assistant, via OpenAI's Responses API web_search
tool (verified against openai SDK types: tool type "web_search",
`search_context_size`; `Response.output_text` aggregates the answer). Uses
the same OPENAI_API_KEY as the voice engine — no extra signup.
"""

from __future__ import annotations

from typing import Any

_PROMPT = """\
You are the research helper for a voice assistant. Search the web and answer \
the question below for someone listening, not reading: 1-3 plain sentences, \
the concrete facts first (hours, numbers, names, dates), no markdown, no \
lists. If results disagree or are stale, say so briefly. End with "Sources:" \
and at most two short domain names.

Question: {query}
"""


class WebSearch:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-5-mini",
        context_size: str = "low",
        timeout: float = 40.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._client = client
        self._model = model
        self._context = context_size if context_size in ("low", "medium", "high") else "low"

    async def search(self, query: str) -> str:
        query = " ".join(str(query).split())
        if not query:
            raise ValueError("web_search needs a question")
        response = await self._client.responses.create(
            model=self._model,
            tools=[{"type": "web_search", "search_context_size": self._context}],
            input=_PROMPT.format(query=query),
        )
        text = (getattr(response, "output_text", "") or "").strip()
        return text or "the search came back empty — try different words"
