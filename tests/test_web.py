"""Web search: the tool path with a fake client — no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistant.brain.tools import ToolExecutor
from assistant.home.fake import FakeHome
from assistant.web import WebSearch


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="Kroger on Vine closes at 11 PM tonight. Sources: kroger.com")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


async def test_web_search_uses_the_responses_web_search_tool() -> None:
    client = FakeClient()
    web = WebSearch("key", model="gpt-5-mini", context_size="low", client=client)
    answer = await web.search("what time does Kroger close")
    assert answer.startswith("Kroger on Vine closes at 11 PM")
    (call,) = client.responses.calls
    assert call["model"] == "gpt-5-mini"
    assert call["tools"] == [{"type": "web_search", "search_context_size": "low"}]
    assert "what time does Kroger close" in call["input"]
    with pytest.raises(ValueError):
        await web.search("   ")


async def test_tool_executor_routes_web_search_and_reports_when_unconfigured() -> None:
    web = WebSearch("key", client=FakeClient())
    text, is_error = await ToolExecutor(FakeHome(), web=web).execute(
        "web_search", {"query": "Kroger hours"}
    )
    assert not is_error and "11 PM" in text

    text, is_error = await ToolExecutor(FakeHome()).execute("web_search", {"query": "x"})
    assert is_error and "isn't configured" in text
