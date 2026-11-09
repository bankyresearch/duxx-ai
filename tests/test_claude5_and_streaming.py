"""Claude 5 compatibility in the Anthropic provider, and tool execution in Agent.stream."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from duxx_ai import Agent, AgentConfig
from duxx_ai.core import llm as llm_module
from duxx_ai.core.llm import AnthropicProvider, LLMConfig, LLMProvider, LLMResponse
from duxx_ai.core.message import Conversation, Message, Role, ToolCall
from duxx_ai.core.tool import tool

MODEL = "claude-test-model"


@pytest.fixture(autouse=True)
def forget_learned_models() -> None:
    llm_module._ANTHROPIC_NO_TEMPERATURE.clear()


def mock_api(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def provider() -> AnthropicProvider:
    return AnthropicProvider(LLMConfig(provider="anthropic", model=MODEL, api_key="test"))


def conversation() -> Conversation:
    c = Conversation()
    c.add(Message(role=Role.USER, content="hello"))
    return c


def message(text: str = "ok") -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "cache_creation": {"ephemeral_5m": 0}, "service_tier": "standard"},
        "model": MODEL,
        "stop_reason": "end_turn",
    }


def rejecting_handler(seen: list[dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "`temperature` is deprecated for this model."}})
        return httpx.Response(200, json=message())

    return handler


async def test_rejected_temperature_is_retried_without_it_and_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    mock_api(monkeypatch, rejecting_handler(seen))

    assert (await provider().complete(conversation())).content == "ok"
    assert ["temperature" in b for b in seen] == [True, False]

    await provider().complete(conversation())
    assert len(seen) == 3 and "temperature" not in seen[2], "the second call skips the failing attempt"


async def test_temperature_still_sent_to_models_that_accept_it(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=message())

    mock_api(monkeypatch, handler)
    await provider().complete(conversation())
    assert seen[0]["temperature"] == 0.7


async def test_other_400s_are_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_api(monkeypatch, lambda _r: httpx.Response(400, json={"error": {"message": "max_tokens too large"}}))
    with pytest.raises(httpx.HTTPStatusError):
        await provider().complete(conversation())


async def test_nested_usage_values_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_api(monkeypatch, lambda _r: httpx.Response(200, json=message()))
    assert (await provider().complete(conversation())).usage == {"input_tokens": 3, "output_tokens": 2}


def sse(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events))


async def test_stream_events_assembles_text_and_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_api(
        monkeypatch,
        lambda _r: sse(
            {"type": "message_start", "message": {"model": MODEL, "usage": {"input_tokens": 5}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check."}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city"'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ': "Pune"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        ),
    )
    items = [i async for i in provider().stream_events(conversation())]
    assert items[:2] == ["Let me ", "check."]
    final = items[-1]
    assert isinstance(final, LLMResponse)
    assert final.content == "Let me check."
    assert [(c.name, c.arguments) for c in final.tool_calls] == [("lookup", {"city": "Pune"})]
    assert final.usage == {"input_tokens": 5, "output_tokens": 9}
    assert final.finish_reason == "tool_use"


class ScriptedProvider(LLMProvider):
    """Asks for the tool once, then answers. Uses the default stream_events."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="local", model="scripted"))
        self.calls = 0

    async def complete(self, conversation: Conversation, tools: Any = None, system_prompt: Any = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(tool_calls=[ToolCall(id="t1", name="lookup", arguments={"city": "Pune"})], usage={"total_tokens": 4})
        result = conversation.messages[-1].tool_results[0].result
        return LLMResponse(content=f"It is {result}.", usage={"input_tokens": 3, "output_tokens": 3})

    async def stream(self, conversation: Conversation, tools: Any = None, system_prompt: Any = None) -> AsyncIterator[str]:
        yield "unused"


async def test_agent_stream_runs_tools_and_records_the_conversation() -> None:
    ran: list[str] = []

    @tool(name="lookup", description="look up")
    def lookup(city: str) -> str:
        ran.append(city)
        return "sunny"

    agent = Agent(config=AgentConfig(name="a", llm=LLMConfig(provider="local", model="scripted")), tools=[lookup])
    agent.provider = ScriptedProvider()

    assert [t async for t in agent.stream("weather?")] == ["It is sunny."]
    assert ran == ["Pune"]
    assert agent.conversation.messages[-1].role == Role.ASSISTANT
    assert agent.conversation.messages[-1].content == "It is sunny."
    assert agent.state.total_tokens == 10, "counts total_tokens and input+output usage"
