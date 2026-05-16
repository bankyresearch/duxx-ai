"""Tests for core BankyAI components."""


import pytest

from duxx_ai.core.message import Conversation, Message, Role, ToolCall, ToolResult
from duxx_ai.core.tool import Tool, ToolParameter, tool


class TestMessage:
    def test_create_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.id is not None

    def test_tool_call_message(self):
        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(name="calc", arguments={"expr": "2+2"})],
        )
        assert msg.is_tool_call()
        assert not msg.is_tool_result()

    def test_tool_result_message(self):
        msg = Message(
            role=Role.TOOL,
            tool_results=[ToolResult(tool_call_id="abc", name="calc", result="4")],
        )
        assert msg.is_tool_result()


class TestConversation:
    def test_add_messages(self):
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="Hi"))
        conv.add(Message(role=Role.ASSISTANT, content="Hello!"))
        assert len(conv.messages) == 2
        assert conv.last_message.role == Role.ASSISTANT

    def test_get_history(self):
        conv = Conversation()
        for i in range(5):
            conv.add(Message(role=Role.USER, content=f"msg {i}"))
        history = conv.get_history(last_n=2)
        assert len(history) == 2
        assert history[0].content == "msg 3"

    def test_to_dicts(self):
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="Hello"))
        conv.add(Message(role=Role.ASSISTANT, content="Hi!"))
        dicts = conv.to_dicts()
        assert dicts == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]


class TestTool:
    def test_create_tool(self):
        t = Tool(
            name="calc",
            description="Calculate math",
            parameters=[ToolParameter(name="expr", type="string")],
        )
        assert t.name == "calc"

    def test_tool_schema(self):
        t = Tool(
            name="calc",
            description="A calculator",
            parameters=[
                ToolParameter(name="expr", type="string", description="Math expression"),
            ],
        )
        schema = t.to_schema()
        assert schema["function"]["name"] == "calc"
        assert "expr" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_tool_execute(self):
        t = Tool(name="add", description="Add two numbers")
        t.bind(lambda a, b: a + b)
        call = ToolCall(name="add", arguments={"a": 2, "b": 3})
        result = await t.execute(call)
        assert result.result == 5
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_execute_error(self):
        t = Tool(name="fail", description="Always fails")
        t.bind(lambda: 1 / 0)
        call = ToolCall(name="fail", arguments={})
        result = await t.execute(call)
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_tool_decorator(self):
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert greet.name == "greet"
        assert len(greet.parameters) == 1

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        import time

        t = Tool(name="slow", description="Slow tool", timeout_seconds=0.1)
        t.bind(lambda: time.sleep(5))
        call = ToolCall(name="slow", arguments={})
        result = await t.execute(call)
        assert result.error is not None
        assert "timed out" in result.error
