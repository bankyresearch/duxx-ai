"""Message types for agent communication."""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    agent_id: str | None = None

    def is_tool_call(self) -> bool:
        return len(self.tool_calls) > 0

    def is_tool_result(self) -> bool:
        return len(self.tool_results) > 0


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def get_history(self, last_n: int | None = None) -> list[Message]:
        if last_n is None:
            return list(self.messages)
        return list(self.messages[-last_n:])

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def to_dicts(self, provider: str = "openai") -> list[dict[str, Any]]:
        """Serialize conversation for an LLM provider API."""
        if provider == "anthropic":
            return self._to_anthropic_dicts()
        return self._to_openai_dicts()

    def _to_openai_dicts(self) -> list[dict[str, Any]]:
        """Serialize for OpenAI chat completions API."""
        result: list[dict[str, Any]] = []
        for m in self.messages:
            if m.role == Role.ASSISTANT and m.tool_calls:
                msg: dict[str, Any] = {"role": "assistant"}
                if m.content:
                    msg["content"] = m.content
                else:
                    msg["content"] = None
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
                result.append(msg)
            elif m.role == Role.TOOL and m.tool_results:
                for tr in m.tool_results:
                    content = tr.error if tr.error else str(tr.result) if tr.result is not None else ""
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": content,
                    })
            elif m.role in (Role.USER, Role.ASSISTANT, Role.SYSTEM):
                if m.content:
                    result.append({"role": m.role.value, "content": m.content})
        return result

    def _to_anthropic_dicts(self) -> list[dict[str, Any]]:
        """Serialize for Anthropic Messages API."""
        result: list[dict[str, Any]] = []
        for m in self.messages:
            if m.role == Role.SYSTEM:
                continue  # Anthropic handles system prompt separately
            elif m.role == Role.ASSISTANT and m.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content_blocks})
            elif m.role == Role.TOOL and m.tool_results:
                tool_result_blocks: list[dict[str, Any]] = []
                for tr in m.tool_results:
                    content = tr.error if tr.error else str(tr.result) if tr.result is not None else ""
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": content,
                    })
                result.append({"role": "user", "content": tool_result_blocks})
            elif m.role == Role.USER:
                result.append({"role": "user", "content": m.content})
            elif m.role == Role.ASSISTANT:
                if m.content:
                    result.append({"role": "assistant", "content": m.content})
        return result
