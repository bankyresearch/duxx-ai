"""Tool abstraction for agent capabilities."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

from duxx_ai.core.message import ToolCall, ToolResult


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None


class Tool(BaseModel):
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    requires_approval: bool = False
    max_retries: int = 0
    timeout_seconds: float = 30.0
    tags: list[str] = Field(default_factory=list)
    _fn: Callable[..., Any] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def bind(self, fn: Callable[..., Any]) -> Tool:
        object.__setattr__(self, "_fn", fn)
        return self

    async def execute(self, call: ToolCall) -> ToolResult:
        fn = object.__getattribute__(self, "_fn")
        if fn is None:
            return ToolResult(
                tool_call_id=call.id, name=call.name, error="Tool has no bound function"
            )

        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await asyncio.wait_for(fn(**call.arguments), self.timeout_seconds)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: fn(**call.arguments)),
                    self.timeout_seconds,
                )
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_call_id=call.id, name=call.name, result=result, duration_ms=duration
            )
        except asyncio.TimeoutError:
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=f"Tool timed out after {self.timeout_seconds}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration,
            )

    def to_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def _python_type_to_json(t: type) -> str:
    mapping: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    return mapping.get(t, "string") if t in mapping else "string"


def tool(
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    def decorator(fn: Callable[..., Any]) -> Tool:
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        params = []
        for pname, param in sig.parameters.items():
            ptype = hints.get(pname, str)
            params.append(
                ToolParameter(
                    name=pname,
                    type=_python_type_to_json(ptype),
                    description="",
                    required=param.default is inspect.Parameter.empty,
                    default=None if param.default is inspect.Parameter.empty else param.default,
                )
            )

        t = Tool(
            name=name or fn.__name__,
            description=description or fn.__doc__ or "",
            parameters=params,
            requires_approval=requires_approval,
            tags=tags or [],
        )
        t.bind(fn)
        return t

    return decorator
