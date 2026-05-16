"""Core Agent abstraction — the fundamental building block of Duxx AI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from duxx_ai.core.llm import LLMConfig, LLMResponse, create_provider
from duxx_ai.core.message import Conversation, Message, Role, ToolResult
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import GuardrailChain
from duxx_ai.observability.tracer import Span, Tracer

logger = logging.getLogger(__name__)

# Default cost per 1K tokens by model family (input/output averaged)
MODEL_COST_PER_1K: dict[str, float] = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.001,
    "claude-sonnet": 0.003,
    "claude-haiku": 0.0008,
    "claude-opus": 0.015,
}


def _estimate_cost(model: str, total_tokens: int) -> float:
    """Estimate cost based on model name and token count."""
    for prefix, cost in MODEL_COST_PER_1K.items():
        if prefix in model:
            return (total_tokens / 1000) * cost
    return 0.0


class RetryConfig(BaseModel):
    """Configuration for automatic retry on LLM failures."""
    max_retries: int = 3
    backoff_factor: float = 1.0
    retry_on: list[str] = Field(default_factory=lambda: ["timeout", "rate_limit", "server_error"])


class AgentConfig(BaseModel):
    name: str = "agent"
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    llm: LLMConfig = Field(default_factory=LLMConfig)
    fallback_llm: LLMConfig | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 10
    max_tokens_per_turn: int = 4096
    max_conversation_messages: int = 100
    tags: dict[str, str] = Field(default_factory=dict)


class AgentState(BaseModel):
    iteration: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    status: str = "idle"
    last_error: str | None = None


class Agent:
    """A single AI agent with tools, memory, guardrails, and observability."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
        guardrails: GuardrailChain | None = None,
        tracer: Tracer | None = None,
        approval_callback: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.provider = create_provider(self.config.llm)
        self.tools: dict[str, Tool] = {}
        self.guardrails = guardrails
        self.tracer = tracer or Tracer()
        self.state = AgentState()
        self.conversation = Conversation()
        self.approval_callback = approval_callback

        self.fallback_provider = (
            create_provider(self.config.fallback_llm) if self.config.fallback_llm else None
        )

        for t in tools or []:
            self.register_tool(t)

    @property
    def name(self) -> str:
        return self.config.name

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    async def _llm_call_with_retry(
        self, conversation: Conversation, tools: list[Tool] | None, system_prompt: str
    ) -> LLMResponse:
        """Call LLM with retry + fallback support."""
        import random

        cfg = self.config.retry
        last_error: Exception | None = None

        for attempt in range(cfg.max_retries + 1):
            try:
                return await self.provider.complete(conversation, tools=tools, system_prompt=system_prompt)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                retryable = any(kw in err_str for kw in ["timeout", "rate", "429", "500", "502", "503"])
                if not retryable or attempt == cfg.max_retries:
                    break
                wait = cfg.backoff_factor * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {wait:.1f}s: {e}")
                await asyncio.sleep(wait)

        # Try fallback provider
        if self.fallback_provider and last_error:
            logger.info("Primary LLM failed, trying fallback provider")
            try:
                return await self.fallback_provider.complete(conversation, tools=tools, system_prompt=system_prompt)
            except Exception as fallback_err:
                logger.error(f"Fallback also failed: {fallback_err}")

        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed with no error captured")

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Stream the agent response token by token."""
        self.conversation.add(Message(role=Role.USER, content=user_input, agent_id=self.name))
        self._trim_conversation()

        async for token in self.provider.stream(
            self.conversation,
            tools=list(self.tools.values()) if self.tools else None,
            system_prompt=self.config.system_prompt,
        ):
            yield token

    async def run(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        """Run the agent with a user input and return the final response."""
        self.state.status = "running"
        self.state.iteration = 0

        self.conversation.add(Message(role=Role.USER, content=user_input, agent_id=self.name))
        self._trim_conversation()

        with self.tracer.span(f"agent.{self.name}.run") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("input", user_input)

            try:
                result = await self._agent_loop(span, context)
                span.set_attribute("output", result)
                self.state.status = "idle"
                return result
            except Exception as e:
                self.state.status = "error"
                self.state.last_error = str(e)
                span.set_attribute("error", str(e))
                raise

    async def _agent_loop(self, parent_span: Span, context: dict[str, Any] | None) -> str:
        tool_list = list(self.tools.values()) if self.tools else None

        for i in range(self.config.max_iterations):
            self.state.iteration = i + 1

            # Input guardrails (only on first iteration — check user input)
            if i == 0 and self.guardrails:
                last = self.conversation.last_message
                if last and last.content:
                    check = await self.guardrails.check_input(last.content)
                    if not check.passed:
                        return f"Blocked by guardrail: {check.reason}"

            with self.tracer.span(f"agent.{self.name}.llm_call") as span:
                response = await self._llm_call_with_retry(
                    self.conversation,
                    tools=tool_list,
                    system_prompt=self.config.system_prompt,
                )
                self._track_usage(response)
                span.set_attribute("model", response.model)
                span.set_attribute("tokens", response.usage)

            # If no tool calls, we have our final response
            if not response.tool_calls:
                if response.content:
                    # Output guardrails
                    if self.guardrails:
                        check = await self.guardrails.check_output(response.content)
                        if not check.passed:
                            return f"Response blocked by guardrail: {check.reason}"

                    self.conversation.add(
                        Message(role=Role.ASSISTANT, content=response.content, agent_id=self.name)
                    )
                return response.content or ""

            # Handle tool calls
            self.conversation.add(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    agent_id=self.name,
                )
            )

            tool_results = await self._execute_tools(response.tool_calls)

            self.conversation.add(
                Message(role=Role.TOOL, tool_results=tool_results, agent_id=self.name)
            )

        return "Agent reached maximum iterations without a final response."

    async def _execute_tools(self, calls: list[Any]) -> list[Any]:
        tasks = []
        for call in calls:
            tool = self.tools.get(call.name)
            if tool is None:
                tasks.append(self._make_unknown_tool_result(call))
            else:
                tasks.append(self._execute_single_tool(tool, call))
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _make_unknown_tool_result(call: Any) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id, name=call.name, error=f"Unknown tool: {call.name}"
        )

    async def _execute_single_tool(self, tool: Tool, call: Any) -> Any:
        with self.tracer.span(f"tool.{call.name}") as span:
            span.set_attribute("tool.name", call.name)
            span.set_attribute("tool.args", str(call.arguments))

            if tool.requires_approval:
                if self.approval_callback:
                    approved = await self.approval_callback(call.name, call.arguments)
                    if not approved:
                        return ToolResult(
                            tool_call_id=call.id,
                            name=call.name,
                            error=f"Tool '{call.name}' execution was denied by approval callback",
                        )
                else:
                    logger.warning(
                        f"Tool '{call.name}' requires approval but no approval_callback is set. "
                        f"Auto-approving. Set approval_callback on the Agent to enforce approvals."
                    )

            result = await tool.execute(call)
            span.set_attribute("tool.result", str(result.result)[:500] if result.result else "")
            if result.error:
                span.set_attribute("tool.error", result.error)
            return result

    def _track_usage(self, response: LLMResponse) -> None:
        total = response.usage.get("total_tokens", 0)
        self.state.total_tokens += total
        self.state.total_cost += _estimate_cost(response.model, total)

    def _trim_conversation(self) -> None:
        """Trim conversation with context compression.

        If conversation exceeds max_conversation_messages:
        1. Summarize older messages into a single context message
        2. Keep the summary + most recent messages
        """
        max_msgs = self.config.max_conversation_messages
        if len(self.conversation.messages) <= max_msgs:
            return

        # Context compression: summarize older messages
        msgs = self.conversation.messages
        keep_recent = max_msgs // 2
        old_messages = msgs[:-keep_recent]

        # Build a compressed summary of old messages
        summary_parts = []
        for m in old_messages:
            if m.content:
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                summary_parts.append(f"[{role}] {m.content[:100]}")

        if summary_parts:
            summary_text = "Previous conversation summary:\n" + "\n".join(summary_parts[-10:])
            summary_msg = Message(
                role=Role.SYSTEM,
                content=summary_text,
                agent_id=self.name,
                metadata={"type": "context_compression", "compressed_count": len(old_messages)},
            )
            self.conversation.messages = [summary_msg] + msgs[-keep_recent:]
        else:
            self.conversation.messages = msgs[-max_msgs:]

    # ── Subagent Spawning ──

    async def spawn_subagent(
        self,
        task: str,
        name: str | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        context: dict[str, Any] | None = None,
        isolated: bool = True,
    ) -> str:
        """Spawn a child agent with isolated context to handle a subtask.

        Args:
            task: The task description for the subagent
            name: Subagent name (auto-generated if None)
            system_prompt: Custom system prompt (inherits parent's if None)
            tools: Tools for subagent (inherits parent's if None)
            context: Additional context to inject (only used if isolated=True)
            isolated: If True, subagent gets a fresh conversation (no parent history)

        Returns:
            The subagent's response string
        """
        sub_name = name or f"{self.name}.sub_{self.state.iteration}"

        sub_config = AgentConfig(
            name=sub_name,
            system_prompt=system_prompt or self.config.system_prompt,
            llm=self.config.llm,
            fallback_llm=self.config.fallback_llm,
            retry=self.config.retry,
            max_iterations=self.config.max_iterations,
        )

        sub_tools = list(tools or list(self.tools.values()))
        sub_agent = Agent(
            config=sub_config,
            tools=sub_tools,
            guardrails=self.guardrails,
            tracer=self.tracer,
            approval_callback=self.approval_callback,
        )

        # Inject context if provided
        if context and not isolated:
            for key, val in context.items():
                sub_agent.conversation.add(Message(
                    role=Role.SYSTEM, content=f"Context ({key}): {val}", agent_id=sub_name
                ))

        with self.tracer.span(f"subagent.{sub_name}") as span:
            span.set_attribute("subagent.name", sub_name)
            span.set_attribute("subagent.task", task[:200])
            span.set_attribute("subagent.isolated", isolated)
            result = await sub_agent.run(task)
            span.set_attribute("subagent.result", result[:500] if result else "")
            # Track child's usage in parent
            self.state.total_tokens += sub_agent.state.total_tokens
            self.state.total_cost += sub_agent.state.total_cost
            return result

    # ── Planning & Task Decomposition ──

    async def plan_and_execute(
        self,
        objective: str,
        max_subtasks: int = 5,
    ) -> dict[str, Any]:
        """Break down a complex objective into subtasks, then execute each.

        Uses the LLM to decompose the objective, then spawns subagents
        for each subtask and collects results.

        Returns:
            {"objective": str, "plan": list, "results": list, "final_answer": str}
        """
        # Step 1: Plan — ask LLM to decompose
        plan_prompt = (
            f"Break down this objective into {max_subtasks} or fewer clear, "
            f"actionable subtasks. Return ONLY a numbered list.\n\n"
            f"Objective: {objective}"
        )
        plan_response = await self.run(plan_prompt)

        # Parse numbered list
        import re
        tasks = re.findall(r"\d+[.)]\s*(.+)", plan_response)
        if not tasks:
            tasks = [objective]

        # Step 2: Execute each subtask via subagents
        results = []
        for i, task in enumerate(tasks[:max_subtasks]):
            result = await self.spawn_subagent(
                task=task,
                name=f"{self.name}.plan_step_{i}",
                isolated=True,
            )
            results.append({"task": task, "result": result})

        # Step 3: Synthesize results
        synthesis_prompt = (
            f"Original objective: {objective}\n\n"
            f"Subtask results:\n" +
            "\n".join(f"- {r['task']}: {r['result'][:200]}" for r in results) +
            "\n\nSynthesize a final answer from all subtask results."
        )
        final = await self.run(synthesis_prompt)

        return {
            "objective": objective,
            "plan": tasks,
            "results": results,
            "final_answer": final,
        }

    # ── Batch Processing ──

    async def batch(
        self,
        inputs: list[str],
        max_concurrency: int = 5,
    ) -> list[str]:
        """Process multiple inputs concurrently with concurrency control.

        Args:
            inputs: List of user messages to process
            max_concurrency: Maximum concurrent agent runs

        Returns:
            List of responses in the same order as inputs
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(user_input: str) -> str:
            async with semaphore:
                # Create a fresh agent for each batch item (isolated context)
                sub = Agent(
                    config=self.config,
                    tools=list(self.tools.values()),
                    guardrails=self.guardrails,
                    tracer=self.tracer,
                )
                return await sub.run(user_input)

        results = await asyncio.gather(*[_run_one(inp) for inp in inputs])
        return list(results)

    # ── Pre/Post Model Hooks ──

    _pre_hooks: list[Callable[[Conversation], Awaitable[Conversation] | Conversation]] = []
    _post_hooks: list[Callable[[LLMResponse], Awaitable[LLMResponse] | LLMResponse]] = []

    def add_pre_hook(self, hook: Callable) -> None:
        """Add a hook that runs BEFORE each LLM call. Receives and returns Conversation."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable) -> None:
        """Add a hook that runs AFTER each LLM call. Receives and returns LLMResponse."""
        self._post_hooks.append(hook)

    def reset(self) -> None:
        self.conversation = Conversation()
        self.state = AgentState()
