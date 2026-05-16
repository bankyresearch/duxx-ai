"""Advanced agent patterns — ReAct, Reflection, Handoffs, Self-Improving, Teachable.

Implements the most advanced agentic patterns from modern AI research,
CrewAI, AutoGen, OpenAI Agents SDK, and Anthropic Agent SDK.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. ReAct Agent — Reasoning + Acting Loop with Self-Correction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ThoughtStep:
    """A single thought-action-observation cycle."""
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class ReActTrace:
    """Complete trace of a ReAct execution."""
    query: str
    steps: list[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    total_iterations: int = 0
    self_corrections: int = 0
    reflection_triggered: bool = False


class ReActAgent:
    """ReAct (Reasoning + Acting) agent with self-correction and reflection.

    Implements the think-act-observe loop with:
    - Explicit reasoning before each action
    - Self-verification after observations
    - Reflection when stuck (re-evaluates approach)
    - Configurable max iterations and reflection threshold

    Usage:
        from duxx_ai.core.patterns import ReActAgent
        from duxx_ai.core.agent import Agent, AgentConfig

        base_agent = Agent(config=AgentConfig(name="researcher"), tools=[search, calc])
        react = ReActAgent(base_agent, max_iterations=10, reflection_threshold=3)
        result = await react.run("What is the GDP of France divided by its population?")
        print(result.final_answer)
        print(f"Steps: {len(result.steps)}, Corrections: {result.self_corrections}")
    """

    REACT_SYSTEM = """You are a reasoning agent. For each query, follow this exact process:

THOUGHT: Think step-by-step about what you need to do.
ACTION: Choose a tool to use (or FINISH if you have the answer).
ACTION_INPUT: Provide the input for the tool as JSON.

After receiving an observation, evaluate:
- Is the observation correct and useful?
- Do I need to try a different approach?
- Am I confident enough to give a final answer?

If you have enough information, respond with:
THOUGHT: I now have enough information.
FINAL_ANSWER: [your answer]

Be precise. Verify your reasoning. If something seems wrong, try a different approach."""

    REFLECTION_PROMPT = """You've been working on this problem for several steps without reaching an answer.

Steps so far:
{steps}

Reflect on your approach:
1. What have you tried?
2. What went wrong?
3. What should you try differently?
4. Is there a simpler way to solve this?

Provide a revised plan as THOUGHT."""

    VERIFY_PROMPT = """Verify this answer before finalizing:

Question: {query}
Proposed answer: {answer}
Reasoning steps: {steps}

Is this answer correct? If not, explain why and provide the correct answer.
Respond with either:
VERIFIED: [the answer is correct]
CORRECTION: [the corrected answer and why]"""

    def __init__(
        self,
        agent: Any,
        max_iterations: int = 10,
        reflection_threshold: int = 3,
        verify_answer: bool = True,
    ):
        self.agent = agent
        self.max_iterations = max_iterations
        self.reflection_threshold = reflection_threshold
        self.verify_answer = verify_answer
        self._original_system = agent.config.system_prompt

    async def run(self, query: str) -> ReActTrace:
        """Execute the ReAct loop."""
        trace = ReActTrace(query=query)

        # Set ReAct system prompt
        self.agent.config.system_prompt = self.REACT_SYSTEM
        self.agent.reset()

        # Build initial prompt
        prompt = f"Query: {query}\n\nTHOUGHT:"
        steps_without_progress = 0

        for i in range(self.max_iterations):
            trace.total_iterations = i + 1

            # Check if reflection needed
            if steps_without_progress >= self.reflection_threshold:
                steps_text = "\n".join(
                    f"Step {j+1}: Thought={s.thought}, Action={s.action}, Obs={s.observation}"
                    for j, s in enumerate(trace.steps)
                )
                prompt = self.REFLECTION_PROMPT.format(steps=steps_text)
                trace.reflection_triggered = True
                steps_without_progress = 0

            # Get agent response
            response = await self.agent.run(prompt)

            # Parse the response
            step = self._parse_response(response)
            trace.steps.append(step)

            if step.is_final:
                # Verify if enabled
                if self.verify_answer and step.thought:
                    verified = await self._verify(query, step.thought, trace.steps)
                    if verified != step.thought:
                        trace.self_corrections += 1
                        step.thought = verified
                trace.final_answer = step.thought or response
                break

            # Execute action if present
            if step.action and step.action_input is not None:
                try:
                    observation = await self.agent.run(
                        f"Execute tool '{step.action}' with input: {json.dumps(step.action_input)}"
                    )
                    step.observation = observation
                    prompt = f"OBSERVATION: {observation}\n\nTHOUGHT:"
                    steps_without_progress = 0
                except Exception as e:
                    step.observation = f"Error: {e}"
                    prompt = f"OBSERVATION: Error occurred: {e}. Try a different approach.\n\nTHOUGHT:"
                    steps_without_progress += 1
            else:
                steps_without_progress += 1
                prompt = f"Continue reasoning. Your last thought: {step.thought}\n\nTHOUGHT:"

        # Restore original system prompt
        self.agent.config.system_prompt = self._original_system
        return trace

    def _parse_response(self, text: str) -> ThoughtStep:
        """Parse a ReAct-formatted response."""
        step = ThoughtStep(thought="")

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("THOUGHT:"):
                step.thought = line[8:].strip()
            elif line.startswith("ACTION:"):
                action = line[7:].strip()
                if action.upper() == "FINISH":
                    step.is_final = True
                else:
                    step.action = action
            elif line.startswith("ACTION_INPUT:"):
                try:
                    step.action_input = json.loads(line[13:].strip())
                except (json.JSONDecodeError, ValueError):
                    step.action_input = {"input": line[13:].strip()}
            elif line.startswith("FINAL_ANSWER:"):
                step.thought = line[13:].strip()
                step.is_final = True
            elif line.startswith("VERIFIED:"):
                step.thought = line[9:].strip()
                step.is_final = True
            elif line.startswith("CORRECTION:"):
                step.thought = line[11:].strip()
                step.is_final = True

        if not step.thought:
            step.thought = text.strip()

        return step

    async def _verify(self, query: str, answer: str, steps: list[ThoughtStep]) -> str:
        """Verify an answer using self-reflection."""
        steps_text = "\n".join(f"- {s.thought}" for s in steps if s.thought)
        prompt = self.VERIFY_PROMPT.format(query=query, answer=answer, steps=steps_text)
        response = await self.agent.run(prompt)

        if "CORRECTION:" in response:
            return response.split("CORRECTION:")[1].strip()
        return answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Agent Handoffs — Tool-Based Agent-to-Agent Delegation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class HandoffResult:
    """Result of an agent handoff."""
    source_agent: str
    target_agent: str
    task: str
    result: str
    tokens_used: int = 0
    cost: float = 0.0


class AgentHandoff:
    """Agent-to-agent delegation via tool-based handoffs.

    Like OpenAI Agents SDK handoffs — one agent can transfer control
    to a specialist agent mid-conversation.

    Usage:
        from duxx_ai.core.patterns import AgentHandoff

        handoff = AgentHandoff()
        handoff.register("refund_agent", refund_agent, "Handles refund requests")
        handoff.register("billing_agent", billing_agent, "Handles billing questions")

        # The triage agent gets handoff tools automatically
        triage = Agent(config=AgentConfig(name="triage"), tools=handoff.as_tools())
        result = await triage.run("I want a refund for my last order")
        # -> triage calls transfer_to_refund_agent tool -> refund_agent handles it
    """

    def __init__(self):
        self._agents: dict[str, tuple[Any, str]] = {}  # name -> (agent, description)
        self._history: list[HandoffResult] = []

    def register(self, name: str, agent: Any, description: str = "") -> AgentHandoff:
        """Register an agent that can receive handoffs."""
        self._agents[name] = (agent, description)
        return self

    def as_tools(self) -> list[Any]:
        """Generate handoff tools for each registered agent."""
        from duxx_ai.core.tool import Tool, ToolParameter

        tools = []
        for name, (agent, desc) in self._agents.items():
            tool = Tool(
                name=f"transfer_to_{name}",
                description=f"Transfer this conversation to {name}. {desc}",
                parameters=[
                    ToolParameter(
                        name="task",
                        type="string",
                        description="The task or question to hand off",
                        required=True,
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Additional context from the current conversation",
                        required=False,
                    ),
                ],
            )

            async def _execute_handoff(task: str, context: str = "", _agent=agent, _name=name) -> str:
                ctx = {"handoff_context": context} if context else {}
                result = await _agent.run(task, context=ctx)
                self._history.append(HandoffResult(
                    source_agent="caller",
                    target_agent=_name,
                    task=task,
                    result=result,
                    tokens_used=_agent.state.total_tokens,
                    cost=_agent.state.total_cost,
                ))
                return result

            tool.bind(_execute_handoff)
            tools.append(tool)

        return tools

    @property
    def history(self) -> list[HandoffResult]:
        return self._history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Self-Improving Agent — Reward-Based Learning Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Reward:
    """Feedback signal for agent improvement."""
    query: str
    response: str
    score: float  # 0.0 to 1.0
    feedback: str = ""
    source: str = "human"  # human, self, evaluator
    timestamp: float = field(default_factory=time.time)


@dataclass
class Skill:
    """A learned skill that can be reused."""
    name: str
    description: str
    examples: list[dict[str, str]] = field(default_factory=list)  # input/output pairs
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)


class SelfImprovingAgent:
    """Agent that learns from rewards and accumulates skills.

    Implements DeepAgents-style self-improvement:
    - Reward signals (human, self-judge, or evaluator)
    - Skill library accumulation from successful interactions
    - Self-judging for automatic reward generation
    - Experience replay from high-reward interactions

    Usage:
        from duxx_ai.core.patterns import SelfImprovingAgent

        sia = SelfImprovingAgent(base_agent, skill_persistence_path="skills.json")

        # Run with automatic self-evaluation
        result = await sia.run("Analyze this financial report")

        # Provide human feedback
        sia.reward(query="Analyze...", response=result, score=0.9, feedback="Great analysis")

        # Agent learns from rewards and improves over time
        # High-scoring interactions become skills
    """

    SELF_JUDGE_PROMPT = """Rate the quality of this response on a scale of 0.0 to 1.0.

Question: {query}
Response: {response}

Evaluate on:
1. Accuracy (0-0.25): Is the information correct?
2. Completeness (0-0.25): Does it fully answer the question?
3. Clarity (0-0.25): Is it well-organized and easy to understand?
4. Usefulness (0-0.25): Is it actionable and helpful?

Respond with ONLY a JSON object:
{{"score": 0.X, "feedback": "brief explanation"}}"""

    SKILL_EXTRACTION_PROMPT = """Extract a reusable skill from this successful interaction.

Question: {query}
Response: {response}
Score: {score}

Describe the skill in a way that can be reused for similar tasks:
{{"name": "skill_name", "description": "what this skill does", "pattern": "when to use it"}}"""

    def __init__(
        self,
        agent: Any,
        self_judge: bool = True,
        skill_threshold: float = 0.8,
        skill_persistence_path: str | None = None,
        max_experience_buffer: int = 100,
    ):
        self.agent = agent
        self.self_judge = self_judge
        self.skill_threshold = skill_threshold
        self.skills: list[Skill] = []
        self.rewards: list[Reward] = []
        self.experience_buffer: list[dict[str, Any]] = []
        self.max_experience_buffer = max_experience_buffer
        self._persistence_path = Path(skill_persistence_path) if skill_persistence_path else None

        if self._persistence_path and self._persistence_path.exists():
            self._load_skills()

    async def run(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Run with skill injection and optional self-evaluation."""
        # Inject relevant skills into context
        relevant_skills = self._find_relevant_skills(query)
        if relevant_skills:
            skill_text = "\n".join(
                f"- {s.name}: {s.description}" for s in relevant_skills
            )
            enhanced_prompt = (
                f"You have learned these skills from past experience:\n{skill_text}\n\n"
                f"Apply relevant skills to answer: {query}"
            )
        else:
            enhanced_prompt = query

        # Execute
        response = await self.agent.run(enhanced_prompt, context=context)

        # Self-judge if enabled
        if self.self_judge:
            reward = await self._self_evaluate(query, response)
            self.rewards.append(reward)

            # Extract skill if high reward
            if reward.score >= self.skill_threshold:
                await self._extract_skill(query, response, reward.score)

        # Store experience
        self.experience_buffer.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
        })
        if len(self.experience_buffer) > self.max_experience_buffer:
            self.experience_buffer.pop(0)

        return response

    def reward(self, query: str, response: str, score: float, feedback: str = "", source: str = "human") -> None:
        """Provide external reward signal."""
        r = Reward(query=query, response=response, score=score, feedback=feedback, source=source)
        self.rewards.append(r)

        if score >= self.skill_threshold:
            asyncio.get_event_loop().run_until_complete(
                self._extract_skill(query, response, score)
            ) if asyncio.get_event_loop().is_running() else None

    async def _self_evaluate(self, query: str, response: str) -> Reward:
        """Self-judge the response quality."""
        prompt = self.SELF_JUDGE_PROMPT.format(query=query, response=response)
        judge_response = await self.agent.run(prompt)

        try:
            data = json.loads(judge_response)
            score = float(data.get("score", 0.5))
            feedback = data.get("feedback", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            feedback = judge_response

        return Reward(
            query=query, response=response,
            score=max(0.0, min(1.0, score)),
            feedback=feedback, source="self",
        )

    async def _extract_skill(self, query: str, response: str, score: float) -> None:
        """Extract a reusable skill from a high-scoring interaction."""
        # Check for duplicate
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        for s in self.skills:
            if query_hash in s.name:
                s.usage_count += 1
                s.success_rate = (s.success_rate + score) / 2
                return

        skill = Skill(
            name=f"skill_{query_hash}",
            description=f"Learned from: {query[:100]}",
            examples=[{"input": query, "output": response[:500]}],
            success_rate=score,
            usage_count=1,
        )
        self.skills.append(skill)

        if self._persistence_path:
            self._save_skills()

    def _find_relevant_skills(self, query: str, top_k: int = 3) -> list[Skill]:
        """Find skills relevant to the current query."""
        if not self.skills:
            return []

        query_words = set(query.lower().split())
        scored = []
        for skill in self.skills:
            desc_words = set(skill.description.lower().split())
            overlap = len(query_words & desc_words) / max(len(query_words), 1)
            score = overlap * skill.success_rate * (1 + skill.usage_count * 0.1)
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k] if _ > 0.1]

    def _save_skills(self) -> None:
        if self._persistence_path:
            data = [
                {
                    "name": s.name, "description": s.description,
                    "examples": s.examples, "success_rate": s.success_rate,
                    "usage_count": s.usage_count, "created_at": s.created_at,
                }
                for s in self.skills
            ]
            self._persistence_path.write_text(json.dumps(data, indent=2))

    def _load_skills(self) -> None:
        if self._persistence_path and self._persistence_path.exists():
            try:
                data = json.loads(self._persistence_path.read_text())
                self.skills = [Skill(**d) for d in data]
            except Exception:
                self.skills = []

    @property
    def stats(self) -> dict[str, Any]:
        """Get improvement statistics."""
        if not self.rewards:
            return {"total_runs": 0, "avg_score": 0, "skills_learned": len(self.skills)}

        scores = [r.score for r in self.rewards]
        recent = scores[-10:] if len(scores) > 10 else scores
        early = scores[:10] if len(scores) > 10 else scores

        return {
            "total_runs": len(self.rewards),
            "avg_score": sum(scores) / len(scores),
            "recent_avg": sum(recent) / len(recent),
            "early_avg": sum(early) / len(early),
            "improvement": (sum(recent) / len(recent)) - (sum(early) / len(early)) if len(scores) > 10 else 0,
            "skills_learned": len(self.skills),
            "self_judged": sum(1 for r in self.rewards if r.source == "self"),
            "human_judged": sum(1 for r in self.rewards if r.source == "human"),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Teachable Agent — Learns from Interactions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Memory:
    """A fact or preference learned from interaction."""
    key: str
    value: str
    source: str  # "told", "inferred", "corrected"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


class TeachableAgent:
    """Agent that learns facts, preferences, and corrections from conversations.

    Like AutoGen's teachable agents — remembers what you tell it.

    Usage:
        from duxx_ai.core.patterns import TeachableAgent

        ta = TeachableAgent(base_agent, memory_path="agent_memory.json")
        await ta.run("My name is Bankatesh and I prefer formal reports")
        # Agent remembers this
        await ta.run("Write me a report")
        # -> Uses formal style, addresses as Bankatesh
    """

    EXTRACT_FACTS_PROMPT = """From this conversation, extract any facts, preferences, or corrections the user mentioned.

User message: {message}
Agent response: {response}

Extract as JSON array. If nothing to learn, return [].
Example: [
    {{"key": "user_name", "value": "Bankatesh", "source": "told"}},
    {{"key": "report_style", "value": "formal with charts", "source": "told"}}
]"""

    def __init__(
        self,
        agent: Any,
        memory_path: str | None = None,
        auto_learn: bool = True,
    ):
        self.agent = agent
        self.memories: list[Memory] = []
        self.auto_learn = auto_learn
        self._memory_path = Path(memory_path) if memory_path else None

        if self._memory_path and self._memory_path.exists():
            self._load_memories()

    async def run(self, message: str, context: dict[str, Any] | None = None) -> str:
        """Run with memory injection and auto-learning."""
        # Inject memories
        relevant = self._recall(message)
        if relevant:
            mem_text = "\n".join(f"- {m.key}: {m.value}" for m in relevant)
            enhanced = (
                f"Things you know about this user:\n{mem_text}\n\n"
                f"User: {message}"
            )
        else:
            enhanced = message

        response = await self.agent.run(enhanced, context=context)

        # Auto-extract learnings
        if self.auto_learn:
            await self._extract_and_store(message, response)

        return response

    def teach(self, key: str, value: str, source: str = "told") -> None:
        """Explicitly teach the agent a fact."""
        # Update existing or add new
        for m in self.memories:
            if m.key == key:
                m.value = value
                m.source = source
                m.confidence = 1.0
                self._save()
                return
        self.memories.append(Memory(key=key, value=value, source=source))
        self._save()

    def correct(self, key: str, value: str) -> None:
        """Correct a previously learned fact."""
        self.teach(key, value, source="corrected")

    def forget(self, key: str) -> bool:
        """Remove a learned fact."""
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.key != key]
        self._save()
        return len(self.memories) < before

    def _recall(self, query: str, top_k: int = 5) -> list[Memory]:
        """Recall relevant memories."""
        if not self.memories:
            return []
        query_words = set(query.lower().split())
        scored = []
        for m in self.memories:
            key_words = set(m.key.lower().split())
            val_words = set(m.value.lower().split())
            overlap = len(query_words & (key_words | val_words))
            score = overlap * m.confidence
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def _extract_and_store(self, message: str, response: str) -> None:
        """Auto-extract facts from conversation."""
        try:
            prompt = self.EXTRACT_FACTS_PROMPT.format(message=message, response=response)
            extract = await self.agent.run(prompt)

            # Try to parse JSON from response
            if "[" in extract:
                json_str = extract[extract.index("["):extract.rindex("]") + 1]
                facts = json.loads(json_str)
                for f in facts:
                    if isinstance(f, dict) and "key" in f and "value" in f:
                        self.teach(f["key"], f["value"], f.get("source", "inferred"))
        except Exception:
            pass  # Silent fail on extraction

    def _save(self) -> None:
        if self._memory_path:
            data = [{"key": m.key, "value": m.value, "source": m.source,
                     "confidence": m.confidence, "created_at": m.created_at}
                    for m in self.memories]
            self._memory_path.write_text(json.dumps(data, indent=2))

    def _load_memories(self) -> None:
        try:
            data = json.loads(self._memory_path.read_text())
            self.memories = [Memory(**d) for d in data]
        except Exception:
            self.memories = []

    @property
    def knowledge(self) -> dict[str, str]:
        """Get all learned facts as a dict."""
        return {m.key: m.value for m in self.memories}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. Evaluator-Optimizer — Self-Correcting Output Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OptimizationResult:
    """Result of an evaluator-optimizer run."""
    original: str
    optimized: str
    iterations: int
    scores: list[float]
    improvements: list[str]
    final_score: float


class EvaluatorOptimizer:
    """Two-agent pattern: one generates, one evaluates, iterate until quality threshold.

    Implements the Anthropic evaluator-optimizer pattern:
    - Generator produces output
    - Evaluator scores and provides specific feedback
    - Generator revises based on feedback
    - Repeat until score >= threshold or max iterations

    Usage:
        from duxx_ai.core.patterns import EvaluatorOptimizer

        eo = EvaluatorOptimizer(
            generator=writer_agent,
            evaluator=critic_agent,
            threshold=0.85,
            max_iterations=3,
        )
        result = await eo.run("Write a technical blog post about RAG pipelines")
        print(f"Final score: {result.final_score}, Iterations: {result.iterations}")
    """

    EVALUATOR_PROMPT = """Evaluate this output on a scale of 0.0 to 1.0.

Task: {task}
Output:
{output}

Score on:
1. Quality (0-0.25): Technical accuracy and depth
2. Completeness (0-0.25): Covers all aspects of the task
3. Clarity (0-0.25): Well-structured and readable
4. Usefulness (0-0.25): Actionable and valuable

Respond with JSON:
{{"score": 0.X, "improvements": ["specific improvement 1", "specific improvement 2"]}}"""

    REVISE_PROMPT = """Revise your output based on this feedback.

Original task: {task}
Your previous output:
{output}

Score: {score}/1.0
Required improvements:
{improvements}

Provide an improved version that addresses ALL feedback points."""

    def __init__(
        self,
        generator: Any,
        evaluator: Any | None = None,
        threshold: float = 0.8,
        max_iterations: int = 3,
    ):
        self.generator = generator
        self.evaluator = evaluator or generator  # Self-evaluate if no separate evaluator
        self.threshold = threshold
        self.max_iterations = max_iterations

    async def run(self, task: str) -> OptimizationResult:
        """Run the evaluator-optimizer loop."""
        # Initial generation
        output = await self.generator.run(task)
        result = OptimizationResult(
            original=output, optimized=output,
            iterations=0, scores=[], improvements=[], final_score=0.0,
        )

        for i in range(self.max_iterations):
            result.iterations = i + 1

            # Evaluate
            eval_prompt = self.EVALUATOR_PROMPT.format(task=task, output=output)
            eval_response = await self.evaluator.run(eval_prompt)

            try:
                data = json.loads(eval_response)
                score = float(data.get("score", 0.5))
                improvements = data.get("improvements", [])
            except (json.JSONDecodeError, ValueError):
                score = 0.5
                improvements = [eval_response]

            result.scores.append(score)
            result.final_score = score

            # Check threshold
            if score >= self.threshold:
                result.optimized = output
                break

            # Revise
            result.improvements.extend(improvements)
            imp_text = "\n".join(f"- {imp}" for imp in improvements)
            revise_prompt = self.REVISE_PROMPT.format(
                task=task, output=output, score=score, improvements=imp_text,
            )
            output = await self.generator.run(revise_prompt)
            result.optimized = output

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. Orchestrator-Worker — Dynamic Task Delegation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class WorkerResult:
    """Result from a worker agent."""
    worker_name: str
    task: str
    result: str
    duration_ms: float = 0.0


class OrchestratorWorker:
    """Dynamic orchestrator that delegates to specialist workers.

    The orchestrator analyzes the task, decomposes it, assigns subtasks
    to the best worker, and synthesizes results.

    Usage:
        from duxx_ai.core.patterns import OrchestratorWorker

        ow = OrchestratorWorker(orchestrator=manager_agent)
        ow.add_worker("researcher", research_agent, "Deep research and analysis")
        ow.add_worker("writer", writer_agent, "Writing and editing")
        ow.add_worker("coder", code_agent, "Code generation and debugging")

        result = await ow.run("Build a Python library for sentiment analysis with docs")
        # Orchestrator decomposes -> assigns to coder + writer -> synthesizes
    """

    DECOMPOSE_PROMPT = """You are an orchestrator. Decompose this task into subtasks and assign each to the best worker.

Task: {task}

Available workers:
{workers}

Respond with JSON:
{{
    "subtasks": [
        {{"task": "description", "worker": "worker_name", "priority": 1}},
        {{"task": "description", "worker": "worker_name", "priority": 2}}
    ]
}}"""

    SYNTHESIZE_PROMPT = """Synthesize these worker results into a final cohesive output.

Original task: {task}

Worker results:
{results}

Provide a complete, well-organized final answer that integrates all worker outputs."""

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.workers: dict[str, tuple[Any, str]] = {}

    def add_worker(self, name: str, agent: Any, description: str) -> OrchestratorWorker:
        self.workers[name] = (agent, description)
        return self

    async def run(self, task: str) -> str:
        """Decompose, delegate, and synthesize."""
        # Decompose
        workers_text = "\n".join(f"- {n}: {d}" for n, (_, d) in self.workers.items())
        decompose = self.DECOMPOSE_PROMPT.format(task=task, workers=workers_text)
        plan_response = await self.orchestrator.run(decompose)

        try:
            plan = json.loads(plan_response)
            subtasks = plan.get("subtasks", [])
        except (json.JSONDecodeError, ValueError):
            # Fallback: send entire task to first available worker
            first_worker = list(self.workers.keys())[0]
            subtasks = [{"task": task, "worker": first_worker, "priority": 1}]

        # Sort by priority
        subtasks.sort(key=lambda x: x.get("priority", 99))

        # Execute
        results: list[WorkerResult] = []
        for st in subtasks:
            worker_name = st.get("worker", "")
            if worker_name not in self.workers:
                worker_name = list(self.workers.keys())[0]  # Fallback

            agent, _ = self.workers[worker_name]
            start = time.time()
            result = await agent.run(st["task"])
            duration = (time.time() - start) * 1000

            results.append(WorkerResult(
                worker_name=worker_name,
                task=st["task"],
                result=result,
                duration_ms=duration,
            ))

        # Synthesize
        results_text = "\n\n".join(
            f"[{r.worker_name}] Task: {r.task}\nResult: {r.result}"
            for r in results
        )
        synthesize = self.SYNTHESIZE_PROMPT.format(task=task, results=results_text)
        final = await self.orchestrator.run(synthesize)

        return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. Parallel Guardrails — Non-Blocking Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GuardrailMode(str, Enum):
    BLOCKING = "blocking"      # Run guardrail first, block if triggered
    NON_BLOCKING = "non_blocking"  # Run in parallel, report after
    PASSIVE = "passive"        # Log only, never block


@dataclass
class GuardrailCheckResult:
    """Result of a guardrail check."""
    name: str
    passed: bool
    mode: GuardrailMode
    message: str = ""
    duration_ms: float = 0.0


class ParallelGuardrails:
    """Run multiple guardrails in parallel with configurable modes.

    Like OpenAI Agents SDK guardrails:
    - BLOCKING: Runs before LLM, blocks if triggered (saves tokens)
    - NON_BLOCKING: Runs in parallel with LLM for best latency
    - PASSIVE: Logs only, never blocks

    Usage:
        from duxx_ai.core.patterns import ParallelGuardrails, GuardrailMode

        pg = ParallelGuardrails()
        pg.add("pii_check", pii_filter_fn, mode=GuardrailMode.BLOCKING)
        pg.add("toxicity", toxicity_fn, mode=GuardrailMode.NON_BLOCKING)
        pg.add("cost_log", cost_logger_fn, mode=GuardrailMode.PASSIVE)

        # Check input before LLM call
        results = await pg.check_input("user message")
        if pg.any_blocked(results):
            return "Input blocked by guardrail"

        # Check output after LLM call (non-blocking ran in parallel)
        results = await pg.check_output("llm response")
    """

    def __init__(self):
        self._guardrails: list[tuple[str, Callable, GuardrailMode]] = []

    def add(self, name: str, check_fn: Callable, mode: GuardrailMode = GuardrailMode.BLOCKING) -> ParallelGuardrails:
        """Add a guardrail check function. fn(text) -> (passed: bool, message: str)"""
        self._guardrails.append((name, check_fn, mode))
        return self

    async def check_input(self, text: str) -> list[GuardrailCheckResult]:
        """Run input guardrails. BLOCKING runs first, then NON_BLOCKING in parallel."""
        results = []

        # Run blocking first
        for name, fn, mode in self._guardrails:
            if mode == GuardrailMode.BLOCKING:
                start = time.time()
                try:
                    if asyncio.iscoroutinefunction(fn):
                        passed, msg = await fn(text)
                    else:
                        passed, msg = fn(text)
                except Exception as e:
                    passed, msg = False, str(e)
                results.append(GuardrailCheckResult(
                    name=name, passed=passed, mode=mode, message=msg,
                    duration_ms=(time.time() - start) * 1000,
                ))
                if not passed:
                    return results  # Short-circuit on blocking failure

        # Run non-blocking in parallel
        async def _run(name, fn, mode):
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(fn):
                    passed, msg = await fn(text)
                else:
                    passed, msg = fn(text)
            except Exception as e:
                passed, msg = False, str(e)
            return GuardrailCheckResult(
                name=name, passed=passed, mode=mode, message=msg,
                duration_ms=(time.time() - start) * 1000,
            )

        non_blocking = [
            _run(name, fn, mode) for name, fn, mode in self._guardrails
            if mode in (GuardrailMode.NON_BLOCKING, GuardrailMode.PASSIVE)
        ]
        if non_blocking:
            nb_results = await asyncio.gather(*non_blocking)
            results.extend(nb_results)

        return results

    async def check_output(self, text: str) -> list[GuardrailCheckResult]:
        """Run output guardrails (all in parallel)."""
        async def _run(name, fn, mode):
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(fn):
                    passed, msg = await fn(text)
                else:
                    passed, msg = fn(text)
            except Exception as e:
                passed, msg = False, str(e)
            return GuardrailCheckResult(
                name=name, passed=passed, mode=mode, message=msg,
                duration_ms=(time.time() - start) * 1000,
            )

        tasks = [_run(name, fn, mode) for name, fn, mode in self._guardrails]
        return await asyncio.gather(*tasks) if tasks else []

    def any_blocked(self, results: list[GuardrailCheckResult]) -> bool:
        """Check if any blocking guardrail failed."""
        return any(
            not r.passed and r.mode == GuardrailMode.BLOCKING
            for r in results
        )

    def summary(self, results: list[GuardrailCheckResult]) -> dict[str, Any]:
        """Get a summary of guardrail results."""
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "blocked": self.any_blocked(results),
            "details": [{"name": r.name, "passed": r.passed, "mode": r.mode.value, "message": r.message} for r in results],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. Agentic RAG — Agent-Driven Query Rewriting & Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RAGResult:
    """Result of an agentic RAG query."""
    original_query: str
    rewritten_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    sources_used: int = 0
    confidence: float = 0.0


class AgenticRAG:
    """Agent-driven RAG with query rewriting and multi-hop retrieval.

    Unlike static RAG, the agent:
    - Rewrites queries for better retrieval
    - Decides which knowledge bases to search
    - Performs multi-hop retrieval (search → read → search again)
    - Verifies answers against sources

    Usage:
        from duxx_ai.core.patterns import AgenticRAG

        rag = AgenticRAG(agent=researcher, retriever=my_retriever)
        result = await rag.query("What was our revenue growth in Q4 vs Q3?")
        print(result.answer)
        print(f"Sources: {result.sources_used}, Confidence: {result.confidence}")
    """

    REWRITE_PROMPT = """Rewrite this query into 2-3 search-optimized queries for a knowledge base.

Original: {query}

Consider:
- Break compound questions into simpler parts
- Use specific keywords that would appear in documents
- Include date/time references if relevant

Respond with JSON: {{"queries": ["query1", "query2", "query3"]}}"""

    ANSWER_PROMPT = """Answer the question using ONLY the provided context. If the context doesn't contain enough information, say so.

Question: {query}
Context:
{context}

Provide your answer with:
1. The answer itself
2. Which sources you used (by number)
3. Your confidence level (0.0-1.0)

Format: {{"answer": "...", "sources_used": [1, 2], "confidence": 0.X}}"""

    def __init__(
        self,
        agent: Any,
        retriever: Any = None,
        max_hops: int = 2,
        top_k: int = 5,
    ):
        self.agent = agent
        self.retriever = retriever
        self.max_hops = max_hops
        self.top_k = top_k

    async def query(self, query: str) -> RAGResult:
        """Execute agentic RAG query."""
        result = RAGResult(original_query=query, rewritten_queries=[], retrieved_chunks=[], answer="")

        # Step 1: Rewrite query
        rewrite_prompt = self.REWRITE_PROMPT.format(query=query)
        rewrite_response = await self.agent.run(rewrite_prompt)
        try:
            data = json.loads(rewrite_response)
            queries = data.get("queries", [query])
        except (json.JSONDecodeError, ValueError):
            queries = [query]
        result.rewritten_queries = queries

        # Step 2: Multi-hop retrieval
        all_chunks = []
        for hop in range(self.max_hops):
            for q in queries:
                if self.retriever:
                    try:
                        if asyncio.iscoroutinefunction(getattr(self.retriever, 'search', None)):
                            chunks = await self.retriever.search(q, top_k=self.top_k)
                        elif hasattr(self.retriever, 'search'):
                            chunks = self.retriever.search(q, top_k=self.top_k)
                        else:
                            chunks = []
                        all_chunks.extend(chunks if isinstance(chunks, list) else [])
                    except Exception:
                        pass

            # Check if we need another hop
            if all_chunks or hop >= self.max_hops - 1:
                break

        # Deduplicate chunks
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            content = str(c.get("content", c) if isinstance(c, dict) else c)
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_chunks.append({"content": content, "id": len(unique_chunks) + 1})

        result.retrieved_chunks = unique_chunks

        # Step 3: Generate answer from context
        if unique_chunks:
            context = "\n\n".join(
                f"[Source {c['id']}]: {c['content']}" for c in unique_chunks
            )
            answer_prompt = self.ANSWER_PROMPT.format(query=query, context=context)
            answer_response = await self.agent.run(answer_prompt)

            try:
                data = json.loads(answer_response)
                result.answer = data.get("answer", answer_response)
                result.sources_used = len(data.get("sources_used", []))
                result.confidence = float(data.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError):
                result.answer = answer_response
                result.sources_used = len(unique_chunks)
                result.confidence = 0.5
        else:
            result.answer = await self.agent.run(query)
            result.confidence = 0.3  # Lower confidence without RAG

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    # ReAct
    "ReActAgent", "ReActTrace", "ThoughtStep",
    # Handoffs
    "AgentHandoff", "HandoffResult",
    # Self-Improving
    "SelfImprovingAgent", "Reward", "Skill",
    # Teachable
    "TeachableAgent", "Memory",
    # Evaluator-Optimizer
    "EvaluatorOptimizer", "OptimizationResult",
    # Orchestrator-Worker
    "OrchestratorWorker", "WorkerResult",
    # Parallel Guardrails
    "ParallelGuardrails", "GuardrailMode", "GuardrailCheckResult",
    # Agentic RAG
    "AgenticRAG", "RAGResult",
]
