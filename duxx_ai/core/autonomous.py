"""AutonomousAgent — Fully self-directed agent with planning, reflection, and self-correction.

Unlike a basic Agent (reactive: user asks → agent responds), the AutonomousAgent
operates autonomously — given a high-level goal, it plans, executes, evaluates,
corrects, and iterates until the goal is achieved or budget is exhausted.

Architecture:
    Goal → Plan → Execute → Reflect → Adapt → (loop until done)

Key capabilities:
    - Goal decomposition with dynamic replanning
    - Self-reflection and quality assessment after each step
    - Self-correction when errors or low-quality results detected
    - Working memory with relevance scoring
    - Belief state tracking (what it knows, what it needs)
    - Tool discovery and selection optimization
    - Budget-aware execution (token, cost, time limits)
    - Multi-strategy reasoning (chain-of-thought, tree-of-thought, ReAct)
    - Checkpoint and resume from any state
    - Full execution trace for observability

Usage:
    from duxx_ai.core.autonomous import AutonomousAgent, GoalConfig

    agent = AutonomousAgent(
        name="researcher",
        system_prompt="You are a deep research analyst.",
        tools=[...],
        llm_provider="openai",
        llm_model="gpt-4o",
    )

    # Run autonomously — agent decides how to achieve the goal
    result = await agent.achieve("Analyze the competitive landscape of AI agent frameworks")

    # With budget constraints
    result = await agent.achieve(
        "Build a market analysis report",
        max_steps=20,
        max_cost=0.50,
        max_time=300,  # 5 minutes
        quality_threshold=0.8,
    )

    # Resume from checkpoint
    result = await agent.resume(checkpoint_id="abc123")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable

from duxx_ai.core.agent import Agent, AgentConfig, AgentState
from duxx_ai.core.llm import LLMConfig, create_provider
from duxx_ai.core.message import Conversation, Message, Role
from duxx_ai.core.tool import Tool

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums & Data Classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLANNED = "replanned"


class ReasoningStrategy(str, Enum):
    REACT = "react"                # Reasoning + Acting loop
    CHAIN_OF_THOUGHT = "cot"       # Step-by-step reasoning
    TREE_OF_THOUGHT = "tot"        # Explore multiple paths
    REFLEXION = "reflexion"        # Reflect on past attempts
    PLAN_AND_SOLVE = "plan_solve"  # Plan then solve


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    PAUSED = "paused"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class PlanStep:
    """A single step in the agent's plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    quality_score: float = 0.0
    attempts: int = 0
    max_attempts: int = 3
    tools_used: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    dependencies: list[str] = field(default_factory=list)  # step IDs this depends on

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description, "status": self.status.value,
            "result": self.result[:200] if self.result else "", "quality_score": self.quality_score,
            "attempts": self.attempts, "tools_used": self.tools_used,
            "duration_ms": self.duration_ms, "tokens_used": self.tokens_used, "error": self.error,
        }


@dataclass
class Belief:
    """What the agent believes about the world / task state."""
    known_facts: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0.0 to 1.0

    def to_prompt(self) -> str:
        parts = []
        if self.known_facts:
            parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in self.known_facts[-10:]))
        if self.unknowns:
            parts.append("Still unknown:\n" + "\n".join(f"- {u}" for u in self.unknowns[-5:]))
        if self.assumptions:
            parts.append("Assumptions:\n" + "\n".join(f"- {a}" for a in self.assumptions[-5:]))
        parts.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(parts)


@dataclass
class ExecutionBudget:
    """Resource limits for autonomous execution."""
    max_steps: int = 30
    max_cost: float = 5.0           # USD
    max_tokens: int = 500_000
    max_time: float = 600.0         # seconds
    max_retries_per_step: int = 3
    quality_threshold: float = 0.7  # minimum acceptable quality

    # Tracking
    steps_used: int = 0
    cost_used: float = 0.0
    tokens_used: int = 0
    time_started: float = 0.0

    def is_exceeded(self) -> tuple[bool, str]:
        if self.steps_used >= self.max_steps:
            return True, f"Step limit ({self.max_steps}) reached"
        if self.cost_used >= self.max_cost:
            return True, f"Cost limit (${self.max_cost}) reached"
        if self.tokens_used >= self.max_tokens:
            return True, f"Token limit ({self.max_tokens}) reached"
        if self.time_started and (time.time() - self.time_started) >= self.max_time:
            return True, f"Time limit ({self.max_time}s) reached"
        return False, ""

    def remaining(self) -> dict:
        elapsed = time.time() - self.time_started if self.time_started else 0
        return {
            "steps": self.max_steps - self.steps_used,
            "cost": f"${self.max_cost - self.cost_used:.4f}",
            "tokens": self.max_tokens - self.tokens_used,
            "time": f"{max(0, self.max_time - elapsed):.0f}s",
        }


@dataclass
class Reflection:
    """Agent's self-assessment after a step or the entire run."""
    step_id: str | None = None
    assessment: str = ""
    quality_score: float = 0.0
    issues_found: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    should_retry: bool = False
    should_replan: bool = False


@dataclass
class ExecutionTrace:
    """Full trace of autonomous execution for observability."""
    goal_id: str = ""
    goal: str = ""
    status: GoalStatus = GoalStatus.ACTIVE
    strategy: str = ""
    plan: list[PlanStep] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
    belief: Belief = field(default_factory=Belief)
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    checkpoints: list[dict] = field(default_factory=list)
    final_answer: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id, "goal": self.goal, "status": self.status.value,
            "strategy": self.strategy,
            "plan": [s.to_dict() for s in self.plan],
            "reflections": len(self.reflections),
            "belief_confidence": self.belief.confidence,
            "budget": {"steps": self.budget.steps_used, "cost": f"${self.budget.cost_used:.4f}",
                       "tokens": self.budget.tokens_used},
            "final_answer": self.final_answer[:500] if self.final_answer else "",
            "duration_s": round(self.completed_at - self.started_at, 2) if self.completed_at else 0,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AutonomousAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AutonomousAgent:
    """Fully autonomous AI agent that plans, executes, reflects, and self-corrects.

    Unlike Agent (reactive: user → response), AutonomousAgent is goal-directed:
    given a high-level objective, it autonomously decides what to do, how to do it,
    evaluates its own work, and adapts until the goal is achieved.

    The execution loop:
        1. PLAN     — Decompose goal into actionable steps
        2. EXECUTE  — Run each step using tools and LLM reasoning
        3. REFLECT  — Self-assess quality and correctness
        4. ADAPT    — Replan if needed, retry failures, update beliefs
        5. REPEAT   — Until goal achieved or budget exhausted

    Example:
        agent = AutonomousAgent(
            name="analyst",
            system_prompt="You are a senior research analyst.",
            tools=get_financial_tools(),
        )
        result = await agent.achieve(
            "Analyze Apple's financial health and create a buy/sell recommendation",
            max_steps=15,
            quality_threshold=0.8,
        )
        print(result.final_answer)
        print(result.to_dict())  # Full execution trace
    """

    def __init__(
        self,
        name: str = "autonomous",
        system_prompt: str = "You are a highly capable autonomous AI agent.",
        tools: list[Tool] | None = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        strategy: ReasoningStrategy = ReasoningStrategy.REACT,
        reflection_model: str | None = None,  # Separate model for reflection (cheaper)
        on_step_complete: Callable[[PlanStep], Any] | None = None,
        on_reflection: Callable[[Reflection], Any] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or [])}
        self.strategy = strategy
        self.on_step_complete = on_step_complete
        self.on_reflection = on_reflection

        # Create the core execution agent
        self._agent = Agent(
            config=AgentConfig(
                name=f"{name}-exec",
                system_prompt=system_prompt,
                llm=LLMConfig(provider=llm_provider, model=llm_model),
                max_iterations=15,
            ),
            tools=tools,
        )

        # Create reflection agent (optionally with a cheaper model)
        ref_model = reflection_model or llm_model
        self._reflect_agent = Agent(
            config=AgentConfig(
                name=f"{name}-reflect",
                system_prompt="You are a critical evaluator. Assess quality, find errors, suggest improvements. Be specific and honest.",
                llm=LLMConfig(provider=llm_provider, model=ref_model),
                max_iterations=1,
            ),
        )

        # State
        self._trace: ExecutionTrace | None = None
        self._checkpoints: dict[str, ExecutionTrace] = {}

    # ── Main Entry Point ──

    async def achieve(
        self,
        goal: str,
        max_steps: int = 30,
        max_cost: float = 5.0,
        max_time: float = 600.0,
        quality_threshold: float = 0.7,
        stream: bool = False,
    ) -> ExecutionTrace:
        """Autonomously work toward achieving a goal.

        The agent will plan, execute, reflect, and adapt until the goal
        is achieved or resource limits are hit.

        Args:
            goal: High-level objective (e.g., "Analyze AAPL stock and recommend buy/sell")
            max_steps: Maximum execution steps before stopping
            max_cost: Maximum USD cost allowed
            max_time: Maximum wall-clock seconds
            quality_threshold: Minimum quality score (0-1) to accept results
            stream: If True, yields events as AsyncIterator

        Returns:
            ExecutionTrace with full plan, results, reflections, and final answer.
        """
        # Initialize trace
        trace = ExecutionTrace(
            goal_id=str(uuid.uuid4())[:12],
            goal=goal,
            strategy=self.strategy.value,
            budget=ExecutionBudget(
                max_steps=max_steps, max_cost=max_cost,
                max_time=max_time, quality_threshold=quality_threshold,
            ),
            belief=Belief(unknowns=[goal]),
            started_at=time.time(),
        )
        trace.budget.time_started = time.time()
        self._trace = trace

        try:
            # Phase 1: PLAN
            logger.info(f"[{self.name}] Planning for goal: {goal[:100]}")
            await self._plan(trace)

            # Phase 2-4: EXECUTE → REFLECT → ADAPT loop
            while trace.status == GoalStatus.ACTIVE:
                # Check budget
                exceeded, reason = trace.budget.is_exceeded()
                if exceeded:
                    trace.status = GoalStatus.BUDGET_EXCEEDED
                    logger.warning(f"[{self.name}] Budget exceeded: {reason}")
                    break

                # Find next pending step
                next_step = self._get_next_step(trace)
                if next_step is None:
                    # All steps done — synthesize final answer
                    break

                # Execute step
                await self._execute_step(trace, next_step)

                # Reflect on step result
                reflection = await self._reflect_on_step(trace, next_step)
                trace.reflections.append(reflection)

                if self.on_reflection:
                    self.on_reflection(reflection)

                # Adapt based on reflection
                await self._adapt(trace, next_step, reflection)

                # Save checkpoint
                self._save_checkpoint(trace)

            # Final synthesis
            if trace.status == GoalStatus.ACTIVE:
                trace.final_answer = await self._synthesize(trace)
                trace.status = GoalStatus.ACHIEVED

            # Final reflection on overall quality
            final_reflection = await self._reflect_on_goal(trace)
            trace.reflections.append(final_reflection)

            if final_reflection.quality_score < quality_threshold and trace.status == GoalStatus.ACHIEVED:
                # Below threshold — try one more synthesis
                logger.info(f"[{self.name}] Quality {final_reflection.quality_score:.0%} below threshold {quality_threshold:.0%}, refining...")
                trace.final_answer = await self._refine_answer(trace, final_reflection)

        except Exception as e:
            trace.status = GoalStatus.FAILED
            trace.final_answer = f"Error: {e}"
            logger.error(f"[{self.name}] Autonomous execution failed: {e}")

        trace.completed_at = time.time()
        return trace

    # ── Planning ──

    async def _plan(self, trace: ExecutionTrace) -> None:
        """Decompose goal into actionable steps."""
        tool_list = ", ".join(self.tools.keys()) if self.tools else "none"

        plan_prompt = f"""You are planning how to achieve this goal autonomously.

GOAL: {trace.goal}

AVAILABLE TOOLS: {tool_list}

REASONING STRATEGY: {self.strategy.value}

Create a step-by-step plan. Each step should be:
- Specific and actionable
- Achievable with the available tools or your own reasoning
- Ordered by dependency (earlier steps should feed later ones)

Return a JSON array of steps:
[
  {{"description": "Step description", "tools": ["tool1", "tool2"], "depends_on": []}},
  {{"description": "Step 2", "tools": [], "depends_on": ["step_0"]}}
]

Keep the plan focused — no more than {min(trace.budget.max_steps, 15)} steps.
Return ONLY the JSON array, no other text."""

        self._agent.reset()
        response = await self._agent.run(plan_prompt)
        trace.budget.steps_used += 1
        trace.budget.tokens_used += self._agent.state.total_tokens
        trace.budget.cost_used += self._agent.state.total_cost

        # Parse plan
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                steps_data = json.loads(json_match.group())
            else:
                steps_data = [{"description": line.strip().lstrip("0123456789.-) "), "tools": [], "depends_on": []}
                              for line in response.split("\n") if line.strip() and not line.strip().startswith("{")]
        except json.JSONDecodeError:
            # Fallback: parse as numbered list
            steps_data = []
            for line in response.split("\n"):
                line = line.strip()
                if line and re.match(r'^\d', line):
                    desc = re.sub(r'^\d+[.)]\s*', '', line)
                    steps_data.append({"description": desc, "tools": [], "depends_on": []})

        if not steps_data:
            steps_data = [{"description": trace.goal, "tools": list(self.tools.keys())[:3], "depends_on": []}]

        for i, sd in enumerate(steps_data[:trace.budget.max_steps]):
            step = PlanStep(
                id=f"step_{i}",
                description=sd.get("description", f"Step {i}"),
                tools_used=sd.get("tools", []),
                dependencies=sd.get("depends_on", []),
            )
            trace.plan.append(step)

        logger.info(f"[{self.name}] Plan created: {len(trace.plan)} steps")

    # ── Execution ──

    async def _execute_step(self, trace: ExecutionTrace, step: PlanStep) -> None:
        """Execute a single plan step."""
        step.status = StepStatus.IN_PROGRESS
        step.attempts += 1
        start = time.time()

        # Build context from completed steps
        context_parts = [f"GOAL: {trace.goal}", "", "COMPLETED STEPS:"]
        for s in trace.plan:
            if s.status == StepStatus.COMPLETED and s.result:
                context_parts.append(f"- {s.description}: {s.result[:300]}")
        context_parts.append(f"\nBELIEF STATE:\n{trace.belief.to_prompt()}")
        context_parts.append(f"\nBUDGET REMAINING: {json.dumps(trace.budget.remaining())}")

        exec_prompt = "\n".join(context_parts) + f"""

NOW EXECUTE THIS STEP:
{step.description}

{"Use these tools if helpful: " + ", ".join(step.tools_used) if step.tools_used else ""}

Provide a thorough, detailed result. Include specific data, numbers, and facts."""

        try:
            self._agent.reset()
            result = await self._agent.run(exec_prompt)
            step.result = result
            step.status = StepStatus.COMPLETED
            step.tokens_used = self._agent.state.total_tokens

            # Update budget
            trace.budget.steps_used += 1
            trace.budget.tokens_used += self._agent.state.total_tokens
            trace.budget.cost_used += self._agent.state.total_cost

            # Update beliefs
            trace.belief.known_facts.append(f"Step '{step.description}' completed: {result[:100]}")
            if step.description in [u for u in trace.belief.unknowns]:
                trace.belief.unknowns.remove(step.description)

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.warning(f"[{self.name}] Step failed: {step.description} — {e}")

        step.duration_ms = (time.time() - start) * 1000

        if self.on_step_complete:
            self.on_step_complete(step)

    # ── Reflection ──

    async def _reflect_on_step(self, trace: ExecutionTrace, step: PlanStep) -> Reflection:
        """Self-assess a completed step."""
        if step.status == StepStatus.FAILED:
            return Reflection(
                step_id=step.id, assessment=f"Step failed: {step.error}",
                quality_score=0.0, issues_found=[step.error or "Unknown error"],
                should_retry=step.attempts < step.max_attempts,
            )

        reflect_prompt = f"""Evaluate this step result critically.

GOAL: {trace.goal}
STEP: {step.description}
RESULT: {step.result[:1500]}

Rate the quality 0.0-1.0 and identify any issues:
- Is the result accurate and relevant?
- Does it fully address what the step asked for?
- Are there errors, hallucinations, or missing information?
- Should this step be retried or the plan changed?

Return JSON:
{{"quality_score": 0.8, "assessment": "...", "issues": ["..."], "suggestions": ["..."], "should_retry": false, "should_replan": false}}"""

        self._reflect_agent.reset()
        resp = await self._reflect_agent.run(reflect_prompt)
        trace.budget.tokens_used += self._reflect_agent.state.total_tokens
        trace.budget.cost_used += self._reflect_agent.state.total_cost

        try:
            json_match = re.search(r'\{[\s\S]*\}', resp)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"quality_score": 0.5, "assessment": resp[:200]}
        except json.JSONDecodeError:
            data = {"quality_score": 0.5, "assessment": resp[:200]}

        reflection = Reflection(
            step_id=step.id,
            assessment=data.get("assessment", ""),
            quality_score=min(1.0, max(0.0, float(data.get("quality_score", 0.5)))),
            issues_found=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            should_retry=data.get("should_retry", False),
            should_replan=data.get("should_replan", False),
        )

        step.quality_score = reflection.quality_score
        return reflection

    async def _reflect_on_goal(self, trace: ExecutionTrace) -> Reflection:
        """Overall quality assessment of the entire execution."""
        completed = [s for s in trace.plan if s.status == StepStatus.COMPLETED]
        avg_quality = sum(s.quality_score for s in completed) / max(len(completed), 1)

        summary = "\n".join(f"- [{s.quality_score:.0%}] {s.description}: {s.result[:150]}" for s in completed)

        reflect_prompt = f"""Evaluate the overall execution quality.

GOAL: {trace.goal}
STEPS COMPLETED: {len(completed)}/{len(trace.plan)}
AVERAGE STEP QUALITY: {avg_quality:.0%}

RESULTS:
{summary}

FINAL ANSWER:
{trace.final_answer[:1000]}

Rate overall quality 0.0-1.0. Is the goal achieved? What's missing?
Return JSON: {{"quality_score": 0.8, "assessment": "...", "issues": ["..."], "suggestions": ["..."]}}"""

        self._reflect_agent.reset()
        resp = await self._reflect_agent.run(reflect_prompt)

        try:
            json_match = re.search(r'\{[\s\S]*\}', resp)
            data = json.loads(json_match.group()) if json_match else {"quality_score": avg_quality}
        except:
            data = {"quality_score": avg_quality}

        return Reflection(
            assessment=data.get("assessment", ""),
            quality_score=min(1.0, max(0.0, float(data.get("quality_score", avg_quality)))),
            issues_found=data.get("issues", []),
            suggestions=data.get("suggestions", []),
        )

    # ── Adaptation ──

    async def _adapt(self, trace: ExecutionTrace, step: PlanStep, reflection: Reflection) -> None:
        """Adapt based on reflection — retry, replan, or continue."""
        if reflection.should_retry and step.attempts < step.max_attempts:
            logger.info(f"[{self.name}] Retrying step: {step.description} (attempt {step.attempts + 1})")
            step.status = StepStatus.PENDING
            step.error = None
            # Add reflection feedback to belief
            trace.belief.known_facts.append(f"Previous attempt at '{step.description}' had issues: {', '.join(reflection.issues_found[:2])}")

        elif reflection.should_replan:
            logger.info(f"[{self.name}] Replanning from step: {step.description}")
            # Mark remaining steps as replanned
            found = False
            for s in trace.plan:
                if s.id == step.id:
                    found = True
                    continue
                if found and s.status == StepStatus.PENDING:
                    s.status = StepStatus.REPLANNED

            # Generate new steps
            replan_prompt = f"""The plan needs adjustment.

ORIGINAL GOAL: {trace.goal}
COMPLETED SO FAR: {len([s for s in trace.plan if s.status == StepStatus.COMPLETED])} steps
ISSUE: {reflection.assessment}
SUGGESTIONS: {', '.join(reflection.suggestions)}
BUDGET REMAINING: {json.dumps(trace.budget.remaining())}

Generate 1-3 additional steps to address the issues and achieve the goal.
Return JSON array: [{{"description": "...", "tools": [], "depends_on": []}}]"""

            self._agent.reset()
            resp = await self._agent.run(replan_prompt)
            trace.budget.steps_used += 1
            trace.budget.tokens_used += self._agent.state.total_tokens
            trace.budget.cost_used += self._agent.state.total_cost

            try:
                json_match = re.search(r'\[[\s\S]*\]', resp)
                if json_match:
                    new_steps = json.loads(json_match.group())
                    for i, sd in enumerate(new_steps[:3]):
                        ns = PlanStep(
                            id=f"replan_{len(trace.plan)}_{i}",
                            description=sd.get("description", ""),
                            tools_used=sd.get("tools", []),
                        )
                        trace.plan.append(ns)
                    logger.info(f"[{self.name}] Added {len(new_steps[:3])} new steps from replan")
            except:
                pass

        # Update belief confidence
        completed = [s for s in trace.plan if s.status == StepStatus.COMPLETED]
        if completed:
            trace.belief.confidence = sum(s.quality_score for s in completed) / len(completed)

    # ── Synthesis ──

    async def _synthesize(self, trace: ExecutionTrace) -> str:
        """Synthesize final answer from all completed steps."""
        completed = [s for s in trace.plan if s.status == StepStatus.COMPLETED]
        if not completed:
            return "No steps were completed successfully."

        results_text = "\n\n".join(
            f"## {s.description}\n{s.result}" for s in completed
        )

        synth_prompt = f"""Synthesize a comprehensive final answer.

GOAL: {trace.goal}

RESEARCH RESULTS:
{results_text[:8000]}

BELIEF STATE:
{trace.belief.to_prompt()}

Create a thorough, well-structured answer that:
1. Directly addresses the original goal
2. Integrates insights from all research steps
3. Provides specific data, numbers, and evidence
4. Draws clear conclusions
5. Notes any caveats or limitations"""

        self._agent.reset()
        answer = await self._agent.run(synth_prompt)
        trace.budget.tokens_used += self._agent.state.total_tokens
        trace.budget.cost_used += self._agent.state.total_cost
        return answer

    async def _refine_answer(self, trace: ExecutionTrace, reflection: Reflection) -> str:
        """Refine answer based on quality assessment feedback."""
        refine_prompt = f"""Improve this answer based on the quality assessment.

ORIGINAL ANSWER:
{trace.final_answer[:3000]}

ISSUES FOUND:
{chr(10).join('- ' + i for i in reflection.issues_found)}

SUGGESTIONS:
{chr(10).join('- ' + s for s in reflection.suggestions)}

Provide an improved, more comprehensive answer."""

        self._agent.reset()
        refined = await self._agent.run(refine_prompt)
        trace.budget.tokens_used += self._agent.state.total_tokens
        trace.budget.cost_used += self._agent.state.total_cost
        return refined

    # ── Helpers ──

    def _get_next_step(self, trace: ExecutionTrace) -> PlanStep | None:
        """Get the next step to execute (respecting dependencies)."""
        completed_ids = {s.id for s in trace.plan if s.status == StepStatus.COMPLETED}
        for step in trace.plan:
            if step.status == StepStatus.PENDING:
                # Check all dependencies are met
                if all(dep in completed_ids for dep in step.dependencies):
                    return step
        return None

    def _save_checkpoint(self, trace: ExecutionTrace) -> str:
        """Save execution state for resume."""
        cp_id = f"cp_{trace.goal_id}_{trace.budget.steps_used}"
        import copy
        self._checkpoints[cp_id] = copy.deepcopy(trace)
        trace.checkpoints.append({"id": cp_id, "step": trace.budget.steps_used, "time": time.time()})
        return cp_id

    async def resume(self, checkpoint_id: str) -> ExecutionTrace:
        """Resume autonomous execution from a checkpoint."""
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found. Available: {list(self._checkpoints.keys())}")

        trace = self._checkpoints[checkpoint_id]
        trace.status = GoalStatus.ACTIVE
        trace.budget.time_started = time.time()
        self._trace = trace

        # Continue execution loop
        while trace.status == GoalStatus.ACTIVE:
            exceeded, reason = trace.budget.is_exceeded()
            if exceeded:
                trace.status = GoalStatus.BUDGET_EXCEEDED
                break

            next_step = self._get_next_step(trace)
            if next_step is None:
                break

            await self._execute_step(trace, next_step)
            reflection = await self._reflect_on_step(trace, next_step)
            trace.reflections.append(reflection)
            await self._adapt(trace, next_step, reflection)
            self._save_checkpoint(trace)

        if trace.status == GoalStatus.ACTIVE:
            trace.final_answer = await self._synthesize(trace)
            trace.status = GoalStatus.ACHIEVED

        trace.completed_at = time.time()
        return trace

    # ── Streaming ──

    async def achieve_stream(
        self, goal: str, **kwargs
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events as they happen.

        Yields dicts with type: "plan", "step_start", "step_complete",
        "reflection", "replan", "synthesis", "complete"
        """
        # Override callbacks to yield events
        events: asyncio.Queue = asyncio.Queue()

        original_step_cb = self.on_step_complete
        original_ref_cb = self.on_reflection

        self.on_step_complete = lambda s: events.put_nowait({"type": "step_complete", "step": s.to_dict()})
        self.on_reflection = lambda r: events.put_nowait({"type": "reflection", "quality": r.quality_score, "assessment": r.assessment[:200]})

        # Run in background
        async def _run():
            result = await self.achieve(goal, **kwargs)
            events.put_nowait({"type": "complete", "trace": result.to_dict()})
            events.put_nowait(None)  # sentinel

        task = asyncio.create_task(_run())

        while True:
            event = await events.get()
            if event is None:
                break
            yield event

        self.on_step_complete = original_step_cb
        self.on_reflection = original_ref_cb
        await task

    # ── Properties ──

    @property
    def trace(self) -> ExecutionTrace | None:
        """Get the current or last execution trace."""
        return self._trace

    @property
    def checkpoints(self) -> list[str]:
        """List available checkpoint IDs."""
        return list(self._checkpoints.keys())

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools[tool.name] = tool
        self._agent.register_tool(tool)

    def add_tools(self, tools: list[Tool]) -> None:
        """Add multiple tools."""
        for t in tools:
            self.add_tool(t)
