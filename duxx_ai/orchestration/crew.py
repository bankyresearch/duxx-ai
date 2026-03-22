"""Role-based crew orchestration — multiple agents collaborating on tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class CrewAgent(BaseModel):
    """An agent within a crew, defined by role and capabilities."""
    name: str
    role: str
    goal: str
    backstory: str = ""
    tools: list[str] = Field(default_factory=list)
    llm_config: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 5
    allow_delegation: bool = True
    _agent: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def bind_agent(self, agent: Any) -> CrewAgent:
        object.__setattr__(self, "_agent", agent)
        return self


class Task(BaseModel):
    id: str
    description: str
    expected_output: str = ""
    assigned_to: str = ""
    dependencies: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    status: str = "pending"


class CrewResult(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Crew:
    """Orchestrates multiple agents working together on a set of tasks."""

    def __init__(
        self,
        name: str = "crew",
        agents: list[CrewAgent] | None = None,
        tasks: list[Task] | None = None,
        strategy: str = "sequential",  # sequential, parallel, hierarchical
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.agents: dict[str, CrewAgent] = {}
        self.tasks: list[Task] = tasks or []
        self.strategy = strategy
        self.verbose = verbose
        self.shared_context: dict[str, Any] = {}

        for a in agents or []:
            self.add_agent(a)

    def add_agent(self, agent: CrewAgent) -> Crew:
        self.agents[agent.name] = agent
        return self

    def add_task(self, task: Task) -> Crew:
        self.tasks.append(task)
        return self

    async def run(self, input_data: dict[str, Any] | None = None) -> CrewResult:
        self.shared_context = input_data or {}

        if self.strategy == "sequential":
            return await self._run_sequential()
        elif self.strategy == "parallel":
            return await self._run_parallel()
        elif self.strategy == "hierarchical":
            return await self._run_hierarchical()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _run_sequential(self) -> CrewResult:
        for task in self.tasks:
            task.status = "running"
            # Wait for dependencies
            for dep_id in task.dependencies:
                dep = next((t for t in self.tasks if t.id == dep_id), None)
                if dep and dep.result:
                    task.context[dep_id] = dep.result

            task.context.update(self.shared_context)
            result = await self._execute_task(task)
            task.result = result
            task.status = "completed"
            self.shared_context[task.id] = result

            if self.verbose:
                logger.info(f"Task '{task.id}' completed by '{task.assigned_to}'")

        return CrewResult(
            tasks=self.tasks,
            final_output=self.tasks[-1].result if self.tasks else "",
        )

    async def _run_parallel(self) -> CrewResult:
        # Group tasks by dependency level
        levels = self._topological_sort()
        for level in levels:
            coros = []
            for task in level:
                task.status = "running"
                task.context.update(self.shared_context)
                coros.append(self._execute_task_and_store(task))
            await asyncio.gather(*coros)

        return CrewResult(
            tasks=self.tasks,
            final_output=self.tasks[-1].result if self.tasks else "",
        )

    async def _run_hierarchical(self) -> CrewResult:
        """Manager agent delegates and reviews work from other agents.

        The first agent acts as the manager. All tasks are executed by their
        assigned agents (or sequentially if unassigned), then the manager
        reviews the collected outputs and produces a final consolidated result.
        """
        agents_list = list(self.agents.values())
        if not agents_list:
            return CrewResult(final_output="No agents in crew")

        manager = agents_list[0]

        # Step 1: Execute all tasks sequentially, collecting results
        task_outputs: dict[str, str] = {}
        for task in self.tasks:
            task.status = "running"
            # Inject dependency results into context
            for dep_id in task.dependencies:
                dep = next((t for t in self.tasks if t.id == dep_id), None)
                if dep and dep.result:
                    task.context[dep_id] = dep.result

            task.context.update(self.shared_context)
            result = await self._execute_task(task)
            task.result = result
            task.status = "completed"
            self.shared_context[task.id] = result
            task_outputs[task.id] = result

            if self.verbose:
                logger.info(f"Task '{task.id}' completed by '{task.assigned_to}'")

        # Step 2: Manager reviews all task outputs and produces a consolidated result
        review_sections = []
        for task in self.tasks:
            review_sections.append(
                f"--- Task '{task.id}' (assigned to '{task.assigned_to}') ---\n"
                f"Description: {task.description}\n"
                f"Result:\n{task.result}"
            )

        manager_prompt = self._build_task_prompt(
            Task(
                id="__manager_review__",
                description=(
                    "You are the manager of this crew. Review the outputs from all tasks below "
                    "and produce a final consolidated result that synthesizes the work.\n\n"
                    + "\n\n".join(review_sections)
                ),
                expected_output="A consolidated summary incorporating all task results.",
                assigned_to=manager.name,
                context=self.shared_context,
            ),
            manager,
        )

        try:
            bound_agent = object.__getattribute__(manager, "_agent")
        except AttributeError:
            bound_agent = None

        if bound_agent is not None:
            final_output = await bound_agent.run(manager_prompt, context=self.shared_context)
        else:
            # No bound LLM — produce a deterministic consolidated summary
            final_output = (
                f"[{manager.role}] Consolidated review:\n\n"
                + "\n\n".join(
                    f"Task '{t.id}': {t.result}" for t in self.tasks
                )
            )

        return CrewResult(
            tasks=self.tasks,
            final_output=final_output,
        )

    async def _execute_task(self, task: Task) -> str:
        agent = self.agents.get(task.assigned_to)
        if agent is None:
            return f"Error: Agent '{task.assigned_to}' not found in crew"

        try:
            bound_agent = object.__getattribute__(agent, "_agent")
        except AttributeError:
            bound_agent = None
        if bound_agent is None:
            # Return a simulated result for agents without bound LLM
            return f"[{agent.role}] Completed: {task.description}"

        prompt = self._build_task_prompt(task, agent)
        return await bound_agent.run(prompt, context=task.context)

    async def _execute_task_and_store(self, task: Task) -> None:
        result = await self._execute_task(task)
        task.result = result
        task.status = "completed"
        self.shared_context[task.id] = result

    def _build_task_prompt(self, task: Task, agent: CrewAgent) -> str:
        parts = [
            f"You are {agent.role}. {agent.backstory}",
            f"Your goal: {agent.goal}",
            f"Task: {task.description}",
        ]
        if task.expected_output:
            parts.append(f"Expected output: {task.expected_output}")
        if task.context:
            parts.append(f"Context from previous tasks: {task.context}")
        return "\n\n".join(parts)

    def _topological_sort(self) -> list[list[Task]]:
        task_map = {t.id: t for t in self.tasks}
        in_degree: dict[str, int] = {t.id: len(t.dependencies) for t in self.tasks}
        levels: list[list[Task]] = []

        while in_degree:
            level = [task_map[tid] for tid, deg in in_degree.items() if deg == 0]
            if not level:
                break
            levels.append(level)
            for t in level:
                del in_degree[t.id]
                for other_id in in_degree:
                    if t.id in task_map[other_id].dependencies:
                        in_degree[other_id] -= 1

        return levels
