"""RL Agent — Self-improving agent that learns from environment interaction.

Combines Duxx AI's AutonomousAgent with RL training loops.
The agent improves its behavior through experience by collecting trajectories,
computing rewards, and updating its strategy.

Usage:
    from duxx_ai.rl.agent import RLAgent
    from duxx_ai.rl.environments import CodingEnv
    from duxx_ai.rl.core import RewardComposer, CorrectnessReward, StepPenalty

    agent = RLAgent(
        name="code-learner",
        env=CodingEnv(difficulty="easy"),
        reward_fn=RewardComposer([
            (CorrectnessReward(), 0.8),
            (StepPenalty(), 0.2),
        ]),
    )

    # Train the agent
    result = await agent.train(n_episodes=100, n_epochs=3)
    print(result.to_dict())

    # Use the improved agent
    answer = await agent.act("Write a function to reverse a string")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from duxx_ai.core.agent import Agent, AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.rl.core import (
    Action,
    Episode,
    Observation,
    RewardComposer,
    RewardFunction,
    RLEnvironment,
    StepResult,
    TrajectoryBuffer,
    batch_rollout,
)
from duxx_ai.rl.training import (
    BestOfNTrainer,
    DPOTrainer,
    GRPOTrainer,
    PPOTrainer,
    REINFORCETrainer,
    TrainingConfig,
    TrainingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RLAgentConfig:
    """Configuration for RL-powered agent."""
    name: str = "rl-agent"
    system_prompt: str = "You are a helpful AI agent learning to improve through experience."
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    trainer_type: str = "grpo"            # grpo, ppo, reinforce, dpo, best_of_n
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    # Experience replay
    max_experience_buffer: int = 10000
    # Self-improvement
    self_improve_interval: int = 50       # Improve system prompt every N episodes
    best_of_n: int = 4                    # For inference-time scaling


class RLAgent:
    """Self-improving agent that learns from environment interaction via RL.

    Combines:
    - Duxx AI Agent (LLM reasoning + tools)
    - RL Environment (reset/step/state)
    - RL Training (GRPO/PPO/REINFORCE/DPO)
    - Experience buffer (learn from past episodes)
    - Self-improving system prompt (adapts based on performance)

    Training loop:
        1. Collect episodes by interacting with environment
        2. Score each episode with reward function
        3. Update agent strategy (prompt/behavior) based on rewards
        4. Repeat — agent gets better over time

    Usage:
        agent = RLAgent(
            name="math-solver",
            env=ReasoningEnv(category="math"),
        )

        # Train
        result = await agent.train(n_episodes=50)
        print(f"Success rate: {result.final_success_rate:.0%}")

        # Use improved agent
        answer = await agent.act("What is 17 * 23?")
    """

    def __init__(
        self,
        name: str = "rl-agent",
        env: RLEnvironment | Callable[[], RLEnvironment] | None = None,
        reward_fn: RewardComposer | RewardFunction | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful AI agent. Give concise, accurate answers.",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        trainer_type: str = "grpo",
        config: RLAgentConfig | None = None,
    ):
        self.config = config or RLAgentConfig(
            name=name, system_prompt=system_prompt,
            llm_provider=llm_provider, llm_model=llm_model,
            trainer_type=trainer_type,
        )
        self.env = env
        self.env_factory = env if callable(env) and not isinstance(env, RLEnvironment) else (lambda: env)
        self.reward_fn = reward_fn
        self.tools = tools or []

        # Core agent
        self._agent = Agent(
            config=AgentConfig(
                name=self.config.name,
                system_prompt=self.config.system_prompt,
                llm=LLMConfig(provider=self.config.llm_provider, model=self.config.llm_model),
                max_iterations=10,
            ),
            tools=self.tools,
        )

        # Experience buffer
        self.experience = TrajectoryBuffer(max_episodes=self.config.max_experience_buffer)

        # Performance tracking
        self._best_episodes: list[Episode] = []
        self._worst_episodes: list[Episode] = []
        self._system_prompt_history: list[str] = [self.config.system_prompt]
        self._training_results: list[TrainingResult] = []

    # ── Core Action ──

    async def act(self, observation: str) -> str:
        """Take action based on observation (inference mode)."""
        self._agent.reset()
        return await self._agent.run(observation)

    async def act_best_of_n(self, observation: str, n: int | None = None) -> str:
        """Generate N responses and return the best (if reward_fn available).

        Inference-time scaling: more samples = better quality.
        """
        n = n or self.config.best_of_n
        responses = []
        for _ in range(n):
            self._agent.reset()
            resp = await self._agent.run(observation)
            responses.append(resp)

        if self.reward_fn and self.env:
            # Score each response
            scored = []
            for resp in responses:
                action = Action(text=resp)
                result = StepResult(observation=Observation(text=observation), reward=0.0, done=True)
                from duxx_ai.rl.core import EnvState
                state = EnvState()
                if isinstance(self.reward_fn, RewardComposer):
                    score = self.reward_fn.compute(action, result, state)
                else:
                    score = self.reward_fn.compute(action, result, state)
                scored.append((resp, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        # No reward function — return longest (heuristic: more detailed)
        return max(responses, key=len)

    # ── Training ──

    async def train(
        self,
        n_episodes: int = 100,
        n_epochs: int = 3,
        trainer_type: str | None = None,
    ) -> TrainingResult:
        """Train the agent via RL in the environment.

        Args:
            n_episodes: Episodes per epoch
            n_epochs: Training epochs
            trainer_type: Override trainer (grpo, ppo, reinforce, dpo, best_of_n)

        Returns:
            TrainingResult with metrics and history.
        """
        if self.env is None:
            raise ValueError("No environment set. Pass env= to RLAgent()")

        tt = trainer_type or self.config.trainer_type
        config = TrainingConfig(
            n_episodes=n_episodes, n_epochs=n_epochs,
            **{k: v for k, v in self.config.training_config.__dict__.items()
               if k not in ('n_episodes', 'n_epochs')},
        )

        # Select trainer
        trainer_cls = {
            "grpo": GRPOTrainer,
            "ppo": PPOTrainer,
            "reinforce": REINFORCETrainer,
            "dpo": DPOTrainer,
            "best_of_n": BestOfNTrainer,
        }.get(tt, GRPOTrainer)

        trainer = trainer_cls(
            agent_fn=self.act,
            env=self.env_factory,
            reward_fn=self.reward_fn,
            config=config,
            on_episode=self._on_episode,
            on_update=self._on_update,
        )

        logger.info(f"[{self.config.name}] Training with {tt.upper()}, {n_episodes} episodes x {n_epochs} epochs")
        result = await trainer.train()

        self._training_results.append(result)
        logger.info(f"[{self.config.name}] Training complete: success={result.final_success_rate:.0%}, best={result.best_episode_reward:.3f}")

        return result

    # ── Self-Improvement ──

    async def self_improve(self) -> str:
        """Analyze experience and improve system prompt.

        Uses best/worst episodes to generate an improved system prompt.
        """
        if len(self.experience) < 10:
            return self.config.system_prompt

        best = self.experience.best_episodes(5)
        worst = self.experience.worst_episodes(5)

        # Analyze patterns
        best_actions = [t.action for ep in best for t in ep.transitions[:2]]
        worst_actions = [t.action for ep in worst for t in ep.transitions[:2]]

        improve_prompt = f"""You are optimizing an AI agent's system prompt based on performance data.

Current system prompt: {self.config.system_prompt}

BEST performing actions (high reward):
{chr(10).join(f'- {a[:100]}' for a in best_actions[:5])}

WORST performing actions (low reward):
{chr(10).join(f'- {a[:100]}' for a in worst_actions[:5])}

Stats:
- Success rate: {self.experience.success_rate():.0%}
- Avg reward: {self.experience.stats().get('reward_mean', 0):.3f}
- Episodes: {len(self.experience)}

Write an IMPROVED system prompt that:
1. Encourages the patterns seen in best actions
2. Discourages patterns seen in worst actions
3. Is concise (under 200 words)

Return ONLY the new system prompt, nothing else."""

        self._agent.reset()
        new_prompt = await self._agent.run(improve_prompt)

        # Update agent
        self.config.system_prompt = new_prompt
        self._agent.config.system_prompt = new_prompt
        self._system_prompt_history.append(new_prompt)

        logger.info(f"[{self.config.name}] Self-improved system prompt (version {len(self._system_prompt_history)})")
        return new_prompt

    # ── Callbacks ──

    def _on_episode(self, episode: Episode) -> None:
        """Called after each episode during training."""
        self.experience.add(episode)

        # Auto self-improve
        interval = self.config.self_improve_interval
        if interval > 0 and len(self.experience) % interval == 0 and len(self.experience) > 0:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.self_improve())
            except RuntimeError:
                pass  # No event loop, skip

    def _on_update(self, metrics: dict) -> None:
        """Called after each policy update."""
        logger.debug(f"[{self.config.name}] Update: {metrics}")

    # ── Evaluation ──

    async def evaluate(self, n_episodes: int = 20) -> dict[str, Any]:
        """Evaluate current agent performance without training.

        Returns:
            Dict with success_rate, avg_reward, avg_steps, etc.
        """
        if self.env is None:
            raise ValueError("No environment set.")

        eval_buffer = await batch_rollout(
            agent_fn=self.act,
            env_factory=self.env_factory,
            n_episodes=n_episodes,
            max_concurrency=5,
            reward_fn=self.reward_fn,
        )

        stats = eval_buffer.stats()
        stats["agent"] = self.config.name
        stats["model"] = self.config.llm_model
        stats["prompt_version"] = len(self._system_prompt_history)
        return stats

    # ── Serialization ──

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive agent stats."""
        return {
            "name": self.config.name,
            "model": self.config.llm_model,
            "trainer": self.config.trainer_type,
            "total_experience": len(self.experience),
            "experience_stats": self.experience.stats() if self.experience else {},
            "prompt_versions": len(self._system_prompt_history),
            "training_runs": len(self._training_results),
            "current_prompt": self.config.system_prompt[:200] + "..." if len(self.config.system_prompt) > 200 else self.config.system_prompt,
        }
