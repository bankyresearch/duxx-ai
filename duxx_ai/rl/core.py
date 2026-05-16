"""RL Core — Gymnasium-style environment interface, typed models, rewards, and trajectory collection.

Provides the foundational abstractions for reinforcement learning with LLM agents:
- RLEnvironment: async reset/step/state interface
- Observation/Action: typed Pydantic models
- RewardFunction: composable reward signals
- TrajectoryBuffer: episode collection for training
- Episode: complete interaction history

Usage:
    from duxx_ai.rl.core import RLEnvironment, Observation, Action, RewardFunction, TrajectoryBuffer

    class MyEnv(RLEnvironment):
        async def reset(self): return Observation(text="Start")
        async def step(self, action): return StepResult(...)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Typed Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Observation(BaseModel):
    """What the agent observes from the environment."""
    text: str = ""                          # Primary text observation
    data: dict[str, Any] = Field(default_factory=dict)  # Structured data
    images: list[str] = Field(default_factory=list)      # Image URLs/paths
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt(self) -> str:
        """Convert observation to LLM prompt text."""
        parts = []
        if self.text:
            parts.append(self.text)
        if self.data:
            parts.append(f"Data: {json.dumps(self.data, default=str)}")
        return "\n".join(parts) if parts else "[empty observation]"


class Action(BaseModel):
    """An action the agent takes in the environment."""
    text: str = ""                          # Free-form text action
    tool_name: str | None = None            # Tool to invoke
    tool_args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str) -> Action:
        return cls(text=text)

    @classmethod
    def from_tool(cls, name: str, **kwargs) -> Action:
        return cls(tool_name=name, tool_args=kwargs)


class StepResult(BaseModel):
    """Result of a single environment step."""
    observation: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False                 # Episode hit max steps
    info: dict[str, Any] = Field(default_factory=dict)
    rewards_breakdown: dict[str, float] = Field(default_factory=dict)  # Per-component rewards


class EnvState(BaseModel):
    """Current environment state metadata."""
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RL Environment Base Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RLEnvironment(ABC):
    """Gymnasium-style async RL environment for training LLM agents.

    Subclass this and implement reset(), step(), and optionally state().

    Usage:
        class MathEnv(RLEnvironment):
            async def reset(self):
                self.target = random.randint(1, 100)
                return Observation(text=f"Guess a number between 1-100")

            async def step(self, action):
                guess = int(action.text)
                diff = abs(guess - self.target)
                if diff == 0:
                    return StepResult(observation=Observation(text="Correct!"), reward=1.0, done=True)
                hint = "higher" if guess < self.target else "lower"
                return StepResult(observation=Observation(text=f"Try {hint}"), reward=-0.1)

        env = MathEnv(max_steps=10)
        obs = await env.reset()
        result = await env.step(Action(text="42"))
    """

    def __init__(
        self,
        name: str = "environment",
        max_steps: int = 100,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.max_steps = max_steps
        self.metadata = metadata or {}
        self._state = EnvState()
        self._history: list[tuple[Action, StepResult]] = []

    @abstractmethod
    async def reset(self, **kwargs) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        ...

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        """Take an action, return (observation, reward, done, info)."""
        ...

    async def state(self) -> EnvState:
        """Get current environment state metadata."""
        self._state.elapsed_seconds = time.time() - self._state.started_at if self._state.started_at else 0
        return self._state

    async def close(self) -> None:
        """Clean up environment resources."""
        pass

    # ── Internal helpers ──

    async def _do_reset(self, **kwargs) -> Observation:
        """Internal reset with state tracking."""
        self._state = EnvState(
            episode_id=str(uuid.uuid4())[:12],
            started_at=time.time(),
        )
        self._history = []
        obs = await self.reset(**kwargs)
        return obs

    async def _do_step(self, action: Action) -> StepResult:
        """Internal step with state tracking and truncation."""
        self._state.step_count += 1
        result = await self.step(action)

        # Track rewards
        self._state.total_reward += result.reward

        # Auto-truncate at max steps
        if self._state.step_count >= self.max_steps and not result.done:
            result.truncated = True
            result.done = True

        self._state.done = result.done
        self._history.append((action, result))
        return result

    @property
    def history(self) -> list[tuple[Action, StepResult]]:
        """Get action-result history for current episode."""
        return self._history

    # ── Context manager ──

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reward Functions (Composable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RewardFunction(ABC):
    """Base class for composable reward functions.

    Subclass and implement compute(). Combine with RewardComposer.

    Usage:
        class AccuracyReward(RewardFunction):
            def compute(self, action, result, state):
                return 1.0 if result.info.get("correct") else 0.0

        class BrevityReward(RewardFunction):
            def compute(self, action, result, state):
                return max(0, 1.0 - len(action.text) / 500)

        composer = RewardComposer([
            (AccuracyReward(), 0.8),
            (BrevityReward(), 0.2),
        ])
        total = composer.compute(action, result, state)
    """

    name: str = "reward"

    @abstractmethod
    def compute(self, action: Action, result: StepResult, state: EnvState) -> float:
        """Compute reward signal. Returns float."""
        ...


class ConstantReward(RewardFunction):
    """Returns a constant reward value."""
    name = "constant"

    def __init__(self, value: float = 0.0):
        self.value = value

    def compute(self, action, result, state):
        return self.value


class SuccessReward(RewardFunction):
    """Binary reward: 1.0 on success (done=True with positive reward), else 0."""
    name = "success"

    def compute(self, action, result, state):
        return 1.0 if result.done and result.reward > 0 else 0.0


class StepPenalty(RewardFunction):
    """Small negative reward per step to encourage efficiency."""
    name = "step_penalty"

    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    def compute(self, action, result, state):
        return self.penalty


class LengthReward(RewardFunction):
    """Reward/penalize based on action text length."""
    name = "length"

    def __init__(self, target_length: int = 100, scale: float = 0.1):
        self.target = target_length
        self.scale = scale

    def compute(self, action, result, state):
        diff = abs(len(action.text) - self.target) / max(self.target, 1)
        return max(0, 1.0 - diff) * self.scale


class ToolUseReward(RewardFunction):
    """Reward for using tools (encourages tool usage)."""
    name = "tool_use"

    def __init__(self, reward_per_tool: float = 0.1):
        self.reward_per_tool = reward_per_tool

    def compute(self, action, result, state):
        return self.reward_per_tool if action.tool_name else 0.0


class CorrectnessReward(RewardFunction):
    """Check result info for 'correct' key."""
    name = "correctness"

    def compute(self, action, result, state):
        return 1.0 if result.info.get("correct", False) else 0.0


class FormatReward(RewardFunction):
    """Reward for following expected format (JSON, numbered list, etc.)."""
    name = "format"

    def __init__(self, expected_format: str = "json"):
        self.expected = expected_format

    def compute(self, action, result, state):
        text = action.text.strip()
        if self.expected == "json":
            try:
                json.loads(text)
                return 1.0
            except:
                return 0.0
        elif self.expected == "numbered_list":
            import re
            lines = [l for l in text.split("\n") if l.strip()]
            numbered = sum(1 for l in lines if re.match(r'^\d+[.)]\s', l.strip()))
            return numbered / max(len(lines), 1)
        return 0.5


class CompletionReward(RewardFunction):
    """Reward for completing the task within step budget."""
    name = "completion"

    def compute(self, action, result, state):
        if result.done and not result.truncated:
            # Bonus for finishing early
            efficiency = 1.0 - (state.step_count / max(state.step_count + 10, 1))
            return 0.5 + 0.5 * efficiency
        return 0.0


class RewardComposer:
    """Compose multiple reward functions with weights.

    Usage:
        composer = RewardComposer([
            (AccuracyReward(), 0.6),
            (BrevityReward(), 0.2),
            (StepPenalty(), 0.2),
        ])
        total, breakdown = composer.compute_with_breakdown(action, result, state)
    """

    def __init__(self, rewards: list[tuple[RewardFunction, float]]):
        self.rewards = rewards
        total_weight = sum(w for _, w in rewards)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Reward weights sum to {total_weight}, not 1.0")

    def compute(self, action: Action, result: StepResult, state: EnvState) -> float:
        """Compute weighted sum of all reward signals."""
        return sum(rf.compute(action, result, state) * w for rf, w in self.rewards)

    def compute_with_breakdown(self, action: Action, result: StepResult, state: EnvState) -> tuple[float, dict[str, float]]:
        """Compute total + per-component breakdown."""
        breakdown = {}
        total = 0.0
        for rf, w in self.rewards:
            raw = rf.compute(action, result, state)
            weighted = raw * w
            breakdown[rf.name] = raw
            total += weighted
        return total, breakdown

    def add(self, reward: RewardFunction, weight: float) -> RewardComposer:
        """Add a reward function (returns self for chaining)."""
        self.rewards.append((reward, weight))
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Trajectory Collection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Transition:
    """A single (state, action, reward, next_state, done) transition."""
    observation: str           # Text observation
    action: str                # Text action
    reward: float
    next_observation: str
    done: bool
    truncated: bool = False
    rewards_breakdown: dict[str, float] = field(default_factory=dict)
    log_prob: float | None = None       # Log probability of action (for PG methods)
    value: float | None = None          # Value estimate (for PPO)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "observation": self.observation[:200], "action": self.action[:200],
            "reward": self.reward, "done": self.done, "truncated": self.truncated,
            "log_prob": self.log_prob, "value": self.value,
        }


@dataclass
class Episode:
    """A complete episode (trajectory) of agent-environment interaction."""
    episode_id: str = ""
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    total_steps: int = 0
    success: bool = False
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def observations(self) -> list[str]:
        return [t.observation for t in self.transitions]

    @property
    def actions(self) -> list[str]:
        return [t.action for t in self.transitions]

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.transitions]

    @property
    def returns(self) -> list[float]:
        """Compute discounted returns (gamma=0.99) for each step."""
        gamma = 0.99
        returns = []
        G = 0.0
        for t in reversed(self.transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)
        return returns

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id, "total_reward": self.total_reward,
            "total_steps": self.total_steps, "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class TrajectoryBuffer:
    """Collects episodes for batch RL training.

    Usage:
        buffer = TrajectoryBuffer(max_episodes=1000)

        # Collect episodes
        for _ in range(100):
            episode = await rollout(agent, env)
            buffer.add(episode)

        # Get training batch
        batch = buffer.sample(batch_size=32)
        transitions = buffer.all_transitions()

        # Stats
        print(buffer.stats())
    """

    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []

    def add(self, episode: Episode) -> None:
        """Add an episode to the buffer."""
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)  # FIFO eviction

    def sample(self, batch_size: int) -> list[Episode]:
        """Random sample of episodes."""
        import random
        return random.sample(self.episodes, min(batch_size, len(self.episodes)))

    def all_transitions(self) -> list[Transition]:
        """Flatten all episodes into a list of transitions."""
        return [t for ep in self.episodes for t in ep.transitions]

    def best_episodes(self, n: int = 10) -> list[Episode]:
        """Get top-n episodes by total reward."""
        return sorted(self.episodes, key=lambda e: e.total_reward, reverse=True)[:n]

    def worst_episodes(self, n: int = 10) -> list[Episode]:
        """Get bottom-n episodes by total reward."""
        return sorted(self.episodes, key=lambda e: e.total_reward)[:n]

    def success_rate(self) -> float:
        """Fraction of successful episodes."""
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all episodes."""
        if not self.episodes:
            return {"episodes": 0}
        rewards = [e.total_reward for e in self.episodes]
        steps = [e.total_steps for e in self.episodes]
        return {
            "episodes": len(self.episodes),
            "success_rate": f"{self.success_rate():.1%}",
            "reward_mean": round(sum(rewards) / len(rewards), 4),
            "reward_std": round((sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5, 4),
            "reward_min": round(min(rewards), 4),
            "reward_max": round(max(rewards), 4),
            "steps_mean": round(sum(steps) / len(steps), 1),
            "total_transitions": sum(e.total_steps for e in self.episodes),
        }

    def to_training_data(self) -> list[dict[str, Any]]:
        """Convert buffer to training-ready format (prompt/completion pairs)."""
        data = []
        for ep in self.episodes:
            for t in ep.transitions:
                data.append({
                    "prompt": t.observation,
                    "completion": t.action,
                    "reward": t.reward,
                    "log_prob": t.log_prob,
                    "value": t.value,
                })
        return data

    def clear(self) -> None:
        self.episodes = []

    def __len__(self) -> int:
        return len(self.episodes)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rollout — Run agent in environment to collect episodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def rollout(
    agent_fn,
    env: RLEnvironment,
    reward_fn: RewardComposer | RewardFunction | None = None,
    max_steps: int | None = None,
    log_probs: bool = False,
) -> Episode:
    """Run one episode: agent interacts with environment until done.

    Args:
        agent_fn: Async callable(observation_text) -> action_text.
                  Can be an Agent.run method or any async function.
        env: RLEnvironment instance.
        reward_fn: Optional reward override (uses env rewards if None).
        max_steps: Override env.max_steps.
        log_probs: If True, agent_fn should return (text, log_prob) tuple.

    Returns:
        Complete Episode with all transitions.

    Usage:
        from duxx_ai.rl.core import rollout, Episode

        # Simple function agent
        async def my_agent(obs_text):
            return "my action"

        episode = await rollout(my_agent, env)

        # Duxx AI Agent
        from duxx_ai import Agent
        agent = Agent(config=AgentConfig(name="rl-agent"), tools=[...])
        episode = await rollout(agent.run, env)
    """
    steps = max_steps or env.max_steps
    start_time = time.time()

    obs = await env._do_reset()
    episode = Episode(episode_id=env._state.episode_id)

    for step in range(steps):
        # Agent decides action
        obs_text = obs.to_prompt()
        if log_probs:
            action_text, log_prob = await agent_fn(obs_text)
        else:
            action_text = await agent_fn(obs_text)
            log_prob = None

        action = Action(text=action_text) if isinstance(action_text, str) else action_text

        # Environment step
        result = await env._do_step(action)

        # Compute reward (override or use env reward)
        if reward_fn:
            state = await env.state()
            if isinstance(reward_fn, RewardComposer):
                reward, breakdown = reward_fn.compute_with_breakdown(action, result, state)
            else:
                reward = reward_fn.compute(action, result, state)
                breakdown = {reward_fn.name: reward}
            result.reward = reward
            result.rewards_breakdown = breakdown

        # Record transition
        transition = Transition(
            observation=obs_text,
            action=action_text if isinstance(action_text, str) else action.text,
            reward=result.reward,
            next_observation=result.observation.to_prompt(),
            done=result.done,
            truncated=result.truncated,
            rewards_breakdown=result.rewards_breakdown,
            log_prob=log_prob,
        )
        episode.transitions.append(transition)

        if result.done:
            break

        obs = result.observation

    # Finalize episode
    episode.total_reward = sum(t.reward for t in episode.transitions)
    episode.total_steps = len(episode.transitions)
    episode.success = any(t.reward > 0 and t.done for t in episode.transitions)
    episode.duration_seconds = time.time() - start_time

    return episode


async def batch_rollout(
    agent_fn,
    env_factory,
    n_episodes: int,
    max_concurrency: int = 5,
    reward_fn: RewardComposer | RewardFunction | None = None,
    buffer: TrajectoryBuffer | None = None,
) -> TrajectoryBuffer:
    """Run multiple episodes concurrently.

    Args:
        agent_fn: Async callable(obs_text) -> action_text.
        env_factory: Callable that creates a new RLEnvironment instance.
        n_episodes: How many episodes to collect.
        max_concurrency: Max parallel episodes.
        reward_fn: Optional reward override.
        buffer: Existing buffer to add to (creates new if None).

    Returns:
        TrajectoryBuffer with all collected episodes.

    Usage:
        buffer = await batch_rollout(
            agent_fn=agent.run,
            env_factory=lambda: CodingEnv(difficulty="easy"),
            n_episodes=100,
            max_concurrency=10,
        )
        print(buffer.stats())
    """
    import asyncio

    buf = buffer or TrajectoryBuffer()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one():
        async with semaphore:
            env = env_factory()
            try:
                ep = await rollout(agent_fn, env, reward_fn=reward_fn)
                buf.add(ep)
            finally:
                await env.close()

    await asyncio.gather(*[_one() for _ in range(n_episodes)])
    return buf
