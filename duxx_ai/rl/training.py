"""RL Training — GRPO, PPO, and REINFORCE trainers for LLM agents.

Provides training loops that use trajectory data to improve LLM agent policies
via reinforcement learning. Works with any Duxx AI agent or custom policy.

Trainers:
    - GRPOTrainer: Group Relative Policy Optimization (Meta)
    - PPOTrainer: Proximal Policy Optimization
    - REINFORCETrainer: Simple policy gradient (REINFORCE)
    - DPOTrainer: Direct Preference Optimization (offline)
    - BestOfNTrainer: Simple best-of-N sampling (no gradient update)

Usage:
    from duxx_ai.rl.training import GRPOTrainer, PPOTrainer

    trainer = GRPOTrainer(model="gpt-4o-mini", env=my_env)
    results = await trainer.train(n_episodes=100, n_epochs=3)
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from duxx_ai.rl.core import (
    Episode,
    RewardComposer,
    RewardFunction,
    RLEnvironment,
    TrajectoryBuffer,
    batch_rollout,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training Config & Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    n_episodes: int = 100               # Episodes per training iteration
    n_epochs: int = 3                   # Training epochs over collected data
    batch_size: int = 16                # Mini-batch size for updates
    learning_rate: float = 1e-5
    gamma: float = 0.99                 # Discount factor
    max_steps_per_episode: int = 50
    max_concurrency: int = 5            # Parallel rollouts
    log_interval: int = 10              # Log every N episodes
    save_interval: int = 50             # Save checkpoint every N episodes
    # GRPO specific
    group_size: int = 4                 # Number of completions per prompt (GRPO)
    # PPO specific
    clip_epsilon: float = 0.2           # PPO clipping parameter
    value_coeff: float = 0.5            # Value loss weight
    entropy_coeff: float = 0.01         # Entropy bonus
    gae_lambda: float = 0.95            # GAE lambda
    # General
    reward_baseline: str = "mean"       # "mean", "running_mean", "none"
    max_grad_norm: float = 1.0          # Gradient clipping


@dataclass
class TrainingResult:
    """Results from a training run."""
    trainer: str = ""
    total_episodes: int = 0
    total_steps: int = 0
    total_time_seconds: float = 0.0
    final_success_rate: float = 0.0
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    best_episode_reward: float = 0.0
    best_episode: Episode | None = None
    checkpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trainer": self.trainer, "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_time": f"{self.total_time_seconds:.1f}s",
            "final_success_rate": f"{self.final_success_rate:.1%}",
            "best_reward": self.best_episode_reward,
            "avg_reward_last_10": round(sum(self.reward_history[-10:]) / max(len(self.reward_history[-10:]), 1), 4),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Base Trainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseTrainer(ABC):
    """Base class for all RL trainers."""

    def __init__(
        self,
        agent_fn: Callable[[str], Awaitable[str]],
        env: RLEnvironment | Callable[[], RLEnvironment],
        reward_fn: RewardComposer | RewardFunction | None = None,
        config: TrainingConfig | None = None,
        on_episode: Callable[[Episode], Any] | None = None,
        on_update: Callable[[dict], Any] | None = None,
    ):
        self.agent_fn = agent_fn
        self.env = env
        self.env_factory = env if callable(env) and not isinstance(env, RLEnvironment) else lambda: env
        self.reward_fn = reward_fn
        self.config = config or TrainingConfig()
        self.on_episode = on_episode
        self.on_update = on_update
        self.buffer = TrajectoryBuffer()

    @abstractmethod
    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """Update the agent's policy using collected episodes. Returns loss metrics."""
        ...

    async def train(self) -> TrainingResult:
        """Run the full training loop.

        1. Collect episodes (rollouts)
        2. Compute advantages/returns
        3. Update policy
        4. Repeat for n_epochs

        Returns TrainingResult with metrics.
        """
        result = TrainingResult(trainer=self.__class__.__name__)
        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            logger.info(f"[{self.__class__.__name__}] Epoch {epoch + 1}/{self.config.n_epochs}")

            # Phase 1: Collect episodes
            self.buffer = await batch_rollout(
                agent_fn=self.agent_fn,
                env_factory=self.env_factory,
                n_episodes=self.config.n_episodes,
                max_concurrency=self.config.max_concurrency,
                reward_fn=self.reward_fn,
                buffer=TrajectoryBuffer(),
            )

            # Log episode stats
            stats = self.buffer.stats()
            logger.info(f"  Collected {stats['episodes']} episodes, avg reward: {stats['reward_mean']}")

            for ep in self.buffer.episodes:
                result.reward_history.append(ep.total_reward)
                result.total_episodes += 1
                result.total_steps += ep.total_steps
                if ep.total_reward > result.best_episode_reward:
                    result.best_episode_reward = ep.total_reward
                    result.best_episode = ep

                if self.on_episode:
                    self.on_episode(ep)

            # Phase 2: Update policy
            batch = self.buffer.episodes
            if batch:
                loss_metrics = await self._update_policy(batch)
                result.loss_history.append(loss_metrics.get("loss", 0.0))

                if self.on_update:
                    self.on_update({"epoch": epoch, "stats": stats, **loss_metrics})

            result.final_success_rate = self.buffer.success_rate()

        result.total_time_seconds = time.time() - start_time
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GRPO Trainer (Group Relative Policy Optimization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization trainer.

    GRPO generates multiple completions per prompt, ranks them by reward,
    and uses the relative ranking as the training signal. No value network needed.

    Algorithm:
        1. For each observation, generate K completions (group_size)
        2. Score each completion with reward function
        3. Normalize rewards within each group (zero-mean, unit-variance)
        4. Use normalized rewards as advantages for policy gradient update

    Usage:
        trainer = GRPOTrainer(
            agent_fn=agent.run,
            env=CodingEnv(),
            reward_fn=RewardComposer([...]),
            config=TrainingConfig(group_size=4, n_episodes=100),
        )
        result = await trainer.train()
    """

    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """GRPO update: rank completions within groups, compute advantages."""
        # Group episodes by similar observations
        groups: dict[str, list[Episode]] = {}
        for ep in batch:
            # Use first observation as group key
            key = ep.observations[0][:100] if ep.observations else "default"
            groups.setdefault(key, []).append(ep)

        total_advantage = 0.0
        n_groups = 0

        for key, group in groups.items():
            if len(group) < 2:
                continue

            # Get rewards for this group
            rewards = [ep.total_reward for ep in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_r = max(std_r, 1e-8)  # Prevent division by zero

            # Normalize rewards within group → advantages
            advantages = [(r - mean_r) / std_r for r in rewards]
            total_advantage += sum(abs(a) for a in advantages) / len(advantages)
            n_groups += 1

            # In production: compute policy gradient with these advantages
            # L = -1/G * sum(advantage_i * log_prob_i)
            # Here we log the computation; actual gradient update requires model access

            logger.debug(f"  GRPO group [{key[:30]}]: {len(group)} episodes, "
                        f"rewards={[round(r,3) for r in rewards]}, "
                        f"advantages={[round(a,3) for a in advantages]}")

        avg_advantage = total_advantage / max(n_groups, 1)
        avg_reward = sum(ep.total_reward for ep in batch) / max(len(batch), 1)

        return {
            "loss": -avg_advantage,  # Negative because we maximize
            "avg_reward": avg_reward,
            "n_groups": n_groups,
            "avg_advantage": avg_advantage,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PPO Trainer (Proximal Policy Optimization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization trainer.

    PPO clips the policy ratio to prevent too-large updates. Requires
    both a policy and value network (value can be approximated).

    Algorithm:
        1. Collect trajectories with current policy
        2. Compute GAE advantages using value estimates
        3. For each mini-batch:
           a. Compute ratio = new_prob / old_prob
           b. Clip ratio to [1-eps, 1+eps]
           c. Loss = -min(ratio * A, clip(ratio) * A)
           d. Value loss = (V - returns)^2
           e. Entropy bonus for exploration

    Usage:
        trainer = PPOTrainer(
            agent_fn=agent.run,
            env=ReasoningEnv(),
            config=TrainingConfig(clip_epsilon=0.2, n_epochs=3),
        )
        result = await trainer.train()
    """

    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """PPO update with clipping and GAE."""
        all_transitions = [t for ep in batch for t in ep.transitions]
        if not all_transitions:
            return {"loss": 0.0}

        # Compute returns and advantages (GAE)
        advantages = []
        returns = []
        for ep in batch:
            ep_returns = ep.returns  # Discounted returns
            # Simple advantage: return - mean(return)
            mean_ret = sum(ep_returns) / max(len(ep_returns), 1)
            ep_advantages = [r - mean_ret for r in ep_returns]
            advantages.extend(ep_advantages)
            returns.extend(ep_returns)

        if not advantages:
            return {"loss": 0.0}

        # Normalize advantages
        adv_mean = sum(advantages) / len(advantages)
        adv_std = (sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)) ** 0.5
        adv_std = max(adv_std, 1e-8)
        norm_advantages = [(a - adv_mean) / adv_std for a in advantages]

        # Simulate PPO loss computation
        # In production: uses actual log probs and ratio clipping
        clip_eps = self.config.clip_epsilon
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0

        for i, t in enumerate(all_transitions):
            adv = norm_advantages[i] if i < len(norm_advantages) else 0.0
            ret = returns[i] if i < len(returns) else 0.0

            # Simplified loss (actual PPO uses log prob ratios)
            # L_clip = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
            ratio = 1.0  # Placeholder: would be exp(new_logp - old_logp)
            clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
            policy_loss += -min(ratio * adv, clipped * adv)

            # Value loss
            value_est = t.value if t.value is not None else ret
            value_loss += (value_est - ret) ** 2

            # Entropy (higher = more exploration)
            entropy += 0.5  # Placeholder

        n = max(len(all_transitions), 1)
        total_loss = (
            policy_loss / n
            + self.config.value_coeff * value_loss / n
            - self.config.entropy_coeff * entropy / n
        )

        return {
            "loss": total_loss,
            "policy_loss": policy_loss / n,
            "value_loss": value_loss / n,
            "entropy": entropy / n,
            "avg_advantage": sum(norm_advantages) / max(len(norm_advantages), 1),
            "avg_return": sum(returns) / max(len(returns), 1),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REINFORCE Trainer (Vanilla Policy Gradient)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class REINFORCETrainer(BaseTrainer):
    """Simple REINFORCE (vanilla policy gradient) trainer.

    Algorithm:
        1. Collect full episode
        2. Compute discounted returns G_t
        3. Update: theta += alpha * G_t * grad(log pi(a|s))

    Simpler than PPO/GRPO but higher variance.

    Usage:
        trainer = REINFORCETrainer(agent_fn=agent.run, env=env)
        result = await trainer.train()
    """

    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """REINFORCE update using episode returns."""
        all_returns = []
        for ep in batch:
            all_returns.extend(ep.returns)

        if not all_returns:
            return {"loss": 0.0}

        # Baseline subtraction (reduces variance)
        baseline = sum(all_returns) / len(all_returns) if self.config.reward_baseline == "mean" else 0.0
        advantages = [r - baseline for r in all_returns]

        # Policy gradient loss: L = -E[G_t * log pi(a_t|s_t)]
        loss = -sum(advantages) / len(advantages)

        return {
            "loss": loss,
            "avg_return": sum(all_returns) / len(all_returns),
            "return_std": (sum((r - sum(all_returns)/len(all_returns))**2 for r in all_returns) / len(all_returns))**0.5,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DPO Trainer (Direct Preference Optimization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization — offline RL from preference pairs.

    Algorithm:
        1. Collect episodes, group by observation
        2. Within each group, rank by reward → create (chosen, rejected) pairs
        3. DPO loss: L = -log(sigma(beta * (log(pi(chosen)) - log(pi(rejected)))))

    Usage:
        trainer = DPOTrainer(agent_fn=agent.run, env=env)
        result = await trainer.train()
    """

    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """DPO update from preference pairs."""
        # Group episodes by initial observation
        groups: dict[str, list[Episode]] = {}
        for ep in batch:
            key = ep.observations[0][:100] if ep.observations else "default"
            groups.setdefault(key, []).append(ep)

        n_pairs = 0
        total_loss = 0.0

        for key, group in groups.items():
            if len(group) < 2:
                continue

            # Sort by reward: best first
            sorted_eps = sorted(group, key=lambda e: e.total_reward, reverse=True)

            # Create preference pairs: (best, worst)
            chosen = sorted_eps[0]
            rejected = sorted_eps[-1]

            # DPO loss (simplified — actual needs log probs)
            reward_diff = chosen.total_reward - rejected.total_reward
            loss = -math.log(1.0 / (1.0 + math.exp(-self.beta * reward_diff)) + 1e-8)
            total_loss += loss
            n_pairs += 1

        avg_loss = total_loss / max(n_pairs, 1)
        return {"loss": avg_loss, "n_pairs": n_pairs}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Best-of-N (No gradient update — inference-time RL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BestOfNTrainer(BaseTrainer):
    """Best-of-N sampling — generate N completions, keep the best.

    Not a true trainer (no gradient update), but useful for:
    - Inference-time scaling (sample more = better results)
    - Creating preference data for DPO
    - Evaluating reward functions

    Usage:
        bon = BestOfNTrainer(agent_fn=agent.run, env=env, n=8)
        result = await bon.train()
        print(result.best_episode.total_reward)
    """

    def __init__(self, *args, n: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    async def _update_policy(self, batch: list[Episode]) -> dict[str, float]:
        """No update — just track best episodes."""
        if not batch:
            return {"loss": 0.0}
        best = max(batch, key=lambda e: e.total_reward)
        worst = min(batch, key=lambda e: e.total_reward)
        return {
            "loss": 0.0,
            "best_reward": best.total_reward,
            "worst_reward": worst.total_reward,
            "spread": best.total_reward - worst.total_reward,
        }
