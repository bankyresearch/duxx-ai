"""Reinforcement Learning module for Duxx AI — train agents via environment interaction.

Gymnasium-style RL environments + composable rewards + trajectory collection + GRPO/PPO training.

Usage:
    from duxx_ai.rl import RLEnvironment, RewardFunction, TrajectoryBuffer
    from duxx_ai.rl.training import GRPOTrainer, PPOTrainer
    from duxx_ai.rl.environments import CodingEnv, ReasoningEnv, GameEnv
    from duxx_ai.rl.agent import RLAgent
"""
