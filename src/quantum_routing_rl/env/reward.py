"""Reward configuration helpers for routing environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardConfig:
    """Weights for multi-objective reward."""

    swap_weight: float = 1.0
    depth_weight: float = 0.0
    noise_weight: float = 0.0
    base_swap_penalty: float = 1.0
    error_penalty_weight: float = 0.0
    time_penalty_weight: float = 0.0
    progress_weight: float = 0.0
    completion_bonus: float = 0.0
    failure_penalty: float = 0.0


@dataclass
class RewardBreakdown:
    """Decomposed reward terms (negative costs)."""

    swap_cost: float = 0.0
    depth_delta: float = 0.0
    noise_proxy_delta: float = 0.0
    progress_bonus: float = 0.0
    completion_bonus: float = 0.0
    failure_penalty: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    def total(self, config: RewardConfig) -> float:
        """Weighted sum of components."""
        return (
            config.swap_weight * self.swap_cost
            + self.progress_bonus
            + self.completion_bonus
            - self.failure_penalty
            + config.depth_weight * self.depth_delta
            + config.noise_weight * self.noise_proxy_delta
        )
