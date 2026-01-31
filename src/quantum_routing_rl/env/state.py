"""State containers for the routing environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


PhysicalSwap = Tuple[int, int]
LogicalGate = Tuple[int, int]


@dataclass(frozen=True)
class RoutingState:
    """Minimal, serialisable state representation."""

    layout: dict[int, int]
    frontier: List[LogicalGate]
    lookahead: List[LogicalGate]
    candidate_swaps: List[PhysicalSwap]
    action_mask: List[bool]
    frontier_phys: Tuple[int, int] | None
    frontier_distance: float | None
    step_count: int
    done: bool
    recent_swaps: List[PhysicalSwap]
