"""Simple swap policy network and feature extraction."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import networkx as nx
import torch
import torch.nn as nn
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics

from quantum_routing_rl.env.state import RoutingState

FEATURE_DIM = 6


def _frontier_physical_pair(state: RoutingState) -> Tuple[int, int] | None:
    if not state.frontier:
        return None
    logical_a, logical_b = state.frontier[0]
    return state.layout[logical_a], state.layout[logical_b]


def candidate_features(state: RoutingState, graph: nx.Graph) -> torch.Tensor:
    """Return feature matrix for each candidate swap."""
    frontier_pair = _frontier_physical_pair(state)
    if frontier_pair is None:
        return torch.zeros((len(state.candidate_swaps), FEATURE_DIM), dtype=torch.float32)

    try:
        current_dist = nx.shortest_path_length(graph, *frontier_pair)
    except nx.NetworkXNoPath:
        current_dist = float("inf")

    feats = []
    path_nodes: Iterable[int] = ()
    try:
        path_nodes = nx.shortest_path(graph, *frontier_pair)
    except nx.NetworkXNoPath:
        path_nodes = ()

    for u, v in state.candidate_swaps:
        touches_a = int(u == frontier_pair[0] or v == frontier_pair[0])
        touches_b = int(u == frontier_pair[1] or v == frontier_pair[1])
        on_shortest = int(u in path_nodes and v in path_nodes)
        # Estimate distance improvement after swap.
        logical_on_u = _logical_on_physical(state, u)
        logical_on_v = _logical_on_physical(state, v)
        new_layout = dict(state.layout)
        if logical_on_u is not None:
            new_layout[logical_on_u] = v
        if logical_on_v is not None:
            new_layout[logical_on_v] = u
        new_phys_pair = (new_layout[logical_a(state)], new_layout[logical_b(state)])
        try:
            new_dist = nx.shortest_path_length(graph, *new_phys_pair)
        except nx.NetworkXNoPath:
            new_dist = float("inf")
        improvement = (current_dist - new_dist) if current_dist != float("inf") else 0.0
        feats.append(
            [
                touches_a,
                touches_b,
                on_shortest,
                float(current_dist if current_dist != float("inf") else 0.0),
                float(new_dist if new_dist != float("inf") else 0.0),
                float(improvement),
            ]
        )
    return torch.tensor(feats, dtype=torch.float32)


def _logical_on_physical(state: RoutingState, physical: int) -> int | None:
    for logical, phys in state.layout.items():
        if phys == physical:
            return logical
    return None


def logical_a(state: RoutingState) -> int:
    return state.frontier[0][0]


def logical_b(state: RoutingState) -> int:
    return state.frontier[0][1]


@dataclass
class PolicyOutput:
    action_index: int
    logits: torch.Tensor


class SwapPolicy(nn.Module):
    """Small MLP scoring candidate swaps."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        return logits


def choose_action(logits: torch.Tensor, mask: torch.Tensor | None = None) -> int:
    """Select highest-scoring legal action."""
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor per candidate")
    if mask is not None:
        logits = logits.clone()
        logits[~mask.bool()] = -1e9
    return int(torch.argmax(logits).item())


def policy_step(model: SwapPolicy, state: RoutingState, graph: nx.Graph) -> PolicyOutput:
    """Compute greedy action from model."""
    with torch.no_grad():
        feats = candidate_features(state, graph)
        logits = model(feats)
        mask_tensor = torch.tensor(state.action_mask, dtype=torch.bool)
        action = choose_action(logits, mask_tensor)
    return PolicyOutput(action_index=action, logits=logits)


def load_swap_policy(
    path: str | "os.PathLike[str]", device: torch.device | None = None
) -> SwapPolicy:
    """Load checkpoint saved by imitation learning or RL."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    model = SwapPolicy(feature_dim=feature_dim)
    state_dict_key = "model_state"
    if state_dict_key not in checkpoint:
        state_dict_key = "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    model.to(device)
    model.eval()
    return model


def route_with_policy(
    model: SwapPolicy,
    circuit,
    coupling_map: CouplingMap | Iterable[Tuple[int, int]],
    *,
    name: str = "il_policy",
    seed: int | None = None,
) -> BaselineResult:
    """Roll out greedy policy to produce a routed circuit."""
    env = RoutingEnv(RoutingEnvConfig())
    state = env.reset(circuit, coupling_map, seed=seed)
    graph = nx.Graph(_normalize_edges(coupling_map))
    start = time.perf_counter()
    while not state.done:
        action = policy_step(model, state, graph).action_index
        state, _, _, _ = env.step(action)
    runtime = time.perf_counter() - start
    routed = env.routed_circuit
    assert_coupling_compatible(routed, _normalize_edges(coupling_map))
    metrics = compute_metrics(routed)
    return BaselineResult(
        name=name,
        circuit=routed,
        metrics=metrics,
        runtime_s=runtime,
        seed=seed,
    )


def _normalize_edges(
    coupling_map: CouplingMap | Iterable[Tuple[int, int]],
) -> list[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [tuple(edge) for edge in coupling_map]
