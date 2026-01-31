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
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.teacher import (
    TeacherPolicy,
    _sabre_initial_layout,
    route_with_teacher,
)
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics

from quantum_routing_rl.env.state import RoutingState

CANDIDATE_FEATURE_DIM = 10
STATE_FEATURE_DIM = 3
FEATURE_DIM = CANDIDATE_FEATURE_DIM + STATE_FEATURE_DIM
DEFAULT_P2_ERROR = 0.005
DEFAULT_T2_NS = 400.0


def _frontier_physical_pair(state: RoutingState) -> Tuple[int, int] | None:
    if not state.frontier:
        return None
    if state.frontier_phys is not None:
        return state.frontier_phys
    logical_a, logical_b = state.frontier[0]
    return state.layout[logical_a], state.layout[logical_b]


def candidate_features(
    state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None = None
) -> torch.Tensor:
    """Return candidate-specific feature matrix (no global state features).

    Feature order:
        0: mapped frontier phys A (normalised)
        1: mapped frontier phys B (normalised)
        2: current frontier shortest-path distance
        3: distance improvement if swap applied (positive is good)
        4: candidate edge lies on a current shortest path (0/1)
        5: candidate edge p2_error
        6: candidate edge two-qubit duration (microseconds)
        7: congestion proxy (avg degree / max degree)
        8: candidate was used recently (within anti-osc window)
        9: mean recent involvement of the two qubits
    """
    frontier_pair = _frontier_physical_pair(state)
    if frontier_pair is None:
        return torch.zeros((len(state.candidate_swaps), CANDIDATE_FEATURE_DIM), dtype=torch.float32)

    try:
        current_dist = state.frontier_distance
        if current_dist is None:
            current_dist = float(nx.shortest_path_length(graph, *frontier_pair))
    except nx.NetworkXNoPath:
        current_dist = float("inf")

    node_norm = max(1, graph.number_of_nodes() - 1)
    feats = []
    path_nodes: Iterable[int] = ()
    try:
        path_nodes = nx.shortest_path(graph, *frontier_pair)
    except nx.NetworkXNoPath:
        path_nodes = ()

    max_degree = max((deg for _, deg in graph.degree()), default=1)

    recent_swaps = getattr(state, "recent_swaps", [])
    window = max(1, len(recent_swaps)) if recent_swaps else 1

    for u, v in state.candidate_swaps:
        on_shortest = int(u in path_nodes and v in path_nodes)
        new_layout = _swapped_layout(state.layout, u, v)
        new_phys_pair = (new_layout[logical_a(state)], new_layout[logical_b(state)])
        try:
            new_dist = nx.shortest_path_length(graph, *new_phys_pair)
        except nx.NetworkXNoPath:
            new_dist = float("inf")

        current_dist_val = (
            float(current_dist)
            if current_dist not in {None, float("inf")}
            else float(graph.number_of_nodes())
        )
        new_dist_val = (
            float(new_dist) if new_dist != float("inf") else float(graph.number_of_nodes())
        )
        improvement = current_dist_val - new_dist_val

        deg_u = graph.degree[u] if u in graph else 0
        deg_v = graph.degree[v] if v in graph else 0
        congestion = ((deg_u + deg_v) / 2) / max_degree if max_degree else 0.0

        edge_props = hardware.get_edge_props(u, v) if hardware else None
        p2_error = edge_props.p2_error if edge_props else DEFAULT_P2_ERROR
        t2_dur_us = (edge_props.t2_duration_ns if edge_props else DEFAULT_T2_NS) / 1000.0
        edge_norm = (u, v) if u <= v else (v, u)
        recent_edge = int(edge_norm in recent_swaps)
        qubit_hits = sum(u in edge for edge in recent_swaps) + sum(
            v in edge for edge in recent_swaps
        )
        qubit_recency = qubit_hits / (2 * window)

        feats.append(
            [
                frontier_pair[0] / node_norm,
                frontier_pair[1] / node_norm,
                current_dist_val,
                improvement,
                float(on_shortest),
                float(p2_error),
                float(t2_dur_us),
                float(congestion),
                float(recent_edge),
                float(qubit_recency),
            ]
        )
    return torch.tensor(feats, dtype=torch.float32)


def state_features(state: RoutingState, graph: nx.Graph) -> torch.Tensor:
    """Global state features tiled across candidates."""
    frontier_pair = _frontier_physical_pair(state)
    try:
        current_dist = state.frontier_distance
        if current_dist is None and frontier_pair is not None:
            current_dist = float(nx.shortest_path_length(graph, *frontier_pair))
    except nx.NetworkXNoPath:
        current_dist = float(graph.number_of_nodes())

    dist_norm = (
        float(current_dist) / max(1, graph.number_of_nodes()) if current_dist is not None else 0.0
    )
    step_norm = state.step_count / max(1, len(state.layout) * 4)
    candidate_norm = len(state.candidate_swaps) / max(1, graph.number_of_edges() or 1)
    return torch.tensor([dist_norm, step_norm, candidate_norm], dtype=torch.float32)


def model_features(
    state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None = None
) -> torch.Tensor:
    """Candidate + state features concatenated for model input."""
    cand = candidate_features(state, graph, hardware)
    state_vec = state_features(state, graph)
    if cand.numel() == 0:
        return torch.zeros((len(state.candidate_swaps), FEATURE_DIM), dtype=torch.float32)
    state_tiled = state_vec.unsqueeze(0).repeat(cand.shape[0], 1)
    return torch.cat([cand, state_tiled], dim=1)


def _logical_on_physical_layout(layout: dict[int, int], physical: int) -> int | None:
    for logical, phys in layout.items():
        if phys == physical:
            return logical
    return None


def logical_a(state: RoutingState) -> int:
    return state.frontier[0][0]


def logical_b(state: RoutingState) -> int:
    return state.frontier[0][1]


def _swapped_layout(layout: dict[int, int], u: int, v: int) -> dict[int, int]:
    """Return a copy of layout after swapping physical qubits u and v."""
    new_layout = dict(layout)
    logical_on_u = _logical_on_physical_layout(layout, u)
    logical_on_v = _logical_on_physical_layout(layout, v)
    if logical_on_u is not None:
        new_layout[logical_on_u] = v
    if logical_on_v is not None:
        new_layout[logical_on_v] = u
    return new_layout


@dataclass
class PolicyOutput:
    action_index: int
    logits: torch.Tensor


class SwapPolicy(nn.Module):
    """Small MLP scoring candidate swaps, optional value head for RL."""

    def __init__(
        self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128, use_value_head: bool = False
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head: nn.Module | None = (
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            if use_value_head
            else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        return logits

    def state_value(self, features: torch.Tensor) -> torch.Tensor:
        if self.value_head is None:
            msg = "Value head not initialised for this policy."
            raise RuntimeError(msg)
        values = self.value_head(features).squeeze(-1)
        return values


def choose_action(logits: torch.Tensor, mask: torch.Tensor | None = None) -> int:
    """Select highest-scoring legal action."""
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor per candidate")
    if mask is not None:
        logits = logits.clone()
        logits[~mask.bool()] = -1e9
    return int(torch.argmax(logits).item())


def policy_step(
    model: SwapPolicy, state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None = None
) -> PolicyOutput:
    """Compute greedy action from model."""
    with torch.no_grad():
        feats = model_features(state, graph, hardware)
        feats = _match_feature_dim(feats, model.net[0].in_features)
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
    state_dict_key = "model_state"
    if state_dict_key not in checkpoint:
        state_dict_key = "model_state_dict"
    state_dict = checkpoint[state_dict_key]
    use_value_head = bool(
        checkpoint.get("use_value_head", False)
        or any(key.startswith("value_head") for key in state_dict.keys())
    )
    model = SwapPolicy(feature_dim=feature_dim, use_value_head=use_value_head)
    state_dict_key = "model_state"
    if state_dict_key not in checkpoint:
        state_dict_key = "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key], strict=False)
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
    env_config: RoutingEnvConfig | None = None,
    max_steps: int | None = None,
    hardware_model: HardwareModel | None = None,
    teacher_mix: float = 0.0,
    teacher_swaps: float | None = None,
    swap_guard_ratio: float | None = None,
) -> BaselineResult:
    """Roll out greedy policy to produce a routed circuit."""
    if env_config is not None:
        effective_config = env_config
    elif swap_guard_ratio is not None or teacher_mix > 0:
        effective_config = RoutingEnvConfig(frontier_size=4)
    else:
        effective_config = RoutingEnvConfig()
    env = RoutingEnv(effective_config)
    try:
        initial_layout = _sabre_initial_layout(circuit, coupling_map, seed=seed)
    except Exception:
        initial_layout = None
    state = env.reset(
        circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
    graph = nx.Graph(_normalize_edges(coupling_map))
    hardware = env.hardware_model
    teacher_for_mix: TeacherPolicy | None = None
    if teacher_mix > 0:
        teacher_for_mix = TeacherPolicy()
        teacher_for_mix.begin_episode(graph)
    teacher_guard: TeacherPolicy | None = None
    guard_target: float | None = None
    if swap_guard_ratio is not None:
        teacher_guard = teacher_for_mix or TeacherPolicy()
        teacher_guard.begin_episode(graph)
        if teacher_swaps is None:
            try:
                baseline = route_with_teacher(
                    circuit, coupling_map, seed=seed, hardware_model=hardware_model
                )
                teacher_swaps = baseline.metrics.swaps
            except Exception:
                teacher_swaps = None
        if teacher_swaps:
            guard_target = float(swap_guard_ratio) * float(teacher_swaps)
    start = time.perf_counter()
    step_budget = max_steps or max(200, len(circuit.data) * 10)
    steps = 0
    while not state.done and steps < step_budget:
        use_teacher = bool(
            teacher_guard is not None
            and guard_target is not None
            and state.step_count >= guard_target
        )
        if use_teacher and teacher_guard is not None:
            action = teacher_guard.select_action(state, graph, hardware)
            chosen_edge = state.candidate_swaps[action]
        else:
            feats = model_features(state, graph, hardware)
            feats = _match_feature_dim(feats, model.net[0].in_features)  # type: ignore[index]
            logits = model(feats)
            mask_tensor = torch.tensor(state.action_mask, dtype=torch.bool)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            if teacher_for_mix is not None:
                scores = torch.tensor(
                    teacher_for_mix.score_candidates(state, graph, hardware),
                    dtype=logits.dtype,
                    device=logits.device,
                )
                teacher_logits = (-scores).masked_fill(~mask_tensor, -1e9)
                logits = logits + float(teacher_mix) * teacher_logits
            action = int(torch.argmax(logits).item())
            chosen_edge = state.candidate_swaps[action]
        state, _, _, _ = env.step(action)
        if teacher_guard is not None:
            teacher_guard.update_after_action(chosen_edge, state.step_count)
        elif teacher_for_mix is not None:
            teacher_for_mix.update_after_action(chosen_edge, state.step_count)
        steps += 1

    # Fallback to a deterministic teacher if the learned policy stalls.
    while not state.done and steps < step_budget * 2:
        action = teacher_fallback_action(state, graph, hardware)
        state, _, _, _ = env.step(action)
        steps += 1

    if not state.done:
        msg = f"Routing did not complete within {step_budget * 2} steps"
        raise RuntimeError(msg)

    runtime = time.perf_counter() - start
    routed = env.routed_circuit
    assert_coupling_compatible(routed, _normalize_edges(coupling_map))
    metrics = compute_metrics(routed, hardware_model=hardware)
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


def teacher_fallback_action(
    state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None = None
) -> int:
    """Deterministic shortest-path action used as a safety net."""
    teacher = TeacherPolicy()
    return teacher.select_action(state, graph, hardware)


def _match_feature_dim(feats: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or truncate features to match model input for backwards compatibility."""
    if feats.ndim != 2:
        return feats
    if feats.shape[1] == target_dim:
        return feats
    if feats.shape[1] > target_dim:
        return feats[:, :target_dim]
    pad = torch.zeros((feats.shape[0], target_dim - feats.shape[1]), device=feats.device)
    return torch.cat([feats, pad], dim=1)
