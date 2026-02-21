"""Residual Top-K policy that refines teacher-ranked candidates."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import networkx as nx
import torch
import torch.nn as nn
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.env.state import RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.policy import (
    candidate_features,
    state_features,
    teacher_fallback_action,
    _match_feature_dim,
)
from quantum_routing_rl.models.teacher import TeacherPolicy, _sabre_initial_layout


# ---------------------------------------------------------------------------
# Scoring network
class ResidualScorer(nn.Module):
    """Lightweight MLP scoring teacher top-K candidates."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 256,
        context_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        input_dim = feature_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cand_features: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return utility logits for each candidate in cand_features."""

        if cand_features.ndim != 2:
            msg = "cand_features must be 2D [num_candidates, feature_dim]"
            raise ValueError(msg)
        feats = cand_features
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            context = context.to(feats.device, feats.dtype)
            context_tiled = context.repeat(feats.shape[0], 1)
            feats = torch.cat([feats, context_tiled], dim=1)
        feats = _match_feature_dim(feats, self.net[0].in_features)  # type: ignore[index]
        logits = self.net(feats).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Utilities / embeddings
def _lookahead_pool(state: RoutingState, graph: nx.Graph) -> torch.Tensor:
    """Mean + max distance over lookahead gates as a small context embedding."""

    dists: list[float] = []
    for gate in state.lookahead[:4]:
        try:
            phys = (state.layout[gate[0]], state.layout[gate[1]])
            dists.append(float(nx.shortest_path_length(graph, *phys)))
        except Exception:
            continue
    if not dists:
        return torch.zeros(2, dtype=torch.float32)
    return torch.tensor([float(sum(dists) / len(dists)), float(max(dists))], dtype=torch.float32)


def _hardware_snapshot_feats(hardware: HardwareModel | None) -> torch.Tensor:
    """Global hardware snapshot statistics (mean/min p2 error + durations)."""

    if hardware is None:
        return torch.zeros(4, dtype=torch.float32)
    errors: list[float] = []
    durations: list[float] = []
    for edge in hardware.adjacency:
        props = hardware.get_edge_props(*edge)
        errors.append(float(props.p2_error))
        durations.append(float(props.t2_duration_ns) / 1000.0)
    if not errors:
        return torch.zeros(4, dtype=torch.float32)
    return torch.tensor(
        [
            float(sum(errors) / len(errors)),
            float(min(errors)),
            float(sum(durations) / len(durations)),
            float(min(durations)),
        ],
        dtype=torch.float32,
    )


def _candidate_context(
    state: RoutingState,
    graph: nx.Graph,
    hardware: HardwareModel | None,
    *,
    include_hardware: bool = True,
) -> torch.Tensor:
    """Concatenate state + lookahead pool + optional hardware stats."""

    state_vec = state_features(state, graph)
    pool = _lookahead_pool(state, graph)
    hardware_vec = _hardware_snapshot_feats(hardware) if include_hardware else torch.zeros(0)
    return torch.cat([state_vec, pool, hardware_vec], dim=0)


def _candidate_and_context(
    state: RoutingState,
    graph: nx.Graph,
    hardware: HardwareModel | None,
    indices: Sequence[int],
    *,
    include_hardware: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    cand_feats = candidate_features(state, graph, hardware)
    if cand_feats.numel() == 0:
        cand = torch.zeros((0, cand_feats.shape[1] if cand_feats.ndim == 2 else 0))
    else:
        cand = cand_feats[list(indices)]
    context = _candidate_context(state, graph, hardware, include_hardware=include_hardware)
    return cand, context


# ---------------------------------------------------------------------------
# Policy + rollout
@dataclass
class ResidualAction:
    action_index: int
    fallback_used: bool
    teacher_choice: int | None
    utilities: torch.Tensor | None


class ResidualTopKPolicy:
    """Select among teacher top-K candidates using a residual scorer."""

    def __init__(
        self,
        scorer: ResidualScorer,
        *,
        top_k: int = 8,
        allow_fallback: bool = True,
        uncertainty_std: float | None = None,
        include_hardware: bool = True,
        teacher_bias: float = 0.0,
    ):
        self.scorer = scorer
        self.top_k = max(1, int(top_k))
        self.allow_fallback = allow_fallback
        self.uncertainty_std = uncertainty_std
        self.include_hardware = include_hardware
        self.teacher = TeacherPolicy()
        self.teacher_bias = max(0.0, float(teacher_bias))

    def select_action(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None = None,
    ) -> ResidualAction:
        try:
            scorer_device = next(self.scorer.parameters()).device
        except StopIteration:
            scorer_device = torch.device("cpu")
        teacher_scores = torch.tensor(
            self.teacher.score_candidates(state, graph, hardware),
            dtype=torch.float32,
            device=scorer_device,
        )
        mask = torch.tensor(state.action_mask, dtype=torch.bool, device=scorer_device)
        if not mask.any():
            return ResidualAction(
                action_index=0, fallback_used=True, teacher_choice=None, utilities=None
            )
        teacher_scores = teacher_scores.masked_fill(~mask, float("inf"))
        valid = torch.isfinite(teacher_scores)
        if not valid.any():
            return ResidualAction(
                action_index=0, fallback_used=True, teacher_choice=None, utilities=None
            )

        sorted_idx = torch.argsort(teacher_scores)
        topk_idx: List[int] = []
        for idx in sorted_idx.tolist():
            if len(topk_idx) >= self.top_k:
                break
            if mask[idx] and torch.isfinite(teacher_scores[idx]):
                topk_idx.append(idx)
        if not topk_idx:
            return ResidualAction(
                action_index=0, fallback_used=True, teacher_choice=None, utilities=None
            )

        cand_feats, context = _candidate_and_context(
            state, graph, hardware, topk_idx, include_hardware=self.include_hardware
        )
        cand_feats = cand_feats.to(scorer_device)
        context = context.to(scorer_device)
        with torch.no_grad():
            utilities = self.scorer(cand_feats, context)
        combined = utilities
        if self.teacher_bias > 0:
            teacher_top = teacher_scores[topk_idx]
            teacher_top = teacher_top.to(utilities.device)
            combined = utilities - float(self.teacher_bias) * teacher_top
        top_choice_local = int(torch.argmax(combined).item()) if utilities.numel() else 0
        chosen_global = int(topk_idx[top_choice_local])

        fallback = False
        teacher_choice = int(sorted_idx[0].item()) if sorted_idx.numel() > 0 else None
        if self.allow_fallback and self.uncertainty_std is not None and utilities.numel():
            if float(torch.std(utilities)) < float(self.uncertainty_std):
                fallback = True
                if teacher_choice is not None:
                    chosen_global = teacher_choice
        return ResidualAction(
            action_index=chosen_global,
            fallback_used=fallback,
            teacher_choice=teacher_choice,
            utilities=utilities,
        )


def route_with_residual_policy(
    policy: ResidualTopKPolicy,
    circuit,
    coupling_map: CouplingMap | Iterable[Sequence[int]],
    *,
    name: str = "residual_topk",
    seed: int | None = None,
    env_config: RoutingEnvConfig | None = None,
    max_steps: int | None = None,
    hardware_model: HardwareModel | None = None,
) -> BaselineResult:
    """Roll out residual policy with optional teacher fallback."""

    env_cfg = env_config or RoutingEnvConfig(frontier_size=4)
    env = RoutingEnv(env_cfg)
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
    policy.teacher.begin_episode(graph)

    start = time.perf_counter()
    base_budget = max_steps or max(200, len(circuit.data) * 10)
    fallback_budget = base_budget * 2 if policy.allow_fallback else base_budget
    steps = 0
    fallback_count = 0
    while not state.done and steps < base_budget:
        action = policy.select_action(state, graph, hardware)
        chosen_idx = action.action_index
        if action.fallback_used and action.teacher_choice is not None:
            fallback_count += 1
            chosen_idx = action.teacher_choice
        swap_edge = state.candidate_swaps[chosen_idx]
        state, _, _, _ = env.step(chosen_idx)
        policy.teacher.update_after_action(swap_edge, state.step_count)
        steps += 1

    # Deterministic teacher fallback if stalled and allowed.
    while not state.done and policy.allow_fallback and steps < fallback_budget:
        fallback_idx = teacher_fallback_action(state, graph, hardware)
        swap_edge = state.candidate_swaps[fallback_idx]
        state, _, _, _ = env.step(fallback_idx)
        policy.teacher.update_after_action(swap_edge, state.step_count)
        steps += 1
        fallback_count += 1

    if not state.done:
        msg = f"Residual policy stalled after {steps} steps (budget={fallback_budget})."
        raise RuntimeError(msg)

    runtime = time.perf_counter() - start
    routed = env.routed_circuit
    assert_coupling_compatible(routed, _normalize_edges(coupling_map))
    metrics = compute_metrics(routed, hardware_model=hardware)
    extra = {
        "top_k": policy.top_k,
        "fallback_count": fallback_count,
        "uncertainty_std": policy.uncertainty_std,
        "residual_include_hardware": policy.include_hardware,
        "teacher_bias": policy.teacher_bias,
    }
    return BaselineResult(
        name=name,
        circuit=routed,
        metrics=metrics,
        runtime_s=runtime,
        seed=seed,
        extra=extra,
    )


def load_residual_policy(
    checkpoint: str | "os.PathLike[str]",
    *,
    top_k: int | None = None,
    allow_fallback: bool = True,
    uncertainty_std: float | None = None,
    include_hardware: bool = True,
    teacher_bias: float = 0.0,
    device: torch.device | None = None,
) -> ResidualTopKPolicy:
    """Load ResidualTopKPolicy + scorer from checkpoint."""

    device = device or torch.device("cpu")
    payload = torch.load(checkpoint, map_location=device)
    state_key = "scorer_state"
    if state_key not in payload:
        state_key = "model_state"
    feature_dim = int(payload.get("feature_dim", 16))
    hidden_dim = int(payload.get("hidden_dim", 256))
    context_dim = int(payload.get("context_dim", 0))
    scorer = ResidualScorer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
    )
    scorer.load_state_dict(payload[state_key])
    scorer.to(device)
    scorer.eval()
    policy = ResidualTopKPolicy(
        scorer,
        top_k=top_k or int(payload.get("top_k", 8)),
        allow_fallback=allow_fallback,
        uncertainty_std=uncertainty_std or payload.get("uncertainty_std"),
        include_hardware=include_hardware,
        teacher_bias=teacher_bias or payload.get("teacher_bias", 0.0),
    )
    return policy


def _normalize_edges(coupling_map: CouplingMap | Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [(int(u), int(v)) for u, v in coupling_map]  # type: ignore[arg-type]
