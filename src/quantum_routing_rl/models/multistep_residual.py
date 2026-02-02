"""Bounded multi-step residual search over teacher-proposed candidates.

This module introduces a light lookahead planner that keeps the teacher
as the proposal distribution but scores short rollouts using a simple
utility that mixes hardware-aware log-success deltas with explicit swap
and duration penalties.  The planner is intentionally shallow (H=2/3)
to stay cheap while still reasoning about near-term consequences – a
response to the observed swap explosion of the 1-step residual_topk.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import torch
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.env.state import RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.policy import teacher_fallback_action
from quantum_routing_rl.models.residual_policy import ResidualScorer, _candidate_context
from quantum_routing_rl.models.teacher import TeacherPolicy, _sabre_initial_layout


# ---------------------------------------------------------------------------
# Utility helpers
def _topk_indices(scores: torch.Tensor, mask: torch.Tensor, k: int) -> list[int]:
    """Return indices of the k best (lowest) scores respecting mask."""

    scores = scores.clone()
    scores = scores.masked_fill(~mask, float("inf"))
    if not torch.isfinite(scores).any():
        return []
    order = torch.argsort(scores)
    out: list[int] = []
    for idx in order.tolist():
        if not mask[idx]:
            continue
        out.append(idx)
        if len(out) >= k:
            break
    return out


def _normalize_edges(coupling_map: CouplingMap | Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [(int(u), int(v)) for u, v in coupling_map]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Value estimator (optional leaf heuristic)
class StateValueNet(torch.nn.Module):
    """Tiny MLP over state context used as a leaf heuristic."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if context.ndim == 1:
            context = context.unsqueeze(0)
        return self.net(context).squeeze(-1)


def load_state_value_model(
    path: str | "os.PathLike[str]", device: torch.device | None = None
) -> StateValueNet:
    payload = torch.load(path, map_location=device or "cpu")
    input_dim = int(payload.get("input_dim", 32))
    hidden = int(payload.get("hidden_dim", 128))
    model = StateValueNet(input_dim=input_dim, hidden=hidden)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_residual_scorer(
    checkpoint: str | "os.PathLike[str]", device: torch.device | None = None
) -> ResidualScorer:
    payload = torch.load(checkpoint, map_location=device or "cpu")
    state_key = "scorer_state" if "scorer_state" in payload else "model_state"
    feature_dim = int(payload.get("feature_dim", 16))
    hidden_dim = int(payload.get("hidden_dim", 256))
    context_dim = int(payload.get("context_dim", 0))
    scorer = ResidualScorer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
    )
    scorer.load_state_dict(payload[state_key])
    scorer.eval()
    return scorer


# ---------------------------------------------------------------------------
@dataclass
class MultiStepConfig:
    top_k: int = 4
    branch_k: int = 2
    horizon: int = 2
    swap_penalty: float = 0.15
    duration_scale: float = 5e-4
    progress_weight: float = 0.1
    include_hardware: bool = True


@dataclass
class MultiStepDecision:
    action_index: int
    utilities: dict[int, float]
    expanded: list[int]


class MultiStepResidualPolicy:
    """Bounded lookahead over teacher proposals with shallow search."""

    def __init__(
        self,
        *,
        config: MultiStepConfig,
        scorer: ResidualScorer | None = None,
        value_model: StateValueNet | None = None,
    ):
        self.config = config
        self.scorer = scorer
        self.value_model = value_model
        self.teacher = TeacherPolicy()

    # ----------------------------------------------------------- public API
    def begin_episode(self, graph: nx.Graph) -> None:
        self.teacher.begin_episode(graph)

    def select_action(
        self,
        env: RoutingEnv,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None,
    ) -> MultiStepDecision:
        """Evaluate top-k candidates with bounded lookahead and pick best."""

        teacher_scores = torch.tensor(self.teacher.score_candidates(state, graph, hardware))
        mask = torch.tensor(state.action_mask, dtype=torch.bool)
        teacher_choice = int(torch.argmin(teacher_scores).item()) if teacher_scores.numel() else 0
        top_indices = _topk_indices(teacher_scores, mask, self.config.top_k)
        if not top_indices:
            return MultiStepDecision(
                action_index=int(torch.argmin(teacher_scores).item()), utilities={}, expanded=[]
            )

        utilities: dict[int, float] = {}
        best_idx = top_indices[0]
        best_util = -float("inf")
        for idx in top_indices:
            util = self._evaluate_candidate(env, state, graph, hardware, idx)
            utilities[idx] = util
            if util > best_util:
                best_util = util
                best_idx = idx
        if teacher_choice not in utilities:
            teacher_util = self._evaluate_candidate(env, state, graph, hardware, teacher_choice)
            utilities[teacher_choice] = teacher_util
        teacher_util = utilities.get(teacher_choice, -float("inf"))
        if best_util < teacher_util:
            best_idx = teacher_choice
            best_util = teacher_util
        return MultiStepDecision(action_index=best_idx, utilities=utilities, expanded=top_indices)

    # ----------------------------------------------------------- internals
    def _evaluate_candidate(
        self,
        env: RoutingEnv,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None,
        action_idx: int,
    ) -> float:
        branch_env = copy.deepcopy(env)
        branch_teacher = copy.deepcopy(self.teacher)
        return self._rollout(
            branch_env,
            branch_teacher,
            graph,
            hardware,
            depth=0,
            initial_state=state,
            forced_action=action_idx,
        )

    def _rollout(
        self,
        env: RoutingEnv,
        teacher: TeacherPolicy,
        graph: nx.Graph,
        hardware: HardwareModel | None,
        *,
        depth: int,
        initial_state: RoutingState | None = None,
        forced_action: int | None = None,
    ) -> float:
        state = initial_state or env._build_state()
        if state.done:
            return self._leaf_value(state, graph, hardware)

        if forced_action is not None:
            action = forced_action
        else:
            scores = torch.tensor(teacher.score_candidates(state, graph, hardware))
            mask = torch.tensor(state.action_mask, dtype=torch.bool)
            top = _topk_indices(scores, mask, self.config.branch_k)
            action = top[0] if top else int(torch.argmin(scores).item())

        step_util, next_env, next_teacher, next_state = self._apply_action(
            env, teacher, state, graph, hardware, action
        )

        if depth + 1 >= self.config.horizon or next_state.done:
            return step_util + self._leaf_value(next_state, graph, hardware)

        scores = torch.tensor(next_teacher.score_candidates(next_state, graph, hardware))
        mask = torch.tensor(next_state.action_mask, dtype=torch.bool)
        children = _topk_indices(scores, mask, self.config.branch_k)
        if not children:
            return step_util + self._leaf_value(next_state, graph, hardware)

        best_future = -float("inf")
        for child in children:
            child_env = copy.deepcopy(next_env)
            child_teacher = copy.deepcopy(next_teacher)
            future = self._rollout(
                child_env,
                child_teacher,
                graph,
                hardware,
                depth=depth + 1,
                initial_state=None,
                forced_action=child,
            )
            if future > best_future:
                best_future = future
        return step_util + best_future

    def _apply_action(
        self,
        env: RoutingEnv,
        teacher: TeacherPolicy,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None,
        action_idx: int,
    ) -> tuple[float, RoutingEnv, TeacherPolicy, RoutingState]:
        swap_edge = state.candidate_swaps[action_idx]
        state_after, _, _, _ = env.step(action_idx)
        teacher.update_after_action(swap_edge, state_after.step_count)
        delta_log, duration_penalty, swap_penalty = self._step_utility(
            state, state_after, swap_edge, graph, hardware
        )
        progress = _frontier_distance(state, graph) - _frontier_distance(state_after, graph)
        step_util = (
            delta_log
            - swap_penalty
            - duration_penalty
            + self.config.progress_weight * float(progress)
        )
        return step_util, env, teacher, state_after

    def _step_utility(
        self,
        before_state: RoutingState,
        after_state: RoutingState,
        swap_edge: tuple[int, int],
        graph: nx.Graph,
        hardware: HardwareModel | None,
    ) -> tuple[float, float, float]:
        """Return (delta_log_success, duration_penalty, swap_penalty)."""

        before_log = self._frontier_success(before_state, graph, hardware)
        after_log = self._frontier_success(after_state, graph, hardware)
        delta_log = after_log - before_log

        duration_cost = 0.0
        if hardware is not None:
            _, dur_ns = hardware.edge_error_and_duration(swap_edge[0], swap_edge[1], directed=True)
            duration_cost = dur_ns * self.config.duration_scale
        swap_penalty = self.config.swap_penalty
        return delta_log, duration_cost, swap_penalty

    def _frontier_success(
        self, state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None
    ) -> float:
        if not state.frontier:
            return 0.0
        pair = state.frontier_phys
        if pair is None:
            try:
                gate = state.frontier[0]
                pair = (state.layout[gate[0]], state.layout[gate[1]])
            except Exception:
                pair = None
        if pair is None:
            return -float(graph.number_of_nodes())
        if hardware is not None and graph.has_edge(*pair):
            err, _ = hardware.edge_error_and_duration(pair[0], pair[1], directed=True)
            return math.log(max(1e-9, 1.0 - min(0.999999, err)))
        try:
            dist = float(nx.shortest_path_length(graph, *pair))
            return -dist
        except Exception:
            return -float(graph.number_of_nodes())

    def _leaf_value(
        self, state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None
    ) -> float:
        if state.done:
            return 0.0

        context = _candidate_context(
            state, graph, hardware, include_hardware=self.config.include_hardware
        )
        if self.value_model is not None:
            with torch.no_grad():
                ctx = context
                target_dim = self.value_model.net[0].in_features  # type: ignore[index]
                if ctx.numel() != target_dim:
                    if ctx.numel() > target_dim:
                        ctx = ctx[:target_dim]
                    else:
                        pad = torch.zeros(target_dim - ctx.numel(), device=ctx.device)
                        ctx = torch.cat([ctx, pad], dim=0)
                return float(self.value_model(ctx).item())

        # Fallback heuristic: prefer states closer to executing frontier gate.
        frontier_dist = state.frontier_distance
        if frontier_dist is None:
            try:
                frontier_pair = state.frontier_phys
                if frontier_pair is not None:
                    frontier_dist = float(nx.shortest_path_length(graph, *frontier_pair))
            except Exception:
                frontier_dist = None
        if frontier_dist is None:
            frontier_dist = float(graph.number_of_nodes())
        return -0.05 * float(frontier_dist)


# ---------------------------------------------------------------------------
# Routing entry point
def route_with_multistep_residual(
    policy: MultiStepResidualPolicy,
    circuit,
    coupling_map: CouplingMap | Iterable[Sequence[int]],
    *,
    name: str = "residual_multistep",
    seed: int | None = None,
    env_config: RoutingEnvConfig | None = None,
    max_steps: int | None = None,
    hardware_model: HardwareModel | None = None,
) -> BaselineResult:
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
    policy.begin_episode(graph)

    start = time.perf_counter()
    budget = max_steps or max(200, len(circuit.data) * 12)
    steps = 0
    while not state.done and steps < budget:
        decision = policy.select_action(env, state, graph, hardware)
        chosen = decision.action_index
        swap_edge = state.candidate_swaps[chosen]
        state, _, _, _ = env.step(chosen)
        policy.teacher.update_after_action(swap_edge, state.step_count)
        steps += 1

    # Deterministic teacher cleanup if stalled.
    fallback_budget = budget * 2
    while not state.done and steps < fallback_budget:
        fallback_idx = teacher_fallback_action(state, graph, hardware)
        swap_edge = state.candidate_swaps[fallback_idx]
        state, _, _, _ = env.step(fallback_idx)
        policy.teacher.update_after_action(swap_edge, state.step_count)
        steps += 1

    if not state.done:
        msg = f"Multistep residual policy stalled after {steps} steps (budget={fallback_budget})."
        raise RuntimeError(msg)

    runtime = time.perf_counter() - start
    routed = env.routed_circuit
    assert_coupling_compatible(routed, _normalize_edges(coupling_map))
    metrics = compute_metrics(routed, hardware_model=hardware)
    extra = {
        "residual_horizon": policy.config.horizon,
        "residual_top_k": policy.config.top_k,
        "residual_branch_k": policy.config.branch_k,
        "residual_swap_penalty": policy.config.swap_penalty,
        "residual_duration_scale": policy.config.duration_scale,
        "residual_progress_weight": policy.config.progress_weight,
        "residual_include_hardware": policy.config.include_hardware,
        "residual_value_model": bool(policy.value_model),
        "residual_scorer": bool(policy.scorer),
    }
    return BaselineResult(
        name=name,
        circuit=routed,
        metrics=metrics,
        runtime_s=runtime,
        seed=seed,
        extra=extra,
    )


__all__ = [
    "MultiStepResidualPolicy",
    "MultiStepConfig",
    "route_with_multistep_residual",
    "StateValueNet",
    "load_state_value_model",
    "load_residual_scorer",
]


def _frontier_distance(state: RoutingState, graph: nx.Graph) -> float:
    if not state.frontier:
        return 0.0
    pair = state.frontier_phys
    if pair is None:
        try:
            logical = state.frontier[0]
            pair = (state.layout[logical[0]], state.layout[logical[1]])
        except Exception:
            pair = None
    if pair is None:
        return float(graph.number_of_nodes())
    try:
        return float(nx.shortest_path_length(graph, *pair))
    except Exception:
        return float(graph.number_of_nodes())
