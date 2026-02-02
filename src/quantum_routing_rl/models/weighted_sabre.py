"""Weighted/noise-aware SABRE variant using hardware-derived distances."""

from __future__ import annotations

import random
import time
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.env.state import LogicalGate, RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.teacher import _normalize_edges, _sabre_initial_layout
from quantum_routing_rl.routing.weighted_distance import (
    WeightedDistanceCache,
    WeightedDistanceParams,
    compute_edge_weight,
)


@dataclass(frozen=True)
class WeightedSabreWeights:
    """Weights for scoring candidate swaps."""

    lookahead_weight: float = 0.5
    decay_weight: float = 0.25
    decay_factor: float = 0.9
    decay_increment: float = 1.0
    stagnation_weight: float = 0.25
    lookahead_limit: int = 8


class WeightedSabreRouter:
    """Heuristic SABRE-style router scored by weighted distances."""

    def __init__(
        self,
        weights: WeightedSabreWeights | None = None,
        distance_params: WeightedDistanceParams | None = None,
        snapshot_mode: str = "avg",
        *,
        rng_seed: int | None = None,
    ) -> None:
        self.weights = weights or WeightedSabreWeights()
        self.distance_params = distance_params or WeightedDistanceParams()
        self.snapshot_mode = snapshot_mode
        self._distance_cache: WeightedDistanceCache | None = None
        self._qubit_decay: dict[int, tuple[float, int]] = {}
        self._graph_key: tuple[tuple[int, int], ...] | None = None
        self._rng = random.Random(rng_seed) if rng_seed is not None else None
        self._hardware: HardwareModel | None = None
        self._distance_scale: float = 1.0

    # ----------------------------------------------------------- lifecycle --
    def begin_episode(self, graph: nx.Graph, hardware: HardwareModel | None) -> None:
        self._graph_key = _graph_key(graph)
        self._qubit_decay = {}
        self._hardware = hardware
        if hardware is not None:
            self._distance_cache = WeightedDistanceCache(hardware, self.distance_params)
            self._distance_scale = _edge_scale(hardware, self.distance_params)
        else:
            self._distance_cache = None
            self._distance_scale = 1.0

    def select_action(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None = None,
    ) -> int:
        if state.done:
            return 0
        if not state.candidate_swaps:
            msg = "No candidate swaps available."
            raise RuntimeError(msg)

        scores = self.score_candidates(state, graph, hardware)
        finite = [(idx, s) for idx, s in enumerate(scores) if s < float("inf")]
        if not finite:
            msg = "All candidate actions are masked."
            raise RuntimeError(msg)

        min_score = min(score for _, score in finite)
        tied = [idx for idx, score in finite if abs(score - min_score) <= 1e-9]
        return min(tied)

    def score_candidates(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None = None,
    ) -> list[float]:
        if self._graph_key != _graph_key(graph):
            self.begin_episode(graph, hardware)

        frontier = state.frontier[:1] if state.frontier else []
        lookahead = state.lookahead[1 : 1 + self.weights.lookahead_limit]
        snapshots = self._snapshots_for_state(state, hardware)
        current_frontier = self._sum_weighted(graph, state.layout, frontier, snapshots)

        scores: list[float] = []
        for idx, edge in enumerate(state.candidate_swaps):
            if not state.action_mask[idx]:
                scores.append(float("inf"))
                continue
            layout_after = _swapped_layout(state.layout, *edge)
            frontier_after = self._sum_weighted(graph, layout_after, frontier, snapshots)
            lookahead_after = self._sum_weighted(graph, layout_after, lookahead, snapshots)
            decay_penalty = self.weights.decay_weight * self._decay_penalty(edge, state.step_count)
            stagnation_penalty = (
                self.weights.stagnation_weight if frontier_after >= current_frontier else 0.0
            )
            score = (
                frontier_after
                + self.weights.lookahead_weight * lookahead_after
                + decay_penalty
                + stagnation_penalty
            )
            scores.append(score)
        return scores

    def update_after_action(self, edge: tuple[int, int], step_count: int) -> None:
        w = self.weights
        for qubit in edge:
            current, last = self._qubit_decay.get(qubit, (0.0, step_count))
            if step_count > last:
                current *= w.decay_factor ** (step_count - last)
            current += w.decay_increment
            self._qubit_decay[qubit] = (current, step_count)

    # --------------------------------------------------------------- scoring --
    def _snapshots_for_state(
        self, state: RoutingState, hardware: HardwareModel | None
    ) -> list[int]:
        if hardware is None:
            return [0]

        if self.snapshot_mode == "bucket":
            snapshot_idx = hardware.snapshot_index_for_time(
                state.step_count * hardware.snapshot_spacing_ns
            )
            return [snapshot_idx]

        num_snapshots = len(hardware.snapshots)
        if num_snapshots <= 1:
            return [0]
        return [0, 1]

    def _sum_weighted(
        self,
        graph: nx.Graph,
        layout: dict[int, int],
        gates: Iterable[LogicalGate],
        snapshots: Sequence[int],
    ) -> float:
        if not gates:
            return 0.0
        total = 0.0
        cache = self._distance_cache
        for gate in gates:
            phys = (layout[gate[0]], layout[gate[1]])
            if cache is None:
                try:
                    dist = float(nx.shortest_path_length(graph, *phys))
                except nx.NetworkXNoPath:
                    dist = float(graph.number_of_nodes())
                total += dist
                continue

            dists = []
            for snap in snapshots:
                try:
                    dists.append(cache.dist(phys[0], phys[1], snapshot_id=snap))
                except IndexError:
                    dists.append(cache.dist(phys[0], phys[1], snapshot_id=0))
            total += self._distance_scale * (sum(dists) / max(1, len(dists)))
        return total

    def _decay_penalty(self, edge: tuple[int, int], step_count: int) -> float:
        return sum(self._effective_decay(q, step_count) for q in edge)

    def _effective_decay(self, qubit: int, step_count: int) -> float:
        value, last = self._qubit_decay.get(qubit, (0.0, step_count))
        if step_count <= last:
            return value
        return value * (self.weights.decay_factor ** (step_count - last))


# ------------------------------------------------------------------ rollout --
def route_with_weighted_sabre(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
    *,
    hardware_model: HardwareModel,
    seed: int | None = None,
    trials: int = 8,
    router_weights: WeightedSabreWeights | None = None,
    distance_params: WeightedDistanceParams | None = None,
    snapshot_mode: str = "avg",
    env_config: RoutingEnvConfig | None = None,
) -> BaselineResult:
    """Run multiple Weighted-SABRE trials and return the best by overall_log_success."""

    if hardware_model is None:
        msg = "hardware_model is required for weighted SABRE routing."
        raise ValueError(msg)

    env_cfg = env_config or RoutingEnvConfig(frontier_size=4)
    base_layout = _sabre_initial_layout(circuit, coupling_map, seed=seed)
    rng = random.Random(seed)

    best: BaselineResult | None = None
    cmap_edges = _normalize_edges(coupling_map)

    for trial in range(max(1, trials)):
        trial_seed = None if seed is None else seed + trial
        layout = _sabre_initial_layout(circuit, coupling_map, seed=trial_seed) or base_layout
        if layout is None:
            layout = base_layout
        if layout is not None and trial > 0 and circuit.num_qubits >= 2 and layout == base_layout:
            layout = _perturb_layout(layout, rng)

        env = RoutingEnv(env_cfg)
        state = env.reset(
            circuit,
            coupling_map,
            seed=trial_seed,
            hardware_model=hardware_model,
            initial_layout=layout,
        )
        graph = env.graph
        if graph is None:
            msg = "RoutingEnv did not provide a coupling graph."
            raise RuntimeError(msg)

        router = WeightedSabreRouter(
            weights=router_weights,
            distance_params=distance_params,
            snapshot_mode=snapshot_mode,
            rng_seed=trial_seed,
        )
        router.begin_episode(graph, env.hardware_model)

        start = time.perf_counter()
        step_budget = max(200, len(circuit.data) * 10)
        steps = 0
        while not state.done and steps < step_budget:
            action_idx = router.select_action(state, graph, env.hardware_model)
            swap_edge = state.candidate_swaps[action_idx]
            state, _, _, _ = env.step(action_idx)
            router.update_after_action(swap_edge, state.step_count)
            steps += 1

        runtime = time.perf_counter() - start
        if not state.done:
            msg = f"Weighted SABRE routing exceeded step budget {step_budget}"
            raise RuntimeError(msg)

        routed = env.routed_circuit
        assert_coupling_compatible(routed, cmap_edges)
        metrics = compute_metrics(routed, hardware_model=hardware_model)
        result = BaselineResult(
            name="weighted_sabre",
            circuit=routed,
            metrics=metrics,
            runtime_s=runtime,
            seed=trial_seed,
            extra={
                "trial": trial,
                "trials_total": max(1, trials),
                "alpha_time": (distance_params.alpha_time if distance_params else 0.0),
                "beta_xtalk": (distance_params.beta_xtalk if distance_params else 0.0),
                "snapshot_mode": snapshot_mode,
            },
        )
        best = _choose_best_trial(best, result)

    assert best is not None
    return best


# --------------------------------------------------------------------------- #
# Helpers


def _choose_best_trial(current: BaselineResult | None, cand: BaselineResult) -> BaselineResult:
    if current is None:
        return cand

    tol = 1e-3
    cur_log = current.metrics.overall_log_success
    cand_log = cand.metrics.overall_log_success
    cur_log = -float("inf") if cur_log is None else float(cur_log)
    cand_log = -float("inf") if cand_log is None else float(cand_log)
    if abs(cur_log - cand_log) > tol:
        return current if cur_log >= cand_log else cand
    if current.metrics.swaps != cand.metrics.swaps:
        return current if current.metrics.swaps <= cand.metrics.swaps else cand
    if current.metrics.two_qubit_depth != cand.metrics.two_qubit_depth:
        return current if current.metrics.two_qubit_depth <= cand.metrics.two_qubit_depth else cand
    return current


def _graph_key(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(u, v), max(u, v)) for u, v in graph.edges()))


def _perturb_layout(layout: dict[int, int], rng: random.Random) -> dict[int, int]:
    if len(layout) < 2:
        return dict(layout)
    new_layout = dict(layout)
    a, b = rng.sample(list(new_layout.keys()), 2)
    new_layout[a], new_layout[b] = new_layout[b], new_layout[a]
    return new_layout


def _edge_scale(hardware: HardwareModel, params: WeightedDistanceParams) -> float:
    weights = []
    for u, v in hardware.adjacency:
        weights.append(compute_edge_weight(u, v, hardware, 0, params))
        weights.append(compute_edge_weight(v, u, hardware, 0, params))
    filtered = [w for w in weights if w > 0]
    if not filtered:
        return 1.0
    median = statistics.median(filtered)
    if median <= 0:
        return 1.0
    scale = 1.0 / median
    return scale if scale >= 1.0 else 1.0


def _swapped_layout(layout: dict[int, int], u: int, v: int) -> dict[int, int]:
    new_layout = dict(layout)
    logical_u = _logical_on_physical(layout, u)
    logical_v = _logical_on_physical(layout, v)
    if logical_u is not None:
        new_layout[logical_u] = v
    if logical_v is not None:
        new_layout[logical_v] = u
    return new_layout


def _logical_on_physical(layout: dict[int, int], physical: int) -> int | None:
    for logical, phys in layout.items():
        if phys == physical:
            return logical
    return None


__all__ = ["WeightedSabreRouter", "WeightedSabreWeights", "route_with_weighted_sabre"]
