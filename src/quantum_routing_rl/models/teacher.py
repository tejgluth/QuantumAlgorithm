"""SABRE-like teacher policy aligned with :class:`RoutingEnv` candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import networkx as nx
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.env.state import LogicalGate, RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, compute_metrics
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.routing.normalize_circuit import normalize_for_routing
from quantum_routing_rl.routing.classical_safe import classical_dependency_ops


@dataclass(frozen=True)
class TeacherWeights:
    """Weights for the teacher scoring function."""

    lookahead_weight: float = 0.5
    decay_weight: float = 0.25
    decay_factor: float = 0.9
    decay_increment: float = 1.0
    hardware_error_weight: float = 0.0
    hardware_time_weight: float = 0.0
    lookahead_limit: int = 8


class TeacherPolicy:
    """SABRE-inspired teacher scored on env candidates with tabu-like decay."""

    def __init__(self, weights: TeacherWeights | None = None):
        self.weights = weights or TeacherWeights()
        self._distance_cache: dict[tuple[Tuple[int, int], ...], dict[int, dict[int, int]]] = {}
        self._current_graph_key: tuple[Tuple[int, int], ...] | None = None
        # Store per-physical-qubit decay (value, last_step)
        self._qubit_decay: Dict[int, tuple[float, int]] = {}

    # ----------------------------------------------------------- lifecycle --
    def begin_episode(self, graph: nx.Graph) -> None:
        """Reset decay bookkeeping and cache distances for a new graph."""

        self._current_graph_key = _graph_key(graph)
        self._ensure_distance_matrix(graph)
        self._qubit_decay = {}

    def select_action(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None = None,
    ) -> int:
        """Return index of the best-scoring legal swap."""
        if state.done:
            return 0
        if not state.candidate_swaps:
            msg = "No candidate swaps available."
            raise RuntimeError(msg)

        scores = self.score_candidates(state, graph, hardware)
        finite_scores = [(idx, s) for idx, s in enumerate(scores) if s < float("inf")]
        if not finite_scores:
            msg = "All candidate actions are masked."
            raise RuntimeError(msg)
        best_idx, _ = min(finite_scores, key=lambda pair: (pair[1], pair[0]))
        return best_idx

    def score_candidates(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None = None,
    ) -> list[float]:
        """Return per-candidate scores aligned with ``state.candidate_swaps``."""

        if self._current_graph_key != _graph_key(graph):
            self.begin_episode(graph)

        frontier_gates = state.frontier[:1] if state.frontier else []
        current_frontier = self._sum_distances(graph, state.layout, frontier_gates)
        scores: list[float] = []
        for idx, edge in enumerate(state.candidate_swaps):
            if not state.action_mask[idx]:
                scores.append(float("inf"))
                continue
            score = self._score_edge(
                state,
                graph,
                hardware,
                edge,
                frontier_gates=frontier_gates,
                current_frontier=current_frontier,
            )
            scores.append(score)
        return scores

    def update_after_action(self, edge: Tuple[int, int], step_count: int) -> None:
        """Apply decay and record the chosen edge for future penalties."""

        w = self.weights
        for qubit in edge:
            current, last_step = self._qubit_decay.get(qubit, (0.0, step_count))
            if step_count > last_step:
                current *= w.decay_factor ** (step_count - last_step)
            current += w.decay_increment
            self._qubit_decay[qubit] = (current, step_count)

    # --------------------------------------------------------------- scoring --
    def _score_edge(
        self,
        state: RoutingState,
        graph: nx.Graph,
        hardware: HardwareModel | None,
        edge: Tuple[int, int],
        *,
        frontier_gates: Iterable[LogicalGate],
        current_frontier: float,
    ) -> float:
        logical_u = _logical_on_physical(state.layout, edge[0])
        logical_v = _logical_on_physical(state.layout, edge[1])
        if logical_u is None and logical_v is None:
            return float("inf")

        layout_after = _swapped_layout(state.layout, *edge)
        frontier_after = self._sum_distances(graph, layout_after, frontier_gates)

        lookahead = state.lookahead[1 : 1 + self.weights.lookahead_limit]
        lookahead_after = self._sum_distances(graph, layout_after, lookahead)

        decay_penalty = self.weights.decay_weight * self._decay_penalty(edge, state.step_count)
        stagnation_penalty = 0.25 if frontier_after >= current_frontier else 0.0

        hardware_penalty = 0.0
        hw_wt = (
            self.weights.hardware_error_weight,
            self.weights.hardware_time_weight,
        )
        if hardware and (hw_wt[0] > 0 or hw_wt[1] > 0):
            props = hardware.get_edge_props(*edge)
            if props is not None:
                hardware_penalty = hw_wt[0] * props.p2_error + hw_wt[1] * (
                    props.t2_duration_ns / 1000.0
                )

        return (
            frontier_after
            + self.weights.lookahead_weight * lookahead_after
            + decay_penalty
            + stagnation_penalty
            + hardware_penalty
        )

    def _sum_distances(
        self, graph: nx.Graph, layout: dict[int, int], gates: Iterable[LogicalGate]
    ) -> float:
        dist = 0.0
        for gate in gates:
            phys = (layout[gate[0]], layout[gate[1]])
            dist += self._distance(graph, *phys)
        return dist

    def _decay_penalty(self, edge: Tuple[int, int], step_count: int) -> float:
        """Tabu-like decay discouraging oscillations on same qubits."""

        return sum(self._effective_decay(qubit, step_count) for qubit in edge)

    def _effective_decay(self, qubit: int, step_count: int) -> float:
        value, last = self._qubit_decay.get(qubit, (0.0, step_count))
        if step_count <= last:
            return value
        return value * (self.weights.decay_factor ** (step_count - last))

    def _distance(self, graph: nx.Graph, a: int, b: int) -> float:
        self._ensure_distance_matrix(graph)
        assert self._current_graph_key is not None
        matrix = self._distance_cache[self._current_graph_key]
        try:
            return float(matrix[a][b])
        except KeyError:
            return float(graph.number_of_nodes())

    def _ensure_distance_matrix(self, graph: nx.Graph) -> None:
        key = _graph_key(graph)
        if key in self._distance_cache:
            self._current_graph_key = key
            return
        dist = {src: dict(lengths) for src, lengths in nx.all_pairs_shortest_path_length(graph)}
        self._distance_cache[key] = dist
        self._current_graph_key = key


# ------------------------------------------------------------------ rollout --
def route_with_teacher(
    circuit,
    coupling_map: CouplingMap | Iterable[Tuple[int, int]],
    *,
    name: str = "teacher_sabre_like",
    seed: int | None = None,
    env_config: RoutingEnvConfig | None = None,
    max_steps: int | None = None,
    hardware_model: HardwareModel | None = None,
) -> BaselineResult:
    """Greedy rollout using :class:`TeacherPolicy`."""
    circuit, norm_meta = normalize_for_routing(circuit)
    norm_extra = {
        "normalized": norm_meta.get("normalized", False),
        "normalization_iters": norm_meta.get("iters", 0),
        "remaining_multiq": norm_meta.get("remaining_multiq", 0),
        "remaining_ops": "|".join(norm_meta.get("remaining_ops", [])),
    }
    if norm_meta.get("remaining_multiq", 0) > 0:
        return BaselineResult(
            name=name,
            circuit=None,
            metrics=None,
            runtime_s=0.0,
            seed=seed,
            baseline_status="SKIPPED",
            skip_reason=norm_meta.get("skipped_reason"),
            extra=norm_extra,
        )
    classical_ops = classical_dependency_ops(circuit)
    if classical_ops:
        return BaselineResult(
            name=name,
            circuit=None,
            metrics=None,
            runtime_s=0.0,
            seed=seed,
            baseline_status="SKIPPED",
            skip_reason="UNSUPPORTED_CLASSICAL_CONTROL",
            extra={**norm_extra, "classical_control_ops": "|".join(classical_ops)},
        )

    env_cfg = env_config or RoutingEnvConfig(frontier_size=4)
    env = RoutingEnv(env_cfg)
    initial_layout = _sabre_initial_layout(circuit, coupling_map, seed=seed)
    state = env.reset(
        circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
    graph = nx.Graph(_normalize_edges(coupling_map))
    hardware = env.hardware_model
    teacher = TeacherPolicy()
    teacher.begin_episode(graph)
    start = time.perf_counter()
    step_budget = max_steps or max(200, len(circuit.data) * 10)
    steps = 0
    while not state.done and steps < step_budget:
        action = teacher.select_action(state, graph, hardware)
        swap_edge = state.candidate_swaps[action]
        state, _, _, _ = env.step(action)
        teacher.update_after_action(swap_edge, state.step_count)
        steps += 1

    if not state.done:
        msg = f"Teacher routing did not complete within {step_budget} steps"
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
        extra=norm_extra,
    )


# ---------------------------------------------------------- helper routines --
def _swapped_layout(layout: dict[int, int], u: int, v: int) -> dict[int, int]:
    new_layout = dict(layout)
    logical_on_u = _logical_on_physical(layout, u)
    logical_on_v = _logical_on_physical(layout, v)
    if logical_on_u is not None:
        new_layout[logical_on_u] = v
    if logical_on_v is not None:
        new_layout[logical_on_v] = u
    return new_layout


def _logical_on_physical(layout: dict[int, int], physical: int) -> int | None:
    for logical, phys in layout.items():
        if phys == physical:
            return logical
    return None


def _normalize_edges(
    coupling_map: CouplingMap | Iterable[Tuple[int, int]],
) -> list[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [tuple(edge) for edge in coupling_map]


def _graph_key(graph: nx.Graph) -> tuple[Tuple[int, int], ...]:
    """Deterministic signature for caching shortest paths."""

    return tuple(sorted((min(u, v), max(u, v)) for u, v in graph.edges()))


def _sabre_initial_layout(
    circuit,
    coupling_map: CouplingMap | Iterable[Tuple[int, int]],
    seed: int | None = None,
) -> dict[int, int] | None:
    """Run SabreLayout to obtain an initial mapping (best-effort)."""

    try:
        cmap_obj = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(list(coupling_map))
        )
        pm = PassManager(SabreLayout(coupling_map=cmap_obj, seed=seed))
        pm.run(circuit)
        layout = pm.property_set.get("layout")
        if layout is None:
            return None
        virt_to_phys = layout.get_virtual_bits()
        mapping: dict[int, int] = {}
        for virt, phys in virt_to_phys.items():
            try:
                logical_idx = circuit.find_bit(virt).index  # type: ignore[assignment]
            except Exception:
                continue
            mapping[int(logical_idx)] = int(phys)
        return mapping
    except Exception:
        return None


__all__ = ["TeacherPolicy", "TeacherWeights", "route_with_teacher"]
