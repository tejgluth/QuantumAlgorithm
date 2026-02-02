"""Weighted distance cache for SABRE-style heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import networkx as nx

from quantum_routing_rl.hardware.model import HardwareModel


@dataclass(frozen=True)
class WeightedDistanceParams:
    """Tunable weights for the hardware-aware distance metric."""

    alpha_time: float = 0.0
    beta_xtalk: float = 0.0
    directional: bool | None = None  # None -> use hardware_model.directional_mode
    min_edge_weight: float = 1.0


def compute_edge_weight(
    u: int,
    v: int,
    hardware_model: HardwareModel,
    snapshot_id: int | None,
    params: WeightedDistanceParams,
) -> float:
    """Return weighted cost of traversing edge u->v under snapshot_id."""

    directed = hardware_model.directional_mode if params.directional is None else params.directional
    try:
        p_error, duration_ns = hardware_model.edge_error_and_duration(
            u, v, snapshot=snapshot_id, directed=directed
        )
    except (KeyError, IndexError):
        p_error, duration_ns = 0.01, 200.0

    error_clamped = min(max(p_error, 0.0), 0.999999)
    base = -math.log(max(1e-12, 1.0 - error_clamped))

    t2_eff = _t2_effective(u, v, hardware_model, snapshot_id)
    time_term = params.alpha_time * (duration_ns / t2_eff)

    ct_term = params.beta_xtalk * _crosstalk_weight(u, v, hardware_model)

    weight = base + time_term + ct_term
    return max(weight, max(0.0, params.min_edge_weight))


class WeightedDistanceCache:
    """Cache of weighted shortest-path distances per hardware snapshot."""

    def __init__(
        self,
        hardware_model: HardwareModel,
        params: WeightedDistanceParams | None = None,
    ) -> None:
        self.hardware_model = hardware_model
        self.params = params or WeightedDistanceParams()
        self._dist_cache: dict[int, dict[int, dict[int, float]]] = {}
        self._nodes = _nodes_from_adjacency(hardware_model)

    # ------------------------------------------------------------------ API --
    def dist(self, u: int, v: int, snapshot_id: int = 0) -> float:
        """Return weighted shortest-path distance between two physical qubits."""

        self._ensure_snapshot(snapshot_id)
        return self._dist_cache[snapshot_id].get(u, {}).get(v, float("inf"))

    # -------------------------------------------------------------- internal --
    def _ensure_snapshot(self, snapshot_id: int) -> None:
        if snapshot_id in self._dist_cache:
            return
        if snapshot_id < 0 or snapshot_id >= len(self.hardware_model.snapshots):
            msg = f"Snapshot index {snapshot_id} out of range."
            raise IndexError(msg)
        graph = self._build_graph(snapshot_id)
        dist: Dict[int, Dict[int, float]] = {}
        for src in graph.nodes():
            dist[src] = nx.single_source_dijkstra_path_length(graph, src, weight="weight")
        self._dist_cache[snapshot_id] = dist

    def _build_graph(self, snapshot_id: int) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self._nodes)
        for a, b in self.hardware_model.adjacency:
            w_ab = compute_edge_weight(a, b, self.hardware_model, snapshot_id, self.params)
            w_ba = compute_edge_weight(b, a, self.hardware_model, snapshot_id, self.params)
            g.add_edge(a, b, weight=w_ab)
            g.add_edge(b, a, weight=w_ba)
        return g


# --------------------------------------------------------------------------- #
# Helpers


def _nodes_from_adjacency(hardware_model: HardwareModel) -> set[int]:
    nodes: set[int] = set()
    for edge in hardware_model.adjacency:
        nodes.update(edge)
    return nodes


def _t2_effective(u: int, v: int, hardware_model: HardwareModel, snapshot_id: int | None) -> float:
    default = 1e6
    try:
        q_u = hardware_model.get_qubit_props(u, snapshot=snapshot_id)
        q_v = hardware_model.get_qubit_props(v, snapshot=snapshot_id)
        t2_u = max(q_u.t2_ns, 1e-6)
        t2_v = max(q_v.t2_ns, 1e-6)
        return min(t2_u, t2_v)
    except (KeyError, IndexError):
        return default


def _crosstalk_weight(u: int, v: int, hardware_model: HardwareModel) -> float:
    if hardware_model.crosstalk_factor <= 0:
        return 0.0
    graph = nx.Graph()
    graph.add_edges_from(hardware_model.adjacency)
    if not graph.nodes:
        return 0.0
    max_degree = max((deg for _, deg in graph.degree()), default=1)
    deg_u = graph.degree[u] if u in graph else 0
    deg_v = graph.degree[v] if v in graph else 0
    return hardware_model.crosstalk_factor * ((deg_u + deg_v) / max(1, max_degree))


__all__ = ["WeightedDistanceParams", "WeightedDistanceCache", "compute_edge_weight"]
