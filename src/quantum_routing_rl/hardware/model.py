"""Hardware noise/timing model used for hardware-aware routing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import networkx as nx

Edge = Tuple[int, int]


@dataclass(frozen=True)
class EdgeProps:
    """Per-edge two-qubit properties."""

    p2_error: float
    t2_duration_ns: float
    # Optional directional overrides (unused for now but allow future extension).
    p2_error_fwd: float | None = None
    p2_error_rev: float | None = None
    t2_duration_fwd_ns: float | None = None
    t2_duration_rev_ns: float | None = None


@dataclass(frozen=True)
class QubitProps:
    """Per-qubit coherence and error properties."""

    t1_ns: float
    t2_ns: float
    p1_error: float
    readout_error: float


class HardwareModel:
    """Container for hardware-specific properties on a coupling graph."""

    def __init__(
        self,
        graph_id: str,
        adjacency: Iterable[Edge],
        edge_props: Dict[Edge, EdgeProps],
        qubit_props: Dict[int, QubitProps],
    ):
        edges = [self._edge_key(*e) for e in adjacency]
        self.graph_id = graph_id
        self.adjacency = sorted(set(edges))
        self.edge_props = {self._edge_key(*k): v for k, v in edge_props.items()}
        self.qubit_props = dict(qubit_props)

    # ------------------------------------------------------------------ build
    @classmethod
    def synthetic(
        cls,
        graph: nx.Graph | Iterable[Edge],
        *,
        seed: int,
        profile: str = "realistic",
    ) -> "HardwareModel":
        """Sample a realistic-ish hardware model deterministically from ``seed``."""
        if isinstance(graph, nx.Graph):
            g = graph.copy()
        else:
            g = nx.Graph()
            g.add_edges_from(graph)
        g = g.to_undirected()
        rng = random.Random(seed)

        edges = [cls._edge_key(u, v) for u, v in g.edges()]
        nodes = sorted(int(n) for n in g.nodes())

        bad_edges = set()
        if edges:
            num_bad = max(1, int(0.15 * len(edges))) if profile == "realistic" else 0
            bad_edges = set(rng.sample(edges, num_bad)) if num_bad > 0 else set()

        edge_props: Dict[Edge, EdgeProps] = {}
        for edge in edges:
            if edge in bad_edges:
                p2_error = rng.uniform(0.02, 0.05)  # noticeably bad edges
            else:
                p2_error = rng.uniform(0.003, 0.008)  # ~0.5% typical
            t2_duration_ns = rng.uniform(200.0, 600.0)
            edge_props[edge] = EdgeProps(p2_error=p2_error, t2_duration_ns=t2_duration_ns)

        qubit_props: Dict[int, QubitProps] = {}
        for q in nodes:
            t1_ns = rng.uniform(20_000.0, 200_000.0)  # 20-200 microseconds in ns
            t2_ns = rng.uniform(t1_ns * 0.4, t1_ns * 0.9)
            p1_error = rng.uniform(1e-4, 5e-3)
            readout_error = rng.uniform(0.01, 0.05)
            qubit_props[q] = QubitProps(
                t1_ns=t1_ns,
                t2_ns=t2_ns,
                p1_error=p1_error,
                readout_error=readout_error,
            )

        graph_id = g.graph.get("name") or getattr(g, "name", None) or "synthetic"
        return cls(
            graph_id=graph_id, adjacency=edges, edge_props=edge_props, qubit_props=qubit_props
        )

    # ----------------------------------------------------------------- helpers
    def get_edge_props(self, u: int, v: int) -> EdgeProps:
        """Return properties for an undirected edge (u, v)."""
        key = self._edge_key(u, v)
        if key not in self.edge_props:
            msg = f"Edge {key} not present in hardware model for {self.graph_id}"
            raise KeyError(msg)
        return self.edge_props[key]

    def get_qubit_props(self, qubit: int) -> QubitProps:
        """Return properties for a qubit."""
        if qubit not in self.qubit_props:
            msg = f"Qubit {qubit} not present in hardware model for {self.graph_id}"
            raise KeyError(msg)
        return self.qubit_props[qubit]

    @staticmethod
    def _edge_key(u: int, v: int) -> Edge:
        return (u, v) if u <= v else (v, u)


__all__ = ["EdgeProps", "QubitProps", "HardwareModel"]
