"""Hardware noise/timing model used for hardware-aware routing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx

Edge = Tuple[int, int]


@dataclass(frozen=True)
class EdgeProps:
    """Per-edge two-qubit properties."""

    p2_error: float
    t2_duration_ns: float
    # Optional directional overrides (used when directional_mode=True).
    p2_error_fwd: float | None = None
    p2_error_rev: float | None = None
    t2_duration_fwd_ns: float | None = None
    t2_duration_rev_ns: float | None = None

    def error_and_duration(self, forward: bool, directional_mode: bool) -> tuple[float, float]:
        """Return (error, duration_ns) respecting directionality when enabled."""
        if directional_mode:
            if forward:
                error = self.p2_error_fwd if self.p2_error_fwd is not None else self.p2_error
                duration = (
                    self.t2_duration_fwd_ns
                    if self.t2_duration_fwd_ns is not None
                    else self.t2_duration_ns
                )
            else:
                error = self.p2_error_rev if self.p2_error_rev is not None else self.p2_error
                duration = (
                    self.t2_duration_rev_ns
                    if self.t2_duration_rev_ns is not None
                    else self.t2_duration_ns
                )
        else:
            error = self.p2_error
            duration = self.t2_duration_ns
        return error, duration


@dataclass(frozen=True)
class QubitProps:
    """Per-qubit coherence and error properties."""

    t1_ns: float
    t2_ns: float
    p1_error: float
    readout_error: float
    p1_duration_ns: float


@dataclass(frozen=True)
class HardwareSnapshot:
    """Piecewise-constant calibration snapshot used to model drift."""

    edge_props: Dict[Edge, EdgeProps]
    qubit_props: Dict[int, QubitProps]
    label: str


class HardwareModel:
    """Container for hardware-specific properties on a coupling graph."""

    def __init__(
        self,
        graph_id: str,
        adjacency: Iterable[Edge],
        edge_props: Dict[Edge, EdgeProps],
        qubit_props: Dict[int, QubitProps],
        *,
        snapshots: List[HardwareSnapshot] | None = None,
        directional_mode: bool = False,
        drift_rate: float = 0.0,
        snapshot_spacing_ns: float = 50_000.0,
        crosstalk_factor: float = 0.0,
    ):
        edges = [self._edge_key(*e) for e in adjacency]
        self.graph_id = graph_id
        self.adjacency = sorted(set(edges))
        self.edge_props = {self._edge_key(*k): v for k, v in edge_props.items()}
        self.qubit_props = dict(qubit_props)
        self.directional_mode = directional_mode
        self.drift_rate = drift_rate
        self.snapshot_spacing_ns = snapshot_spacing_ns
        self.crosstalk_factor = max(0.0, float(crosstalk_factor))
        base_snapshot = HardwareSnapshot(
            edge_props=self.edge_props, qubit_props=self.qubit_props, label="t0"
        )
        self.snapshots = snapshots or [base_snapshot]

    # ------------------------------------------------------------------ build
    @classmethod
    def synthetic(
        cls,
        graph: nx.Graph | Iterable[Edge],
        *,
        seed: int,
        profile: str = "realistic",
        directional: bool = False,
        drift_rate: float = 0.0,
        snapshots: int = 1,
        snapshot_spacing_ns: float = 50_000.0,
        crosstalk_factor: float = 0.01,
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
            if directional:
                # Directional variations keep errors/durations in a plausible range.
                fwd_scale = rng.uniform(0.8, 1.2)
                rev_scale = rng.uniform(0.8, 1.2)
                edge_props[edge] = EdgeProps(
                    p2_error=p2_error,
                    t2_duration_ns=t2_duration_ns,
                    p2_error_fwd=p2_error * fwd_scale,
                    p2_error_rev=p2_error * rev_scale,
                    t2_duration_fwd_ns=t2_duration_ns * rng.uniform(0.9, 1.15),
                    t2_duration_rev_ns=t2_duration_ns * rng.uniform(0.9, 1.15),
                )
            else:
                edge_props[edge] = EdgeProps(p2_error=p2_error, t2_duration_ns=t2_duration_ns)

        qubit_props: Dict[int, QubitProps] = {}
        for q in nodes:
            t1_ns = rng.uniform(20_000.0, 200_000.0)  # 20-200 microseconds in ns
            t2_ns = rng.uniform(t1_ns * 0.4, t1_ns * 0.9)
            p1_error = rng.uniform(1e-4, 5e-3)
            readout_error = rng.uniform(0.01, 0.05)
            p1_duration_ns = rng.uniform(20.0, 80.0)
            qubit_props[q] = QubitProps(
                t1_ns=t1_ns,
                t2_ns=t2_ns,
                p1_error=p1_error,
                readout_error=readout_error,
                p1_duration_ns=p1_duration_ns,
            )

        graph_id = g.graph.get("name") or getattr(g, "name", None) or "synthetic"
        base_model = cls(
            graph_id=graph_id,
            adjacency=edges,
            edge_props=edge_props,
            qubit_props=qubit_props,
            directional_mode=directional,
            drift_rate=drift_rate,
            snapshot_spacing_ns=snapshot_spacing_ns,
            crosstalk_factor=crosstalk_factor if profile == "realistic" else 0.0,
        )
        if snapshots <= 1 or drift_rate <= 0:
            return base_model

        snap_list: List[HardwareSnapshot] = []
        for idx in range(snapshots):
            if idx == 0:
                snap_list.append(
                    HardwareSnapshot(edge_props=edge_props, qubit_props=qubit_props, label="t0")
                )
                continue
            drift_rng = random.Random(seed + idx)
            drifted_edges: Dict[Edge, EdgeProps] = {}
            for edge, props in edge_props.items():
                drifted_edges[edge] = _drift_edge(props, drift_rate, drift_rng)
            drifted_qubits: Dict[int, QubitProps] = {}
            for q, props in qubit_props.items():
                drifted_qubits[q] = _drift_qubit(props, drift_rate, drift_rng)
            snap_list.append(
                HardwareSnapshot(
                    edge_props=drifted_edges, qubit_props=drifted_qubits, label=f"t{idx}"
                )
            )

        base_model.snapshots = snap_list
        return base_model

    # ----------------------------------------------------------------- helpers
    # Snapshot access -----------------------------------------------------
    def get_edge_props(self, u: int, v: int, snapshot: int | None = None) -> EdgeProps:
        """Return properties for an undirected edge (u, v)."""
        key = self._edge_key(u, v)
        snap = self._snapshot(snapshot)
        if key not in snap.edge_props:
            msg = f"Edge {key} not present in hardware model for {self.graph_id}"
            raise KeyError(msg)
        return snap.edge_props[key]

    def get_qubit_props(self, qubit: int, snapshot: int | None = None) -> QubitProps:
        """Return properties for a qubit."""
        snap = self._snapshot(snapshot)
        if qubit not in snap.qubit_props:
            msg = f"Qubit {qubit} not present in hardware model for {self.graph_id}"
            raise KeyError(msg)
        return snap.qubit_props[qubit]

    def edge_error_and_duration(
        self, u: int, v: int, *, snapshot: int | None = None, directed: bool = False
    ) -> tuple[float, float]:
        """Return (p_error, duration_ns) for an oriented edge."""
        key = self._edge_key(u, v)
        props = self.get_edge_props(u, v, snapshot=snapshot)
        forward = (u, v) == key
        error, duration = props.error_and_duration(forward=forward, directional_mode=directed)
        return error, duration

    def snapshot_index_for_time(self, time_ns: float) -> int:
        """Map a start time to a calibration snapshot index."""
        if self.snapshot_spacing_ns <= 0:
            return 0
        idx = int(time_ns // self.snapshot_spacing_ns)
        return min(max(idx, 0), len(self.snapshots) - 1)

    @staticmethod
    def _edge_key(u: int, v: int) -> Edge:
        return (u, v) if u <= v else (v, u)

    # Internal helpers ----------------------------------------------------
    def _snapshot(self, idx: int | None) -> HardwareSnapshot:
        if idx is None:
            return self.snapshots[0]
        if idx < 0 or idx >= len(self.snapshots):
            msg = f"Snapshot index {idx} out of range for {self.graph_id}"
            raise IndexError(msg)
        return self.snapshots[idx]


def _drift_edge(props: EdgeProps, rate: float, rng: random.Random) -> EdgeProps:
    def _jitter(val: float) -> float:
        delta = rng.uniform(-rate, rate)
        return max(1e-6, val * (1.0 + delta))

    return EdgeProps(
        p2_error=min(0.5, _jitter(props.p2_error)),
        t2_duration_ns=_jitter(props.t2_duration_ns),
        p2_error_fwd=_jitter(props.p2_error_fwd) if props.p2_error_fwd is not None else None,
        p2_error_rev=_jitter(props.p2_error_rev) if props.p2_error_rev is not None else None,
        t2_duration_fwd_ns=_jitter(props.t2_duration_fwd_ns)
        if props.t2_duration_fwd_ns is not None
        else None,
        t2_duration_rev_ns=_jitter(props.t2_duration_rev_ns)
        if props.t2_duration_rev_ns is not None
        else None,
    )


def _drift_qubit(props: QubitProps, rate: float, rng: random.Random) -> QubitProps:
    def _jitter(val: float) -> float:
        delta = rng.uniform(-rate, rate)
        return max(1e-6, val * (1.0 + delta))

    return QubitProps(
        t1_ns=_jitter(props.t1_ns),
        t2_ns=_jitter(props.t2_ns),
        p1_error=min(0.5, _jitter(props.p1_error)),
        readout_error=min(0.5, _jitter(props.readout_error)),
        p1_duration_ns=_jitter(props.p1_duration_ns),
    )


__all__ = ["EdgeProps", "QubitProps", "HardwareModel", "HardwareSnapshot"]
