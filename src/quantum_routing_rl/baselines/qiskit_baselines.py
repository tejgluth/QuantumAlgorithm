"""Qiskit SABRE-family baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.eval.metrics import (
    CircuitMetrics,
    assert_coupling_compatible,
    compute_metrics,
)


@dataclass
class BaselineResult:
    """Container with routed circuit and metrics."""

    name: str
    circuit: QuantumCircuit
    metrics: CircuitMetrics
    runtime_s: float
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Flatten into a dictionary suitable for CSV/JSON logging."""
        record: dict[str, Any] = {
            "baseline": self.name,
            "runtime_s": self.runtime_s,
            "seed": self.seed,
        }
        record.update(self.metrics.as_dict())
        record.update(self.extra)
        return record


def run_sabre_layout_swap(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    optimization_level: int = 1,
) -> BaselineResult:
    """SabreLayout + SabreSwap baseline via ``transpile``."""
    transpile_opts = dict(
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    return _run_baseline("sabre_layout_swap", circuit, transpile_opts, coupling_map)


def run_best_available_sabre(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
) -> BaselineResult:
    """Use Qiskit's highest preset (optimization_level=3) with SABRE routing."""
    transpile_opts = dict(
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=3,
        seed_transpiler=seed,
    )
    return _run_baseline("qiskit_sabre_best", circuit, transpile_opts, coupling_map)


def _run_baseline(
    name: str,
    circuit: QuantumCircuit,
    transpile_opts: dict[str, Any],
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
) -> BaselineResult:
    start = time.perf_counter()
    routed = transpile(circuit, **transpile_opts)
    runtime = time.perf_counter() - start

    cmap_edges: list[tuple[int, int]] | None = None
    if coupling_map is not None:
        cmap_edges = _normalize_edges(coupling_map)
        assert_coupling_compatible(routed, cmap_edges)

    metrics = compute_metrics(routed)
    extra = {
        k: v
        for k, v in transpile_opts.items()
        if k not in {"seed_transpiler", "coupling_map"} and v is not None
    }
    if cmap_edges is not None:
        extra["coupling_map"] = cmap_edges
    return BaselineResult(
        name=name,
        circuit=routed,
        metrics=metrics,
        runtime_s=runtime,
        seed=transpile_opts.get("seed_transpiler"),
        extra=extra,
    )


def _normalize_edges(coupling: CouplingMap | Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(coupling, CouplingMap):
        return [tuple(edge) for edge in coupling.get_edges()]
    return [tuple(edge) for edge in coupling]
