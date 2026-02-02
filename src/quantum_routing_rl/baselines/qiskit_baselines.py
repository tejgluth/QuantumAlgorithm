"""Qiskit SABRE-family baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

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
    hardware_model=None,
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
    return _run_baseline(
        "sabre_layout_swap", circuit, transpile_opts, coupling_map, hardware_model=hardware_model
    )


def run_best_available_sabre(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    hardware_model=None,
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
    return _run_baseline(
        "qiskit_sabre_best", circuit, transpile_opts, coupling_map, hardware_model=hardware_model
    )


def run_qiskit_sabre_trials(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    trials: int = 8,
    hardware_model=None,
    trial_runner: Any | None = None,
) -> BaselineResult:
    """Run Qiskit SABRE multiple times and pick best by swap/depth/duration.

    Args:
        circuit: Circuit to route.
        coupling_map: Coupling constraint.
        basis_gates: Optional basis gates passed to ``transpile``.
        seed: Base seed; individual trials offset by +trial index.
        trials: Number of SABRE attempts to run.
        hardware_model: Optional hardware model for noise-aware metrics.
        trial_runner: Internal hook (tests) to supply pre-routed trials.
    """

    trial_budget = max(1, int(trials))
    baseline_name = f"qiskit_sabre_trials{trial_budget}"
    trial_seeds = [None if seed is None else seed + i for i in range(trial_budget)]

    best: BaselineResult | None = None
    for trial_idx, trial_seed in enumerate(trial_seeds):
        if trial_runner is not None:
            result: BaselineResult = trial_runner(trial_idx, trial_seed)
        else:
            transpile_opts = dict(
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=3,
                seed_transpiler=trial_seed,
            )
            result = _run_baseline(
                baseline_name,
                circuit,
                transpile_opts,
                coupling_map,
                hardware_model=hardware_model,
            )
        # Normalize metadata for downstream logging/selection.
        result.name = baseline_name
        extra = dict(getattr(result, "extra", {}) or {})
        extra.update({"trial": trial_idx, "trials_total": trial_budget})
        result.extra = extra
        best = _choose_qiskit_best_trial(best, result)

    if best is None:  # pragma: no cover - defensive
        msg = "No SABRE trials produced a result."
        raise RuntimeError(msg)
    return best


def _run_baseline(
    name: str,
    circuit: QuantumCircuit,
    transpile_opts: dict[str, Any],
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    hardware_model=None,
) -> BaselineResult:
    fallback_used = False
    start = time.perf_counter()
    try:
        routed = transpile(circuit, **transpile_opts)
    except TranspilerError:
        fallback_used = True
        fallback_opts = dict(transpile_opts)
        fallback_opts["optimization_level"] = 1
        routed = transpile(circuit, **fallback_opts)
    runtime = time.perf_counter() - start

    cmap_edges: list[tuple[int, int]] | None = None
    if coupling_map is not None:
        cmap_edges = _normalize_edges(coupling_map)
        assert_coupling_compatible(routed, cmap_edges)

    metrics = compute_metrics(routed, hardware_model=hardware_model)
    extra = {
        k: v
        for k, v in transpile_opts.items()
        if k not in {"seed_transpiler", "coupling_map"} and v is not None
    }
    if cmap_edges is not None:
        extra["coupling_map"] = cmap_edges
    if fallback_used:
        extra["fallback"] = "optimization_level_1"
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


def _choose_qiskit_best_trial(
    current: BaselineResult | None, candidate: BaselineResult
) -> BaselineResult:
    """Select the better trial by swaps -> depth -> duration -> twoq_depth."""
    if current is None:
        return candidate

    if candidate.metrics.swaps != current.metrics.swaps:
        return candidate if candidate.metrics.swaps < current.metrics.swaps else current

    if candidate.metrics.depth != current.metrics.depth:
        return candidate if candidate.metrics.depth < current.metrics.depth else current

    cur_dur = current.metrics.total_duration_ns
    cand_dur = candidate.metrics.total_duration_ns
    if cur_dur is not None or cand_dur is not None:
        if cur_dur is None:
            return candidate
        if cand_dur is None:
            return current
        if cand_dur != cur_dur:
            return candidate if cand_dur < cur_dur else current

    if candidate.metrics.two_qubit_depth != current.metrics.two_qubit_depth:
        return (
            candidate
            if candidate.metrics.two_qubit_depth < current.metrics.two_qubit_depth
            else current
        )

    return current
