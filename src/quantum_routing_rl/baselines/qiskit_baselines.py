"""Qiskit routing baselines (SABRE family + controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quantum_routing_rl.eval.metrics import (
    CircuitMetrics,
    assert_coupling_compatible,
    compute_metrics,
)
from quantum_routing_rl.routing.normalize_circuit import normalize_for_routing


@dataclass
class BaselineResult:
    """Container with routed circuit and metrics."""

    name: str
    circuit: QuantumCircuit | None
    metrics: CircuitMetrics | None
    runtime_s: float
    seed: int | None = None
    baseline_status: str = "ok"
    fallback_used: bool = False
    fallback_reason: str | None = None
    skip_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Flatten into a dictionary suitable for CSV/JSON logging."""
        record: dict[str, Any] = {
            "baseline": self.name,
            "runtime_s": self.runtime_s,
            "seed": self.seed,
            "baseline_status": self.baseline_status,
            "fallback_used": self.fallback_used,
        }
        if self.fallback_reason:
            record["fallback_reason"] = self.fallback_reason
        if self.skip_reason:
            record["skip_reason"] = self.skip_reason
        if self.metrics is not None:
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


def run_basic_swap(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    """BasicSwap routing baseline."""
    transpile_opts = dict(
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        routing_method="basic",
        optimization_level=1,
        seed_transpiler=seed,
    )
    return _run_baseline(
        "qiskit_basic_swap", circuit, transpile_opts, coupling_map, hardware_model=hardware_model
    )


def run_lookahead_swap(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    """LookaheadSwap routing baseline."""
    transpile_opts = dict(
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        routing_method="lookahead",
        optimization_level=2,
        seed_transpiler=seed,
    )
    return _run_baseline(
        "qiskit_lookahead_swap",
        circuit,
        transpile_opts,
        coupling_map,
        hardware_model=hardware_model,
    )


def run_commuting_2q_router(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    """Commuting2qGateRouter baseline (best-effort)."""
    name = "qiskit_commuting_2q_router"
    circuit, norm_meta = normalize_for_routing(circuit)
    norm_extra = {
        "normalized": norm_meta.get("normalized", False),
        "normalization_iters": norm_meta.get("iters", 0),
        "remaining_multiq": norm_meta.get("remaining_multiq", 0),
        "remaining_ops": "|".join(norm_meta.get("remaining_ops", [])),
    }
    if norm_meta.get("remaining_multiq", 0) > 0:
        return BaselineResult(
            name,
            circuit=None,
            metrics=None,
            runtime_s=0.0,
            seed=seed,
            baseline_status="SKIPPED",
            skip_reason=norm_meta.get("skipped_reason"),
            extra=norm_extra,
        )
    try:
        from qiskit.transpiler.passes import Commuting2qBlock, Commuting2qGateRouter
        from qiskit.transpiler import PassManager
    except Exception as exc:  # pragma: no cover - optional import path
        return _finalize_result(
            name,
            None,
            coupling_map,
            runtime=0.0,
            seed=seed,
            hardware_model=hardware_model,
            baseline_status="not_available",
            fallback_reason=str(exc),
            extra={},
        )

    if not any(instr.operation.num_qubits == 2 for instr in circuit.data):
        return _finalize_result(
            name,
            circuit,
            coupling_map,
            runtime=0.0,
            seed=seed,
            hardware_model=hardware_model,
            baseline_status="not_applicable",
            fallback_reason="no_two_qubit_gates",
            extra=norm_extra,
        )

    pm = PassManager([Commuting2qBlock(), Commuting2qGateRouter(coupling_map=coupling_map)])
    start = time.perf_counter()
    try:
        routed = pm.run(circuit)
    except Exception as exc:  # pragma: no cover - defensive
        runtime = time.perf_counter() - start
        return _finalize_result(
            name,
            None,
            coupling_map,
            runtime=runtime,
            seed=seed,
            hardware_model=hardware_model,
            baseline_status="failed",
            fallback_reason=str(exc),
            extra=norm_extra,
        )
    runtime = time.perf_counter() - start
    norm_extra["pass_manager"] = "Commuting2qGateRouter"
    return _finalize_result(
        name,
        routed,
        coupling_map,
        runtime=runtime,
        seed=seed,
        hardware_model=hardware_model,
        extra=norm_extra,
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
    """Run Qiskit SABRE multiple times and pick best by swaps/twoq_depth/depth/duration."""

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
        result.name = baseline_name
        extra = dict(getattr(result, "extra", {}) or {})
        extra.update({"trial": trial_idx, "trials_total": trial_budget})
        result.extra = extra
        best = _choose_qiskit_best_trial(best, result)

    if best is None:  # pragma: no cover - defensive
        msg = "No SABRE trials produced a result."
        raise RuntimeError(msg)
    return best


def run_preset_opt3(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    basis_gates: Sequence[str] | None = None,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    """Preset pass manager at optimization level 3 (non-SABRE control)."""
    baseline_name = "qiskit_preset_opt3"
    try:
        pm = generate_preset_pass_manager(
            optimization_level=3,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            seed_transpiler=seed,
        )
        return _run_pass_manager(
            baseline_name,
            circuit,
            pm,
            coupling_map,
            seed=seed,
            hardware_model=hardware_model,
        )
    except Exception as exc:
        transpile_opts = dict(
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )
        return _run_baseline(
            baseline_name,
            circuit,
            transpile_opts,
            coupling_map,
            hardware_model=hardware_model,
            fallback_reason=str(exc),
            baseline_status="fallback",
        )


# --------------------------------------------------------------------------- helpers
def _run_baseline(
    name: str,
    circuit: QuantumCircuit,
    transpile_opts: dict[str, Any],
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    hardware_model=None,
    *,
    fallback_reason: str | None = None,
    baseline_status: str = "ok",
) -> BaselineResult:
    circuit, norm_meta = normalize_for_routing(circuit)
    norm_extra = {
        "normalized": norm_meta.get("normalized", False),
        "normalization_iters": norm_meta.get("iters", 0),
        "remaining_multiq": norm_meta.get("remaining_multiq", 0),
        "remaining_ops": "|".join(norm_meta.get("remaining_ops", [])),
    }
    if norm_meta.get("remaining_multiq", 0) > 0:
        return BaselineResult(
            name,
            circuit=None,
            metrics=None,
            runtime_s=0.0,
            seed=transpile_opts.get("seed_transpiler"),
            baseline_status="SKIPPED",
            skip_reason=norm_meta.get("skipped_reason"),
            extra=norm_extra,
        )

    fallback_used = False
    start = time.perf_counter()
    routed: QuantumCircuit | None = None
    try:
        routed = transpile(circuit, **transpile_opts)
    except TranspilerError as exc:
        fallback_used = True
        fallback_reason = fallback_reason or f"transpile_error:{exc.__class__.__name__}"
        fallback_opts = dict(transpile_opts)
        fallback_opts["optimization_level"] = 1
        try:
            routed = transpile(circuit, **fallback_opts)
        except Exception as exc2:  # pragma: no cover - defensive
            runtime = time.perf_counter() - start
            return _finalize_result(
                name,
                None,
                coupling_map,
                runtime=runtime,
                seed=transpile_opts.get("seed_transpiler"),
                hardware_model=hardware_model,
                baseline_status="failed",
                fallback_used=True,
                fallback_reason=str(exc2),
                skip_reason=None,
                extra=dict(norm_extra),
            )
    runtime = time.perf_counter() - start

    extra = {
        k: v
        for k, v in transpile_opts.items()
        if k not in {"seed_transpiler", "coupling_map"} and v is not None
    }
    extra.update(norm_extra)
    return _finalize_result(
        name,
        routed,
        coupling_map,
        runtime=runtime,
        seed=transpile_opts.get("seed_transpiler"),
        hardware_model=hardware_model,
        baseline_status=baseline_status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        skip_reason=None,
        extra=extra,
    )


def _run_pass_manager(
    name: str,
    circuit: QuantumCircuit,
    pass_manager,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    circuit, norm_meta = normalize_for_routing(circuit)
    norm_extra = {
        "normalized": norm_meta.get("normalized", False),
        "normalization_iters": norm_meta.get("iters", 0),
        "remaining_multiq": norm_meta.get("remaining_multiq", 0),
        "remaining_ops": "|".join(norm_meta.get("remaining_ops", [])),
    }
    if norm_meta.get("remaining_multiq", 0) > 0:
        return BaselineResult(
            name,
            circuit=None,
            metrics=None,
            runtime_s=0.0,
            seed=seed,
            baseline_status="SKIPPED",
            skip_reason=norm_meta.get("skipped_reason"),
            extra=norm_extra,
        )

    start = time.perf_counter()
    try:
        routed = pass_manager.run(circuit)
    except Exception as exc:  # pragma: no cover - defensive
        runtime = time.perf_counter() - start
        return _finalize_result(
            name,
            None,
            coupling_map,
            runtime=runtime,
            seed=seed,
            hardware_model=hardware_model,
            baseline_status="failed",
            fallback_reason=str(exc),
            skip_reason=None,
            extra=dict(norm_extra),
        )
    runtime = time.perf_counter() - start
    extra = {"pass_manager": pass_manager.__class__.__name__}
    extra.update(norm_extra)
    return _finalize_result(
        name,
        routed,
        coupling_map,
        runtime=runtime,
        seed=seed,
        hardware_model=hardware_model,
        skip_reason=None,
        extra=extra,
    )


def _finalize_result(
    name: str,
    circuit: QuantumCircuit | None,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    runtime: float,
    seed: int | None,
    hardware_model=None,
    baseline_status: str = "ok",
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    skip_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> BaselineResult:
    cmap_edges: list[tuple[int, int]] | None = None
    if coupling_map is not None:
        cmap_edges = _normalize_edges(coupling_map)
    status = baseline_status
    if circuit is not None and cmap_edges is not None:
        try:
            assert_coupling_compatible(circuit, cmap_edges)
        except AssertionError as exc:  # pragma: no cover - defensive
            status = "invalid_coupling"
            fallback_reason = fallback_reason or str(exc)
    metrics = (
        compute_metrics(circuit, hardware_model=hardware_model) if circuit is not None else None
    )
    merged_extra = dict(extra or {})
    if cmap_edges is not None:
        merged_extra.setdefault("coupling_map", cmap_edges)
    return BaselineResult(
        name=name,
        circuit=circuit,
        metrics=metrics,
        runtime_s=runtime,
        seed=seed,
        baseline_status=status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        skip_reason=skip_reason,
        extra=merged_extra,
    )


def _normalize_edges(coupling: CouplingMap | Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(coupling, CouplingMap):
        return [tuple(edge) for edge in coupling.get_edges()]
    return [tuple(edge) for edge in coupling]


def _choose_qiskit_best_trial(
    current: BaselineResult | None, candidate: BaselineResult
) -> BaselineResult:
    """Select the better trial by swaps -> twoq_depth -> depth -> duration."""
    if current is None:
        return candidate
    if candidate.metrics is None:
        return current
    if current.metrics is None:
        return candidate

    if candidate.metrics.swaps != current.metrics.swaps:
        return candidate if candidate.metrics.swaps < current.metrics.swaps else current

    if candidate.metrics.two_qubit_depth != current.metrics.two_qubit_depth:
        return (
            candidate
            if candidate.metrics.two_qubit_depth < current.metrics.two_qubit_depth
            else current
        )

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

    return current
