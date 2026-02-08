"""Best-effort adapters for external routing libraries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult


def _normalize_edges(coupling: CouplingMap | Sequence[Sequence[int]] | None):
    if coupling is None:
        return None
    if isinstance(coupling, CouplingMap):
        return [tuple(edge) for edge in coupling.get_edges()]
    return [tuple(edge) for edge in coupling]


def _maybe_import(module: str, pip_name: str) -> tuple[object | None, str | None]:
    try:
        mod = __import__(module)
        return mod, None
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mod = __import__(module)
            return mod, None
        except Exception as exc2:
            return None, f"{module} unavailable: {exc2}"


def _skip_result(
    name: str,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    seed: int | None,
    reason: str,
    status: str = "not_available",
) -> BaselineResult:
    extra = {"external_reason": reason}
    cmap_edges = _normalize_edges(coupling_map)
    if cmap_edges is not None:
        extra["coupling_map"] = cmap_edges
    return BaselineResult(
        name=name,
        circuit=None,
        metrics=None,
        runtime_s=0.0,
        seed=seed,
        baseline_status=status,
        fallback_used=False,
        fallback_reason=reason,
        extra=extra,
    )


def run_cirq_routecqc(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    module, err = _maybe_import("cirq", "cirq")
    if module is None:
        return _skip_result("cirq_routecqc", coupling_map, seed, err or "cirq missing")
    return _skip_result(
        "cirq_routecqc",
        coupling_map,
        seed,
        "implementation pending; conversion not yet wired",
        status="not_available",
    )


def run_bqskit_generalized_sabre(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    module, err = _maybe_import("bqskit", "bqskit")
    if module is None:
        return _skip_result("bqskit_generalized_sabre", coupling_map, seed, err or "bqskit missing")
    return _skip_result(
        "bqskit_generalized_sabre",
        coupling_map,
        seed,
        "implementation pending; conversion not yet wired",
        status="not_available",
    )


def run_pytket_router(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap | Sequence[Sequence[int]] | None,
    *,
    seed: int | None = None,
    hardware_model=None,
) -> BaselineResult:
    module, err = _maybe_import("pytket", "pytket")
    if module is None:
        return _skip_result("pytket_router", coupling_map, seed, err or "pytket missing")
    return _skip_result(
        "pytket_router",
        coupling_map,
        seed,
        "implementation pending; conversion not yet wired",
        status="not_available",
    )


def write_status(status: dict, out_path: Path) -> None:
    """Persist external baseline status without breaking evaluation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(status, indent=2))
