"""Utilities to make circuits safe for routing passes.

The SABRE-style routers in this project only handle single- and two-qubit
operations. Some benchmark circuits include wider gates (e.g., multi-controlled
operations). Those should be decomposed into the target basis before routing so
we do not crash on platforms that eagerly validate ``instruction.num_qubits``.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from qiskit import transpile
from qiskit.circuit import CircuitInstruction, QuantumCircuit


AllowedMultiQ = {"barrier"}


def _unsupported_multiq(inst: CircuitInstruction) -> bool:
    """Return True when an instruction is an unsupported >2Q operation."""

    if inst.operation.name in AllowedMultiQ:
        return False
    return inst.operation.num_qubits is not None and inst.operation.num_qubits > 2


def _remaining_multiq(circuit: QuantumCircuit) -> list[CircuitInstruction]:
    return [inst for inst in circuit.data if _unsupported_multiq(inst)]


def _norm_metadata(
    normalized: bool,
    iters: int,
    remaining: Sequence[CircuitInstruction],
) -> dict:
    return {
        "normalized": bool(normalized),
        "iters": int(iters),
        "remaining_multiq": len(remaining),
        "remaining_ops": sorted({inst.operation.name for inst in remaining}),
        "skipped_reason": "UNSUPPORTED_MULTI_QUBIT" if remaining else None,
    }


def normalize_for_routing(
    circuit: QuantumCircuit,
    basis_1q: Iterable[str] | None = None,
    basis_2q: Iterable[str] | None = None,
    max_iters: int = 6,
) -> Tuple[QuantumCircuit, dict]:
    """Best-effort normalization before routing.

    Returns a **copy** of the input circuit plus a metadata dictionary. The
    caller is responsible for skipping circuits when ``remaining_multiq`` is
    non-zero.
    """

    one_q = list(basis_1q) if basis_1q is not None else ["rz", "sx", "x"]
    two_q = list(basis_2q) if basis_2q is not None else ["cx"]
    basis_gates = one_q + two_q + ["measure", "reset", "barrier"]

    working = circuit.copy()
    normalized = False
    iterations = 0

    for _ in range(max(1, max_iters)):
        remaining = _remaining_multiq(working)
        if not remaining:
            break

        iterations += 1
        normalized = True

        transpiled = None
        try:
            transpiled = transpile(working, basis_gates=basis_gates, optimization_level=0)
        except Exception:
            transpiled = None

        if transpiled is not None:
            working = transpiled
            continue

        try:
            working = working.decompose(reps=1)
        except Exception:
            # Give up for this iteration; we'll exit once max_iters is hit.
            pass

    remaining = _remaining_multiq(working)
    metadata = _norm_metadata(normalized, iterations, remaining)
    return working, metadata
