"""Helpers to strip and rebuild circuits while handling classical bits safely."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

from qiskit.circuit import ClassicalRegister, QuantumCircuit


# Names of operations that should be removed before routing.
_NONUNITARY_NAMES = {"measure", "reset", "barrier", "delay"}


@dataclass(frozen=True)
class MeasurementMeta:
    """Lightweight record of a measurement."""

    index: int
    qubit: int
    clbit: int


@dataclass(frozen=True)
class NonUnitaryMeta:
    """Metadata for other non-unitary instructions (e.g., reset, barrier)."""

    index: int
    operation: Any
    qubits: Tuple[int, ...]
    clbits: Tuple[int, ...]


def strip_nonunitary(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[MeasurementMeta], list[NonUnitaryMeta]]:
    """Return a quantum-only copy of ``circuit`` plus metadata for restoration.

    The returned circuit contains only unitary operations (1Q/2Q) with all
    classical bits removed. Measurements and other non-unitary operations are
    captured in metadata so they can be reattached after routing.
    """

    quantum_only = QuantumCircuit(*circuit.qregs, name=circuit.name)
    measurements: list[MeasurementMeta] = []
    nonunitary: list[NonUnitaryMeta] = []

    for idx, inst in enumerate(circuit.data):
        name = inst.operation.name
        qubit_indices = tuple(circuit.find_bit(qb).index for qb in inst.qubits)
        clbit_indices = tuple(circuit.find_bit(cb).index for cb in inst.clbits)

        if name == "measure":
            # Expect 1:1 measure ops; record mapping by index.
            if not qubit_indices or not clbit_indices:
                continue
            measurements.append(MeasurementMeta(idx, qubit_indices[0], clbit_indices[0]))
            continue

        if name in _NONUNITARY_NAMES:
            nonunitary.append(NonUnitaryMeta(idx, inst.operation, qubit_indices, clbit_indices))
            continue

        # Drop any classical bits to avoid cross-circuit clbit issues.
        quantum_only.append(inst.operation, inst.qubits, [])

    return quantum_only, measurements, nonunitary


def rebuild_with_classical(
    routed_quantum: QuantumCircuit,
    original_circuit: QuantumCircuit,
    measurements: Iterable[MeasurementMeta],
    *,
    other_nonunitary: Iterable[NonUnitaryMeta] | None = None,
    layout_map: Dict[int, int] | None = None,
) -> QuantumCircuit:
    """Reattach classical structure and measurements to a routed quantum circuit.

    Args:
        routed_quantum: Output of the router (no classical bits).
        original_circuit: Circuit providing classical register structure.
        measurements: Metadata produced by :func:`strip_nonunitary`.
        other_nonunitary: Optional metadata for resets/barriers/etc to re-attach.
        layout_map: Final logical->physical layout. When provided, measurements
            (and other non-unitary ops) are mapped using this layout.
    """

    rebuilt = routed_quantum.copy()

    # Add classical registers mirroring the original structure.
    creg_map: Dict[ClassicalRegister, ClassicalRegister] = {}
    clbit_lookup: List[Any] = []
    for creg in original_circuit.cregs:
        new_creg = ClassicalRegister(creg.size, creg.name)
        rebuilt.add_register(new_creg)
        creg_map[creg] = new_creg
    for clbit in original_circuit.clbits:
        loc = original_circuit.find_bit(clbit)
        if not loc.registers:
            # Fall back to flat indexing if register metadata is missing.
            clbit_lookup.append(clbit)
            continue
        parent, offset = loc.registers[0]
        new_creg = creg_map[parent]
        clbit_lookup.append(new_creg[offset])

    def _map_qubit(logical_idx: int) -> Any:
        if layout_map is None:
            return rebuilt.qubits[logical_idx]
        return rebuilt.qubits[layout_map.get(logical_idx, logical_idx)]

    # Re-attach other non-unitary operations (e.g., reset/barrier) in original order.
    if other_nonunitary:
        for meta in sorted(other_nonunitary, key=lambda m: m.index):
            mapped_qubits = tuple(_map_qubit(q) for q in meta.qubits)
            mapped_clbits = tuple(clbit_lookup[c] for c in meta.clbits)
            rebuilt.append(meta.operation, mapped_qubits, mapped_clbits)

    # Append measurements using the provided mapping.
    for meta in sorted(measurements, key=lambda m: m.index):
        rebuilt.measure(_map_qubit(meta.qubit), clbit_lookup[meta.clbit])

    return rebuilt
