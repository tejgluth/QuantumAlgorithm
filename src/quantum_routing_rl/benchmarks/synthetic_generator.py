"""Synthetic circuit generator for creating routing pressure."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from qiskit import QuantumCircuit

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit


@dataclass(frozen=True)
class SyntheticSpec:
    """Specification of a synthetic pressure circuit."""

    name: str
    n_qubits: int
    twoq_layers: int
    seed: int


def generate_pressure_circuit(spec: SyntheticSpec) -> QasmCircuit:
    """Generate a circuit with deliberately nonlocal 2Q interactions."""
    rng = random.Random(spec.seed)
    qc = QuantumCircuit(spec.n_qubits, name=spec.name)
    for layer in range(spec.twoq_layers):
        control = rng.randrange(spec.n_qubits)
        # Enforce non-locality by picking a target at least 2 steps away.
        offset = rng.randrange(2, spec.n_qubits)
        target = (control + offset) % spec.n_qubits
        if target == control:
            target = (target + 1) % spec.n_qubits
        qc.cx(control, target)
        # Sprinkle single-qubit mixing to avoid degenerate layouts.
        for qb in range(spec.n_qubits):
            if rng.random() < 0.25:
                qc.h(qb)
    path = Path(f"synthetic/{spec.name}.qasm")
    return QasmCircuit(spec.name, path, qc)


def pressure_suite(seed: int = 0) -> List[QasmCircuit]:
    """Return a small suite of pressure-inducing circuits."""
    specs: Iterable[SyntheticSpec] = [
        SyntheticSpec("pressure_ring8", 8, 18, seed),
        SyntheticSpec("pressure_grid9", 9, 18, seed + 1),
        SyntheticSpec("pressure_hex15", 15, 22, seed + 2),
    ]
    return [generate_pressure_circuit(spec) for spec in specs]
