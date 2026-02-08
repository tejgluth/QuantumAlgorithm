"""Synthetic circuit generator for creating routing pressure."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

from qiskit import QuantumCircuit

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit


@dataclass(frozen=True)
class SyntheticSpec:
    """Specification of a synthetic pressure circuit."""

    name: str
    n_qubits: int
    twoq_layers: int
    seed: int


@dataclass(frozen=True)
class StructuredSpec:
    """Specification for structured hard synthetic circuits."""

    name: str
    n_qubits: int
    generator: Callable[[int, int], QuantumCircuit]
    seed: int
    metadata: dict[str, object] | None = None


def _qasm_entry(name: str, circuit: QuantumCircuit) -> QasmCircuit:
    """Wrap a circuit with a stable synthetic path."""
    path = Path(f"synthetic/{name}.qasm")
    circuit.name = name
    return QasmCircuit(name, path, circuit)


def generate_pressure_circuit(spec: SyntheticSpec) -> QasmCircuit:
    """Generate a circuit with deliberately nonlocal 2Q interactions."""
    rng = random.Random(spec.seed)
    qc = QuantumCircuit(spec.n_qubits, name=spec.name)
    for _ in range(spec.twoq_layers):
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
    return _qasm_entry(spec.name, qc)


def pressure_suite(seed: int = 0) -> List[QasmCircuit]:
    """Return a small suite of pressure-inducing circuits."""
    specs: Iterable[SyntheticSpec] = [
        SyntheticSpec("pressure_ring8", 8, 18, seed),
        SyntheticSpec("pressure_grid9", 9, 18, seed + 1),
        SyntheticSpec("pressure_hex15", 15, 22, seed + 2),
    ]
    return [generate_pressure_circuit(spec) for spec in specs]


# -------------------------------------------------------------------- Structured
def qaoa_like(n_qubits: int, p: int, entangler_pattern: str, seed: int) -> QuantumCircuit:
    """QAOA-inspired layer stack with deterministic angles."""
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for qb in range(n_qubits):
        qc.h(qb)
    for layer in range(p):
        if entangler_pattern == "ring":
            pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        elif entangler_pattern == "line":
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        else:  # random matching
            order = list(range(n_qubits))
            rng.shuffle(order)
            pairs = [(order[i], order[i + 1]) for i in range(0, n_qubits - 1, 2)]
        for control, target in pairs:
            qc.cx(control, target)
            qc.rz(0.35 + 0.1 * (layer + 1), target)
            qc.cx(control, target)
        for qb in range(n_qubits):
            qc.rx(0.15 * (layer + 1) + 0.05 * math.sin(layer + qb), qb)
    return qc


def commuting_blocks(n_qubits: int, blocks: int, density: float, seed: int) -> QuantumCircuit:
    """Blocks of commuting CX gates on disjoint pairs."""
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for block in range(blocks):
        available = list(range(n_qubits))
        rng.shuffle(available)
        while len(available) >= 2:
            if rng.random() > density:
                break
            control = available.pop()
            target = available.pop()
            qc.cx(control, target)
        for qb in range(n_qubits):
            if rng.random() < 0.3:
                qc.s(qb)
        qc.barrier()
    return qc


def clifford_t_style(n_qubits: int, depth: int, cx_rate: float, seed: int) -> QuantumCircuit:
    """Rough Clifford+T patterned circuit with CX sprinkling."""
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    clifford_gates = [qc.h, qc.s, qc.t]
    for _ in range(depth):
        for qb in range(n_qubits):
            gate = clifford_gates[rng.randrange(len(clifford_gates))]
            gate(qb)
        if rng.random() < cx_rate:
            c, t = rng.sample(range(n_qubits), 2)
            qc.cx(c, t)
        qc.barrier()
    return qc


def adversarial_routing_pressure(
    n_qubits: int, layers: int, pattern_seed: int, *, long_range: bool = True
) -> QuantumCircuit:
    """Designed to maximize routing conflicts via shuffling far-apart pairs."""
    rng = random.Random(pattern_seed)
    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        perm = list(range(n_qubits))
        rng.shuffle(perm)
        mid = len(perm) // 2
        lhs, rhs = perm[:mid], perm[mid:]
        rhs = list(reversed(rhs))
        for control, target in zip(lhs, rhs):
            if long_range and abs(control - target) <= 1:
                target = (target + 2) % n_qubits
            qc.cx(control, target)
        for qb in range(n_qubits):
            qc.rx(0.2 + 0.05 * layer, qb)
    return qc


def _structured_spec_factory(name: str, n_qubits: int, rng: random.Random) -> StructuredSpec:
    choice = rng.choice(["qaoa", "commuting", "clifford_t", "adversarial"])
    seed = rng.randrange(1_000_000)
    if choice == "qaoa":
        p = rng.randint(2, 4)
        pattern = rng.choice(["ring", "line", "matching"])
        return StructuredSpec(
            name=f"{name}_qaoa_p{p}",
            n_qubits=n_qubits,
            generator=lambda nq, s: qaoa_like(nq, p=p, entangler_pattern=pattern, seed=s),
            seed=seed,
            metadata={"family": "qaoa_like", "p": p, "pattern": pattern},
        )
    if choice == "commuting":
        blocks = rng.randint(3, 6)
        density = rng.uniform(0.4, 0.9)
        return StructuredSpec(
            name=f"{name}_comm_blocks",
            n_qubits=n_qubits,
            generator=lambda nq, s: commuting_blocks(nq, blocks=blocks, density=density, seed=s),
            seed=seed,
            metadata={"family": "commuting_blocks", "blocks": blocks, "density": density},
        )
    if choice == "clifford_t":
        depth = rng.randint(8, 14)
        cx_rate = rng.uniform(0.3, 0.8)
        return StructuredSpec(
            name=f"{name}_clifford_t",
            n_qubits=n_qubits,
            generator=lambda nq, s: clifford_t_style(nq, depth=depth, cx_rate=cx_rate, seed=s),
            seed=seed,
            metadata={"family": "clifford_t_style", "depth": depth, "cx_rate": cx_rate},
        )
    layers = rng.randint(6, 10)
    return StructuredSpec(
        name=f"{name}_adv",
        n_qubits=n_qubits,
        generator=lambda nq, s: adversarial_routing_pressure(
            nq, layers=layers, pattern_seed=s, long_range=True
        ),
        seed=seed,
        metadata={"family": "adversarial", "layers": layers},
    )


def structured_hard_suite(tier: str, seed: int = 0) -> list[QasmCircuit]:
    """Generate tiered structured-hard synthetic suites."""
    tier = tier.lower()
    configs = {
        "hard_small": {"n_range": (6, 12), "twoq_budget": (30, 70), "count": 12},
        "hard_medium": {"n_range": (12, 20), "twoq_budget": (60, 140), "count": 14},
        "hard_large": {"n_range": (20, 32), "twoq_budget": (120, 220), "count": 14},
    }
    if tier not in configs:
        msg = f"Unknown structured hard tier '{tier}'"
        raise ValueError(msg)

    cfg = configs[tier]
    rng = random.Random(seed)
    circuits: list[QasmCircuit] = []
    for idx in range(cfg["count"]):
        n_qubits = rng.randint(cfg["n_range"][0], cfg["n_range"][1])
        spec = _structured_spec_factory(f"{tier}_{idx}", n_qubits, rng)
        qc = spec.generator(spec.n_qubits, spec.seed)
        circuits.append(_qasm_entry(spec.name, qc))
    return circuits
