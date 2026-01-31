"""Routing evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from qiskit.circuit import CircuitInstruction, QuantumCircuit
from quantum_routing_rl.hardware.model import HardwareModel


@dataclass
class CircuitMetrics:
    """Lightweight container for routing quality numbers."""

    swaps: int
    two_qubit_count: int
    two_qubit_depth: int
    depth: int
    size: int
    success_prob: float | None
    log_success_proxy: float | None = None
    duration_proxy: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        """Dictionary view useful for DataFrame construction."""
        return asdict(self)


def count_swaps(circuit: QuantumCircuit) -> int:
    """Count SWAP operations in the circuit."""
    return sum(1 for inst in circuit.data if inst.operation.name == "swap")


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count all two-qubit operations (including swaps)."""
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Depth considering only two-qubit gates."""

    def _filter(inst: CircuitInstruction) -> bool:
        return inst.operation.num_qubits == 2

    return circuit.depth(filter_function=_filter)


def success_probability_proxy(
    circuit: QuantumCircuit,
    backend_properties: Any | None = None,
    default_error_rate: float = 0.0,
) -> float | None:
    """Noise-aware success proxy.

    Multiplies (1 - error) for each two-qubit gate. If ``backend_properties`` is
    available we query actual errors; otherwise ``default_error_rate`` is used.
    Returns ``None`` when no two-qubit gates are present.
    """
    if default_error_rate < 0 or default_error_rate >= 1:
        msg = "default_error_rate must be in [0, 1)."
        raise ValueError(msg)

    probabilities: list[float] = []
    for inst in circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        error = _lookup_gate_error(inst, circuit, backend_properties, default_error_rate)
        probabilities.append(max(0.0, min(1.0, 1.0 - error)))

    if not probabilities:
        return None

    log_prob = sum(math.log(p) for p in probabilities)
    return math.exp(log_prob)


def _lookup_gate_error(
    inst: CircuitInstruction,
    circuit: QuantumCircuit,
    backend_properties: Any | None,
    fallback: float,
) -> float:
    """Best-effort gate error lookup with graceful fallback."""
    if backend_properties is None:
        return fallback

    qubit_indices = tuple(circuit.find_bit(qb).index for qb in inst.qubits)

    gate_error = getattr(backend_properties, "gate_error", None)
    if callable(gate_error):
        try:
            error_val = gate_error(inst.operation.name, qubit_indices)
            if error_val is not None:
                return float(error_val)
        except Exception:
            pass

    instr_props = getattr(backend_properties, "instruction_properties", None)
    if callable(instr_props):
        try:
            props = instr_props(inst.operation.name, qubit_indices)
            if props and getattr(props, "error", None) is not None:
                return float(props.error)
        except Exception:
            pass

    return fallback


def compute_metrics(
    circuit: QuantumCircuit,
    *,
    backend_properties: Any | None = None,
    default_error_rate: float = 0.0,
    hardware_model: HardwareModel | None = None,
) -> CircuitMetrics:
    """Compute routing metrics for a circuit."""
    swaps = count_swaps(circuit)
    two_q = count_two_qubit_gates(circuit)
    two_q_depth_val = two_qubit_depth(circuit)
    log_proxy = None
    duration_proxy = None
    if hardware_model is not None:
        log_proxy, duration_proxy = _hardware_noise_proxies(circuit, hardware_model)
    success = (
        math.exp(log_proxy)
        if log_proxy is not None
        else success_probability_proxy(
            circuit,
            backend_properties=backend_properties,
            default_error_rate=default_error_rate,
        )
    )
    return CircuitMetrics(
        swaps=swaps,
        two_qubit_count=two_q,
        two_qubit_depth=two_q_depth_val,
        depth=circuit.depth(),
        size=circuit.size(),
        success_prob=success,
        log_success_proxy=log_proxy,
        duration_proxy=duration_proxy,
    )


def assert_coupling_compatible(circuit: QuantumCircuit, edges: Sequence[Sequence[int]]) -> None:
    """Raise if any two-qubit gate violates the coupling map."""
    allowed = {tuple(edge) for edge in _symmetrize_edges(edges)}
    for inst in circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        qubits = tuple(circuit.find_bit(qb).index for qb in inst.qubits)
        if qubits not in allowed:
            msg = f"Gate {inst.operation.name} on {qubits} not in coupling map."
            raise ValueError(msg)


def _symmetrize_edges(edges: Iterable[Sequence[int]]) -> set[tuple[int, int]]:
    """Return undirected edge set."""
    undirected: set[tuple[int, int]] = set()
    for u, v in edges:
        undirected.add((u, v))
        undirected.add((v, u))
    return undirected


def _hardware_noise_proxies(
    circuit: QuantumCircuit, hardware_model: HardwareModel
) -> tuple[float | None, float | None]:
    """Compute log success and decoherence proxies from a HardwareModel."""
    log_terms: list[float] = []
    durations_ns: list[float] = []
    t1_samples: list[float] = []
    for inst in circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        qubits = tuple(circuit.find_bit(qb).index for qb in inst.qubits)
        try:
            edge_props = hardware_model.get_edge_props(*qubits)
        except KeyError:
            continue
        error = max(0.0, min(0.999999, edge_props.p2_error))
        log_terms.append(math.log(1.0 - error))
        durations_ns.append(edge_props.t2_duration_ns)
        for q in qubits:
            try:
                t1_samples.append(hardware_model.get_qubit_props(q).t1_ns)
            except KeyError:
                continue

    log_proxy = sum(log_terms) if log_terms else None
    duration_proxy = None
    if durations_ns:
        total_duration = sum(durations_ns)
        if t1_samples:
            duration_proxy = -total_duration / (sum(t1_samples) / len(t1_samples))
        else:
            duration_proxy = -total_duration / 1_000_000.0
    return log_proxy, duration_proxy
