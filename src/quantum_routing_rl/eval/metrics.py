"""Routing evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence
from collections import defaultdict

import networkx as nx

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
    overall_log_success: float | None = None
    total_duration_ns: float | None = None
    decoherence_penalty: float | None = None

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
    overall_log_success = None
    total_duration_ns = None
    decoherence_penalty = None
    if hardware_model is not None:
        (
            log_proxy,
            duration_proxy,
            overall_log_success,
            total_duration_ns,
            decoherence_penalty,
        ) = _hardware_noise_proxies(circuit, hardware_model)
    success = (
        math.exp(overall_log_success)
        if overall_log_success is not None
        else math.exp(log_proxy)
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
        overall_log_success=overall_log_success,
        total_duration_ns=total_duration_ns,
        decoherence_penalty=decoherence_penalty,
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
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Compute noise-aware proxies with time, drift, and decoherence."""

    schedule = _time_aware_schedule(circuit, hardware_model)
    if not schedule:
        return None, None, None, None, None

    graph = nx.Graph()
    graph.add_edges_from(hardware_model.adjacency)

    log_success_2q = 0.0
    log_success_1q = 0.0
    readout_penalty = 0.0
    decoherence_penalty = 0.0
    events_by_time: dict[float, list[_ScheduledOp]] = defaultdict(list)
    for op in schedule:
        bucket_time = round(op.start_ns, 6)
        events_by_time[bucket_time].append(op)
        if op.kind == "twoq":
            error, _ = hardware_model.edge_error_and_duration(
                op.qubits[0], op.qubits[1], snapshot=op.snapshot, directed=True
            )
            log_success_2q += math.log(max(1e-9, 1.0 - min(0.999999, error)))
            decoherence_penalty += _decoherence_cost(op, hardware_model)
        elif op.kind == "oneq":
            props = hardware_model.get_qubit_props(op.qubits[0], snapshot=op.snapshot)
            log_success_1q += math.log(max(1e-9, 1.0 - min(0.999999, props.p1_error)))
            decoherence_penalty += _decoherence_cost(op, hardware_model)
        elif op.kind == "measure":
            props = hardware_model.get_qubit_props(op.qubits[0], snapshot=op.snapshot)
            readout_penalty += math.log(max(1e-9, 1.0 - min(0.999999, props.readout_error)))
            decoherence_penalty += _decoherence_cost(op, hardware_model)
        else:
            decoherence_penalty += _decoherence_cost(op, hardware_model)

    # Crosstalk proxy: penalties for concurrent nearby two-qubit gates.
    if hardware_model.crosstalk_factor > 0:
        ct_factor = hardware_model.crosstalk_factor
        for ops in events_by_time.values():
            twoq_ops = [op for op in ops if op.kind == "twoq"]
            if len(twoq_ops) < 2:
                continue
            for i in range(len(twoq_ops)):
                for j in range(i + 1, len(twoq_ops)):
                    if _edges_adjacent(twoq_ops[i].qubits, twoq_ops[j].qubits, graph):
                        log_success_2q += math.log(max(1e-9, 1.0 - ct_factor))

    total_duration = max(op.end_ns for op in schedule)
    overall_log_success = log_success_2q + log_success_1q + decoherence_penalty + readout_penalty

    # Backward-compat fields
    log_proxy = log_success_2q if log_success_2q != 0.0 else None
    duration_proxy = decoherence_penalty if decoherence_penalty != 0.0 else -total_duration / 1e6

    return (
        log_proxy,
        duration_proxy,
        overall_log_success,
        total_duration,
        decoherence_penalty,
    )


# --------------------------------------------------------------------------- #
# Scheduling utilities


@dataclass(frozen=True)
class _ScheduledOp:
    start_ns: float
    duration_ns: float
    end_ns: float
    qubits: tuple[int, ...]
    snapshot: int
    kind: str  # "twoq", "oneq", "measure", "other"


def _time_aware_schedule(
    circuit: QuantumCircuit, hardware_model: HardwareModel
) -> list[_ScheduledOp]:
    """Deterministic ASAP schedule using per-qubit availability."""

    if circuit.num_qubits == 0:
        return []

    available: dict[int, float] = {i: 0.0 for i in range(circuit.num_qubits)}
    schedule: list[_ScheduledOp] = []

    for inst in circuit.data:
        qubits = tuple(circuit.find_bit(qb).index for qb in inst.qubits)
        start_time = max(available[q] for q in qubits) if qubits else 0.0
        snapshot_idx = hardware_model.snapshot_index_for_time(start_time)
        duration = _duration_for_inst(inst, qubits, hardware_model, snapshot_idx)
        end_time = start_time + duration
        kind = _classify_inst(inst)
        schedule.append(
            _ScheduledOp(
                start_ns=start_time,
                duration_ns=duration,
                end_ns=end_time,
                qubits=qubits,
                snapshot=snapshot_idx,
                kind=kind,
            )
        )
        for q in qubits:
            available[q] = end_time
    return schedule


def _duration_for_inst(
    inst: CircuitInstruction,
    qubits: tuple[int, ...],
    hardware_model: HardwareModel,
    snapshot_idx: int,
) -> float:
    """Return a deterministic duration in nanoseconds for the instruction."""

    if inst.operation.num_qubits == 2 and len(qubits) == 2:
        _, duration = hardware_model.edge_error_and_duration(
            qubits[0], qubits[1], snapshot=snapshot_idx, directed=True
        )
        return duration
    if inst.operation.num_qubits == 1 and len(qubits) == 1:
        props = hardware_model.get_qubit_props(qubits[0], snapshot=snapshot_idx)
        return props.p1_duration_ns
    return 0.0


def _classify_inst(inst: CircuitInstruction) -> str:
    if inst.operation.num_qubits == 2:
        return "twoq"
    if inst.operation.name == "measure":
        return "measure"
    if inst.operation.num_qubits == 1:
        return "oneq"
    return "other"


def _edges_adjacent(edge_a: tuple[int, int], edge_b: tuple[int, int], graph: nx.Graph) -> bool:
    """Return True if the edges act on nearby qubits (distance <= 1)."""
    for qa in edge_a:
        for qb in edge_b:
            if qa == qb:
                return True
            if graph.has_edge(qa, qb):
                return True
    return False


def _decoherence_cost(op: _ScheduledOp, hardware_model: HardwareModel) -> float:
    """Approximate decoherence penalty contributed by this operation."""
    penalty = 0.0
    for q in op.qubits:
        props = hardware_model.get_qubit_props(q, snapshot=op.snapshot)
        if props.t1_ns > 0:
            penalty -= op.duration_ns / props.t1_ns
        if props.t2_ns > 0:
            penalty -= op.duration_ns / props.t2_ns
    return penalty
