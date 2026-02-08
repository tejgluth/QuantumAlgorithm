from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import (
    BaselineResult,
    run_basic_swap,
    run_best_available_sabre,
    run_commuting_2q_router,
    run_lookahead_swap,
    run_preset_opt3,
    run_qiskit_sabre_trials,
    run_sabre_layout_swap,
)
from quantum_routing_rl.eval.metrics import CircuitMetrics


def _pair(inst, circuit):
    return tuple(circuit.find_bit(qb).index for qb in inst.qubits)


def _allowed_pairs(coupling):
    edges = {(u, v) for u, v in coupling}
    edges |= {(v, u) for u, v in coupling}
    return edges


def test_basic_and_lookahead_respect_coupling():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])

    basic = run_basic_swap(qc, cmap, seed=3)
    look = run_lookahead_swap(qc, cmap, seed=4)

    allowed = _allowed_pairs(cmap.get_edges())
    for result in (basic, look):
        assert isinstance(result, BaselineResult)
        assert result.baseline_status == "ok"
        for inst in result.circuit.data:
            if inst.operation.num_qubits != 2:
                continue
            assert _pair(inst, result.circuit) in allowed


def test_commuting_router_not_applicable_without_twoq():
    qc = QuantumCircuit(2)
    qc.h(0)
    cmap = CouplingMap([[0, 1]])

    result = run_commuting_2q_router(qc, cmap, seed=5)

    assert result.baseline_status in {"not_applicable", "not_available"}
    assert result.fallback_used is False
    # Metrics still computed when possible.
    if result.metrics:
        assert result.metrics.two_qubit_count == 0


def test_best_available_sabre_runs_and_records_metrics():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(0)

    cmap = CouplingMap([[0, 1]])
    result = run_best_available_sabre(qc, cmap, seed=11)

    record = result.as_record()
    assert record["baseline"] == "qiskit_sabre_best"
    assert "depth" in record
    assert result.metrics.two_qubit_count >= 1
    assert "baseline_status" in record


def _metrics(swaps: int, depth: int, *, twoq_depth: int = 0, duration: float | None = None):
    return CircuitMetrics(
        swaps=swaps,
        two_qubit_count=max(0, twoq_depth),
        two_qubit_depth=twoq_depth,
        depth=depth,
        size=depth,
        success_prob=1.0,
        log_success_proxy=None,
        duration_proxy=None,
        overall_log_success=None,
        total_duration_ns=duration,
        decoherence_penalty=None,
    )


def test_qiskit_trials_prefers_twoq_depth_then_depth():
    qc = QuantumCircuit(2)
    cmap = CouplingMap([[0, 1]])
    trials = [
        BaselineResult("t0", qc, _metrics(1, 10, twoq_depth=12), runtime_s=0.01, seed=0),
        BaselineResult("t1", qc, _metrics(1, 12, twoq_depth=8), runtime_s=0.01, seed=1),
    ]

    result = run_qiskit_sabre_trials(
        qc,
        cmap,
        seed=7,
        trials=len(trials),
        trial_runner=lambda idx, s: trials[idx],
    )

    assert result.name == f"qiskit_sabre_trials{len(trials)}"
    assert result.metrics.two_qubit_depth == 8
    assert result.extra["trial"] == 1
    assert result.extra["trials_total"] == len(trials)


def test_qiskit_trials_prefers_depth_before_duration():
    qc = QuantumCircuit(2)
    cmap = CouplingMap([[0, 1]])
    trials = [
        BaselineResult("t0", qc, _metrics(2, 8, duration=300), runtime_s=0.01, seed=0),
        BaselineResult("t1", qc, _metrics(2, 10, duration=100), runtime_s=0.01, seed=1),
    ]

    result = run_qiskit_sabre_trials(
        qc,
        cmap,
        trials=2,
        trial_runner=lambda idx, s: trials[idx],
    )

    assert result.metrics.depth == 8


def test_preset_opt3_runs_and_respects_coupling():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    cmap = CouplingMap([[0, 1]])

    result = run_preset_opt3(qc, cmap, seed=5)

    assert result.name == "qiskit_preset_opt3"
    assert result.metrics.two_qubit_count >= 1
    allowed = {(0, 1), (1, 0)}
    for inst in result.circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        pair = _pair(inst, result.circuit)
        assert pair in allowed


def test_sabre_layout_swap_respects_coupling():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)

    cmap = CouplingMap([[0, 1], [1, 2]])
    result = run_sabre_layout_swap(qc, cmap, seed=7)

    assert isinstance(result, BaselineResult)
    assert result.runtime_s >= 0
    allowed = _allowed_pairs(cmap.get_edges())
    for inst in result.circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        assert _pair(inst, result.circuit) in allowed
