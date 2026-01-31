from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import (
    BaselineResult,
    run_best_available_sabre,
    run_sabre_layout_swap,
)


def _pair(inst, circuit):
    return tuple(circuit.find_bit(qb).index for qb in inst.qubits)


def _allowed_pairs(coupling):
    edges = {(u, v) for u, v in coupling}
    edges |= {(v, u) for u, v in coupling}
    return edges


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
