import pandas as pd
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import run_basic_swap
from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit
from quantum_routing_rl.eval.run_eval import _result_record
from quantum_routing_rl.routing.normalize_circuit import normalize_for_routing


def test_normalize_handles_multi_qubit_gate():
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)

    normalized, meta = normalize_for_routing(qc)

    assert meta["remaining_multiq"] == 0
    assert any(inst.operation.num_qubits == 2 for inst in normalized.data)


def test_barrier_not_flagged_and_custom_gate_skipped():
    barrier_circ = QuantumCircuit(5)
    barrier_circ.barrier(range(5))

    _, barrier_meta = normalize_for_routing(barrier_circ)
    assert barrier_meta["remaining_multiq"] == 0

    custom = QuantumCircuit(3)
    custom.append(Gate("g3", 3, []), [0, 1, 2])

    normalized, meta = normalize_for_routing(custom, max_iters=2)
    assert meta["remaining_multiq"] >= 1
    assert normalized is not None


def test_eval_smoke_skips_unsupported(tmp_path):
    normal = QuantumCircuit(2)
    normal.cx(0, 1)

    multi = QuantumCircuit(3)
    multi.append(Gate("g3", 3, []), [0, 1, 2])

    cmap = CouplingMap([[0, 1], [1, 2]])

    qc_normal = QasmCircuit(path=tmp_path / "normal.qasm", circuit_id="normal", circuit=normal)
    qc_multi = QasmCircuit(path=tmp_path / "multi.qasm", circuit_id="multi", circuit=multi)

    res_ok = run_basic_swap(normal, cmap, seed=1)
    res_skip = run_basic_swap(multi, cmap, seed=1)

    records = [
        _result_record(
            qc_normal,
            "line3",
            res_ok,
            seed=1,
            suite="test",
            hardware_seed=None,
            hardware_profile=None,
        ),
        _result_record(
            qc_multi,
            "line3",
            res_skip,
            seed=1,
            suite="test",
            hardware_seed=None,
            hardware_profile=None,
        ),
    ]

    df = pd.DataFrame(records)
    assert (df.loc[df.circuit_id == "multi", "baseline_status"] == "SKIPPED").all()
    assert not df.loc[df.circuit_id == "normal", "baseline_status"].eq("SKIPPED").any()
