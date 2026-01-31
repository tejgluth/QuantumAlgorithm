from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit
from quantum_routing_rl.models.il_train import (
    collect_dataset,
    record_traces_for_circuit,
    save_checkpoint,
    save_dataset,
    train_model,
)


def _dummy_circuit() -> QasmCircuit:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    return QasmCircuit("dummy", Path("dummy.qasm"), circuit)


def test_trace_recording_is_deterministic():
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    traces_a = record_traces_for_circuit(circuit, cmap, seed=42, graph_id="line")
    traces_b = record_traces_for_circuit(circuit, cmap, seed=42, graph_id="line")
    assert len(traces_a) == len(traces_b)
    for a, b in zip(traces_a, traces_b, strict=True):
        assert a.label == b.label
        assert a.meta["candidate_swaps"] == b.meta["candidate_swaps"]
        assert a.features.equal(b.features)


def test_training_smoke(tmp_path):
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    samples = collect_dataset([circuit], {"line": cmap}, seed=7)
    dataset_path = tmp_path / "il_traces.pt"
    save_dataset(samples, dataset_path, seed=7)

    model, history = train_model(samples, epochs=1, lr=1e-2)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, ckpt_path, epochs=1, seed=7, loss_history=history)

    assert ckpt_path.exists()
    assert len(history) == 1
