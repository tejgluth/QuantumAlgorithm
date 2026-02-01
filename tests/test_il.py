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
from quantum_routing_rl.models.teacher import route_with_teacher
from quantum_routing_rl.models.policy import route_with_policy


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
        assert a.candidate_features.equal(b.candidate_features)
        assert a.state_features.equal(b.state_features)
        assert a.teacher_scores.equal(b.teacher_scores)
        assert a.action_mask.equal(b.action_mask)


def test_training_smoke(tmp_path):
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    samples = collect_dataset(
        {"line": [circuit]},
        {"line": cmap},
        seed=7,
        hardware_models={"line": [(None, None)]},
    )
    dataset_path = tmp_path / "il_traces.pt"
    save_dataset(
        samples,
        dataset_path,
        metadata={"graphs": ["line"], "num_circuit_seeds": 1, "num_hardware_seeds": 1},
    )

    model, history = train_model(samples, epochs=1, lr=1e-2, tau=1.0)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        model,
        ckpt_path,
        epochs=1,
        seed=7,
        loss_history=history,
        tau=1.0,
        hard_weight=0.2,
        mse_weight=1.0,
        rank_weight=0.5,
        score_mode="min",
    )

    assert ckpt_path.exists()
    assert len(history) == 1


def test_il_rollout_compares_to_teacher(tmp_path):
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    samples = collect_dataset(
        {"line": [circuit]},
        {"line": cmap},
        seed=5,
        hardware_models={"line": [(None, None)]},
    )
    model, _ = train_model(samples, epochs=1, lr=1e-2, tau=1.0)

    teacher_result = route_with_teacher(circuit.circuit, cmap, seed=11)
    il_result = route_with_policy(
        model,
        circuit.circuit,
        cmap,
        name="il_policy_test",
        seed=11,
        max_steps=100,
    )
    assert il_result.metrics.swaps <= max(teacher_result.metrics.swaps * 3, 3)
