from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit
from quantum_routing_rl.env.reward import RewardConfig
from quantum_routing_rl.models.rl_train import save_checkpoint, train_rl


def _dummy_circuit() -> QasmCircuit:
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    return QasmCircuit("dummy", Path("dummy.qasm"), qc)


def test_rl_training_smoke(tmp_path):
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    model, history = train_rl(
        [circuit],
        {"line": cmap},
        episodes=2,
        seed=1,
        lr=1e-2,
        gamma=0.9,
        reward_config=RewardConfig(),
        il_checkpoint=None,
    )
    ckpt_path = tmp_path / "rl.pt"
    save_checkpoint(
        model,
        ckpt_path,
        history=history,
        episodes=2,
        seed=1,
        gamma=0.9,
        reward_config=RewardConfig(),
        il_checkpoint=None,
    )
    assert ckpt_path.exists()
    assert history
