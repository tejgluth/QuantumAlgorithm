from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from types import SimpleNamespace

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit
from quantum_routing_rl.models.rl_train import save_checkpoint, train_rl


def _dummy_circuit() -> QasmCircuit:
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    return QasmCircuit("dummy", Path("dummy.qasm"), qc)


def test_rl_training_smoke(tmp_path):
    cmap = CouplingMap([[0, 1], [1, 2]])
    circuit = _dummy_circuit()
    args = SimpleNamespace(
        episodes=2,
        seed=1,
        lr=1e-2,
        gamma=0.9,
        gae_lambda=0.9,
        entropy_coef=0.0,
        value_coef=0.0,
        max_grad_norm=0.5,
        clip_eps=0.2,
        ppo_epochs=1,
        eval_interval=0,
        swap_guard=1.5,
        curriculum=False,
        curriculum_switch=0,
        curriculum_ramp=1,
        max_restarts=0,
        swap_weight=1.0,
        depth_weight=0.0,
        noise_weight=0.0,
        base_swap=1.0,
        error_penalty=0.0,
        time_penalty=0.0,
        progress_weight=0.0,
        completion_bonus=0.0,
        failure_penalty=0.0,
    )
    model, history = train_rl(
        [circuit],
        {"line": cmap},
        {"line": [None]},
        args,
        il_checkpoint=None,
    )
    ckpt_path = tmp_path / "rl.pt"
    save_checkpoint(
        model,
        ckpt_path,
        history=history,
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        il_checkpoint=None,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        clip_eps=args.clip_eps,
        curriculum=args.curriculum,
        curriculum_switch=args.curriculum_switch,
        curriculum_ramp=args.curriculum_ramp,
        ppo_epochs=args.ppo_epochs,
    )
    assert ckpt_path.exists()
    assert history
