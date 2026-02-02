import torch
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.residual_policy import ResidualScorer
from experiments.legacy_rl.residual_train import (
    _topk_indices,
    _utility_components,
    main as residual_main,
)
from quantum_routing_rl.models.teacher import TeacherPolicy
from quantum_routing_rl.models.policy import candidate_features
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig


def _toy_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.cx(0, 2)  # blocked on line_3 to ensure swaps are needed
    qc.cx(0, 1)
    return qc


def test_residual_scorer_shape_and_determinism():
    torch.manual_seed(0)
    scorer = ResidualScorer(feature_dim=5, hidden_dim=16, context_dim=2)
    cand = torch.randn(4, 5)
    ctx = torch.randn(2)
    out_a = scorer(cand, ctx)
    out_b = scorer(cand, ctx)
    assert out_a.shape == (4,)
    assert torch.allclose(out_a, out_b)


def test_collect_samples_topk_alignment():
    cmap = CouplingMap([[0, 1], [1, 2]])
    graph = HardwareModel.synthetic(nx.Graph(list(cmap.get_edges())), seed=3)
    teacher = TeacherPolicy()
    env = RoutingEnv(RoutingEnvConfig(frontier_size=4))
    initial_layout = {0: 0, 1: 2, 2: 1}  # blocked for the first CX
    state = env.reset(
        _toy_circuit(),
        cmap,
        seed=7,
        hardware_model=graph,
        initial_layout=initial_layout,
    )
    graph_nx = nx.Graph(list(cmap.get_edges()))
    teacher.begin_episode(graph_nx)
    cand_feats = candidate_features(state, graph_nx, graph)
    scores = torch.tensor(teacher.score_candidates(state, graph_nx, graph))
    mask = torch.tensor(state.action_mask, dtype=torch.bool)
    topk = _topk_indices(scores, mask, 4)
    assert topk
    utilities = [
        _utility_components(state, graph_nx, graph, state.candidate_swaps[idx]) for idx in topk
    ]
    assert cand_feats[topk].shape[0] == len(topk)
    assert len(utilities) == len(topk)


def test_residual_training_smoke(tmp_path):
    out = tmp_path / "artifacts"
    exit_code = residual_main(
        [
            "--out",
            str(out),
            "--epochs",
            "1",
            "--max-circuits",
            "1",
            "--max-steps",
            "8",
        ]
    )
    assert exit_code == 0
    ckpt = out / "checkpoints" / "residual_topk.pt"
    assert ckpt.exists()
