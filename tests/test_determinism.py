import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import run_qiskit_sabre_trials
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.weighted_sabre import (
    WeightedDistanceParams,
    WeightedSabreWeights,
    route_with_weighted_sabre,
)


def _hardware():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    return HardwareModel.synthetic(graph, seed=123, profile="realistic")


def test_qiskit_trials_deterministic_with_seed():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    cmap = CouplingMap([[0, 1]])
    hw = _hardware()

    res1 = run_qiskit_sabre_trials(qc, cmap, seed=5, trials=2, hardware_model=hw)
    res2 = run_qiskit_sabre_trials(qc, cmap, seed=5, trials=2, hardware_model=hw)

    assert res1.metrics.swaps == res2.metrics.swaps
    assert res1.metrics.two_qubit_depth == res2.metrics.two_qubit_depth


def test_weighted_sabre_deterministic_with_seed():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    cmap = CouplingMap([[0, 1]])
    hw = _hardware()

    params = WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2, min_edge_weight=1.0)
    weights = WeightedSabreWeights(
        lookahead_weight=0.5, decay_weight=0.25, stagnation_weight=0.25, decay_factor=0.9
    )

    res1 = route_with_weighted_sabre(
        qc,
        cmap,
        hardware_model=hw,
        seed=7,
        trials=2,
        distance_params=params,
        snapshot_mode="avg",
        router_weights=weights,
    )
    res2 = route_with_weighted_sabre(
        qc,
        cmap,
        hardware_model=hw,
        seed=7,
        trials=2,
        distance_params=params,
        snapshot_mode="avg",
        router_weights=weights,
    )

    assert res1.metrics.swaps == res2.metrics.swaps
    assert res1.metrics.two_qubit_depth == res2.metrics.two_qubit_depth
