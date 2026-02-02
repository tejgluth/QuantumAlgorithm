from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult
from quantum_routing_rl.eval.metrics import CircuitMetrics, assert_coupling_compatible
from quantum_routing_rl.env.routing_env import RoutingEnvConfig
from quantum_routing_rl.hardware.model import EdgeProps, HardwareModel, QubitProps
from quantum_routing_rl.models.teacher import route_with_teacher
from quantum_routing_rl.models.weighted_sabre import (
    WeightedDistanceParams,
    WeightedSabreWeights,
    _choose_best_trial,
    route_with_weighted_sabre,
)


def _uniform_hardware() -> HardwareModel:
    edge_props = {
        (0, 1): EdgeProps(p2_error=0.005, t2_duration_ns=200.0),
        (1, 2): EdgeProps(p2_error=0.005, t2_duration_ns=200.0),
    }
    qubit_props = {
        q: QubitProps(
            t1_ns=1e6,
            t2_ns=1e6,
            p1_error=1e-4,
            readout_error=1e-3,
            p1_duration_ns=50.0,
        )
        for q in range(3)
    }
    return HardwareModel(
        graph_id="line3", adjacency=[(0, 1), (1, 2)], edge_props=edge_props, qubit_props=qubit_props
    )


def test_weighted_sabre_routes_coupling_valid():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])
    hardware = HardwareModel.synthetic(cmap, seed=7, directional=True, drift_rate=0.01, snapshots=2)

    result = route_with_weighted_sabre(
        circuit, cmap, hardware_model=hardware, seed=5, trials=2, snapshot_mode="avg"
    )
    assert_coupling_compatible(result.circuit, cmap.get_edges())
    assert result.metrics.two_qubit_count >= 2


def test_trial_selection_prefers_success_then_swaps():
    qc = QuantumCircuit(1)
    metrics_bad = CircuitMetrics(
        swaps=5,
        two_qubit_count=0,
        two_qubit_depth=0,
        depth=0,
        size=0,
        success_prob=None,
        log_success_proxy=None,
        duration_proxy=None,
        overall_log_success=-10.0,
        total_duration_ns=100.0,
        decoherence_penalty=None,
    )
    metrics_good = CircuitMetrics(
        swaps=2,
        two_qubit_count=0,
        two_qubit_depth=0,
        depth=0,
        size=0,
        success_prob=None,
        log_success_proxy=None,
        duration_proxy=None,
        overall_log_success=-5.0,
        total_duration_ns=50.0,
        decoherence_penalty=None,
    )

    res_bad = BaselineResult("weighted_sabre", qc, metrics_bad, runtime_s=0.1, seed=1)
    res_good = BaselineResult("weighted_sabre", qc, metrics_good, runtime_s=0.1, seed=2)

    best = _choose_best_trial(res_bad, res_good)
    assert best is res_good


def test_neutral_weights_track_teacher_behaviour():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(2, 0)
    circuit.cx(0, 1)
    cmap = CouplingMap([[0, 1], [1, 2]])
    hardware = _uniform_hardware()

    teacher_result = route_with_teacher(
        circuit,
        cmap,
        seed=3,
        env_config=RoutingEnvConfig(frontier_size=4),
        hardware_model=hardware,
    )
    weighted_result = route_with_weighted_sabre(
        circuit,
        cmap,
        hardware_model=hardware,
        seed=3,
        trials=1,
        router_weights=WeightedSabreWeights(),
        distance_params=WeightedDistanceParams(alpha_time=0.0, beta_xtalk=0.0),
        snapshot_mode="avg",
        env_config=RoutingEnvConfig(frontier_size=4),
    )

    assert abs(weighted_result.metrics.swaps - teacher_result.metrics.swaps) <= 1
    assert (
        abs(weighted_result.metrics.two_qubit_depth - teacher_result.metrics.two_qubit_depth) <= 1
    )
