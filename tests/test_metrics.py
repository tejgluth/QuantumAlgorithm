import math

import networkx as nx
from qiskit import QuantumCircuit

from quantum_routing_rl.eval.metrics import (
    assert_coupling_compatible,
    compute_metrics,
    success_probability_proxy,
)
from quantum_routing_rl.hardware.model import (
    EdgeProps,
    HardwareModel,
    HardwareSnapshot,
    QubitProps,
)


def test_counts_and_depths():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.swap(0, 1)

    metrics = compute_metrics(qc, default_error_rate=0.0)

    assert metrics.swaps == 1
    assert metrics.two_qubit_count == 2
    assert metrics.two_qubit_depth == 2
    assert metrics.depth == 2
    assert metrics.size == 2
    assert metrics.success_prob == 1.0
    assert metrics.overall_log_success is None
    assert metrics.total_duration_ns is None
    assert metrics.decoherence_penalty is None


def test_success_proxy_with_default_error():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)

    prob = success_probability_proxy(qc, default_error_rate=0.1)
    assert prob is not None
    assert abs(prob - (0.9**2)) < 1e-9


def test_assert_coupling_compatible_passes_for_valid_routing():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.swap(1, 2)

    # Coupling is linear: 0-1-2
    edges = [(0, 1), (1, 2)]
    assert_coupling_compatible(qc, edges)


def test_time_aware_metrics_use_direction_and_drift():
    edge_props_0 = {
        (0, 1): EdgeProps(
            p2_error=0.1,
            t2_duration_ns=100.0,
            p2_error_fwd=0.1,
            p2_error_rev=0.2,
            t2_duration_fwd_ns=100.0,
            t2_duration_rev_ns=120.0,
        )
    }
    edge_props_1 = {
        (0, 1): EdgeProps(
            p2_error=0.15,
            t2_duration_ns=110.0,
            p2_error_fwd=0.15,
            p2_error_rev=0.3,
            t2_duration_fwd_ns=110.0,
            t2_duration_rev_ns=150.0,
        )
    }
    qubit_props = {
        0: QubitProps(
            t1_ns=2000.0, t2_ns=2000.0, p1_error=0.001, readout_error=0.02, p1_duration_ns=20.0
        ),
        1: QubitProps(
            t1_ns=2000.0, t2_ns=2000.0, p1_error=0.001, readout_error=0.02, p1_duration_ns=20.0
        ),
    }
    snapshot0 = HardwareSnapshot(edge_props=edge_props_0, qubit_props=qubit_props, label="t0")
    snapshot1 = HardwareSnapshot(edge_props=edge_props_1, qubit_props=qubit_props, label="t1")
    hw = HardwareModel(
        graph_id="line",
        adjacency=[(0, 1)],
        edge_props=edge_props_0,
        qubit_props=qubit_props,
        snapshots=[snapshot0, snapshot1],
        directional_mode=True,
        snapshot_spacing_ns=50.0,
    )

    qc = QuantumCircuit(2)
    qc.cx(0, 1)  # forward, snapshot 0
    qc.cx(1, 0)  # reverse, snapshot 1 due to spacing and duration

    metrics = compute_metrics(qc, hardware_model=hw)

    # Duration: 100 + 150 = 250 ns
    assert metrics.total_duration_ns is not None
    assert abs(metrics.total_duration_ns - 250.0) < 1e-6

    # Directional: first gate uses 0.1, second uses 0.3
    expected_log_proxy = math.log(0.9) + math.log(0.7)
    assert metrics.log_success_proxy is not None
    assert abs(metrics.log_success_proxy - expected_log_proxy) < 1e-6

    # Decoherence: per gate, per qubit -duration/2000 *2 qubits *2 gates
    expected_decoherence = -0.5
    assert metrics.decoherence_penalty is not None
    assert abs(metrics.decoherence_penalty - expected_decoherence) < 1e-9

    overall = expected_log_proxy + expected_decoherence  # no 1q or readout contributions
    assert metrics.overall_log_success is not None
    assert abs(metrics.overall_log_success - overall) < 1e-6


def test_schedule_is_deterministic():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    hw = HardwareModel.synthetic(nx.path_graph(2), seed=7)

    m1 = compute_metrics(qc, hardware_model=hw)
    m2 = compute_metrics(qc, hardware_model=hw)

    assert m1.total_duration_ns == m2.total_duration_ns
    assert m1.log_success_proxy == m2.log_success_proxy
