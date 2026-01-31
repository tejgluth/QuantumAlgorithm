from qiskit import QuantumCircuit

from quantum_routing_rl.eval.metrics import (
    assert_coupling_compatible,
    compute_metrics,
    success_probability_proxy,
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
