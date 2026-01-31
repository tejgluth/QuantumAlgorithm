from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import run_sabre_layout_swap
from quantum_routing_rl.benchmarks.synthetic_generator import pressure_suite


def test_pressure_suite_shapes_and_ids():
    circuits = pressure_suite(seed=3)
    assert len(circuits) >= 3
    assert {c.circuit.num_qubits for c in circuits} >= {8, 9}
    assert circuits[0].circuit_id == "pressure_ring8"


def test_pressure_ring8_induces_swaps():
    circuit = pressure_suite(seed=1)[0]
    ring_edges = [[i, (i + 1) % 8] for i in range(8)]
    cmap = CouplingMap(ring_edges)
    result = run_sabre_layout_swap(circuit.circuit, cmap, seed=11)
    assert result.metrics.swaps > 0
