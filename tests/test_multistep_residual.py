import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.models.multistep_residual import (
    MultiStepConfig,
    MultiStepResidualPolicy,
    route_with_multistep_residual,
)


def _toy_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.cx(0, 2)  # requires at least one swap on line_3
    qc.cx(0, 1)
    return qc


def test_multistep_residual_smoke():
    cmap = CouplingMap([[0, 1], [1, 2]])
    config = MultiStepConfig(top_k=4, branch_k=2, horizon=2)
    policy = MultiStepResidualPolicy(config=config, scorer=None)
    # exercise routing end-to-end; synthetic hardware is provided by env
    result = route_with_multistep_residual(
        policy,
        _toy_circuit(),
        cmap,
        name="residual_multistep_test",
        seed=7,
    )
    assert result.circuit is not None
    assert result.metrics.swaps >= 0
    assert result.metrics.overall_log_success is not None
    # ensure hardware graph matches coupling map size
    graph = nx.Graph(list(cmap.get_edges()))
    assert result.circuit.num_qubits >= graph.number_of_nodes()
