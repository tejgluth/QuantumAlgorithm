from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.eval.metrics import assert_coupling_compatible
from quantum_routing_rl.models.policy import SwapPolicy
from quantum_routing_rl.qiskit_pass.learned_swap_router import LearnedSwapRouter


def test_router_uses_property_coupling_map_and_falls_back_to_sabre():
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    dag = circuit_to_dag(qc)
    cmap = CouplingMap([[0, 1], [1, 2]])

    router = LearnedSwapRouter(checkpoint_path="does_not_exist.pt", seed=5)
    router.property_set["coupling_map"] = cmap
    routed_dag = router.run(dag)

    routed_circuit = dag_to_circuit(routed_dag)
    assert_coupling_compatible(routed_circuit, cmap.get_edges())
    assert "routing_result" in router.property_set
    assert router.property_set["routing_result"].name == "sabre_layout_swap"


def test_router_accepts_policy_instance():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)
    cmap = CouplingMap([[0, 1]])

    router = LearnedSwapRouter(policy=SwapPolicy())
    router.property_set["coupling_map"] = cmap
    routed = router.run(dag)
    circ = dag_to_circuit(routed)
    assert_coupling_compatible(circ, cmap.get_edges())
