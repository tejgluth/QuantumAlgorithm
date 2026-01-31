import random

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.env.routing_env import RoutingEnv
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, count_two_qubit_gates


def _non_swap_twoq_count(circuit: QuantumCircuit) -> int:
    return sum(
        1
        for inst in circuit.data
        if inst.operation.num_qubits == 2 and inst.operation.name != "swap"
    )


def test_routing_env_routes_cnot_on_line():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])

    env = RoutingEnv()
    state = env.reset(circuit, cmap, seed=7)
    graph = nx.Graph(list(cmap.get_edges()))

    while not state.done:
        logical_a, logical_b = state.frontier[0]
        phys_pair = (state.layout[logical_a], state.layout[logical_b])
        if graph.has_edge(*phys_pair):
            break
        path = nx.shortest_path(graph, phys_pair[0], phys_pair[1])
        next_edge = (path[0], path[1])
        if next_edge not in state.candidate_swaps and next_edge[::-1] in state.candidate_swaps:
            next_edge = next_edge[::-1]
        action_idx = state.candidate_swaps.index(next_edge)
        state, _, _, _ = env.step(action_idx)

    routed = env.routed_circuit
    assert_coupling_compatible(routed, cmap.get_edges())
    assert _non_swap_twoq_count(routed) == _non_swap_twoq_count(circuit)


def test_random_policy_rollout_produces_valid_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])

    env = RoutingEnv()
    rng = random.Random(123)
    state = env.reset(circuit, cmap)

    while not state.done:
        valid_actions = [i for i, ok in enumerate(state.action_mask) if ok]
        action_idx = rng.choice(valid_actions)
        state, _, _, _ = env.step(action_idx)

    routed = env.routed_circuit
    assert_coupling_compatible(routed, cmap.get_edges())
    assert count_two_qubit_gates(routed) >= count_two_qubit_gates(circuit)
    assert _non_swap_twoq_count(routed) == _non_swap_twoq_count(circuit)


def test_reset_schedules_ready_gates_without_swaps():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])

    env = RoutingEnv()
    state = env.reset(circuit, cmap)
    assert state.done
    routed = env.routed_circuit
    assert _non_swap_twoq_count(routed) == 2
    assert count_two_qubit_gates(routed) == 2
