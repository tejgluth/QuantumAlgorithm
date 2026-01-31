import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig


def _norm(edge):
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _old_candidates(graph: nx.Graph, frontier_phys, neighbor_hops: int = 1):
    target_nodes = set()
    for a, b in frontier_phys:
        target_nodes.update({a, b})
        if neighbor_hops >= 1:
            for node in (a, b):
                target_nodes.update(graph.neighbors(node))
    edges = set()
    for u, v in graph.edges():
        if u in target_nodes or v in target_nodes:
            edges.add(_norm((u, v)))
    return edges


def test_candidates_follow_shortest_path():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 3)
    cmap = CouplingMap([[0, 1], [1, 2], [2, 3]])

    env = RoutingEnv(RoutingEnvConfig(neighbor_hops=0, candidate_k_shortest=1))
    state = env.reset(circuit, cmap, seed=1)

    expected = {(0, 1), (1, 2), (2, 3)}
    assert set(state.candidate_swaps) == expected


def test_candidates_shrink_against_old_method():
    circuit = QuantumCircuit(6)
    circuit.cx(0, 5)
    cmap = CouplingMap([[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]])

    env = RoutingEnv(RoutingEnvConfig(neighbor_hops=0, candidate_k_shortest=2, candidate_cap=24))
    state = env.reset(circuit, cmap, seed=2)

    graph = nx.Graph(list(cmap.get_edges()))
    frontier_phys = (state.layout[0], state.layout[5])
    old_set = _old_candidates(graph, [frontier_phys], neighbor_hops=1)
    new_set = set(state.candidate_swaps)

    assert new_set.issubset(old_set)
    assert len(new_set) < len(old_set)


def test_anti_oscillation_masks_recent_edge_without_progress():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 3)
    cmap = CouplingMap([[0, 1], [1, 2], [2, 3]])

    env = RoutingEnv(RoutingEnvConfig(neighbor_hops=0, anti_osc_window=2))
    state = env.reset(circuit, cmap, seed=3)

    first_idx = state.candidate_swaps.index((1, 2))
    state, _, _, _ = env.step(first_idx)

    assert (1, 2) in state.candidate_swaps
    back_idx = state.candidate_swaps.index((1, 2))
    assert state.action_mask[back_idx] is False
