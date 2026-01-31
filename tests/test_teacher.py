from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.eval.metrics import assert_coupling_compatible
from quantum_routing_rl.models.teacher import TeacherPolicy, route_with_teacher


def test_teacher_is_deterministic_on_same_state():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(0, 1)
    cmap = CouplingMap([[0, 1], [1, 2]])
    env = RoutingEnv()
    state = env.reset(circuit, cmap, seed=5)
    graph = env.graph
    assert graph is not None

    teacher = TeacherPolicy()
    teacher.begin_episode(graph)
    first = teacher.select_action(state, graph)
    second = teacher.select_action(state, graph)
    assert first == second


def test_teacher_prefers_swap_reducing_frontier_and_lookahead():
    """Swap (0,1) should be chosen over (1,2) for frontier (0,2) lookahead (0,1)."""

    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(0, 1)
    cmap = CouplingMap([[0, 1], [1, 2]])
    env = RoutingEnv()
    state = env.reset(circuit, cmap, seed=7)
    graph = env.graph
    assert graph is not None
    teacher = TeacherPolicy()
    teacher.begin_episode(graph)

    action_idx = teacher.select_action(state, graph)
    assert state.candidate_swaps[action_idx] == (0, 1)


def test_teacher_smoke_routes_under_step_cap():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    cmap = CouplingMap([[0, 1], [1, 2]])

    result = route_with_teacher(
        circuit,
        cmap,
        seed=11,
        env_config=RoutingEnvConfig(max_steps=10),
        name="teacher_sabre_like",
    )
    assert_coupling_compatible(result.circuit, cmap.get_edges())
    assert result.metrics.two_qubit_count >= 1
