"""Simple qubit-routing environment producing valid routed circuits."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.env.reward import RewardBreakdown, RewardConfig
from quantum_routing_rl.env.state import LogicalGate, PhysicalSwap, RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, count_two_qubit_gates


@dataclass
class RoutingEnvConfig:
    """Configuration knobs for :class:`RoutingEnv`."""

    frontier_size: int = 2
    neighbor_hops: int = 1
    reward: RewardConfig = field(default_factory=RewardConfig)


class RoutingEnv:
    """Environment inserting SWAPs until all gates are executable."""

    def __init__(self, config: RoutingEnvConfig | None = None):
        self.config = config or RoutingEnvConfig()
        self._rng = random.Random()
        self._graph: nx.Graph | None = None
        self._circuit: QuantumCircuit | None = None
        self._routed: QuantumCircuit | None = None
        self._layout: dict[int, int] = {}
        self._inst_ptr = 0
        self._step_count = 0
        self._done = False
        self._coupling_edges: List[tuple[int, int]] = []

    # ------------------------------------------------------------------ API --
    def reset(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | Sequence[Sequence[int]],
        *,
        seed: int | None = None,
    ) -> RoutingState:
        """Reset environment for a new circuit."""
        if seed is not None:
            self._rng.seed(seed)
        self._circuit = circuit
        self._inst_ptr = 0
        self._step_count = 0
        self._done = False

        edges = _normalize_edges(coupling_map)
        self._coupling_edges = edges
        self._graph = nx.Graph()
        self._graph.add_edges_from(edges)

        num_phys = 1 + max(max(u, v) for u, v in edges)
        if circuit.num_qubits > num_phys:
            msg = "Coupling map must have at least as many physical qubits as the circuit."
            raise ValueError(msg)

        self._layout = {logical: logical for logical in range(circuit.num_qubits)}
        self._routed = QuantumCircuit(num_phys, name=f"{circuit.name or 'circuit'}_routed")

        self._apply_ready_gates()
        return self._build_state()

    def step(self, action: int | PhysicalSwap) -> tuple[RoutingState, float, bool, dict]:
        """Apply a swap action and schedule any newly executable gates."""
        if self._done:
            msg = "Environment already terminated."
            raise RuntimeError(msg)

        edge = self._resolve_action(action)
        depth_before = self._routed.depth() if self._routed is not None else 0
        twoq_before = count_two_qubit_gates(self._routed) if self._routed is not None else 0

        self._apply_swap(edge)
        self._apply_ready_gates()
        self._step_count += 1

        depth_after = self._routed.depth() if self._routed is not None else depth_before
        twoq_after = (
            count_two_qubit_gates(self._routed) if self._routed is not None else twoq_before
        )
        noise_delta = -(twoq_after - twoq_before)
        breakdown = RewardBreakdown(
            swap_cost=-1.0,
            depth_delta=-(depth_after - depth_before),
            noise_proxy_delta=noise_delta,
        )
        reward = breakdown.total(self.config.reward)
        self._done = self._inst_ptr >= len(self._circuit.data)  # type: ignore[arg-type]

        info = {
            "reward_breakdown": breakdown.__dict__,
            "swap": edge,
            "step_count": self._step_count,
        }
        return self._build_state(), reward, self._done, info

    def render_debug(self) -> str:
        """Return a textual snapshot helpful for debugging."""
        state = self._build_state()
        return (
            f"step={state.step_count} done={state.done} layout={state.layout} "
            f"frontier={state.frontier} candidates={state.candidate_swaps}"
        )

    # ------------------------------------------------------------- Helpers --
    def _apply_swap(self, edge: PhysicalSwap) -> None:
        """Update layout and append SWAP gate to routed circuit."""
        if self._routed is None:
            raise RuntimeError("reset must be called first.")
        a, b = edge
        logical_a = _logical_on_physical(self._layout, a)
        logical_b = _logical_on_physical(self._layout, b)
        if logical_a is not None:
            self._layout[logical_a] = b
        if logical_b is not None:
            self._layout[logical_b] = a
        self._routed.swap(self._routed.qubits[a], self._routed.qubits[b])

    def _apply_ready_gates(self) -> None:
        """Schedule executable gates sequentially."""
        assert self._circuit is not None
        assert self._routed is not None
        while self._inst_ptr < len(self._circuit.data):
            inst: CircuitInstruction = self._circuit.data[self._inst_ptr]
            if inst.operation.num_qubits == 1:
                self._append_mapped(inst)
                self._inst_ptr += 1
                continue

            if inst.operation.num_qubits != 2:
                msg = f"Unsupported gate with {inst.operation.num_qubits} qubits."
                raise ValueError(msg)

            logical_qubits = tuple(self._circuit.find_bit(qb).index for qb in inst.qubits)
            phys = tuple(self._layout[q] for q in logical_qubits)
            if _is_edge(phys, self._coupling_edges):
                self._append_mapped(inst, phys)
                self._inst_ptr += 1
                continue
            break

        if self._inst_ptr >= len(self._circuit.data):
            assert_coupling_compatible(self._routed, self._coupling_edges)
            self._done = True

    def _append_mapped(
        self, inst: CircuitInstruction, phys_qubits: Tuple[int, int] | None = None
    ) -> None:
        """Append an instruction using current layout."""
        assert self._routed is not None
        if phys_qubits is None:
            phys_qubits = tuple(
                self._layout[self._circuit.find_bit(qb).index] for qb in inst.qubits
            )  # type: ignore[arg-type]
        mapped_qubits = [self._routed.qubits[i] for i in phys_qubits]
        self._routed.append(inst.operation, mapped_qubits, inst.clbits)

    def _build_state(self) -> RoutingState:
        frontier = self._frontier_gates(limit=self.config.frontier_size)
        frontier_phys = {_logical_to_physical(self._layout, g) for g in frontier}
        candidate_swaps = self._candidate_edges(frontier_phys)
        mask = [True] * len(candidate_swaps)
        return RoutingState(
            layout=dict(self._layout),
            frontier=frontier,
            lookahead=frontier,
            candidate_swaps=candidate_swaps,
            action_mask=mask,
            step_count=self._step_count,
            done=self._done,
        )

    def _frontier_gates(self, limit: int) -> List[LogicalGate]:
        """Return next ``limit`` two-qubit gates."""
        assert self._circuit is not None
        frontier: List[LogicalGate] = []
        idx = self._inst_ptr
        while idx < len(self._circuit.data) and len(frontier) < limit:
            inst = self._circuit.data[idx]
            if inst.operation.num_qubits == 2:
                logical = tuple(self._circuit.find_bit(qb).index for qb in inst.qubits)
                frontier.append((int(logical[0]), int(logical[1])))
            idx += 1
        return frontier

    def _candidate_edges(self, frontier_phys: Iterable[tuple[int, int]]) -> List[PhysicalSwap]:
        """Edges incident to frontier qubits (plus one-hop neighbours)."""
        assert self._graph is not None
        if not frontier_phys:
            return sorted(self._graph.edges())

        target_nodes: set[int] = set()
        for a, b in frontier_phys:
            target_nodes.update({a, b})
            if self.config.neighbor_hops >= 1:
                for node in (a, b):
                    target_nodes.update(self._graph.neighbors(node))

        edges: set[PhysicalSwap] = set()
        for u, v in self._graph.edges():
            if u in target_nodes or v in target_nodes:
                edges.add((u, v))
        return sorted(edges)

    def _resolve_action(self, action: int | PhysicalSwap) -> PhysicalSwap:
        state = self._build_state()
        if isinstance(action, int):
            if action < 0 or action >= len(state.candidate_swaps):
                msg = f"Action index {action} out of range."
                raise IndexError(msg)
            edge = state.candidate_swaps[action]
        else:
            edge = action
            if edge not in state.candidate_swaps:
                msg = f"Swap {edge} is not a legal action."
                raise ValueError(msg)
        return edge

    # ---------------------------------------------------------- Introspection --
    @property
    def routed_circuit(self) -> QuantumCircuit:
        if self._routed is None:
            raise RuntimeError("reset must be called before accessing routed circuit.")
        return self._routed


def _normalize_edges(coupling_map: CouplingMap | Sequence[Sequence[int]]) -> List[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [tuple(edge) for edge in coupling_map]


def _is_edge(pair: Tuple[int, int], edges: Sequence[Tuple[int, int]]) -> bool:
    return pair in edges or (pair[1], pair[0]) in edges


def _logical_on_physical(layout: dict[int, int], physical: int) -> int | None:
    for logical, phys in layout.items():
        if phys == physical:
            return logical
    return None


def _logical_to_physical(layout: dict[int, int], gate: LogicalGate) -> tuple[int, int]:
    return (layout[gate[0]], layout[gate[1]])
