"""Simple qubit-routing environment producing valid routed circuits."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.env.reward import RewardBreakdown, RewardConfig
from quantum_routing_rl.env.state import LogicalGate, PhysicalSwap, RoutingState
from quantum_routing_rl.eval.metrics import assert_coupling_compatible, count_two_qubit_gates


@dataclass
class RoutingEnvConfig:
    """Configuration knobs for :class:`RoutingEnv`."""

    frontier_size: int = 2
    neighbor_hops: int = 1
    candidate_k_shortest: int = 2
    candidate_cap: int = 24
    anti_osc_window: int = 2
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_steps: int | None = None


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
        self._hardware_model: HardwareModel | None = None
        self._max_steps: int | None = None
        self._recent_swaps: list[PhysicalSwap] = []

    # ------------------------------------------------------------------ API --
    def reset(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | Sequence[Sequence[int]],
        *,
        seed: int | None = None,
        hardware_model: HardwareModel | None = None,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutingState:
        """Reset environment for a new circuit."""
        if seed is not None:
            self._rng.seed(seed)
        self._circuit = circuit
        self._inst_ptr = 0
        self._step_count = 0
        self._done = False
        self._recent_swaps = []

        edges = _normalize_edges(coupling_map)
        self._coupling_edges = edges
        self._graph = nx.Graph()
        self._graph.add_edges_from(edges)

        num_phys = 1 + max(max(u, v) for u, v in edges)
        if circuit.num_qubits > num_phys:
            msg = "Coupling map must have at least as many physical qubits as the circuit."
            raise ValueError(msg)

        if initial_layout is not None:
            self._layout = _coerce_initial_layout(initial_layout, circuit.num_qubits, num_phys)
        else:
            self._layout = {logical: logical for logical in range(circuit.num_qubits)}
        self._routed = QuantumCircuit(num_phys, name=f"{circuit.name or 'circuit'}_routed")
        if hardware_model is None:
            self._hardware_model = HardwareModel.synthetic(
                self._graph, seed=seed or self._rng.randrange(1_000_000)
            )
        else:
            self._hardware_model = hardware_model

        self._apply_ready_gates()
        twoq_total = count_two_qubit_gates(self._circuit)
        default_cap = min(200, 8 * twoq_total + 30)
        self._max_steps = self.config.max_steps or default_cap
        return self._build_state()

    def step(self, action: int | PhysicalSwap) -> tuple[RoutingState, float, bool, dict]:
        """Apply a swap action and schedule any newly executable gates."""
        if self._done:
            msg = "Environment already terminated."
            raise RuntimeError(msg)

        edge = self._resolve_action(action)
        depth_before = self._routed.depth() if self._routed is not None else 0
        twoq_before = count_two_qubit_gates(self._routed) if self._routed is not None else 0
        inst_ptr_before = self._inst_ptr
        if self._hardware_model is not None:
            p2_error, t2_duration_ns = self._hardware_model.edge_error_and_duration(
                edge[0], edge[1], directed=True
            )
        else:
            p2_error, t2_duration_ns = 0.0, 0.0

        self._apply_swap(edge)
        self._apply_ready_gates()
        self._record_swap(edge)
        self._step_count += 1

        depth_after = self._routed.depth() if self._routed is not None else depth_before
        twoq_after = (
            count_two_qubit_gates(self._routed) if self._routed is not None else twoq_before
        )
        noise_delta = -(twoq_after - twoq_before)
        progress_twoq = _count_twoq_in_slice(self._circuit, inst_ptr_before, self._inst_ptr)
        swap_penalty = -(
            self.config.reward.base_swap_penalty
            + self.config.reward.error_penalty_weight * p2_error
            + self.config.reward.time_penalty_weight * (t2_duration_ns / 1000.0)
        )
        breakdown = RewardBreakdown(
            swap_cost=swap_penalty,
            depth_delta=-(depth_after - depth_before),
            noise_proxy_delta=noise_delta,
            progress_bonus=self.config.reward.progress_weight * progress_twoq,
        )
        self._done = self._inst_ptr >= len(self._circuit.data)  # type: ignore[arg-type]
        hit_cap = bool(self._max_steps and self._step_count >= self._max_steps and not self._done)
        if self._done and not hit_cap:
            breakdown.completion_bonus = self.config.reward.completion_bonus
        if hit_cap:
            self._done = True
            breakdown.failure_penalty = self.config.reward.failure_penalty

        reward = breakdown.total(self.config.reward)

        info = {
            "reward_breakdown": breakdown.__dict__,
            "swap": edge,
            "step_count": self._step_count,
            "hit_step_cap": hit_cap,
            "max_steps": self._max_steps,
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
            if inst.operation.name == "barrier":
                self._append_mapped(inst)
                self._inst_ptr += 1
                continue
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
        candidate_swaps, mask = self._candidates_and_mask(frontier)
        frontier_phys = _logical_to_physical(self._layout, frontier[0]) if frontier else None
        frontier_distance = None
        if frontier_phys is not None and self._graph is not None:
            try:
                frontier_distance = float(nx.shortest_path_length(self._graph, *frontier_phys))
            except nx.NetworkXNoPath:
                frontier_distance = None
        return RoutingState(
            layout=dict(self._layout),
            frontier=frontier,
            lookahead=frontier,
            candidate_swaps=candidate_swaps,
            action_mask=mask,
            frontier_phys=frontier_phys,
            frontier_distance=frontier_distance,
            step_count=self._step_count,
            done=self._done,
            recent_swaps=list(self._recent_swaps),
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

    def _candidates_and_mask(
        self, frontier: List[LogicalGate]
    ) -> tuple[List[PhysicalSwap], List[bool]]:
        """Return SABRE-like candidate set plus legality mask."""

        candidates = self._candidate_edges(frontier)
        mask = [self._is_action_allowed(edge, frontier) for edge in candidates]
        return candidates, mask

    def _candidate_edges(self, frontier: Iterable[LogicalGate]) -> List[PhysicalSwap]:
        """Edges on shortest paths for blocked frontier gates with small neighbourhood."""

        assert self._graph is not None
        frontier_list = list(frontier)
        if not frontier_list:
            return sorted(_normalize_edge(edge) for edge in self._graph.edges())

        blocked_pairs: list[tuple[LogicalGate, tuple[int, int]]] = []
        for gate in frontier_list:
            phys = _logical_to_physical(self._layout, gate)
            if not self._graph.has_edge(*phys):
                blocked_pairs.append((gate, phys))

        if not blocked_pairs:
            gate = frontier_list[0]
            blocked_pairs.append((gate, _logical_to_physical(self._layout, gate)))

        path_nodes: set[int] = set()
        candidate_edges: set[PhysicalSwap] = set()
        k_paths = max(1, int(self.config.candidate_k_shortest))
        for _gate, phys in blocked_pairs:
            try:
                paths = nx.shortest_simple_paths(self._graph, phys[0], phys[1])
                for path_idx, path in zip(range(k_paths), paths):
                    path_nodes.update(path)
                    candidate_edges.update(_edges_from_path(path))
            except (nx.NetworkXNoPath, nx.NetworkXError):
                continue

        if self.config.neighbor_hops > 0 and path_nodes:
            neighbour_nodes = set(path_nodes)
            shell = set(path_nodes)
            for _ in range(self.config.neighbor_hops):
                expansion: set[int] = set()
                for node in shell:
                    expansion.update(self._graph.neighbors(node))
                neighbour_nodes.update(expansion)
                shell = expansion
            for node in neighbour_nodes:
                for nbr in self._graph.neighbors(node):
                    candidate_edges.add(_normalize_edge((node, nbr)))

        if not candidate_edges:
            candidate_edges = {edge for edge in (_normalize_edge(e) for e in self._graph.edges())}

        capped = self._cap_candidates(candidate_edges, frontier_list)
        return sorted(capped)

    def _cap_candidates(
        self,
        candidates: Iterable[PhysicalSwap],
        frontier_full: list[LogicalGate],
    ) -> List[PhysicalSwap]:
        candidates_list = list(candidates)
        if len(candidates_list) <= self.config.candidate_cap:
            return candidates_list

        scored: list[tuple[float, PhysicalSwap]] = []
        for edge in candidates_list:
            score = self._candidate_score(edge, frontier_full)
            scored.append((score, edge))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [edge for _, edge in scored[: self.config.candidate_cap]]

    def _candidate_score(self, edge: PhysicalSwap, frontier_full: list[LogicalGate]) -> float:
        """Distance-based score used only when capping candidates."""

        layout_after = _swapped_layout(self._layout, *edge)
        total_dist = 0.0
        for idx, gate in enumerate(frontier_full):
            phys_after = (layout_after[gate[0]], layout_after[gate[1]])
            try:
                dist = float(nx.shortest_path_length(self._graph, *phys_after))
            except (nx.NetworkXNoPath, nx.NetworkXError):
                dist = float(self._graph.number_of_nodes())
            weight = 1.0 if idx == 0 else 0.5
            total_dist += weight * dist
        if _normalize_edge(edge) in self._recent_swaps:
            total_dist += 0.25
        return total_dist

    def _is_action_allowed(self, edge: PhysicalSwap, frontier: List[LogicalGate]) -> bool:
        """Anti-oscillation guard: forbid swap reversal without progress."""

        normalized = _normalize_edge(edge)
        recent_window = self._recent_swaps[-max(1, self.config.anti_osc_window) :]
        if normalized not in recent_window:
            return True
        return self._would_enable_frontier(frontier, edge)

    def _would_enable_frontier(self, frontier: List[LogicalGate], edge: PhysicalSwap) -> bool:
        if not frontier:
            return False
        layout_after = _swapped_layout(self._layout, *edge)
        for gate in frontier:
            phys = (layout_after[gate[0]], layout_after[gate[1]])
            if _is_edge(phys, self._coupling_edges):
                return True
        return False

    def _resolve_action(self, action: int | PhysicalSwap) -> PhysicalSwap:
        state = self._build_state()
        if isinstance(action, int):
            if action < 0 or action >= len(state.candidate_swaps):
                msg = f"Action index {action} out of range."
                raise IndexError(msg)
            edge = state.candidate_swaps[action]
            if not state.action_mask[action]:
                msg = f"Action {action} on edge {edge} is masked out."
                raise ValueError(msg)
        else:
            edge = action
            if edge not in state.candidate_swaps:
                msg = f"Swap {edge} is not a legal action."
                raise ValueError(msg)
            idx = state.candidate_swaps.index(edge)
            if not state.action_mask[idx]:
                msg = f"Swap {edge} is masked out."
                raise ValueError(msg)
        return edge

    def _record_swap(self, edge: PhysicalSwap) -> None:
        normalized = _normalize_edge(edge)
        self._recent_swaps.append(normalized)
        window = max(1, self.config.anti_osc_window)
        if len(self._recent_swaps) > window:
            self._recent_swaps = self._recent_swaps[-window:]

    # ---------------------------------------------------------- Introspection --
    @property
    def routed_circuit(self) -> QuantumCircuit:
        if self._routed is None:
            raise RuntimeError("reset must be called before accessing routed circuit.")
        return self._routed

    @property
    def hardware_model(self) -> HardwareModel | None:
        return self._hardware_model

    @property
    def graph(self) -> nx.Graph | None:
        return self._graph


def _normalize_edges(coupling_map: CouplingMap | Sequence[Sequence[int]]) -> List[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [tuple(edge) for edge in coupling_map]


def _is_edge(pair: Tuple[int, int], edges: Sequence[Tuple[int, int]]) -> bool:
    return pair in edges or (pair[1], pair[0]) in edges


def _edges_from_path(path: Sequence[int]) -> List[PhysicalSwap]:
    if len(path) < 2:
        return []
    return [_normalize_edge((a, b)) for a, b in zip(path[:-1], path[1:])]


def _logical_on_physical(layout: dict[int, int], physical: int) -> int | None:
    for logical, phys in layout.items():
        if phys == physical:
            return logical
    return None


def _logical_to_physical(layout: dict[int, int], gate: LogicalGate) -> tuple[int, int]:
    return (layout[gate[0]], layout[gate[1]])


def _normalize_edge(edge: PhysicalSwap) -> PhysicalSwap:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _swapped_layout(layout: dict[int, int], u: int, v: int) -> dict[int, int]:
    new_layout = dict(layout)
    logical_on_u = _logical_on_physical(layout, u)
    logical_on_v = _logical_on_physical(layout, v)
    if logical_on_u is not None:
        new_layout[logical_on_u] = v
    if logical_on_v is not None:
        new_layout[logical_on_v] = u
    return new_layout


def _count_twoq_in_slice(circuit: QuantumCircuit, start: int, end: int) -> int:
    """Count two-qubit gates between indices [start, end)."""
    return sum(
        1
        for inst in circuit.data[start:end]
        if inst.operation.num_qubits == 2 and inst.operation.name != "swap"
    )


def _coerce_initial_layout(
    layout: dict[int, int], num_logical: int, num_phys: int
) -> dict[int, int]:
    """Validate and normalise an externally supplied initial layout."""

    keys = set(layout.keys())
    required = set(range(num_logical))
    if keys != required:
        msg = "initial_layout must provide a mapping for every logical qubit."
        raise ValueError(msg)
    values = list(layout.values())
    if len(values) != len(set(values)):
        msg = "initial_layout physical qubits must be unique."
        raise ValueError(msg)
    if any(v < 0 or v >= num_phys for v in values):
        msg = "initial_layout contains out-of-range physical indices."
        raise ValueError(msg)
    return {int(k): int(v) for k, v in layout.items()}
