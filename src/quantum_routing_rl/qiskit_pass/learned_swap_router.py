"""Qiskit TransformationPass wrapping the learned swap router."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from quantum_routing_rl.baselines.qiskit_baselines import BaselineResult, run_sabre_layout_swap
from quantum_routing_rl.env.reward import RewardConfig
from quantum_routing_rl.env.routing_env import RoutingEnvConfig
from quantum_routing_rl.eval.metrics import assert_coupling_compatible
from quantum_routing_rl.models.policy import SwapPolicy, load_swap_policy, route_with_policy


class LearnedSwapRouter(TransformationPass):
    """TransformationPass that routes using a learned swap policy."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        policy: SwapPolicy | None = None,
        reward_config: RewardConfig | None = None,
        env_config: RoutingEnvConfig | None = None,
        seed: int | None = None,
        coupling_map: CouplingMap | Sequence[Sequence[int]] | None = None,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self._policy = policy
        self.reward_config = reward_config or RewardConfig()
        self.env_config = env_config or RoutingEnvConfig(reward=self.reward_config)
        self.seed = seed
        self._explicit_coupling = _as_coupling_map(coupling_map) if coupling_map else None

    # ------------------------------------------------------------------ API --
    def run(self, dag: DAGCircuit) -> DAGCircuit | QuantumCircuit:
        """Route the given DAG/QuantumCircuit respecting the coupling map."""
        circuit = dag if isinstance(dag, QuantumCircuit) else dag_to_circuit(dag)
        coupling_map = self._resolve_coupling_map()
        policy = self._load_policy()

        if policy is None:
            result = run_sabre_layout_swap(circuit, coupling_map, seed=self.seed)
        else:
            result = self._route_with_learned(policy, circuit, coupling_map)

        assert_coupling_compatible(result.circuit, _normalize_edges(coupling_map))
        if hasattr(self, "property_set"):
            self.property_set["routing_result"] = result
            self.property_set["routing_method"] = "learned_swap_router"
            self.property_set["coupling_map"] = coupling_map
        return circuit_to_dag(result.circuit) if isinstance(dag, DAGCircuit) else result.circuit

    # ------------------------------------------------------------- Helpers --
    def _load_policy(self) -> SwapPolicy | None:
        if self._policy is not None:
            return self._policy
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._policy = load_swap_policy(self.checkpoint_path)
            return self._policy
        return None

    def _resolve_coupling_map(self) -> CouplingMap:
        if self._explicit_coupling is not None:
            return self._explicit_coupling

        cmap = None
        if hasattr(self, "property_set"):
            cmap = self.property_set.get("coupling_map")
            if cmap is None:
                target = self.property_set.get("target")
                if isinstance(target, Target) and hasattr(target, "build_coupling_map"):
                    cmap = target.build_coupling_map()

        if cmap is None:
            msg = "LearnedSwapRouter requires a coupling map or target in the transpiler context."
            raise TranspilerError(msg)
        return _as_coupling_map(cmap)

    def _route_with_learned(
        self, policy: SwapPolicy, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> BaselineResult:
        return route_with_policy(
            policy,
            circuit,
            coupling_map,
            name="learned_policy",
            seed=self.seed,
            env_config=self.env_config,
        )


def _as_coupling_map(coupling: CouplingMap | Sequence[Sequence[int]]) -> CouplingMap:
    if isinstance(coupling, CouplingMap):
        return coupling
    return CouplingMap(coupling)


def _normalize_edges(coupling_map: CouplingMap | Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(coupling_map, CouplingMap):
        return [tuple(edge) for edge in coupling_map.get_edges()]
    return [tuple(edge) for edge in coupling_map]
