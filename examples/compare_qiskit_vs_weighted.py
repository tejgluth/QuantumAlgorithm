"""Quick comparison between Qiskit SABRE and weighted SABRE on a small circuit."""

from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl import (
    WeightedDistanceParams,
    run_best_available_sabre,
    make_synthetic_hardware,
    route_with_weighted_sabre,
)


def _demo_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="bridge")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 3)  # forces a non-trivial layout
    return qc


def _print_metrics(label: str, result) -> None:
    m = result.metrics
    print(
        f"{label}: swaps={m.swaps}, twoq_depth={m.two_qubit_depth}, "
        f"overall_log_success={m.overall_log_success:.3f}, "
        f"duration_ns={m.total_duration_ns}"
    )


def main() -> None:
    circuit = _demo_circuit()
    coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]])
    graph = nx.Graph(coupling_map.get_edges())
    hardware = make_synthetic_hardware(
        graph,
        seed=23,
        profile="realistic",
        drift_rate=0.05,
        snapshots=2,
        snapshot_spacing_ns=50_000.0,
        crosstalk_factor=0.01,
    )

    qiskit_result = run_best_available_sabre(
        circuit, coupling_map, seed=13, hardware_model=hardware
    )
    weighted_result = route_with_weighted_sabre(
        circuit,
        coupling_map,
        hardware_model=hardware,
        seed=13,
        trials=4,
        distance_params=WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2),
        snapshot_mode="avg",
    )

    print("Comparison on a 4-qubit circuit (trials=4):")
    _print_metrics("Qiskit SABRE", qiskit_result)
    _print_metrics("Weighted SABRE", weighted_result)


if __name__ == "__main__":
    main()
