"""Minimal example: run weighted SABRE on a tiny circuit."""

from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl import (
    WeightedDistanceParams,
    make_synthetic_hardware,
    route_with_weighted_sabre,
)


def _demo_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="ghz_line")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


def main() -> None:
    circuit = _demo_circuit()
    coupling_map = CouplingMap([[0, 1], [1, 2]])
    graph = nx.Graph(coupling_map.get_edges())
    hardware = make_synthetic_hardware(
        graph,
        seed=11,
        profile="realistic",
        drift_rate=0.05,
        snapshots=2,
        snapshot_spacing_ns=50_000.0,
        crosstalk_factor=0.01,
    )

    result = route_with_weighted_sabre(
        circuit,
        coupling_map,
        hardware_model=hardware,
        seed=7,
        trials=4,
        distance_params=WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2),
        snapshot_mode="avg",
    )
    metrics = result.metrics
    trial_field = result.extra.get("trial")
    trial = trial_field + 1 if isinstance(trial_field, int) else "?"
    print("Weighted SABRE on demo circuit")
    print(f"  baseline_name: {result.name}")
    print(f"  selected_trial: {trial} / {result.extra.get('trials_total')}")
    print(f"  swaps: {metrics.swaps}")
    print(f"  twoq_depth: {metrics.two_qubit_depth}")
    print(f"  overall_log_success: {metrics.overall_log_success:.3f}")
    if metrics.total_duration_ns is not None:
        print(f"  total_duration_ns: {metrics.total_duration_ns:.1f}")


if __name__ == "__main__":
    main()
