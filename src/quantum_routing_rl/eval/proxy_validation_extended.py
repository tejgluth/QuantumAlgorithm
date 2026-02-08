"""Extended proxy validation: compare proxy scores against noisy Aer simulation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
from qiskit import transpile

try:  # Import Aer lazily to give a clear error if missing.
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
        thermal_relaxation_error,
    )
except Exception as exc:  # pragma: no cover - handled in main
    AerSimulator = None
    AerImportError = exc
else:
    AerImportError = None

from quantum_routing_rl.baselines.qiskit_baselines import (
    BaselineResult,
    run_best_available_sabre,
    run_qiskit_sabre_trials,
)
from quantum_routing_rl.benchmarks.gauntlet_manager import build_suite
from quantum_routing_rl.env.routing_env import RoutingEnvConfig
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.weighted_sabre import (
    WeightedDistanceParams,
    WeightedSabreWeights,
    route_with_weighted_sabre,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    parser.add_argument("--max-circuits", type=int, default=20, help="Max (circuit, topo) pairs.")
    parser.add_argument("--max-qubits", type=int, default=10, help="Skip circuits above this size.")
    parser.add_argument("--shots", type=int, default=1000, help="Simulation shots.")
    parser.add_argument(
        "--selection-seed", type=int, default=7, help="Deterministic selection seed."
    )
    parser.add_argument(
        "--hardware-seed", type=int, default=401, help="Seed for synthetic hardware."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Optional QASMBench root; defaults to env/QASMBENCH_ROOT or tests fixtures.",
    )
    parser.add_argument(
        "--include-weighted",
        action="store_true",
        help="Include weighted_sabre in validation (requires hardware model).",
    )
    return parser.parse_args(argv)


def _default_qasm_root() -> Path:
    env_path = os.environ.get("QASMBENCH_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "qasmbench"


def _hardware_for_graph(graph_id: str, coupling_map, seed: int) -> HardwareModel:
    edges = [tuple(e) for e in coupling_map.get_edges()]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return HardwareModel.synthetic(graph, seed=seed, profile="realistic")


def _noise_model_from_hardware(hw: HardwareModel, *, seed: int) -> NoiseModel:
    rng = random.Random(seed)
    nm = NoiseModel()
    for (u, v), props in hw.edge_props.items():
        p2 = props.p2_error if props.p2_error is not None else rng.uniform(0.003, 0.02)
        dur = (
            props.t2_duration_ns if props.t2_duration_ns is not None else rng.uniform(200.0, 600.0)
        )
        nm.add_quantum_error(depolarizing_error(min(0.3, p2 * 1.1), 2), "cx", [u, v])
    for qb, props in hw.qubit_props.items():
        ro = props.readout_error if props.readout_error is not None else rng.uniform(0.01, 0.05)
        nm.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [qb])
        p1 = props.p1_error if props.p1_error is not None else rng.uniform(1e-4, 5e-3)
        dur = props.p1_duration_ns if props.p1_duration_ns is not None else rng.uniform(20.0, 80.0)
        nm.add_quantum_error(depolarizing_error(min(0.1, p1 * 1.1), 1), "id", [qb])
        nm.add_quantum_error(
            thermal_relaxation_error(
                t1=props.t1_ns if props.t1_ns is not None else dur * 1000,
                t2=props.t2_ns if props.t2_ns is not None else dur * 800,
                time=dur,
            ),
            "id",
            [qb],
        )
    return nm


def _empirical_success(sim, circuit, shots: int) -> float | None:
    """Return empirical success prob = max count / shots for measured circuits."""
    if circuit.num_clbits == 0:
        return None
    compiled = transpile(circuit, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    if not counts:
        return None
    return max(counts.values()) / shots


def _select_slice(entries: Iterable[tuple[str, object]], limit: int, seed: int):
    rng = random.Random(seed)
    ordered = list(entries)
    rng.shuffle(ordered)
    return ordered[:limit]


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for graph_id, group in df.groupby("graph_id"):
        if len(group) < 2 or group["proxy"].isna().all() or group["empirical"].isna().all():
            rows.append({"graph_id": graph_id, "spearman": math.nan, "n": len(group)})
            continue
        corr, _ = spearmanr(group["proxy"], group["empirical"], nan_policy="omit")
        rows.append({"graph_id": graph_id, "spearman": corr, "n": len(group)})
    # overall
    if len(df) >= 2:
        corr, _ = spearmanr(df["proxy"], df["empirical"], nan_policy="omit")
        rows.append({"graph_id": "overall", "spearman": corr, "n": len(df)})
    return pd.DataFrame(rows)


def _delta_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = df.pivot_table(
        index=["circuit_id", "graph_id"],
        columns="baseline_name",
        values=["proxy", "empirical"],
    )
    required = [
        ("proxy", "weighted_sabre"),
        ("proxy", "qiskit_sabre_best"),
        ("empirical", "weighted_sabre"),
        ("empirical", "qiskit_sabre_best"),
    ]
    if any(col not in pivot for col in required):
        return pd.DataFrame()
    delta_proxy = pivot["proxy", "weighted_sabre"] - pivot["proxy", "qiskit_sabre_best"]
    delta_emp = pivot["empirical", "weighted_sabre"] - pivot["empirical", "qiskit_sabre_best"]
    mask = (~delta_proxy.isna()) & (~delta_emp.isna())
    if mask.sum() >= 2:
        corr, _ = spearmanr(delta_proxy[mask], delta_emp[mask], nan_policy="omit")
        rows.append({"graph_id": "overall", "spearman": corr, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if AerSimulator is None:
        skip_path = out_dir / "skipped.txt"
        skip_path.write_text(f"Aer not available: {AerImportError}")
        print(f"[proxy_validation] skipping: {AerImportError}")
        return 0

    qasm_root = args.qasm_root or _default_qasm_root()
    suite = build_suite(
        "qasmbench_small",
        qasm_root=qasm_root,
        selection_seed=args.selection_seed,
    )
    pair_candidates = []
    for circuit in suite.circuits:
        if circuit.circuit.num_qubits > args.max_qubits:
            continue
        for topo_id, cmap in suite.topologies.items():
            pair_candidates.append((circuit, topo_id, cmap))
    selected = _select_slice(pair_candidates, args.max_circuits, args.selection_seed)

    weighted_params = WeightedDistanceParams(alpha_time=0.5, beta_xtalk=0.2, min_edge_weight=1.0)
    weighted_weights = WeightedSabreWeights(
        lookahead_weight=0.5, decay_weight=0.25, stagnation_weight=0.25, decay_factor=0.9
    )
    env_cfg = RoutingEnvConfig(frontier_size=4)

    rows = []
    for circuit, topo_id, cmap in selected:
        hw = _hardware_for_graph(topo_id, cmap, seed=args.hardware_seed)
        noise_model = _noise_model_from_hardware(hw, seed=args.hardware_seed)
        sim = AerSimulator(noise_model=noise_model)
        baselines = [
            run_best_available_sabre(
                circuit.circuit, cmap, seed=args.selection_seed, hardware_model=hw
            ),
            run_qiskit_sabre_trials(
                circuit.circuit,
                cmap,
                seed=args.selection_seed,
                trials=8,
                hardware_model=hw,
            ),
        ]
        if args.include_weighted:
            try:
                weighted = route_with_weighted_sabre(
                    circuit.circuit,
                    cmap,
                    hardware_model=hw,
                    seed=args.selection_seed,
                    trials=8,
                    distance_params=weighted_params,
                    snapshot_mode="avg",
                    router_weights=weighted_weights,
                    env_config=env_cfg,
                )
                baselines.append(weighted)
            except Exception as exc:  # pragma: no cover - defensive
                baselines.append(
                    BaselineResult(
                        name="weighted_sabre",
                        circuit=None,
                        metrics=None,
                        runtime_s=0.0,
                        seed=args.selection_seed,
                        baseline_status="failed",
                        fallback_reason=str(exc),
                    )
                )
        for result in baselines:
            proxy = None if result.metrics is None else result.metrics.overall_log_success
            empirical = None
            if result.circuit is not None:
                try:
                    empirical = _empirical_success(sim, result.circuit, shots=args.shots)
                except Exception:
                    empirical = None
            rows.append(
                {
                    "circuit_id": circuit.circuit_id,
                    "graph_id": topo_id,
                    "baseline_name": result.name,
                    "proxy": proxy,
                    "empirical": empirical,
                    "swaps": None if result.metrics is None else result.metrics.swaps,
                    "runtime_s": result.runtime_s,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "proxy_validation_records.csv", index=False)
    corr_df = _correlations(df)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    delta_df = _delta_correlations(df)
    delta_df.to_csv(out_dir / "delta_correlations.csv", index=False)

    summary = {
        "num_pairs": len(df),
        "graphs": sorted(df["graph_id"].unique()),
        "correlations_csv": str(out_dir / "correlations.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[proxy_validation] wrote {len(df)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
