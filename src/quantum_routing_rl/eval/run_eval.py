"""Evaluation harness running SABRE-family baselines on QASMBench."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import (
    run_best_available_sabre,
    run_sabre_layout_swap,
)
from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit, load_suite
from quantum_routing_rl.models.policy import load_swap_policy, route_with_policy


REQUIRED_COLUMNS = {
    "n_qubits",
    "graph_id",
    "baseline_name",
    "swaps_inserted",
    "twoq_count",
    "depth",
    "twoq_depth",
    "routing_runtime_s",
    "noise_proxy_score",
    "seed",
    "qiskit_version",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dev", "full"], default="dev")
    parser.add_argument("--dev-limit", type=int, default=20, help="Max circuits for dev suite.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Path to local QASMBench root (defaults to env QASMBENCH_ROOT or tests fixtures).",
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed for SABRE baselines.")
    parser.add_argument(
        "--il-checkpoint",
        type=Path,
        help="Optional path to an IL checkpoint to evaluate alongside baselines.",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        help="Optional path to an RL checkpoint to evaluate alongside baselines.",
    )
    return parser.parse_args(argv)


def _default_qasm_root() -> Path:
    env_path = os.environ.get("QASMBENCH_ROOT")
    if env_path:
        return Path(env_path).expanduser()

    # Fallback to in-repo fixtures for dev usage.
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "qasmbench"


def _build_coupling_maps() -> dict[str, CouplingMap]:
    return {
        "line_3": CouplingMap([[0, 1], [1, 2]]),
        "square_4": CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]]),
    }


def _coupling_size(map_obj: CouplingMap) -> int:
    if hasattr(map_obj, "size"):
        try:
            return int(map_obj.size())
        except Exception:
            pass
    edges = map_obj.get_edges()
    return max(max(edge) for edge in edges) + 1


def _collect_metadata() -> dict[str, object]:
    try:
        import torch
    except Exception:
        torch = None

    return {
        "python_version": sys.version,
        "qiskit_version": qiskit_version,
        "torch_version": getattr(torch, "__version__", None),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_available": bool(torch and torch.cuda.is_available()),
    }


def _validate_results(results_path: Path) -> None:
    if not results_path.exists():
        msg = f"Missing results file: {results_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(results_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    has_id = "circuit_id" in df.columns or "path" in df.columns
    if not has_id:
        missing.append("circuit_id or path")
    if missing:
        msg = f"results.csv missing columns: {missing}"
        raise ValueError(msg)


def _filter_coupling_maps(
    coupling_maps: dict[str, CouplingMap], circuit: QasmCircuit
) -> Iterable[tuple[str, CouplingMap]]:
    for name, cmap in coupling_maps.items():
        if circuit.circuit.num_qubits <= _coupling_size(cmap):
            yield name, cmap


def _result_record(circuit: QasmCircuit, graph_id: str, result) -> dict[str, object]:
    metrics = result.metrics
    return {
        "circuit_id": circuit.circuit_id,
        "path": circuit.path.as_posix(),
        "n_qubits": circuit.circuit.num_qubits,
        "graph_id": graph_id,
        "baseline_name": result.name,
        "swaps_inserted": metrics.swaps,
        "twoq_count": metrics.two_qubit_count,
        "depth": metrics.depth,
        "twoq_depth": metrics.two_qubit_depth,
        "routing_runtime_s": result.runtime_s,
        "noise_proxy_score": metrics.success_prob,
        "seed": result.seed,
        "qiskit_version": qiskit_version,
    }


def _evaluate_circuit(
    circuit: QasmCircuit,
    *,
    coupling_maps: dict[str, CouplingMap],
    seed: int,
    il_model=None,
    rl_model=None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    applicable_maps = list(_filter_coupling_maps(coupling_maps, circuit))
    if not applicable_maps:
        msg = f"No coupling map large enough for circuit {circuit.circuit_id}."
        raise ValueError(msg)

    for graph_id, coupling in applicable_maps:
        baselines = [
            run_sabre_layout_swap(circuit.circuit, coupling, seed=seed),
            run_best_available_sabre(circuit.circuit, coupling, seed=seed),
        ]
        for result in baselines:
            rows.append(_result_record(circuit, graph_id, result))
        if il_model is not None:
            il_result = route_with_policy(
                il_model, circuit.circuit, coupling, name="il_policy", seed=seed
            )
            rows.append(_result_record(circuit, graph_id, il_result))
        if rl_model is not None:
            rl_result = route_with_policy(
                rl_model, circuit.circuit, coupling, name="rl_policy", seed=seed
            )
            rows.append(_result_record(circuit, graph_id, rl_result))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    qasm_root = Path(args.qasm_root or _default_qasm_root()).expanduser()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite(qasm_root, suite=args.suite, dev_limit=args.dev_limit)
    coupling_maps = _build_coupling_maps()
    il_model = None
    if args.il_checkpoint:
        il_model = load_swap_policy(args.il_checkpoint)
    rl_model = None
    if args.rl_checkpoint:
        rl_model = load_swap_policy(args.rl_checkpoint)

    records: list[dict[str, object]] = []
    for circuit in suite:
        records.extend(
            _evaluate_circuit(
                circuit,
                coupling_maps=coupling_maps,
                seed=args.seed,
                il_model=il_model,
                rl_model=rl_model,
            )
        )

    results_path = out_dir / "results.csv"
    df = pd.DataFrame(records)
    df.to_csv(results_path, index=False)

    metadata_path = out_dir / "metadata.json"
    existing_meta = {}
    if metadata_path.exists():
        try:
            existing_meta = json.loads(metadata_path.read_text())
        except Exception:
            existing_meta = {}
    metadata = _collect_metadata()
    metadata.update(existing_meta)
    metadata.update({"suite": args.suite, "dev_limit": args.dev_limit, "qasm_root": str(qasm_root)})
    if args.il_checkpoint:
        metadata["il_checkpoint"] = str(args.il_checkpoint)
    if args.rl_checkpoint:
        metadata["rl_checkpoint"] = str(args.rl_checkpoint)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    _validate_results(results_path)
    print(f"Wrote {len(df)} rows to {results_path}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
