"""Evaluation harness for SABRE baselines and learned routing policies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import networkx as nx
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.baselines.qiskit_baselines import (
    run_best_available_sabre,
    run_sabre_layout_swap,
)
from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit, load_suite
from quantum_routing_rl.benchmarks.synthetic_generator import pressure_suite
from quantum_routing_rl.models.policy import load_swap_policy, route_with_policy
from quantum_routing_rl.models.teacher import route_with_teacher
from quantum_routing_rl.hardware.model import HardwareModel

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
    "log_success_proxy",
    "duration_proxy",
    "seed",
    "hardware_seed",
    "qiskit_version",
}


# --------------------------------------------------------------------------- CLI
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dev", "pressure", "full"], default="dev")
    parser.add_argument("--dev-limit", type=int, default=20, help="Max circuits for dev suite.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Path to local QASMBench root (defaults to env QASMBENCH_ROOT or tests fixtures).",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="results.csv",
        help="Filename for the results CSV written inside --out.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        help="Optional summary filename (defaults to results-name with 'summary').",
    )
    parser.add_argument("--seed", type=int, help="Single seed (deprecated, prefer --seeds).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Seeds for baselines and learned policies (>=3 for full).",
    )
    parser.add_argument(
        "--pressure-seed",
        type=int,
        default=99,
        help="Seed for synthetic pressure circuit generation.",
    )
    parser.add_argument(
        "--pressure-qasm",
        type=int,
        default=6,
        help="Number of QASMBench circuits to include in the pressure suite.",
    )
    parser.add_argument(
        "--il-checkpoint",
        type=Path,
        help="Optional path to an IL checkpoint to evaluate alongside baselines.",
    )
    parser.add_argument(
        "--il-name",
        type=str,
        help="Name to use for the IL policy in outputs (defaults based on checkpoint).",
    )
    parser.add_argument(
        "--il-teacher-mix",
        type=float,
        default=0.0,
        help="Blend factor for teacher scores when running the IL policy (0 = off).",
    )
    parser.add_argument(
        "--il-swap-guard",
        type=float,
        help="If set, fall back to teacher when IL swaps exceed guard * teacher swaps.",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        help="Optional path to an RL checkpoint to evaluate alongside baselines.",
    )
    parser.add_argument(
        "--rl-name",
        type=str,
        help="Name to use for the RL policy in outputs (defaults based on checkpoint).",
    )
    parser.add_argument(
        "--rl-teacher-mix",
        type=float,
        default=0.0,
        help="Blend factor for teacher scores when running the RL policy (0 = off).",
    )
    parser.add_argument(
        "--rl-swap-guard",
        type=float,
        help="If set, fall back to teacher when RL swaps exceed guard * teacher swaps.",
    )
    parser.add_argument(
        "--include-teacher",
        action="store_true",
        help="Evaluate the in-env teacher policy alongside baselines.",
    )
    parser.add_argument(
        "--hardware-samples",
        type=int,
        default=1,
        help="Number of hardware model draws per coupling graph (noise eval).",
    )
    parser.add_argument(
        "--hardware-seed-base",
        type=int,
        default=101,
        help="Base seed for synthetic hardware models.",
    )
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default="realistic",
        help="Synthetic hardware profile to sample.",
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------- Suite helpers
def _default_qasm_root() -> Path:
    env_path = os.environ.get("QASMBENCH_ROOT")
    if env_path:
        return Path(env_path).expanduser()

    # Fallback to in-repo fixtures for dev usage.
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "qasmbench"


def _build_coupling_maps(suite: str) -> dict[str, CouplingMap]:
    base = {
        "line_3": CouplingMap([[0, 1], [1, 2]]),
        "square_4": CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]]),
    }
    pressure = {
        "ring_8": CouplingMap([[i, (i + 1) % 8] for i in range(8)]),
        "grid_3x3": CouplingMap(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [0, 3],
                [3, 6],
                [1, 4],
                [4, 7],
                [2, 5],
                [5, 8],
            ]
        ),
        "heavy_hex_15": CouplingMap(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [2, 6],
                [4, 7],
                [5, 6],
                [6, 7],
                [5, 8],
                [6, 9],
                [7, 10],
                [8, 9],
                [9, 10],
                [8, 11],
                [9, 12],
                [10, 13],
                [11, 12],
                [12, 13],
                [11, 14],
                [13, 14],
            ]
        ),
    }
    if suite == "dev":
        return base
    if suite == "pressure":
        return pressure
    merged = dict(base)
    merged.update(pressure)
    return merged


def _coupling_size(map_obj: CouplingMap) -> int:
    if hasattr(map_obj, "size"):
        try:
            return int(map_obj.size())
        except Exception:
            pass
    edges = map_obj.get_edges()
    return max(max(edge) for edge in edges) + 1


def _build_hardware_models(
    coupling_maps: dict[str, CouplingMap], seeds: list[int], profile: str
) -> dict[str, list[HardwareModel]]:
    models: dict[str, list[HardwareModel]] = {}
    for graph_id, cmap in coupling_maps.items():
        edges = [tuple(edge) for edge in cmap.get_edges()]
        graph = nx.Graph()
        graph.add_edges_from(edges)
        models[graph_id] = [
            HardwareModel.synthetic(graph, seed=seed, profile=profile) for seed in seeds
        ]
    return models


def _filter_coupling_maps(
    coupling_maps: dict[str, CouplingMap], circuit: QasmCircuit
) -> Iterable[tuple[str, CouplingMap]]:
    for name, cmap in coupling_maps.items():
        if circuit.circuit.num_qubits <= _coupling_size(cmap):
            yield name, cmap


def _load_benchmarks(
    suite: str, qasm_root: Path, *, dev_limit: int, pressure_seed: int, pressure_qasm: int
) -> list[QasmCircuit]:
    circuits: list[QasmCircuit] = []
    seen_ids: set[str] = set()

    def _append_unique(entries: Iterable[QasmCircuit]) -> None:
        for entry in entries:
            if entry.circuit_id in seen_ids:
                continue
            seen_ids.add(entry.circuit_id)
            circuits.append(entry)

    if suite in {"dev", "full"}:
        _append_unique(
            load_suite(
                qasm_root,
                suite="dev" if suite == "dev" else "full",
                dev_limit=dev_limit,
            )
        )
    if suite in {"pressure", "full"}:
        _append_unique(pressure_suite(seed=pressure_seed))
        # A few real circuits to complement synthetic pressure cases.
        _append_unique(load_suite(qasm_root, suite="dev", dev_limit=pressure_qasm))
    return circuits


# -------------------------------------------------------------------- Metadata
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


def _result_record(
    circuit: QasmCircuit,
    graph_id: str,
    result,
    *,
    seed: int,
    suite: str,
    hardware_seed: int | None = None,
    hardware_profile: str | None = None,
) -> dict[str, object]:
    metrics = result.metrics
    return {
        "suite": suite,
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
        "log_success_proxy": metrics.log_success_proxy,
        "duration_proxy": metrics.duration_proxy,
        "seed": seed,
        "hardware_seed": hardware_seed,
        "hardware_profile": hardware_profile,
        "qiskit_version": qiskit_version,
    }


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    metrics = [
        "swaps_inserted",
        "twoq_count",
        "twoq_depth",
        "depth",
        "routing_runtime_s",
        "noise_proxy_score",
        "log_success_proxy",
        "duration_proxy",
    ]
    agg = df.groupby(["graph_id", "baseline_name"]).agg({m: ["mean", "std"] for m in metrics})
    agg.columns = ["_".join(filter(None, col)).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)


def _default_checkpoint(out_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = out_dir / "checkpoints" / f"{name}.pt"
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------------- Main
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or ([args.seed] if args.seed is not None else [13])
    qasm_root = Path(args.qasm_root or _default_qasm_root()).expanduser()
    circuits = _load_benchmarks(
        args.suite,
        qasm_root,
        dev_limit=args.dev_limit,
        pressure_seed=args.pressure_seed,
        pressure_qasm=args.pressure_qasm,
    )
    coupling_maps = _build_coupling_maps(args.suite)
    hardware_seeds = [args.hardware_seed_base + i for i in range(max(1, args.hardware_samples))]
    hardware_models = _build_hardware_models(coupling_maps, hardware_seeds, args.hardware_profile)

    il_ckpt = args.il_checkpoint or _default_checkpoint(out_dir, ["il_soft", "il"])
    rl_ckpt = args.rl_checkpoint or _default_checkpoint(out_dir, ["rl_ppo", "rl"])
    il_name = args.il_name
    if il_name is None:
        if il_ckpt and "il_soft" in il_ckpt.name:
            il_name = "il_soft"
        else:
            il_name = "il_policy"
    rl_name = args.rl_name
    if rl_name is None:
        if rl_ckpt and "rl_ppo" in rl_ckpt.name:
            rl_name = "rl_ppo"
        else:
            rl_name = "rl_policy"
    il_model = load_swap_policy(il_ckpt) if il_ckpt and Path(il_ckpt).exists() else None
    rl_model = load_swap_policy(rl_ckpt) if rl_ckpt and Path(rl_ckpt).exists() else None

    records: list[dict[str, object]] = []
    for circuit in circuits:
        applicable_maps = list(_filter_coupling_maps(coupling_maps, circuit))
        if not applicable_maps:
            continue
        for seed in seeds:
            for graph_id, coupling in applicable_maps:
                hw_list = hardware_models.get(graph_id, [])
                hw_iter = list(zip(hardware_seeds, hw_list)) if hw_list else [(None, None)]
                for hw_seed, hw_model in hw_iter:
                    teacher_swaps = None
                    baselines = [
                        run_sabre_layout_swap(
                            circuit.circuit, coupling, seed=seed, hardware_model=hw_model
                        ),
                        run_best_available_sabre(
                            circuit.circuit, coupling, seed=seed, hardware_model=hw_model
                        ),
                    ]
                    for result in baselines:
                        records.append(
                            _result_record(
                                circuit,
                                graph_id,
                                result,
                                seed=seed,
                                suite=args.suite,
                                hardware_seed=hw_seed,
                                hardware_profile=args.hardware_profile,
                            )
                        )
                    if args.include_teacher:
                        teacher_result = route_with_teacher(
                            circuit.circuit,
                            coupling,
                            name="teacher_sabre_like",
                            seed=seed,
                            hardware_model=hw_model,
                        )
                        teacher_swaps = teacher_result.metrics.swaps
                        records.append(
                            _result_record(
                                circuit,
                                graph_id,
                                teacher_result,
                                seed=seed,
                                suite=args.suite,
                                hardware_seed=hw_seed,
                                hardware_profile=args.hardware_profile,
                            )
                        )
                    if il_model is not None:
                        try:
                            il_result = route_with_policy(
                                il_model,
                                circuit.circuit,
                                coupling,
                                name=il_name,
                                seed=seed,
                                hardware_model=hw_model,
                                teacher_mix=args.il_teacher_mix,
                                teacher_swaps=teacher_swaps,
                                swap_guard_ratio=args.il_swap_guard,
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            print(
                                f"[warn] IL policy failed on {circuit.circuit_id} {graph_id}: {exc}"
                            )
                        else:
                            records.append(
                                _result_record(
                                    circuit,
                                    graph_id,
                                    il_result,
                                    seed=seed,
                                    suite=args.suite,
                                    hardware_seed=hw_seed,
                                    hardware_profile=args.hardware_profile,
                                )
                            )
                    if rl_model is not None:
                        try:
                            rl_result = route_with_policy(
                                rl_model,
                                circuit.circuit,
                                coupling,
                                name=rl_name,
                                seed=seed,
                                hardware_model=hw_model,
                                teacher_mix=args.rl_teacher_mix,
                                teacher_swaps=teacher_swaps,
                                swap_guard_ratio=args.rl_swap_guard,
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            print(
                                f"[warn] RL policy failed on {circuit.circuit_id} {graph_id}: {exc}"
                            )
                        else:
                            records.append(
                                _result_record(
                                    circuit,
                                    graph_id,
                                    rl_result,
                                    seed=seed,
                                    suite=args.suite,
                                    hardware_seed=hw_seed,
                                    hardware_profile=args.hardware_profile,
                                )
                            )

    if not records:
        msg = "No evaluation records were produced; check suite and coupling map settings."
        raise RuntimeError(msg)

    results_path = out_dir / args.results_name
    df = pd.DataFrame(records)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    summary_name = args.summary_name or args.results_name.replace("results", "summary")
    summary_path = out_dir / summary_name
    _write_summary(df, summary_path)

    metadata_path = out_dir / "metadata.json"
    existing_meta = {}
    if metadata_path.exists():
        try:
            existing_meta = json.loads(metadata_path.read_text())
        except Exception:
            existing_meta = {}
    metadata = _collect_metadata()
    metadata.update(existing_meta)
    metadata.update(
        {
            "suite": args.suite,
            "dev_limit": args.dev_limit,
            "qasm_root": str(qasm_root),
            "results": str(results_path),
            "summary": str(summary_path),
            "seeds": seeds,
            "circuits_evaluated": len(circuits),
            "coupling_maps": list(coupling_maps.keys()),
            "il_checkpoint": str(il_ckpt) if il_ckpt else None,
            "rl_checkpoint": str(rl_ckpt) if rl_ckpt else None,
            "il_name": il_name if il_model else None,
            "rl_name": rl_name if rl_model else None,
            "il_teacher_mix": args.il_teacher_mix if il_model else None,
            "rl_teacher_mix": args.rl_teacher_mix if rl_model else None,
            "il_swap_guard": args.il_swap_guard if il_model else None,
            "rl_swap_guard": args.rl_swap_guard if rl_model else None,
            "include_teacher": args.include_teacher,
            "hardware_seeds": hardware_seeds,
            "hardware_profile": args.hardware_profile,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    _validate_results(results_path)
    print(f"Wrote {len(df)} rows to {results_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
