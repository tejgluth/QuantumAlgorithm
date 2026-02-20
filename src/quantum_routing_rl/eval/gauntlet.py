"""Gauntlet orchestrator for Weighted SABRE evaluation suites."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import pandas as pd

from quantum_routing_rl.benchmarks.fetch_qasmbench import ensure_qasmbench
from quantum_routing_rl.benchmarks.qasm_discovery import (
    discover_qasm_root,
    is_fixture_like,
    pretty_print_discovery,
)
from quantum_routing_rl.eval import run_eval


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["small", "full", "industrial"],
        default="small",
        help="Gauntlet tier to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Base output directory (default artifacts/gauntlet/<timestamp>).",
    )
    parser.add_argument("--hardware-draws", type=int, default=10)
    parser.add_argument("--hardware-seed-base", type=int, default=211)
    parser.add_argument("--hardware-profile", type=str, default="realistic")
    parser.add_argument("--hardware-snapshots", type=int, default=1)
    parser.add_argument("--hardware-drift", type=float, default=0.0)
    parser.add_argument("--hardware-crosstalk", type=float, default=0.01)
    parser.add_argument("--hardware-directional", action="store_true")
    parser.add_argument("--hardware-snapshot-spacing", type=float, default=50_000.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    parser.add_argument("--qasm-root", type=Path, help="(Deprecated) direct QASMBench root.")
    parser.add_argument(
        "--qasmbench-root",
        type=Path,
        help="Datasets directory that contains QASMBench; discovery picks the best .qasm subtree.",
    )
    parser.add_argument(
        "--require-qasmbench",
        action="store_true",
        help="Fail fast unless a real QASMBench dataset (not fixtures) is found.",
    )
    parser.add_argument(
        "--min-qasm",
        type=int,
        help="Override minimum required .qasm files (default depends on --mode).",
    )
    parser.add_argument(
        "--auto-download-qasmbench",
        action="store_true",
        help="Attempt to fetch QASMBench if discovery fails or is too small.",
    )
    parser.add_argument(
        "--qasmbench-url",
        type=str,
        default="https://github.com/pnnl/QASMBench",
        help="Repository/zip URL for auto-download.",
    )
    parser.add_argument(
        "--qasmbench-dest",
        type=Path,
        default=Path("artifacts/benchmarks/qasmbench_src"),
        help="Destination folder for auto-downloaded QASMBench.",
    )
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument(
        "--qiskit-trials",
        type=int,
        nargs="+",
        default=[1, 8, 16],
        help="Qiskit SABRE trial budgets to evaluate (default: 1 8 16).",
    )
    parser.add_argument(
        "--weighted-trials",
        type=int,
        default=8,
        help="Weighted SABRE trial count (default: 8).",
    )
    parser.add_argument(
        "--max-circuits-per-suite",
        type=int,
        help="Optional cap passed to run_eval --max-circuits for faster validation runs.",
    )
    parser.add_argument(
        "--max-qubits-per-suite",
        type=int,
        help="Optional width cap passed to run_eval --max-qubits.",
    )
    parser.add_argument(
        "--max-ops-per-circuit",
        type=int,
        help="Optional operation cap passed to run_eval --max-ops.",
    )
    parser.add_argument(
        "--circuit-selection",
        choices=["stable", "smallest"],
        default="stable",
        help="Ordering passed to run_eval before per-suite slicing.",
    )
    return parser.parse_args(argv)


def _timestamp_dir(base: Path | None = None) -> Path:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    root = base or Path("artifacts") / "gauntlet"
    return root / ts


def _suite_list(mode: str) -> list[str]:
    if mode == "small":
        return ["dev", "pressure", "qasmbench_small", "hard_small"]
    if mode == "full":
        return [
            "dev",
            "pressure",
            "qasmbench_small",
            "qasmbench_medium",
            "hard_small",
            "hard_medium",
        ]
    return ["qasmbench_hard", "hard_large"]


def _min_qasm_required(mode: str, override: int | None) -> int:
    if override is not None:
        return override
    if mode == "industrial":
        # Upstream pnnl/QASMBench currently provides ~252 .qasm files.
        # Keep strict mode meaningful while remaining compatible with the
        # canonical public dataset.
        return 240
    if mode == "full":
        return 200
    return 0


def _should_require_qasm(args: argparse.Namespace) -> bool:
    return args.require_qasmbench or args.mode in {"full", "industrial"}


def _effective_qasm_root(args: argparse.Namespace) -> Path | None:
    """Resolve dataset root with discovery, enforcement, and optional fetching."""

    require = _should_require_qasm(args)
    min_qasm = _min_qasm_required(args.mode, args.min_qasm)
    env_root = (
        Path(os.environ["QASMBENCH_ROOT"]).expanduser() if "QASMBENCH_ROOT" in os.environ else None
    )
    general_root = args.qasmbench_root or args.qasm_root or env_root

    if not require and general_root is None:
        return args.qasm_root  # allow fixtures when explicitly not required

    if general_root is None and args.auto_download_qasmbench:
        fetched = ensure_qasmbench(args.qasmbench_dest, repo=args.qasmbench_url)
        if fetched:
            general_root = fetched

    if general_root is None:
        msg = (
            "[gauntlet] QASMBench dataset required for this mode. "
            "Provide --qasmbench-root (datasets directory) or set QASMBENCH_ROOT, "
            "or enable --auto-download-qasmbench."
        )
        raise SystemExit(msg)

    discovery = discover_qasm_root(general_root)
    print(pretty_print_discovery(discovery))
    selected = discovery.get("selected_root")
    count = discovery.get("qasm_count", 0)
    if selected and not is_fixture_like(selected, count) and count >= min_qasm:
        return Path(selected)

    if args.auto_download_qasmbench:
        fetched = ensure_qasmbench(args.qasmbench_dest, repo=args.qasmbench_url)
        if fetched:
            discovery = discover_qasm_root(fetched)
            print("[gauntlet] after auto-download:")
            print(pretty_print_discovery(discovery))
            selected = discovery.get("selected_root")
            count = discovery.get("qasm_count", 0)
            if selected and not is_fixture_like(selected, count) and count >= min_qasm:
                return Path(selected)

    candidates = "\n".join(
        f"- {path} ({cnt} qasm)" for path, cnt in discovery.get("candidates", [])
    )
    expl = (
        "[gauntlet] Failed to locate a real QASMBench dataset.\n"
        "The path you provided looks like the project repo or fixtures.\n"
        "Provide the datasets folder that contains QASMBench (not this repo), "
        "or enable --auto-download-qasmbench.\n"
        f"Minimum required for mode '{args.mode}': {min_qasm} qasm files.\n"
        "Top candidates seen:\n"
        f"{candidates or '  (none)'}"
    )
    raise SystemExit(expl)


def _write_summary(df: pd.DataFrame, mean_path: Path, std_path: Path) -> None:
    metrics = [
        "swaps_inserted",
        "twoq_count",
        "twoq_depth",
        "depth",
        "routing_runtime_s",
        "noise_proxy_score",
        "log_success_proxy",
        "duration_proxy",
        "overall_log_success",
        "total_duration_ns",
        "decoherence_penalty",
    ]
    agg = df.groupby(["graph_id", "baseline_name"]).agg({m: ["mean", "std"] for m in metrics})
    mean_df = agg.xs("mean", level=1, axis=1)
    mean_df.columns = [f"{col}_mean" for col in mean_df.columns]
    std_df = agg.xs("std", level=1, axis=1)
    std_df.columns = [f"{col}_std" for col in std_df.columns]
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.reset_index().to_csv(mean_path, index=False)
    std_path.parent.mkdir(parents=True, exist_ok=True)
    std_df.reset_index().to_csv(std_path, index=False)


def _baseline_comparison(df: pd.DataFrame, out_path: Path) -> None:
    out_lines = ["# Baseline Comparison", ""]
    grouped = df.groupby("baseline_name")["overall_log_success"].mean().sort_values(ascending=False)
    best = grouped.iloc[0]
    for name, val in grouped.items():
        delta = val - best
        out_lines.append(f"- {name}: {val:.3f} (Δ vs best {delta:+.3f})")
    out_path.write_text("\n".join(out_lines))


def _run_single_suite(
    suite: str, args: argparse.Namespace, out_root: Path, selection_seed: int
) -> Path:
    suite_dir = out_root / suite
    results_name = f"results_{suite}.csv"
    summary_name = f"summary_{suite}.csv"
    summary_std_name = f"summary_{suite}_std.csv"
    argv = (
        [
            "--suite",
            f"gauntlet:{suite}",
            "--out",
            str(suite_dir),
            "--results-name",
            results_name,
            "--summary-name",
            summary_name,
            "--summary-std-name",
            summary_std_name,
            "--seeds",
        ]
        + [str(s) for s in args.seeds]
        + [
            "--hardware-samples",
            str(args.hardware_draws),
            "--hardware-seed-base",
            str(args.hardware_seed_base),
            "--hardware-profile",
            args.hardware_profile,
            "--hardware-snapshots",
            str(args.hardware_snapshots),
            "--hardware-drift",
            str(args.hardware_drift),
            "--hardware-snapshot-spacing",
            str(args.hardware_snapshot_spacing),
            "--hardware-crosstalk",
            str(args.hardware_crosstalk),
            "--qiskit-trials",
            *[str(max(1, int(v))) for v in args.qiskit_trials],
            "--run-weighted-sabre",
            "--weighted-alpha-time",
            "0.5",
            "--weighted-beta-xtalk",
            "0.2",
            "--weighted-snapshot-mode",
            "avg",
            "--weighted-trials",
            str(max(1, int(args.weighted_trials))),
            "--run-preset-opt3",
            "--selection-seed",
            str(selection_seed),
        ]
    )
    if args.qasm_root:
        argv += ["--qasm-root", str(args.qasm_root)]
    if args.max_circuits_per_suite is not None:
        argv += ["--max-circuits", str(max(0, int(args.max_circuits_per_suite)))]
    if args.max_qubits_per_suite is not None:
        argv += ["--max-qubits", str(max(0, int(args.max_qubits_per_suite)))]
    if args.max_ops_per_circuit is not None:
        argv += ["--max-ops", str(max(0, int(args.max_ops_per_circuit)))]
    argv += ["--circuit-selection", args.circuit_selection]
    if _should_require_qasm(args):
        argv.append("--require-qasmbench")
    if args.hardware_directional:
        argv.append("--hardware-directional")
    run_eval.main(argv)
    return suite_dir / results_name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.qasm_root = _effective_qasm_root(args)
    out_root = _timestamp_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    selection_seed = args.selection_seed
    suites = _suite_list(args.mode)
    result_paths: list[Path] = []
    for suite in suites:
        path = _run_single_suite(suite, args, out_root, selection_seed)
        result_paths.append(path)
    if not result_paths:
        print("[gauntlet] no suites executed")
        return 1
    df = pd.concat(pd.read_csv(p) for p in result_paths)
    combined_results = out_root / f"results_gauntlet_{args.mode}.csv"
    df.to_csv(combined_results, index=False)
    mean_path = out_root / f"summary_gauntlet_{args.mode}.csv"
    std_path = out_root / f"summary_gauntlet_{args.mode}_std.csv"
    _write_summary(df, mean_path, std_path)
    _baseline_comparison(df, out_root / "baseline_comparison.md")
    print(f"[gauntlet] wrote combined results to {combined_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
