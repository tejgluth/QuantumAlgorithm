"""Gauntlet orchestrator for Weighted SABRE evaluation suites."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--qasm-root", type=Path, help="Optional QASMBench root.")
    parser.add_argument("--selection-seed", type=int, default=7)
    return parser.parse_args(argv)


def _timestamp_dir(base: Path | None = None) -> Path:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
            "1",
            "8",
            "16",
            "--run-weighted-sabre",
            "--weighted-alpha-time",
            "0.5",
            "--weighted-beta-xtalk",
            "0.2",
            "--weighted-snapshot-mode",
            "avg",
            "--weighted-trials",
            "8",
            "--run-preset-opt3",
            "--selection-seed",
            str(selection_seed),
        ]
    )
    if args.qasm_root:
        argv += ["--qasm-root", str(args.qasm_root)]
    if args.hardware_directional:
        argv.append("--hardware-directional")
    run_eval.main(argv)
    return suite_dir / results_name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
