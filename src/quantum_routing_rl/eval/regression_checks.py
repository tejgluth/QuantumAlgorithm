"""Regression checks for learned routers versus teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary CSV.")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline name to compare against the teacher.",
    )
    parser.add_argument("--teacher", type=str, default="teacher_sabre_like")
    parser.add_argument(
        "--graphs",
        type=str,
        nargs="*",
        help="Optional list of graph_ids to check (defaults to all in summary).",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.25,
        help="Maximum allowed ratio baseline/teacher for all metrics.",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Fail if the baseline is missing for any graph.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["swaps_inserted"],
        help="Metrics (mean columns) to ratio-check, e.g., swaps_inserted twoq_depth total_duration_ns.",
    )
    return parser.parse_args(argv)


def _ratio_failures(
    df: pd.DataFrame,
    *,
    baseline: str,
    teacher: str,
    graphs: Iterable[str],
    max_ratio: float,
    require_baseline: bool,
    metrics: Iterable[str],
) -> list[str]:
    failures: list[str] = []
    missing_cols: list[str] = []
    col_cache: dict[str, str] = {}

    def _col(metric: str) -> str:
        if metric in col_cache:
            return col_cache[metric]
        for candidate in (f"{metric}_mean", metric):
            if candidate in df.columns:
                col_cache[metric] = candidate
                return candidate
        missing_cols.append(metric)
        return metric

    # Pre-flight column presence.
    for metric in metrics:
        _col(metric)
    if missing_cols:
        failures.append(f"Missing columns for metrics: {', '.join(sorted(set(missing_cols)))}")
        return failures

    for graph in graphs:
        teacher_row = df[(df["graph_id"] == graph) & (df["baseline_name"] == teacher)]
        if teacher_row.empty:
            failures.append(f"{graph}: missing teacher results")
            continue
        baseline_row = df[(df["graph_id"] == graph) & (df["baseline_name"] == baseline)]
        if baseline_row.empty:
            if require_baseline:
                failures.append(f"{graph}: missing baseline '{baseline}'")
            continue
        for metric in metrics:
            col = _col(metric)
            teacher_val = float(teacher_row[col].iloc[0])
            base_val = float(baseline_row[col].iloc[0])
            if teacher_val <= 0:
                continue
            ratio = base_val / teacher_val
            if ratio > max_ratio:
                failures.append(
                    f"{graph}: {baseline} {metric} {base_val:.2f} > {max_ratio:.2f}x teacher {teacher_val:.2f}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.summary.exists():
        print(f"[gate] Missing summary: {args.summary}")
        return 1
    df = pd.read_csv(args.summary)
    graphs = args.graphs or sorted(df["graph_id"].unique())
    failures = _ratio_failures(
        df,
        baseline=args.baseline,
        teacher=args.teacher,
        graphs=graphs,
        max_ratio=args.max_ratio,
        require_baseline=args.require_baseline,
        metrics=args.metrics,
    )
    if failures:
        print("[gate] Regression check failed:")
        for msg in failures:
            print(f" - {msg}")
        return 1
    print(
        f"[gate] Swap ratio check passed for {args.baseline} vs {args.teacher} "
        f"on graphs {', '.join(graphs)} (max_ratio={args.max_ratio})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
