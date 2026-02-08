"""Invariant checks to guard against regressions (determinism, coupling validity, fallback logging)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

METRIC_COLS = [
    "swaps_inserted",
    "twoq_count",
    "twoq_depth",
    "depth",
    "overall_log_success",
    "total_duration_ns",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Path to results.csv.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (defaults to results parent / invariants).",
    )
    return parser.parse_args(argv)


def _determinism(df: pd.DataFrame) -> list[dict]:
    issues = []
    key_cols = ["circuit_id", "graph_id", "baseline_name", "seed", "hardware_seed"]
    for key, group in df.groupby(key_cols):
        for col in METRIC_COLS:
            vals = group[col].dropna()
            if vals.empty:
                continue
            if vals.max() != vals.min():
                issues.append({"type": "determinism", "key": key, "metric": col})
                break
    return issues


def _coupling_validity(df: pd.DataFrame) -> list[dict]:
    issues = []
    mask = df["twoq_count"].notna() & df["twoq_depth"].notna()
    bad = df[mask & (df["twoq_count"] < df["twoq_depth"])]
    for _, row in bad.iterrows():
        issues.append(
            {
                "type": "coupling_validity",
                "circuit_id": row["circuit_id"],
                "graph_id": row["graph_id"],
                "baseline_name": row["baseline_name"],
            }
        )
    return issues


def _fallback_presence(df: pd.DataFrame) -> list[dict]:
    issues = []
    if "fallback_used" not in df.columns or "baseline_status" not in df.columns:
        issues.append({"type": "schema", "detail": "fallback columns missing"})
        return issues
    missing_reason = df[(df["fallback_used"] == True) & (df["fallback_reason"].isna())]  # noqa: E712
    for _, row in missing_reason.iterrows():
        issues.append(
            {
                "type": "fallback_reason_missing",
                "baseline_name": row["baseline_name"],
                "circuit_id": row["circuit_id"],
                "graph_id": row["graph_id"],
            }
        )
    return issues


def _metric_sanity(df: pd.DataFrame) -> list[dict]:
    issues = []
    negative = df[(df["swaps_inserted"].fillna(0) < 0) | (df["depth"].fillna(0) < 0)]
    for _, row in negative.iterrows():
        issues.append(
            {
                "type": "metric_sanity",
                "baseline_name": row["baseline_name"],
                "circuit_id": row["circuit_id"],
                "graph_id": row["graph_id"],
                "detail": "negative metric",
            }
        )
    return issues


def _summarize(issues: Iterable[dict]) -> dict:
    issues_list = list(issues)
    grouped: dict[str, int] = {}
    for item in issues_list:
        grouped[item["type"]] = grouped.get(item["type"], 0) + 1
    return {"issues": issues_list, "counts": grouped, "passed": len(issues_list) == 0}


def _write_report(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Invariants Report", ""]
    if summary["passed"]:
        report_lines.append("- All invariants passed.")
    else:
        report_lines.append(f"- Issues found: {summary['counts']}")
        for issue in summary["issues"]:
            parts = [issue["type"]]
            for key, val in issue.items():
                if key == "type":
                    continue
                parts.append(f"{key}={val}")
            report_lines.append(f"  - {'; '.join(parts)}")
    (out_dir / "report.md").write_text("\n".join(report_lines))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_path = args.results.expanduser()
    out_dir = args.out or results_path.parent / "invariants"
    df = pd.read_csv(results_path)
    issues = []
    issues.extend(_determinism(df))
    issues.extend(_coupling_validity(df))
    issues.extend(_metric_sanity(df))
    issues.extend(_fallback_presence(df))
    summary = _summarize(issues)
    _write_report(out_dir, summary)
    print(f"[invariants] wrote report to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
