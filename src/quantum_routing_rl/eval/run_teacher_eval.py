"""Teacher-focused evaluation with SABRE parity guard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from quantum_routing_rl.eval import run_eval

TEACHER_NAME = "teacher_sabre_like"
SABRE_BASELINES = ("sabre_layout_swap", "qiskit_sabre_best")
PARITY_GRAPHS = ("ring_8", "grid_3x3", "heavy_hex_15")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--suite",
        choices=["pressure", "full", "dev"],
        default="pressure",
        help="Benchmark suite to run (default: pressure).",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="results_teacher.csv",
        help="Filename for raw results written inside --out.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary_teacher.csv",
        help="Filename for summary written inside --out.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13])
    parser.add_argument("--pressure-seed", type=int, default=99)
    parser.add_argument("--pressure-qasm", type=int, default=6)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--hardware-samples", type=int, default=1)
    parser.add_argument("--hardware-seed-base", type=int, default=101)
    parser.add_argument("--hardware-profile", type=str, default="realistic")
    parser.add_argument("--qasm-root", type=Path)
    return parser.parse_args(argv)


def _parity_failures(summary_path: Path) -> list[str]:
    if not summary_path.exists():
        return [f"Missing summary file: {summary_path}"]
    df = pd.read_csv(summary_path)
    if "swaps_inserted_mean" not in df.columns:
        return ["summary missing swaps_inserted_mean column"]

    failures: list[str] = []
    for graph in PARITY_GRAPHS:
        teacher_row = df[(df["graph_id"] == graph) & (df["baseline_name"] == TEACHER_NAME)]
        if teacher_row.empty:
            failures.append(f"{graph}: missing teacher results")
            continue
        sabre_rows = df[(df["graph_id"] == graph) & (df["baseline_name"].isin(SABRE_BASELINES))]
        if sabre_rows.empty:
            failures.append(f"{graph}: missing SABRE baselines")
            continue

        teacher_swaps = float(teacher_row["swaps_inserted_mean"].iloc[0])
        sabre_best = float(sabre_rows["swaps_inserted_mean"].min())
        if sabre_best <= 0:
            continue
        if teacher_swaps > 2 * sabre_best:
            failures.append(
                f"{graph}: teacher swaps {teacher_swaps:.2f} > 2x SABRE {sabre_best:.2f}"
            )
    return failures


def _build_eval_args(args: argparse.Namespace) -> list[str]:
    eval_args = [
        "--suite",
        args.suite,
        "--out",
        str(args.out),
        "--results-name",
        args.results_name,
        "--summary-name",
        args.summary_name,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--include-teacher",
        "--hardware-samples",
        str(args.hardware_samples),
        "--hardware-seed-base",
        str(args.hardware_seed_base),
        "--hardware-profile",
        args.hardware_profile,
        "--pressure-seed",
        str(args.pressure_seed),
        "--pressure-qasm",
        str(args.pressure_qasm),
        "--dev-limit",
        str(args.dev_limit),
    ]
    if args.qasm_root:
        eval_args.extend(["--qasm-root", str(args.qasm_root)])
    return eval_args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)

    eval_args = _build_eval_args(args)
    status = run_eval.main(eval_args)
    if status != 0:
        return status

    summary_path = args.out / args.summary_name
    failures = _parity_failures(summary_path)
    if failures:
        print("Teacher parity check failed:")
        for msg in failures:
            print(f" - {msg}")
        return 1

    print("Teacher parity check passed on pressure graphs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
