"""Scalar objective computation for routing evaluations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

PRESSURE_GRAPHS = ("ring_8", "grid_3x3", "heavy_hex_15")
CONSTRAINT_METRICS = {
    "swaps": "swaps_inserted_mean",
    "twoq_depth": "twoq_depth_mean",
    "total_duration_ns": "total_duration_ns_mean",
}


@dataclass
class ScalarObjective:
    """Container for the primary score and constraint checks."""

    primary: float
    constraints_ok: bool
    constraint_details: dict[str, dict[str, float]]
    mean_ratios: dict[str, float]
    primary_by_graph: dict[str, float]
    policy: str
    sabre: str
    graphs: list[str]
    missing_policy_graphs: list[str]
    missing_sabre_graphs: list[str]
    timestamp_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary,
            "constraints_ok": self.constraints_ok,
            "constraint_details": self.constraint_details,
            "mean_ratios": self.mean_ratios,
            "primary_by_graph": self.primary_by_graph,
            "policy": self.policy,
            "sabre": self.sabre,
            "graphs": self.graphs,
            "missing_policy_graphs": self.missing_policy_graphs,
            "missing_sabre_graphs": self.missing_sabre_graphs,
            "timestamp_utc": self.timestamp_utc,
        }


def compute_scalar_objective(
    summary: pd.DataFrame,
    *,
    policy_name: str,
    sabre_name: str = "sabre_layout_swap",
    graphs: Iterable[str] = PRESSURE_GRAPHS,
    primary_column: str = "overall_log_success_mean",
    constraint_threshold: float = 1.5,
) -> ScalarObjective:
    """Compute the scalar objective and constraint satisfaction.

    Args:
        summary: Aggregated summary CSV (one row per graph/baseline).
        policy_name: Baseline name to score (e.g., ``rl_ppo``).
        sabre_name: SABRE baseline row used for constraints.
        graphs: Graph ids considered the pressure set.
        primary_column: Column used for the primary score.
        constraint_threshold: Max allowed ratio vs SABRE.
    """

    graph_list = list(graphs)
    policy_rows = _rows_by_graph(summary, policy_name, graph_list)
    sabre_rows = _rows_by_graph(summary, sabre_name, graph_list)

    primary_by_graph: dict[str, float] = {}
    constraint_details: dict[str, dict[str, float]] = {name: {} for name in CONSTRAINT_METRICS}
    mean_ratios: dict[str, float] = {}
    constraints_ok = True

    for g in graph_list:
        policy_row = policy_rows.get(g)
        sabre_row = sabre_rows.get(g)
        if policy_row is None or sabre_row is None:
            continue
        primary_val = float(policy_row.get(primary_column, float("nan")))
        primary_by_graph[g] = primary_val
        for label, col in CONSTRAINT_METRICS.items():
            ratio = _safe_ratio(policy_row.get(col), sabre_row.get(col))
            constraint_details[label][g] = ratio
            if ratio > constraint_threshold:
                constraints_ok = False

    mean_ratios = {
        label: float(pd.Series(list(vals.values())).mean()) if vals else float("nan")
        for label, vals in constraint_details.items()
    }

    primary = float(pd.Series(primary_by_graph.values()).mean())
    timestamp = datetime.now(timezone.utc).isoformat()

    return ScalarObjective(
        primary=primary,
        constraints_ok=constraints_ok,
        constraint_details=constraint_details,
        mean_ratios=mean_ratios,
        primary_by_graph=primary_by_graph,
        policy=policy_name,
        sabre=sabre_name,
        graphs=graph_list,
        missing_policy_graphs=_missing_graphs(policy_rows, graph_list),
        missing_sabre_graphs=_missing_graphs(sabre_rows, graph_list),
        timestamp_utc=timestamp,
    )


def _rows_by_graph(
    summary: pd.DataFrame, baseline_name: str, graphs: Iterable[str]
) -> Mapping[str, dict[str, object]]:
    filtered = summary[
        (summary["baseline_name"] == baseline_name) & (summary["graph_id"].isin(graphs))
    ]
    return {row["graph_id"]: dict(row) for row in filtered.to_dict("records")}


def _missing_graphs(rows: Mapping[str, dict[str, object]], graphs: Iterable[str]) -> list[str]:
    return [g for g in graphs if g not in rows]


def _safe_ratio(policy_val: object, sabre_val: object) -> float:
    try:
        p = float(policy_val)
        s = float(sabre_val)
    except (TypeError, ValueError):
        return float("inf")
    if s == 0:
        return float("inf")
    return p / s


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Summary CSV to score.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Baseline name to score (e.g., rl_ppo).",
    )
    parser.add_argument(
        "--sabre",
        type=str,
        default="sabre_layout_swap",
        help="Baseline name used as SABRE reference.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=list(PRESSURE_GRAPHS),
        help="Graph ids to include in the pressure set.",
    )
    parser.add_argument(
        "--primary-column",
        type=str,
        default="overall_log_success_mean",
        help="Column used for the primary score.",
    )
    parser.add_argument(
        "--constraint-threshold",
        type=float,
        default=1.5,
        help="Max allowed ratio vs SABRE for constraints.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the JSON result.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = pd.read_csv(args.summary)
    result = compute_scalar_objective(
        summary,
        policy_name=args.policy,
        sabre_name=args.sabre,
        graphs=args.graphs,
        primary_column=args.primary_column,
        constraint_threshold=args.constraint_threshold,
    ).as_dict()
    result["summary_path"] = str(args.summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
