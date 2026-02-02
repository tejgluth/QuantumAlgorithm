"""Yield vs routing overhead plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _build_points(
    summary: pd.DataFrame,
    weighted_name: str,
    baseline_name: str,
) -> pd.DataFrame:
    weighted = summary[summary["baseline_name"] == weighted_name]
    baseline = summary[summary["baseline_name"] == baseline_name]
    merged = weighted.merge(
        baseline,
        on="graph_id",
        suffixes=("_weighted", "_baseline"),
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "graph_id": row["graph_id"],
                "method": baseline_name,
                "swap_ratio": 1.0,
                "overall_log_success": row["overall_log_success_mean_baseline"],
            }
        )
        rows.append(
            {
                "graph_id": row["graph_id"],
                "method": weighted_name,
                "swap_ratio": row["swaps_inserted_mean_weighted"]
                / row["swaps_inserted_mean_baseline"],
                "overall_log_success": row["overall_log_success_mean_weighted"],
            }
        )
    return pd.DataFrame(rows)


def _plot(points: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"qiskit_sabre_best": "#4c78a8", "weighted_sabre": "#f58518"}
    markers = {"grid_3x3": "o", "ring_8": "s", "heavy_hex_15": "D"}
    for _, row in points.iterrows():
        marker = markers.get(row["graph_id"], "o")
        color = colors.get(row["method"], "#333333")
        ax.scatter(row["swap_ratio"], row["overall_log_success"], color=color, marker=marker, s=60)
        ax.annotate(
            row["graph_id"],
            (row["swap_ratio"], row["overall_log_success"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.axvline(1.3, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Swap ratio vs qiskit_sabre_best")
    ax.set_ylabel("overall_log_success (mean)")
    ax.set_title("Yield vs routing overhead")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="qiskit_sabre_best",
            markerfacecolor=colors["qiskit_sabre_best"],
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="weighted_sabre",
            markerfacecolor=colors["weighted_sabre"],
        ),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "yield_vs_overhead.png", dpi=200)
    fig.savefig(out_dir / "yield_vs_overhead.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot yield improvement vs routing overhead.")
    parser.add_argument("--summary", type=Path, required=True, help="Summary CSV path.")
    parser.add_argument(
        "--weighted-name", type=str, default="weighted_sabre", help="Weighted SABRE name."
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="qiskit_sabre_best",
        help="Baseline name to compare against.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/plots_yield_overhead"), help="Output dir."
    )
    args = parser.parse_args(argv)

    summary = pd.read_csv(args.summary)
    points = _build_points(summary, args.weighted_name, args.baseline_name)
    args.out.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out / "yield_vs_overhead_points.csv", index=False)
    _plot(points, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
