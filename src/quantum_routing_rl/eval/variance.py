"""Variance decomposition for routing evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sum_squares(means: np.ndarray, counts: np.ndarray, grand_mean: float) -> float:
    return float(np.sum(counts * (means - grand_mean) ** 2))


def variance_breakdown(
    df: pd.DataFrame,
    *,
    value_col: str = "overall_log_success",
    group_cols: Iterable[str] = ("graph_id", "baseline_name"),
    center_by_circuit: bool = True,
) -> pd.DataFrame:
    """Return variance components by graph/baseline."""

    required = {"hardware_seed", "seed", value_col}
    missing = required - set(df.columns)
    if missing:
        msg = f"results missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    results: list[dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols)):
        working = sub.copy()
        value_field = value_col
        if center_by_circuit and "circuit_id" in working.columns:
            working["_centered_value"] = working[value_col] - working.groupby("circuit_id")[
                value_col
            ].transform("mean")
            value_field = "_centered_value"

        values = working[value_field].to_numpy(dtype=float)
        if values.size < 2:
            continue
        grand_mean = float(np.mean(values))
        total_ss = float(np.sum((values - grand_mean) ** 2))
        denom = max(len(values) - 1, 1)
        total_var = total_ss / denom

        hw_groups = working.groupby("hardware_seed")[value_field]
        hw_means = hw_groups.mean().to_numpy()
        hw_counts = hw_groups.count().to_numpy()
        between_hw_ss = _sum_squares(hw_means, hw_counts, grand_mean)

        between_seed_ss = 0.0
        residual_ss = 0.0
        for _, hw_sub in working.groupby("hardware_seed"):
            hw_mean = hw_sub[value_field].mean()
            for _, seed_sub in hw_sub.groupby("seed"):
                seed_mean = seed_sub[value_field].mean()
                count = len(seed_sub)
                between_seed_ss += count * (seed_mean - hw_mean) ** 2
                residual_ss += float(np.sum((seed_sub[value_field] - seed_mean) ** 2))

        residual_ss = max(residual_ss, 0.0)

        between_hw_var = between_hw_ss / denom
        between_seed_var = between_seed_ss / denom
        residual_var = residual_ss / denom

        hw_draws = hw_groups.ngroups
        seeds = working["seed"].nunique()
        row = {
            "graph_id": key[0] if isinstance(key, tuple) else key,
            "baseline_name": key[1] if isinstance(key, tuple) else "unknown",
            "total_variance": total_var,
            "between_hardware_variance": between_hw_var,
            "between_seed_variance": between_seed_var,
            "residual_variance": residual_var,
            "hardware_draws": hw_draws,
            "seeds": seeds,
        }
        denom_total = total_var if total_var else 1.0
        row["between_hardware_frac"] = between_hw_var / denom_total
        row["between_seed_frac"] = between_seed_var / denom_total
        row["residual_frac"] = residual_var / denom_total
        results.append(row)

    return pd.DataFrame(results)


def _plot_variance(df: pd.DataFrame, out_dir: Path) -> None:
    components = [
        ("between_hardware_frac", "#4c78a8", "Hardware"),
        ("between_seed_frac", "#f58518", "Seed"),
        ("residual_frac", "#72b7b2", "Residual"),
    ]
    graphs = sorted(df["graph_id"].unique())
    for graph in graphs:
        sub = df[df["graph_id"] == graph]
        fig, ax = plt.subplots(figsize=(6, 4))
        bottom = np.zeros(len(sub))
        x = np.arange(len(sub))
        for key, color, label in components:
            ax.bar(
                x,
                sub[key].to_numpy(),
                bottom=bottom,
                label=label,
                color=color,
            )
            bottom += sub[key].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels(sub["baseline_name"], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Variance fraction")
        ax.set_title(f"Variance breakdown: {graph}")
        ax.legend()
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"variance_{graph}.png", dpi=200)
        fig.savefig(out_dir / f"variance_{graph}.pdf")
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Variance decomposition for routing results.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results CSV.")
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="*",
        default=None,
        help="Optional baseline names to include (default all).",
    )
    parser.add_argument(
        "--no-center-by-circuit",
        action="store_true",
        help="Disable centering by circuit_id before variance decomposition.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/variance"), help="Output directory."
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.results)
    if args.baselines:
        df = df[df["baseline_name"].isin(args.baselines)]
    breakdown = variance_breakdown(
        df,
        center_by_circuit=not args.no_center_by_circuit,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(args.out / "variance_breakdown.csv", index=False)
    _plot_variance(breakdown, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
