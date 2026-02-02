"""Paired delta analysis between weighted SABRE and a baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def compute_paired_deltas(
    df: pd.DataFrame,
    *,
    weighted_name: str = "weighted_sabre",
    baseline_name: str = "qiskit_sabre_best",
    value_col: str = "overall_log_success",
    key_cols: Iterable[str] = ("circuit_id", "graph_id", "seed", "hardware_seed"),
) -> pd.DataFrame:
    """Return paired deltas keyed by circuit/graph/seed/hardware draw."""

    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        msg = f"results missing required key columns: {missing}"
        raise ValueError(msg)
    if value_col not in df.columns:
        msg = f"results missing value column {value_col}"
        raise ValueError(msg)

    weighted = df[df["baseline_name"] == weighted_name].copy()
    baseline = df[df["baseline_name"] == baseline_name].copy()
    if weighted.empty or baseline.empty:
        msg = f"no rows for weighted={weighted_name} or baseline={baseline_name}"
        raise ValueError(msg)

    pair_keys = list(key_cols)
    merged = pd.merge(
        weighted,
        baseline,
        on=pair_keys,
        suffixes=("_weighted", "_baseline"),
        how="inner",
    )
    if merged.empty:
        msg = "no paired rows after merge; check seeds/hardware draws"
        raise ValueError(msg)

    merged["delta_success"] = merged[f"{value_col}_weighted"] - merged[f"{value_col}_baseline"]
    keep_cols = pair_keys + [
        f"{value_col}_weighted",
        f"{value_col}_baseline",
        "delta_success",
    ]
    if "trial_weighted" in merged.columns:
        keep_cols.append("trial_weighted")
    if "trials_total_weighted" in merged.columns:
        keep_cols.append("trials_total_weighted")
    if "alpha_time_weighted" in merged.columns:
        keep_cols.append("alpha_time_weighted")
    if "beta_xtalk_weighted" in merged.columns:
        keep_cols.append("beta_xtalk_weighted")
    if "snapshot_mode_weighted" in merged.columns:
        keep_cols.append("snapshot_mode_weighted")
    return merged[keep_cols]


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 1234,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for the mean."""

    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        msg = "bootstrap_ci received no data"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_resamples, data.size), replace=True)
    means = samples.mean(axis=1)
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return lower, upper


def _cliffs_delta(values: Iterable[float]) -> float:
    """Return Cliff's delta effect size for paired deltas."""

    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        msg = "cliffs_delta received no data"
        raise ValueError(msg)

    pos = np.sum(vals > 0)
    neg = np.sum(vals < 0)
    total = vals.size
    return float((pos - neg) / total)


def _cohens_d(values: Iterable[float]) -> float:
    """Return Cohen's d for paired deltas."""

    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        msg = "cohens_d received no data"
        raise ValueError(msg)
    std = np.std(vals, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(vals) / std)


def _summaries(deltas: pd.DataFrame, *, n_resamples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for graph_id, sub in [("all", deltas)] + list(deltas.groupby("graph_id")):
        vals = sub["delta_success"].to_numpy()
        mean_delta = float(np.mean(vals))
        lower, upper = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        positive_frac = float((vals > 0).mean())
        rows.append(
            {
                "graph_id": graph_id,
                "mean_delta": mean_delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "positive_frac": positive_frac,
                "count": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def _stats_tables(
    deltas: pd.DataFrame, *, n_resamples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (significance_df, effect_df)."""

    sig_rows: list[dict[str, float | int | str]] = []
    eff_rows: list[dict[str, float | int | str]] = []
    groups = [("all", deltas)] + list(deltas.groupby("graph_id"))
    for graph_id, sub in groups:
        vals = sub["delta_success"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        mean_delta = float(np.mean(vals))
        ci_lower, ci_upper = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        try:
            stat, p_value = stats.wilcoxon(
                vals,
                zero_method="pratt",
                alternative="greater",
                method="approx",
            )
        except Exception:
            # Fallback to exact when approximation is unsupported for tiny samples.
            stat, p_value = stats.wilcoxon(
                vals,
                zero_method="pratt",
                alternative="greater",
                method="exact",
            )
        sig_rows.append(
            {
                "graph_id": graph_id,
                "pairs": int(len(vals)),
                "mean_delta": mean_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "positive_frac": float((vals > 0).mean()),
            }
        )

        eff_rows.append(
            {
                "graph_id": graph_id,
                "pairs": int(len(vals)),
                "cliffs_delta": _cliffs_delta(vals),
                "cohens_d": _cohens_d(vals),
            }
        )
    return pd.DataFrame(sig_rows), pd.DataFrame(eff_rows)


def _plot_histograms(
    deltas: pd.DataFrame, out_dir: Path, stats_df: pd.DataFrame | None = None
) -> None:
    graphs = sorted(deltas["graph_id"].unique())
    cols = max(1, min(3, len(graphs)))
    rows = int(np.ceil(len(graphs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, graph in zip(axes.ravel(), graphs):
        ax.axis("on")
        vals = deltas.loc[deltas["graph_id"] == graph, "delta_success"]
        ax.hist(vals, bins=20, color="#2f4b7c", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{graph} paired deltas")
        ax.set_xlabel("Δ overall_log_success (weighted - qiskit)")
        ax.set_ylabel("Count")
        if stats_df is not None:
            row = stats_df[stats_df["graph_id"] == graph]
            if not row.empty:
                mean_delta = row["mean_delta"].iloc[0]
                ci_lower = row["ci_lower"].iloc[0]
                ci_upper = row["ci_upper"].iloc[0]
                p_val = row["p_value"].iloc[0]
                text = (
                    f"mean={mean_delta:+.3f}\n"
                    f"95% CI [{ci_lower:+.3f}, {ci_upper:+.3f}]\n"
                    f"p={p_val:.2e}"
                )
                ax.text(
                    0.97,
                    0.97,
                    text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "hist_delta_success.png", dpi=300)
    fig.savefig(out_dir / "hist_delta_success.pdf")
    plt.close(fig)


def _plot_cdf(deltas: pd.DataFrame, out_dir: Path) -> None:
    vals = np.sort(deltas["delta_success"].to_numpy())
    y = np.linspace(0, 1, len(vals), endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(vals, y, where="post", color="#277da1")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ overall_log_success (weighted - qiskit)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of paired deltas (all graphs)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cdf_delta_success.png", dpi=300)
    fig.savefig(out_dir / "cdf_delta_success.pdf")
    plt.close(fig)


def _plot_positive_fraction(deltas: pd.DataFrame, out_dir: Path) -> None:
    fractions = (
        deltas.groupby("graph_id")["delta_success"].apply(lambda s: (s > 0).mean()).reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fractions["graph_id"], fractions["delta_success"], color="#90be6d")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Fraction Δ success > 0")
    ax.set_title("Fraction of paired draws where Weighted > Qiskit")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "positive_fraction.png", dpi=300)
    fig.savefig(out_dir / "positive_fraction.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired delta analysis for weighted SABRE.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results CSV.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/deltas"), help="Output dir.")
    parser.add_argument(
        "--plots-out", type=Path, default=Path("artifacts/plots_paired_deltas"), help="Plot dir."
    )
    parser.add_argument(
        "--weighted-name",
        type=str,
        default="weighted_sabre",
        help="Name for weighted SABRE baseline in results.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="qiskit_sabre_best",
        help="Baseline name to compare against.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for CIs.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("artifacts/statistics"),
        help="Directory for significance/effect-size CSVs.",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.results)
    deltas = compute_paired_deltas(
        df,
        weighted_name=args.weighted_name,
        baseline_name=args.baseline_name,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(args.out / "paired_deltas.csv", index=False)

    summary_df = _summaries(deltas, n_resamples=args.bootstrap_samples, seed=args.seed)
    summary_df.to_csv(args.out / "paired_delta_summary.csv", index=False)

    sig_df, eff_df = _stats_tables(deltas, n_resamples=args.bootstrap_samples, seed=args.seed)
    args.stats_out.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(args.stats_out / "significance.csv", index=False)
    eff_df.to_csv(args.stats_out / "effect_size.csv", index=False)

    _plot_histograms(deltas, args.plots_out, stats_df=sig_df)
    _plot_cdf(deltas, args.plots_out)
    _plot_positive_fraction(deltas, args.plots_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
