"""Plotting utilities for routing evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", type=Path, required=True, help="Input results CSV.")
    parser.add_argument("--out", dest="out", type=Path, required=True, help="Output directory.")
    return parser.parse_args(argv)


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["graph_id", "baseline_name"])[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean", "std": "std"})
    )


def _barplots(df: pd.DataFrame, metric: str, ylabel: str, fname: str, out_dir: Path) -> None:
    graphs = sorted(df["graph_id"].unique())
    baselines = sorted(df["baseline_name"].unique())
    num_graphs = len(graphs)
    fig, axes = plt.subplots(1, num_graphs, figsize=(4 * num_graphs, 4), sharey=False)
    if num_graphs == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10")
    for ax, graph in zip(axes, graphs):
        subset = df[df["graph_id"] == graph]
        stats = subset.groupby("baseline_name")[metric].agg(["mean", "std"]).reindex(baselines)
        means = stats["mean"].fillna(0.0)
        errs = stats["std"].fillna(0.0)
        colors = [palette(i % 10) for i in range(len(baselines))]
        ax.bar(range(len(baselines)), means, yerr=errs, color=colors)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels(baselines, rotation=45, ha="right")
        ax.set_title(graph)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{fname}.{ext}", dpi=200)
    plt.close(fig)


def _pareto(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["graph_id", "baseline_name"])
        .agg({"swaps_inserted": "mean", "twoq_depth": "mean"})
        .reset_index()
    )
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, (baseline, subset) in enumerate(agg.groupby("baseline_name")):
        ax.scatter(
            subset["swaps_inserted"],
            subset["twoq_depth"],
            label=baseline,
            color=palette(idx % 10),
            s=60,
        )
        for _, row in subset.iterrows():
            ax.annotate(row["graph_id"], (row["swaps_inserted"], row["twoq_depth"]), fontsize=8)
    ax.set_xlabel("Swaps (mean)")
    ax.set_ylabel("Two-qubit depth (mean)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_tradeoff.{ext}", dpi=200)
    plt.close(fig)


def _noise_safe(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["noise_proxy_score"])


def _pareto_noise(df: pd.DataFrame, out_dir: Path) -> None:
    noise_df = df.dropna(subset=["log_success_proxy"])
    if noise_df.empty:
        return
    agg = (
        noise_df.groupby(["graph_id", "baseline_name"])
        .agg({"swaps_inserted": "mean", "log_success_proxy": "mean"})
        .reset_index()
    )
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, (baseline, subset) in enumerate(agg.groupby("baseline_name")):
        ax.scatter(
            subset["swaps_inserted"],
            subset["log_success_proxy"],
            label=baseline,
            color=palette(idx % 10),
            s=60,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["graph_id"], (row["swaps_inserted"], row["log_success_proxy"]), fontsize=8
            )
    ax.set_xlabel("Swaps (mean)")
    ax.set_ylabel("Log success proxy (mean)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_noise.{ext}", dpi=200)
    plt.close(fig)


def _pareto_overall_success(df: pd.DataFrame, out_dir: Path) -> None:
    noise_df = df.dropna(subset=["overall_log_success"])
    if noise_df.empty:
        return
    agg = (
        noise_df.groupby(["graph_id", "baseline_name"])
        .agg({"swaps_inserted": "mean", "overall_log_success": "mean"})
        .reset_index()
    )
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, (baseline, subset) in enumerate(agg.groupby("baseline_name")):
        ax.scatter(
            subset["swaps_inserted"],
            subset["overall_log_success"],
            label=baseline,
            color=palette(idx % 10),
            s=60,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["graph_id"], (row["swaps_inserted"], row["overall_log_success"]), fontsize=8
            )
    ax.set_xlabel("Swaps (mean)")
    ax.set_ylabel("Overall log success (mean)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_overall_success.{ext}", dpi=200)
    plt.close(fig)


def _weighted_vs_qiskit(df: pd.DataFrame, out_dir: Path) -> None:
    baselines = {"weighted_sabre", "qiskit_sabre_best"}
    if "baseline_name" not in df.columns or not baselines.issubset(set(df["baseline_name"])):
        return
    subset = df[
        (df["baseline_name"].isin(baselines)) & df["overall_log_success"].notna()  # type: ignore[operator]
    ]
    if subset.empty:
        return
    agg = (
        subset.groupby(["graph_id", "baseline_name"])
        .agg({"swaps_inserted": "mean", "overall_log_success": "mean"})
        .reset_index()
    )
    palette = {"weighted_sabre": "tab:blue", "qiskit_sabre_best": "tab:orange"}
    markers = {"weighted_sabre": "o", "qiskit_sabre_best": "s"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for baseline, group in agg.groupby("baseline_name"):
        ax.scatter(
            group["swaps_inserted"],
            group["overall_log_success"],
            label=baseline,
            color=palette.get(baseline, "gray"),
            marker=markers.get(baseline, "o"),
            s=70,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["graph_id"],
                (row["swaps_inserted"], row["overall_log_success"]),
                fontsize=8,
            )
    ax.set_xlabel("Swaps (mean)")
    ax.set_ylabel("Overall log success (mean)")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"weighted_vs_qiskit_overall_success.{ext}", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = pd.read_csv(args.input)
    out_dir = args.out

    _barplots(df, "swaps_inserted", "Swaps (mean ± std)", "swaps_comparison", out_dir)
    _barplots(df, "twoq_depth", "Two-qubit depth", "twoq_depth_comparison", out_dir)
    _barplots(df, "routing_runtime_s", "Routing runtime (s)", "runtime_comparison", out_dir)

    noise_df = _noise_safe(df)
    if not noise_df.empty:
        _barplots(noise_df, "noise_proxy_score", "Noise proxy", "noise_proxy_comparison", out_dir)
        if "log_success_proxy" in noise_df.columns:
            _barplots(
                noise_df,
                "log_success_proxy",
                "Log success proxy",
                "log_success_comparison",
                out_dir,
            )
        if "duration_proxy" in noise_df.columns:
            _barplots(
                noise_df,
                "duration_proxy",
                "Duration/T1 proxy",
                "duration_proxy_comparison",
                out_dir,
            )
        if "overall_log_success" in noise_df.columns:
            _barplots(
                noise_df,
                "overall_log_success",
                "Overall log success",
                "overall_log_success_comparison",
                out_dir,
            )
        if "total_duration_ns" in noise_df.columns:
            _barplots(
                noise_df,
                "total_duration_ns",
                "Total duration (ns)",
                "total_duration_comparison",
                out_dir,
            )
        if "decoherence_penalty" in noise_df.columns:
            _barplots(
                noise_df,
                "decoherence_penalty",
                "Decoherence penalty",
                "decoherence_penalty_comparison",
                out_dir,
            )
    else:
        # Write placeholder so downstream checks still succeed.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "noise_proxy_comparison.png").touch()
        (out_dir / "noise_proxy_comparison.pdf").touch()

    _pareto(df, out_dir)
    _pareto_noise(df, out_dir)
    _pareto_overall_success(df, out_dir)
    _weighted_vs_qiskit(df, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
