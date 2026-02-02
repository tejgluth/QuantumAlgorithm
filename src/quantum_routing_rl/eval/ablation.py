"""Ablation runs for Weighted SABRE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from quantum_routing_rl.eval import run_eval


Ablation = tuple[str, float, float]


def _run_single(ablation: Ablation, args: argparse.Namespace) -> dict[str, Path]:
    tag, alpha, beta = ablation
    results_name = f"results_ablation_{tag}.csv"
    summary_name = f"summary_ablation_{tag}.csv"
    summary_std_name = f"summary_ablation_{tag}_std.csv"
    run_args = [
        "--suite",
        "pressure",
        "--out",
        str(args.out),
        "--results-name",
        results_name,
        "--summary-name",
        summary_name,
        "--summary-std-name",
        summary_std_name,
        "--seeds",
    ] + [str(s) for s in args.seeds]
    run_args += [
        "--hardware-samples",
        str(args.hardware_draws),
        "--hardware-seed-base",
        str(args.hardware_seed_base),
        "--hardware-profile",
        "realistic",
        "--hardware-snapshots",
        "2",
        "--hardware-drift",
        "0.05",
        "--hardware-directional",
        "--hardware-snapshot-spacing",
        "50000",
        "--hardware-crosstalk",
        "0.01",
        "--run-weighted-sabre",
        "--weighted-alpha-time",
        str(alpha),
        "--weighted-beta-xtalk",
        str(beta),
        "--weighted-snapshot-mode",
        args.snapshot_mode,
        "--weighted-trials",
        str(args.trials),
    ]
    if args.include_teacher:
        run_args.append("--include-teacher")
    run_eval.main(run_args)
    return {
        "results": args.out / results_name,
        "summary": args.out / summary_name,
        "summary_std": args.out / summary_std_name,
    }


def _merge_summaries(
    ablations: Iterable[Ablation], out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    std_frames = []
    for tag, alpha, beta in ablations:
        summary_path = out_dir / f"summary_ablation_{tag}.csv"
        std_path = out_dir / f"summary_ablation_{tag}_std.csv"
        df = pd.read_csv(summary_path)
        df["ablation"] = tag
        df["alpha_time"] = alpha
        df["beta_xtalk"] = beta
        summary_frames.append(df)

        std_df = pd.read_csv(std_path)
        std_df["ablation"] = tag
        std_frames.append(std_df)

    return pd.concat(summary_frames, ignore_index=True), pd.concat(std_frames, ignore_index=True)


def _plot_success(weighted: pd.DataFrame, weighted_std: pd.DataFrame, out_dir: Path) -> None:
    merged = weighted.merge(
        weighted_std,
        on=["graph_id", "baseline_name", "ablation"],
        suffixes=("", "_std"),
    )
    graphs = sorted(merged["graph_id"].unique())
    fig, axes = plt.subplots(1, len(graphs), figsize=(4 * len(graphs), 4), sharey=True)
    if len(graphs) == 1:
        axes = [axes]
    for ax, graph in zip(axes, graphs):
        sub = merged[merged["graph_id"] == graph].sort_values("ablation")
        ax.errorbar(
            sub["ablation"],
            sub["overall_log_success_mean"],
            yerr=sub["overall_log_success_std"],
            fmt="o-",
            color="#2f4b7c",
            capsize=4,
        )
        ax.axhline(
            merged.loc[
                (merged["graph_id"] == graph) & (merged["ablation"] == merged["ablation"].min()),
                "overall_log_success_mean",
            ].mean(),
            color="black",
            linestyle=":",
            linewidth=1,
        )
        ax.set_title(graph)
        ax.set_ylabel("overall_log_success (mean ± std)")
        ax.set_xlabel("Ablation")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "overall_log_success_by_ablation.png", dpi=200)
    fig.savefig(out_dir / "overall_log_success_by_ablation.pdf")
    plt.close(fig)


def _plot_overhead_ratios(ratios: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("swap_ratio", "Swap ratio"),
        ("twoq_depth_ratio", "2Q depth ratio"),
        ("duration_ratio", "Duration ratio"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 10))
    graphs = sorted(ratios["graph_id"].unique())
    colors = plt.get_cmap("tab10").colors
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        for g_idx, graph in enumerate(graphs):
            sub = ratios[ratios["graph_id"] == graph].sort_values("ablation")
            ax.plot(
                sub["ablation"],
                sub[col],
                marker="o",
                label=graph if idx == 0 else None,
                color=colors[g_idx % len(colors)],
            )
        ax.axhline(1.3, color="red", linestyle="--", linewidth=1)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel(label)
        ax.set_ylim(0.6, 1.4)
    axes[-1].set_xlabel("Ablation")
    if graphs:
        axes[0].legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "overhead_ratios.png", dpi=200)
    fig.savefig(out_dir / "overhead_ratios.pdf")
    plt.close(fig)


def _compute_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    weighted = summary[summary["baseline_name"] == "weighted_sabre"]
    baseline = summary[summary["baseline_name"] == "qiskit_sabre_best"]
    merged = weighted.merge(
        baseline,
        on=["graph_id", "ablation"],
        suffixes=("_weighted", "_baseline"),
    )
    merged["swap_ratio"] = (
        merged["swaps_inserted_mean_weighted"] / merged["swaps_inserted_mean_baseline"]
    )
    merged["twoq_depth_ratio"] = (
        merged["twoq_depth_mean_weighted"] / merged["twoq_depth_mean_baseline"]
    )
    merged["duration_ratio"] = (
        merged["total_duration_ns_mean_weighted"] / merged["total_duration_ns_mean_baseline"]
    )
    cols = [
        "graph_id",
        "ablation",
        "swap_ratio",
        "twoq_depth_ratio",
        "duration_ratio",
        "overall_log_success_mean_weighted",
        "overall_log_success_mean_baseline",
    ]
    return merged[cols]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Weighted SABRE ablations.")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="Output root.")
    parser.add_argument("--hardware-draws", type=int, default=30, help="Hardware samples.")
    parser.add_argument("--hardware-seed-base", type=int, default=211, help="Hardware seed base.")
    parser.add_argument(
        "--snapshot-mode",
        type=str,
        default="avg",
        choices=["avg", "bucket"],
        help="Weighted SABRE snapshot mode.",
    )
    parser.add_argument("--trials", type=int, default=8, help="Weighted SABRE trials.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for time/error weight.")
    parser.add_argument("--beta", type=float, default=0.2, help="Beta for crosstalk weight.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[13, 17, 23], help="Evaluation seeds."
    )
    parser.add_argument("--include-teacher", action="store_true", help="Include teacher baseline.")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist.")
    args = parser.parse_args(argv)

    ablations: list[Ablation] = [
        ("A0_hop", 0.0, 0.0),
        ("A1_time", args.alpha, 0.0),
        ("A2_xtalk", 0.0, args.beta),
        ("A3_full", args.alpha, args.beta),
    ]

    results_dir = args.out
    results_dir.mkdir(parents=True, exist_ok=True)

    for ablation in ablations:
        tag, alpha, beta = ablation
        expected = results_dir / f"results_ablation_{tag}.csv"
        if expected.exists() and not args.force:
            continue
        _run_single(ablation, args)

    summary_all, std_all = _merge_summaries(ablations, results_dir)
    summary_all.to_csv(results_dir / "summary_ablation_raw.csv", index=False)

    filtered = summary_all[
        summary_all["baseline_name"].isin(["weighted_sabre", "qiskit_sabre_best"])
    ]
    filtered.to_csv(results_dir / "summary_ablation.csv", index=False)

    std_filtered = std_all[std_all["baseline_name"].isin(["weighted_sabre"])]
    weighted = filtered[filtered["baseline_name"] == "weighted_sabre"]
    _plot_success(weighted, std_filtered, results_dir / "plots_ablation")

    ratios = _compute_ratios(filtered)
    ratios.to_csv(results_dir / "summary_ablation_ratios.csv", index=False)
    _plot_overhead_ratios(ratios, results_dir / "plots_ablation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
