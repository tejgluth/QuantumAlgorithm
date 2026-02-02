"""Paper-ready tables and figures for Weighted SABRE results."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

BASELINE_NAME = "qiskit_sabre_best"
WEIGHTED_NAME = "weighted_sabre"
PALETTE = {BASELINE_NAME: "#f58518", WEIGHTED_NAME: "#2f4b7c"}
VARIANCE_COLORS = {
    "between_hardware_frac": "#4c78a8",
    "between_seed_frac": "#f58518",
    "residual_frac": "#72b7b2",
}


# --------------------------------------------------------------------------- utils
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mean_ci(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    n = max(len(values), 1)
    mean = float(np.mean(values))
    if n == 1:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    ci_half = 1.96 * std / math.sqrt(n)
    return mean, ci_half


def _format_mean_ci(mean: float, ci_half: float, *, bold: bool, latex: bool) -> str:
    text = f"{mean:+.3f} ± {ci_half:.3f}"
    if bold:
        return f"\\textbf{{{text}}}" if latex else f"**{text}**"
    return text


def _ratios(summary: pd.DataFrame, graph: str) -> dict[str, float]:
    weighted = summary[(summary["graph_id"] == graph) & (summary["baseline_name"] == WEIGHTED_NAME)]
    baseline = summary[(summary["graph_id"] == graph) & (summary["baseline_name"] == BASELINE_NAME)]
    if weighted.empty or baseline.empty:
        msg = f"Missing weighted/baseline rows for graph={graph}"
        raise ValueError(msg)
    return {
        "swap_ratio": float(
            weighted["swaps_inserted_mean"].iloc[0] / baseline["swaps_inserted_mean"].iloc[0]
        ),
        "twoq_depth_ratio": float(
            weighted["twoq_depth_mean"].iloc[0] / baseline["twoq_depth_mean"].iloc[0]
        ),
        "duration_ratio": float(
            weighted["total_duration_ns_mean"].iloc[0] / baseline["total_duration_ns_mean"].iloc[0]
        ),
    }


def _load_ablation_results(root: Path) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tag in ["A0_hop", "A1_time", "A2_xtalk", "A3_full"]:
        path = root / f"results_ablation_{tag}.csv"
        if not path.exists():
            msg = f"Missing ablation results: {path}"
            raise FileNotFoundError(msg)
        results[tag] = pd.read_csv(path)
    return results


def _ablation_notes() -> dict[str, str]:
    return {
        "A0_hop": "Hop-distance only (α=0, β=0, trials=8)",
        "A1_time": "Adds duration weighting only (α=0.5, β=0)",
        "A2_xtalk": "Adds crosstalk weighting only (α=0, β=0.2)",
        "A3_full": "Full weighting (α=0.5, β=0.2)",
    }


# --------------------------------------------------------------------------- tables
def build_main_table(results: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    graphs = sorted(summary["graph_id"].unique())
    for graph in graphs:
        base_vals = results[
            (results["graph_id"] == graph) & (results["baseline_name"] == BASELINE_NAME)
        ]["overall_log_success"]
        weighted_vals = results[
            (results["graph_id"] == graph) & (results["baseline_name"] == WEIGHTED_NAME)
        ]["overall_log_success"]
        base_mean, base_ci = _mean_ci(base_vals)
        w_mean, w_ci = _mean_ci(weighted_vals)
        ratios = _ratios(summary, graph)
        rows.append(
            {
                "Graph": graph,
                "Qiskit overall_log_success": base_mean,
                "Qiskit ci": base_ci,
                "Weighted overall_log_success": w_mean,
                "Weighted ci": w_ci,
                "Swap ratio": ratios["swap_ratio"],
                "2Q depth ratio": ratios["twoq_depth_ratio"],
                "Duration ratio": ratios["duration_ratio"],
                "Weighted better": w_mean > base_mean,
            }
        )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, name: str, out_dir: Path, *, latex: bool = True) -> None:
    _ensure_dir(out_dir)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    if latex:
        df.to_latex(out_dir / f"{name}.tex", index=False, escape=False)


def write_main_table(raw: pd.DataFrame, out_dir: Path) -> None:
    formatted_rows = []
    for _, row in raw.iterrows():
        bold = bool(row["Weighted better"])
        formatted_rows.append(
            {
                "Graph": row["Graph"],
                "Qiskit overall_log_success (95% CI)": _format_mean_ci(
                    row["Qiskit overall_log_success"],
                    row["Qiskit ci"],
                    bold=False,
                    latex=False,
                ),
                "Weighted overall_log_success (95% CI)": _format_mean_ci(
                    row["Weighted overall_log_success"],
                    row["Weighted ci"],
                    bold=bold,
                    latex=False,
                ),
                "Swap ratio": f"{row['Swap ratio']:.3f}",
                "2Q depth ratio": f"{row['2Q depth ratio']:.3f}",
                "Duration ratio": f"{row['Duration ratio']:.3f}",
            }
        )
    formatted = pd.DataFrame(formatted_rows)
    write_table(formatted, "main_results", out_dir)

    latex_rows = []
    for _, row in raw.iterrows():
        bold = bool(row["Weighted better"])
        latex_rows.append(
            {
                "Graph": row["Graph"],
                "Qiskit overall_log_success (95% CI)": _format_mean_ci(
                    row["Qiskit overall_log_success"],
                    row["Qiskit ci"],
                    bold=False,
                    latex=True,
                ),
                "Weighted overall_log_success (95% CI)": _format_mean_ci(
                    row["Weighted overall_log_success"],
                    row["Weighted ci"],
                    bold=bold,
                    latex=True,
                ),
                "Swap ratio": f"{row['Swap ratio']:.3f}",
                "2Q depth ratio": f"{row['2Q depth ratio']:.3f}",
                "Duration ratio": f"{row['Duration ratio']:.3f}",
            }
        )
    latex_df = pd.DataFrame(latex_rows)
    write_table(latex_df, "main_results_latex", out_dir)


def build_ablation_table(
    ablation_results: dict[str, pd.DataFrame],
    ratios: pd.DataFrame,
) -> pd.DataFrame:
    notes = _ablation_notes()
    rows: list[dict[str, object]] = []
    for tag, df in ablation_results.items():
        weighted_vals = df[df["baseline_name"] == WEIGHTED_NAME]["overall_log_success"]
        base_vals = df[df["baseline_name"] == BASELINE_NAME]["overall_log_success"]
        w_mean, w_ci = _mean_ci(weighted_vals)
        base_mean, _ = _mean_ci(base_vals)

        ratio_rows = ratios[ratios["ablation"] == tag]
        swap_ratio = float(ratio_rows["swap_ratio"].mean())
        depth_ratio = float(ratio_rows["twoq_depth_ratio"].mean())
        duration_ratio = float(ratio_rows["duration_ratio"].mean())
        rows.append(
            {
                "Ablation": tag,
                "Weighted overall_log_success (95% CI)": _format_mean_ci(
                    w_mean, w_ci, bold=w_mean > base_mean, latex=False
                ),
                "Swap ratio": f"{swap_ratio:.3f}",
                "2Q depth ratio": f"{depth_ratio:.3f}",
                "Duration ratio": f"{duration_ratio:.3f}",
                "Δ vs Qiskit": f"{w_mean - base_mean:+.3f}",
                "Interpretation": notes.get(tag, ""),
            }
        )
    return pd.DataFrame(rows)


def build_variance_table(variance_df: pd.DataFrame) -> pd.DataFrame:
    filtered = variance_df[variance_df["baseline_name"].isin([WEIGHTED_NAME, BASELINE_NAME])].copy()
    filtered["Hardware %"] = (filtered["between_hardware_frac"] * 100).round(1)
    filtered["Seed %"] = (filtered["between_seed_frac"] * 100).round(1)
    filtered["Residual %"] = (filtered["residual_frac"] * 100).round(1)
    keep_cols = ["graph_id", "baseline_name", "Hardware %", "Seed %", "Residual %"]
    renamed = filtered[keep_cols].rename(columns={"graph_id": "Graph", "baseline_name": "Baseline"})
    return renamed.sort_values(["Graph", "Baseline"])


# --------------------------------------------------------------------------- plots
def plot_main_results(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    graphs = list(raw["Graph"])
    x = np.arange(len(graphs))
    width = 0.35
    base_means = raw["Qiskit overall_log_success"].to_numpy()
    base_err = raw["Qiskit ci"].to_numpy()
    w_means = raw["Weighted overall_log_success"].to_numpy()
    w_err = raw["Weighted ci"].to_numpy()

    ax.bar(
        x - width / 2,
        base_means,
        width,
        yerr=base_err,
        label="Qiskit SABRE",
        color=PALETTE[BASELINE_NAME],
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        w_means,
        width,
        yerr=w_err,
        label="Weighted SABRE",
        color=PALETTE[WEIGHTED_NAME],
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.set_ylabel("Overall log success (mean ± 95% CI)")
    ax.set_title("Figure 1: Weighted SABRE vs Qiskit SABRE")
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_dir)
    fig.savefig(out_dir / "Fig1_MainResults.png", dpi=300)
    fig.savefig(out_dir / "Fig1_MainResults.pdf")
    plt.close(fig)


def plot_yield_overhead(points_path: Path, out_dir: Path) -> None:
    points = pd.read_csv(points_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"grid_3x3": "o", "heavy_hex_15": "s", "ring_8": "D"}
    for _, row in points.iterrows():
        ax.scatter(
            row["swap_ratio"],
            row["overall_log_success"],
            color=PALETTE.get(row["method"], "#333333"),
            marker=markers.get(row["graph_id"], "o"),
            s=70,
            label=None,
        )
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.axvline(1.3, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Swap ratio vs Qiskit SABRE")
    ax.set_ylabel("Overall log success (mean)")
    ax.set_title("Figure 3: Yield vs routing overhead")
    method_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label="Qiskit SABRE", markerfacecolor="#f58518"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", label="Weighted SABRE", markerfacecolor="#2f4b7c"
        ),
    ]
    graph_handles = [
        plt.Line2D([0], [0], marker=m, color="k", label=g, linestyle="None")
        for g, m in markers.items()
    ]
    handles = method_handles + graph_handles
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="lower right", fontsize=9)
    fig.tight_layout()
    _ensure_dir(out_dir)
    fig.savefig(out_dir / "Fig3_YieldVsOverhead.png", dpi=300)
    fig.savefig(out_dir / "Fig3_YieldVsOverhead.pdf")
    plt.close(fig)


def plot_ablation(ablation_table: pd.DataFrame, ratios: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["A0_hop", "A1_time", "A2_xtalk", "A3_full"]

    # Success subplot
    ax = axes[0]
    x = np.arange(len(order))
    success = []
    err = []
    for tag in order:
        row = ablation_table[ablation_table["Ablation"] == tag]
        if row.empty:
            continue
        text = row["Weighted overall_log_success (95% CI)"].iloc[0]
        clean = text.replace("*", "").strip()
        mean_str, ci_str = [part.strip() for part in clean.split("±")]
        mean = float(mean_str)
        ci = float(ci_str)
        success.append(mean)
        err.append(ci)
    ax.errorbar(
        x,
        success,
        yerr=err,
        fmt="o",
        color=PALETTE[WEIGHTED_NAME],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Weighted overall log success (mean ± 95% CI)")
    ax.set_title("Ablation impact on success")
    ax.grid(True, linestyle=":", axis="y", alpha=0.5)

    # Overhead subplot
    ax2 = axes[1]
    metrics = [
        ("swap_ratio", "Swap"),
        ("twoq_depth_ratio", "2Q depth"),
        ("duration_ratio", "Duration"),
    ]
    colors = ["#2f4b7c", "#43aa8b", "#f58518"]
    for color, (col, label) in zip(colors, metrics):
        vals = [ratios.loc[ratios["ablation"] == tag, col].mean() for tag in order]
        ax2.plot(order, vals, marker="o", color=color, label=label)
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax2.axhline(1.3, color="red", linestyle="--", linewidth=1)
    ax2.set_ylabel("Ratio vs Qiskit SABRE")
    ax2.set_title("Overhead ratios by ablation")
    ax2.legend()
    ax2.grid(True, linestyle=":", axis="y", alpha=0.5)

    fig.suptitle("Figure 4: Weighted SABRE ablation study", y=1.02)
    fig.tight_layout()
    _ensure_dir(out_dir)
    fig.savefig(out_dir / "Fig4_Ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig4_Ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_variance(variance_df: pd.DataFrame, out_dir: Path) -> None:
    filtered = variance_df[variance_df["baseline_name"].isin([WEIGHTED_NAME, BASELINE_NAME])]
    graphs = sorted(filtered["graph_id"].unique())
    fig, axes = plt.subplots(1, len(graphs), figsize=(4 * len(graphs), 4), squeeze=False)
    for ax, graph in zip(axes.ravel(), graphs):
        sub = filtered[filtered["graph_id"] == graph]
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for key in ["between_hardware_frac", "between_seed_frac", "residual_frac"]:
            ax.bar(
                x,
                sub[key].to_numpy(),
                bottom=bottom,
                label=key.replace("_frac", "").replace("_", " ").title(),
                color=VARIANCE_COLORS[key],
            )
            bottom += sub[key].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels(sub["baseline_name"], rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Variance fraction")
        ax.set_title(graph)
    axes.ravel()[0].legend()
    fig.suptitle("Figure 5: Variance attribution (hardware vs seed vs residual)", y=1.02)
    fig.tight_layout()
    _ensure_dir(out_dir)
    fig.savefig(out_dir / "Fig5_VarianceBreakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig5_VarianceBreakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_paired_delta_plot(source_dir: Path, out_dir: Path) -> None:
    src = source_dir / "hist_delta_success.png"
    dst_png = out_dir / "Fig2_PairedDeltas.png"
    dst_pdf = out_dir / "Fig2_PairedDeltas.pdf"
    if not src.exists():
        msg = f"Missing paired delta histogram: {src}"
        raise FileNotFoundError(msg)
    _ensure_dir(out_dir)
    shutil.copyfile(src, dst_png)
    pdf_src = source_dir / "hist_delta_success.pdf"
    if pdf_src.exists():
        shutil.copyfile(pdf_src, dst_pdf)


# --------------------------------------------------------------------------- metadata
def update_version_metadata(meta_path: Path) -> None:
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    try:
        import torch
    except Exception:
        torch = None

    platform_full = platform.platform()
    os_name = platform.system()

    version_block = {
        "python": sys.version,
        "qiskit": meta.get("qiskit_version"),
        "torch": getattr(torch, "__version__", None),
        "os": os_name,
        "platform": platform_full,
        "device": "mps"
        if torch and torch.backends.mps.is_available()
        else ("cuda" if torch and torch.cuda.is_available() else "cpu"),
    }
    meta["version_pinning"] = version_block
    meta_path.write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- main
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Main results CSV.")
    parser.add_argument("--summary", type=Path, required=True, help="Main summary CSV.")
    parser.add_argument("--variance", type=Path, required=True, help="Variance breakdown CSV path.")
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing ablation result CSVs.",
    )
    parser.add_argument(
        "--ablation-ratios",
        type=Path,
        default=Path("artifacts/summary_ablation_ratios.csv"),
        help="Ablation ratios CSV path.",
    )
    parser.add_argument(
        "--tables-out", type=Path, default=Path("artifacts/tables"), help="Tables directory."
    )
    parser.add_argument(
        "--plots-out", type=Path, default=Path("artifacts/plots_final"), help="Plots directory."
    )
    parser.add_argument(
        "--paired-deltas",
        type=Path,
        default=Path("artifacts/plots_paired_deltas"),
        help="Directory containing paired delta plots.",
    )
    parser.add_argument(
        "--yield-points",
        type=Path,
        default=Path("artifacts/plots_yield_overhead/yield_vs_overhead_points.csv"),
        help="Yield vs overhead points CSV.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/metadata.json"),
        help="Metadata JSON to update with version pinning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.summary)
    variance_df = pd.read_csv(args.variance)
    ablation_results = _load_ablation_results(args.ablation_root)
    ablation_ratios = pd.read_csv(args.ablation_ratios)

    main_raw = build_main_table(results, summary)
    write_main_table(main_raw, args.tables_out)

    ablation_table = build_ablation_table(ablation_results, ablation_ratios)
    write_table(ablation_table, "ablation", args.tables_out)

    variance_table = build_variance_table(variance_df)
    write_table(variance_table, "variance", args.tables_out)

    plot_main_results(main_raw, args.plots_out)
    plot_yield_overhead(args.yield_points, args.plots_out)
    plot_ablation(ablation_table, ablation_ratios, args.plots_out)
    plot_variance(variance_df, args.plots_out)
    copy_paired_delta_plot(args.paired_deltas, args.plots_out)

    update_version_metadata(args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
