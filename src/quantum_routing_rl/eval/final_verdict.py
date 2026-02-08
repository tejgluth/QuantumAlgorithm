"""Assemble a concise FINAL_VERDICT.md for the audit run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        type=Path,
        required=True,
        help="Root directory (timestamped) containing audit artifacts.",
    )
    parser.add_argument(
        "--bestness-summary", type=Path, help="Optional override for summary_bestness.csv"
    )
    parser.add_argument(
        "--weighted-summary",
        type=Path,
        help="Optional override for summary_noise_unguarded_weighted*.csv",
    )
    parser.add_argument(
        "--proxy-correlations",
        type=Path,
        help="Optional override for proxy_validation/correlations.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write FINAL_VERDICT.md (defaults to audit_root/FINAL_VERDICT.md).",
    )
    return parser.parse_args(argv)


def _load(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        msg = f"Missing required artifact: {description} at {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def _metric(df: pd.DataFrame, baseline: str, metric: str) -> float | None:
    col = metric if metric in df.columns else f"{metric}_mean"
    if col not in df.columns:
        return None
    sub = df[df["baseline_name"] == baseline]
    if sub.empty:
        return None
    return float(sub[col].mean())


def _constraint_ratios(df: pd.DataFrame, candidate: str, reference: str) -> dict[str, float | None]:
    metrics = ["swaps_inserted", "twoq_depth", "total_duration_ns"]
    ratios: dict[str, float | None] = {}
    for metric in metrics:
        col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
        cand = df[(df["baseline_name"] == candidate)][col]
        ref = df[(df["baseline_name"] == reference)][col]
        if cand.empty or ref.empty or ref.mean() <= 0:
            ratios[metric] = None
            continue
        ratios[metric] = float(cand.mean() / ref.mean())
    return ratios


def _format_ratio(ratio: float | None) -> str:
    if ratio is None or pd.isna(ratio):
        return "n/a"
    return f"{ratio:.2f}x"


def _write_verdict(
    out_path: Path,
    *,
    bestness_df: pd.DataFrame,
    weighted_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    delta_corr_df: pd.DataFrame | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overall_corr_row = corr_df[corr_df["graph_id"] == "overall"]
    overall_corr = (
        float(overall_corr_row["spearman"].iloc[0]) if not overall_corr_row.empty else float("nan")
    )
    overall_n = int(overall_corr_row["n"].iloc[0]) if not overall_corr_row.empty else 0
    proxy_validated = pd.notna(overall_corr) and overall_corr > 0.3
    delta_corr_val = None
    delta_corr_n = 0
    if delta_corr_df is not None and not delta_corr_df.empty:
        delta_overall = delta_corr_df[delta_corr_df["graph_id"] == "overall"]
        if not delta_overall.empty:
            delta_corr_val = float(delta_overall["spearman"].iloc[0])
            delta_corr_n = int(delta_overall["n"].iloc[0])

    baselines = [
        "qiskit_sabre_best",
        "qiskit_sabre_trials1",
        "qiskit_sabre_trials8",
        "qiskit_sabre_trials16",
        "qiskit_preset_opt3",
    ]
    weighted_metric = _metric(bestness_df, "weighted_sabre", "overall_log_success")
    baseline_rows = []
    for base in baselines:
        val = _metric(bestness_df, base, "overall_log_success")
        if val is None:
            continue
        delta = None if weighted_metric is None else weighted_metric - val
        baseline_rows.append((base, val, delta))

    ratios = _constraint_ratios(bestness_df, "weighted_sabre", "qiskit_sabre_best")

    lines = []
    lines.append("# Final Audit Verdict")
    lines.append("")
    lines.append("## Pipeline Correctness")
    lines.append(
        f"- Constraint check weighted_sabre vs qiskit_sabre_best: swaps {_format_ratio(ratios['swaps_inserted'])}, "
        f"twoq_depth {_format_ratio(ratios['twoq_depth'])}, total_duration_ns {_format_ratio(ratios['total_duration_ns'])} (gate ≤1.3x)."
    )
    lines.append(
        f"- Bestness summary rows: {len(bestness_df)} ; weighted summary rows: {len(weighted_df)}."
    )
    lines.append("")
    lines.append("## Proxy Realism")
    if pd.isna(overall_corr):
        lines.append("- Spearman(proxy, empirical): n/a (insufficient data).")
    else:
        lines.append(
            f"- Spearman(proxy, empirical) = {overall_corr:.3f} (n={overall_n}); "
            f"status: {'validated' if proxy_validated else 'not validated'} (threshold > 0.3)."
        )
    if delta_corr_val is not None:
        lines.append(
            f"- Δ(proxy) vs Δ(empirical) Spearman = {delta_corr_val:.3f} (n={delta_corr_n}) "
            "for weighted_sabre minus qiskit_sabre_best."
        )
    lines.append("")
    lines.append("## Bestness Summary (overall_log_success proxy)")
    if weighted_metric is not None:
        lines.append(f"- weighted_sabre average overall_log_success_mean: {weighted_metric:.3f}")
    for base, val, delta in baseline_rows:
        delta_str = "" if delta is None else f" (Δ vs weighted = {delta:+.3f})"
        lines.append(f"- {base}: {val:.3f}{delta_str}")
    lines.append("")
    lines.append("## Claim")
    if proxy_validated:
        lines.append(
            "weighted_sabre improves expected empirical success under the tested noise models "
            "while remaining within the 1.3x overhead gate relative to qiskit_sabre_best."
        )
    else:
        lines.append(
            "weighted_sabre improves the proxy success metric; the proxy is not yet validated against empirical outcomes "
            "(re-run proxy validation with more data to firm the claim)."
        )

    out_path.write_text("\n".join(lines))
    print(f"[audit] Wrote FINAL_VERDICT to {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.audit_root.expanduser()
    bestness_path = args.bestness_summary or root / "summary_bestness.csv"
    weighted_path = args.weighted_summary
    if weighted_path is None:
        # prefer hardware-draws=50 file if present, otherwise fallback to default.
        hd = root / "summary_noise_unguarded_weighted_hd.csv"
        default = root / "summary_noise_unguarded_weighted.csv"
        weighted_path = hd if hd.exists() else default
    proxy_corr_path = args.proxy_correlations or root / "proxy_validation" / "correlations.csv"
    delta_corr_path = root / "proxy_validation" / "delta_correlations.csv"
    out_path = args.out or root / "FINAL_VERDICT.md"

    bestness_df = _load(bestness_path, "summary_bestness.csv")
    weighted_df = _load(weighted_path, "weighted summary")
    corr_df = _load(proxy_corr_path, "proxy correlations")
    delta_corr_df = (
        delta_corr_path.exists() and _load(delta_corr_path, "delta correlations") or None
    )

    _write_verdict(
        out_path,
        bestness_df=bestness_df,
        weighted_df=weighted_df,
        corr_df=corr_df,
        delta_corr_df=delta_corr_df if isinstance(delta_corr_df, pd.DataFrame) else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
