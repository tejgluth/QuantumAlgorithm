"""Cross-platform task runner for quantum-routing-rl.

All commands use the currently active Python interpreter (sys.executable) so they work
on POSIX and Windows without Make.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ARTIFACTS_ENV = "ARTIFACTS"
HARDWARE_DRAWS_ENV = "HARDWARE_DRAWS"
DEFAULT_ARTIFACTS = "artifacts"


class RunError(Exception):
    """Raised when an invoked command fails."""


def _log(msg: str) -> None:
    print(f"[run] {msg}")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    _log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RunError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _artifacts_root() -> Path:
    return Path(os.environ.get(ARTIFACTS_ENV, DEFAULT_ARTIFACTS))


def _hardware_draws(default: int = 10) -> int:
    val = os.environ.get(HARDWARE_DRAWS_ENV)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        _log(f"invalid {HARDWARE_DRAWS_ENV}={val!r}; falling back to {default}")
        return default


def _latest_gauntlet_run(root: Path | None = None) -> Path | None:
    base = root or (_artifacts_root() / "gauntlet")
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _latest_gauntlet_file(run_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(run_dir.glob(f"{prefix}_gauntlet_*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def cmd_lint(_: argparse.Namespace) -> None:
    _run([sys.executable, "-m", "ruff", "check", "."])
    _run([sys.executable, "-m", "ruff", "format", "--check", "."])


def cmd_format(_: argparse.Namespace) -> None:
    _run([sys.executable, "-m", "ruff", "format", "."])
    _run([sys.executable, "-m", "ruff", "check", "--fix", "."])


def cmd_test(args: argparse.Namespace) -> None:
    pytest_cmd = [sys.executable, "-m", "pytest"]
    if args.quiet:
        pytest_cmd.append("-q")
    _run(pytest_cmd)


def _gauntlet(mode: str, args: argparse.Namespace) -> None:
    draws = args.hardware_draws or _hardware_draws()
    out_root = args.out or (_artifacts_root() / "gauntlet")
    cmd = [
        sys.executable,
        "-m",
        "quantum_routing_rl.eval.gauntlet",
        "--mode",
        mode,
        "--hardware-draws",
        str(draws),
        "--hardware-seed-base",
        str(args.hardware_seed_base),
        "--hardware-profile",
        args.hardware_profile,
        "--hardware-snapshots",
        str(args.hardware_snapshots),
        "--hardware-drift",
        str(args.hardware_drift),
        "--hardware-crosstalk",
        str(args.hardware_crosstalk),
        "--hardware-snapshot-spacing",
        str(args.hardware_snapshot_spacing),
        "--out",
        str(out_root),
    ]
    if args.hardware_directional:
        cmd.append("--hardware-directional")
    if args.seeds:
        cmd += ["--seeds", *[str(s) for s in args.seeds]]
    if args.qasm_root:
        cmd += ["--qasm-root", str(args.qasm_root)]
    if args.selection_seed is not None:
        cmd += ["--selection-seed", str(args.selection_seed)]
    _run(cmd)
    latest = _latest_gauntlet_run(out_root)
    if latest:
        _log(f"latest gauntlet run: {latest}")


def cmd_gauntlet_small(args: argparse.Namespace) -> None:
    _gauntlet("small", args)


def cmd_gauntlet_full(args: argparse.Namespace) -> None:
    _gauntlet("full", args)


def cmd_gauntlet_industrial(args: argparse.Namespace) -> None:
    _gauntlet("industrial", args)


def cmd_invariants(args: argparse.Namespace) -> None:
    results = args.results
    if results is None:
        latest_run = _latest_gauntlet_run()
        if not latest_run:
            raise RunError("No gauntlet runs found; provide --results explicitly")
        results = _latest_gauntlet_file(latest_run, "results")
    if results is None or not results.exists():
        raise RunError("Could not locate gauntlet results CSV; provide --results")
    out_dir = args.out or (results.parent / "invariants")
    _run(
        [
            sys.executable,
            "-m",
            "quantum_routing_rl.eval.invariants",
            "--results",
            str(results),
            "--out",
            str(out_dir),
        ]
    )


def cmd_validate_proxy(args: argparse.Namespace) -> None:
    out_dir = args.out
    if out_dir is None:
        latest_run = _latest_gauntlet_run()
        if not latest_run:
            raise RunError("No gauntlet runs found; provide --out for proxy validation")
        out_dir = latest_run / "proxy_validation_extended"
    cmd = [
        sys.executable,
        "-m",
        "quantum_routing_rl.eval.proxy_validation_extended",
        "--out",
        str(out_dir),
        "--max-circuits",
        str(args.max_circuits),
        "--max-qubits",
        str(args.max_qubits),
        "--shots",
        str(args.shots),
        "--selection-seed",
        str(args.selection_seed),
    ]
    if args.qasm_root:
        cmd += ["--qasm-root", str(args.qasm_root)]
    if args.include_weighted:
        cmd.append("--include-weighted")
    _run(cmd)


def _find_verdict_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    audit_root = args.audit_root
    if audit_root is None:
        audit_root = _latest_gauntlet_run()
        if audit_root is None:
            raise RunError("No gauntlet runs found; specify --audit-root")
    summary_candidates = []
    gauntlet_summaries = sorted(
        audit_root.glob("summary_gauntlet_*.csv"),
        key=lambda p: ("_std" in p.stem, -p.stat().st_mtime if p.exists() else 0),
    )
    summary_candidates.extend(gauntlet_summaries)
    summary_candidates.append(audit_root / "summary.csv")
    summary_candidates.append(audit_root / "summary_bestness.csv")
    bestness_summary = args.bestness_summary
    if bestness_summary is None:
        for path in summary_candidates:
            if path.exists():
                bestness_summary = path
                break
    weighted_summary = args.weighted_summary or bestness_summary

    proxy_corr = args.proxy_correlations
    delta_corr = args.delta_correlations
    if proxy_corr is None or delta_corr is None:
        for candidate_dir in (
            audit_root / "proxy_validation",
            audit_root / "proxy_validation_extended",
        ):
            corr_path = candidate_dir / "correlations.csv"
            delta_path = candidate_dir / "delta_correlations.csv"
            if proxy_corr is None and corr_path.exists():
                proxy_corr = corr_path
            if delta_corr is None and delta_path.exists():
                delta_corr = delta_path
    return audit_root, bestness_summary, weighted_summary, proxy_corr, delta_corr


def cmd_verdict(args: argparse.Namespace) -> None:
    audit_root, bestness_summary, weighted_summary, proxy_corr, delta_corr = _find_verdict_inputs(
        args
    )
    if not bestness_summary or not bestness_summary.exists():
        raise RunError(
            "bestness/summary CSV not found; run gauntlet first or pass --bestness-summary"
        )
    if not proxy_corr or not proxy_corr.exists():
        raise RunError(
            "proxy correlations not found; run validate-proxy-extended or pass --proxy-correlations"
        )
    out_path = args.out or (audit_root / "FINAL_VERDICT.md")
    cmd = [
        sys.executable,
        "-m",
        "quantum_routing_rl.eval.final_verdict",
        "--audit-root",
        str(audit_root),
        "--bestness-summary",
        str(bestness_summary),
        "--weighted-summary",
        str(weighted_summary),
        "--proxy-correlations",
        str(proxy_corr),
        "--out",
        str(out_path),
    ]
    if delta_corr and delta_corr.exists():
        cmd += ["--delta-correlations", str(delta_corr)]
    _run(cmd)
    final_copy = _artifacts_root() / "FINAL_VERDICT.md"
    if out_path != final_copy:
        final_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, final_copy)
        _log(f"copied verdict to {final_copy}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    lint_p = sub.add_parser("lint", help="Run ruff checks")
    lint_p.set_defaults(func=cmd_lint)

    fmt_p = sub.add_parser("format", help="Apply ruff format and fixes")
    fmt_p.set_defaults(func=cmd_format)

    test_p = sub.add_parser("test", help="Run pytest")
    test_p.add_argument("--quiet", action="store_true", help="Quiet pytest output")
    test_p.set_defaults(func=cmd_test)

    def add_gauntlet(name: str, mode: str):
        gp = sub.add_parser(name, help=f"Run gauntlet ({mode})")
        gp.add_argument("--hardware-draws", type=int, help="Override HARDWARE_DRAWS env")
        gp.add_argument("--hardware-seed-base", type=int, default=211)
        gp.add_argument("--hardware-profile", type=str, default="realistic")
        gp.add_argument("--hardware-snapshots", type=int, default=1)
        gp.add_argument("--hardware-drift", type=float, default=0.0)
        gp.add_argument("--hardware-crosstalk", type=float, default=0.01)
        gp.add_argument("--hardware-directional", action="store_true")
        gp.add_argument("--hardware-snapshot-spacing", type=float, default=50_000.0)
        gp.add_argument("--seeds", type=int, nargs="+", help="Override gauntlet seeds")
        gp.add_argument("--qasm-root", type=Path, help="Optional QASMBench root")
        gp.add_argument("--selection-seed", type=int, help="Selection seed for QASMBench suites")
        gp.add_argument("--out", type=Path, help="Base output directory (timestamp is appended)")
        gp.set_defaults(func=lambda a, m=mode: _gauntlet(m, a))

    add_gauntlet("gauntlet-small", "small")
    add_gauntlet("gauntlet-full", "full")
    add_gauntlet("gauntlet-industrial", "industrial")

    inv = sub.add_parser("invariants", help="Run invariant checks on results CSV")
    inv.add_argument(
        "--results", type=Path, help="Path to results CSV (defaults to latest gauntlet results)"
    )
    inv.add_argument("--out", type=Path, help="Output directory for report")
    inv.set_defaults(func=cmd_invariants)

    proxy = sub.add_parser("validate-proxy-extended", help="Run proxy validation vs noisy Aer")
    proxy.add_argument(
        "--out", type=Path, help="Output directory (defaults to latest gauntlet run)"
    )
    proxy.add_argument("--max-circuits", type=int, default=20)
    proxy.add_argument("--max-qubits", type=int, default=10)
    proxy.add_argument("--shots", type=int, default=1000)
    proxy.add_argument("--selection-seed", type=int, default=7)
    proxy.add_argument("--qasm-root", type=Path)
    proxy.add_argument("--include-weighted", action="store_true", help="Include weighted SABRE")
    proxy.set_defaults(func=cmd_validate_proxy)

    verdict = sub.add_parser("verdict", help="Assemble FINAL_VERDICT.md")
    verdict.add_argument("--audit-root", type=Path, help="Root directory with gauntlet artifacts")
    verdict.add_argument("--bestness-summary", type=Path, help="Override bestness/summary CSV")
    verdict.add_argument("--weighted-summary", type=Path, help="Override weighted summary CSV")
    verdict.add_argument("--proxy-correlations", type=Path, help="Override proxy correlations CSV")
    verdict.add_argument("--delta-correlations", type=Path, help="Override delta correlations CSV")
    verdict.add_argument("--out", type=Path, help="Path for FINAL_VERDICT.md")
    verdict.set_defaults(func=cmd_verdict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RunError as exc:  # pragma: no cover - simple CLI error
        _log(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
