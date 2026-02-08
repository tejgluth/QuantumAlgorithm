import pandas as pd

from quantum_routing_rl.eval import run_eval


def test_run_eval_emits_qiskit_trials_baseline(tmp_path):
    out_dir = tmp_path / "artifacts"
    args = [
        "--suite",
        "dev",
        "--dev-limit",
        "1",
        "--out",
        str(out_dir),
        "--results-name",
        "results.csv",
        "--summary-name",
        "summary.csv",
        "--seeds",
        "5",
        "--hardware-samples",
        "1",
        "--qiskit-trials",
        "1",
        "2",
    ]

    exit_code = run_eval.main(args)
    assert exit_code == 0

    df = pd.read_csv(out_dir / "results.csv")
    baselines = set(df["baseline_name"])
    assert "qiskit_sabre_trials1" in baselines
    assert "qiskit_sabre_trials2" in baselines
    assert "qiskit_sabre_best" in baselines
    assert not (run_eval.REQUIRED_COLUMNS - set(df.columns))
    # At least one row per baseline.
    assert len(df) >= len(baselines)
