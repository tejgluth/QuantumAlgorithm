from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.gauntlet_manager import GauntletSuite
from quantum_routing_rl.benchmarks.qasmbench_loader import (
    QasmCircuit,
    load_suite,
    resolve_qasm_root,
)
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
        "--torch-device",
        "cpu",
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


def test_run_eval_max_circuits_caps_loaded_set(tmp_path):
    out_dir = tmp_path / "artifacts_cap"
    args = [
        "--suite",
        "dev",
        "--dev-limit",
        "5",
        "--max-circuits",
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
    ]

    exit_code = run_eval.main(args)
    assert exit_code == 0
    df = pd.read_csv(out_dir / "results.csv")
    assert df["circuit_id"].nunique() == 1


def test_run_eval_smallest_selection_prefers_smaller_circuits(tmp_path):
    out_dir = tmp_path / "artifacts_smallest"
    qasm_root = resolve_qasm_root(None, allow_fixtures=True, strict=False)
    suite = load_suite(qasm_root, suite="dev", dev_limit=6, selection_seed=0)
    expected = sorted(
        suite,
        key=lambda entry: (
            entry.circuit.num_qubits,
            len(entry.circuit.data),
            entry.circuit_id,
        ),
    )[:2]
    expected_ids = [entry.circuit_id for entry in expected]

    args = [
        "--suite",
        "dev",
        "--dev-limit",
        "6",
        "--max-circuits",
        "2",
        "--circuit-selection",
        "smallest",
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
    ]

    exit_code = run_eval.main(args)
    assert exit_code == 0
    df = pd.read_csv(out_dir / "results.csv")
    observed_ids = df["circuit_id"].drop_duplicates().tolist()
    assert observed_ids == expected_ids


def test_run_eval_max_qubits_filters_circuits(tmp_path):
    out_dir = tmp_path / "artifacts_max_qubits"
    qasm_root = resolve_qasm_root(None, allow_fixtures=True, strict=False)
    suite = load_suite(qasm_root, suite="dev", dev_limit=6, selection_seed=0)
    threshold = min(entry.circuit.num_qubits for entry in suite)

    args = [
        "--suite",
        "dev",
        "--dev-limit",
        "6",
        "--max-qubits",
        str(threshold),
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
    ]

    exit_code = run_eval.main(args)
    assert exit_code == 0
    df = pd.read_csv(out_dir / "results.csv")
    assert int(df["n_qubits"].max()) <= threshold


def test_run_eval_compile_only_suite_skips_heavy_baselines(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts_compile_only"
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    suite = GauntletSuite(
        name="hard_large",
        circuits=[QasmCircuit("toy/hard_large", Path("toy/hard_large.qasm"), qc)],
        topologies={"line_3": CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])},
        metadata={"compile_only": True},
    )

    monkeypatch.setattr(run_eval, "build_suite", lambda *args, **kwargs: suite)

    args = [
        "--suite",
        "gauntlet:hard_large",
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
        "--include-teacher",
        "--run-weighted-sabre",
    ]

    exit_code = run_eval.main(args)
    assert exit_code == 0
    df = pd.read_csv(out_dir / "results.csv")
    assert set(df["baseline_name"]) == {"qiskit_sabre_best", "qiskit_sabre_trials1"}
