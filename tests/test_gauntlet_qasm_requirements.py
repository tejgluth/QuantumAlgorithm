from pathlib import Path

import pandas as pd
import pytest

from quantum_routing_rl.benchmarks.qasm_discovery import discover_qasm_root, is_fixture_like
from quantum_routing_rl.eval import gauntlet


def _write_qasm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("OPENQASM 2.0;\nqreg q[1];\n")


def _stub_results(out_dir: Path, results_name: str) -> None:
    metrics = {
        "graph_id": "g0",
        "baseline_name": "baseline",
        "swaps_inserted": 0,
        "twoq_count": 0,
        "twoq_depth": 0,
        "depth": 0,
        "routing_runtime_s": 0.0,
        "noise_proxy_score": 0.0,
        "log_success_proxy": 0.0,
        "duration_proxy": 0.0,
        "overall_log_success": 0.0,
        "total_duration_ns": 0.0,
        "decoherence_penalty": 0.0,
    }
    df = pd.DataFrame([metrics])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / results_name, index=False)


def test_discovery_ignores_fixtures(tmp_path: Path) -> None:
    fixture_root = tmp_path / "tests" / "fixtures" / "qasmbench"
    real_root = tmp_path / "datasets" / "QASMBench"
    for i in range(20):
        _write_qasm(fixture_root / f"fixture_{i}.qasm")
    real_dir = real_root / "subset"
    for i in range(600):
        _write_qasm(real_dir / f"circuit_{i}.qasm")

    result = discover_qasm_root(tmp_path)
    selected = Path(result["selected_root"])
    assert not is_fixture_like(selected, result["qasm_count"])
    assert selected.is_relative_to(real_root)
    assert result["qasm_count"] >= 600


def test_require_fails_on_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_root = tmp_path / "tests" / "fixtures" / "qasmbench"
    for i in range(20):
        _write_qasm(fixture_root / f"fixture_{i}.qasm")

    with pytest.raises(SystemExit):
        gauntlet.main(
            [
                "--mode",
                "full",
                "--qasmbench-root",
                str(tmp_path),
                "--require-qasmbench",
            ]
        )


def test_auto_download_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "downloaded" / "qasmbench"

    def fake_fetch(target: Path, repo: str) -> Path:
        for i in range(550):
            _write_qasm(target / "QASMBench" / "subset" / f"circuit_{i}.qasm")
        return target

    def fake_run_eval(argv: list[str]) -> None:
        out_dir = Path(argv[argv.index("--out") + 1])
        results_name = argv[argv.index("--results-name") + 1]
        _stub_results(out_dir, results_name)

    monkeypatch.setattr(gauntlet, "ensure_qasmbench", fake_fetch)
    monkeypatch.setattr(gauntlet.run_eval, "main", fake_run_eval)

    exit_code = gauntlet.main(
        [
            "--mode",
            "full",
            "--qasmbench-root",
            str(tmp_path),
            "--require-qasmbench",
            "--auto-download-qasmbench",
            "--qasmbench-dest",
            str(dest),
            "--out",
            str(tmp_path / "gauntlet_out"),
            "--seeds",
            "1",
        ]
    )

    assert exit_code == 0
    base_out = tmp_path / "gauntlet_out"
    assert base_out.exists()
    runs = list(base_out.iterdir())
    assert runs, "timestamped gauntlet output was not created"


def test_run_eval_receives_fast_validation_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qasm_root = tmp_path / "datasets" / "QASMBench"
    for i in range(60):
        _write_qasm(qasm_root / "small" / f"toy_{i}.qasm")
    calls: list[list[str]] = []

    def fake_run_eval(argv: list[str]) -> None:
        calls.append(argv)
        out_dir = Path(argv[argv.index("--out") + 1])
        results_name = argv[argv.index("--results-name") + 1]
        _stub_results(out_dir, results_name)

    monkeypatch.setattr(gauntlet.run_eval, "main", fake_run_eval)

    exit_code = gauntlet.main(
        [
            "--mode",
            "small",
            "--qasmbench-root",
            str(qasm_root),
            "--out",
            str(tmp_path / "gauntlet_out"),
            "--seeds",
            "1",
            "--max-circuits-per-suite",
            "1",
            "--max-qubits-per-suite",
            "12",
            "--max-ops-per-circuit",
            "500",
            "--circuit-selection",
            "smallest",
            "--qiskit-trials",
            "1",
            "2",
            "--weighted-trials",
            "3",
            "--jobs",
            "1",
            "--torch-device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert calls, "run_eval.main was not invoked"
    for argv in calls:
        assert "--max-circuits" in argv
        assert "--max-qubits" in argv
        assert "--max-ops" in argv
        assert "--circuit-selection" in argv
        assert "--qiskit-trials" in argv
        assert "--weighted-trials" in argv
        assert "--torch-device" in argv


def test_gauntlet_skips_suite_when_filters_remove_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qasm_root = tmp_path / "datasets" / "QASMBench"
    for i in range(60):
        _write_qasm(qasm_root / "small" / f"toy_{i}.qasm")

    def fake_run_eval(argv: list[str]) -> None:
        suite_name = argv[argv.index("--suite") + 1]
        if suite_name == "gauntlet:pressure":
            raise RuntimeError("No circuits selected after applying suite/filters.")
        out_dir = Path(argv[argv.index("--out") + 1])
        results_name = argv[argv.index("--results-name") + 1]
        _stub_results(out_dir, results_name)

    monkeypatch.setattr(gauntlet.run_eval, "main", fake_run_eval)

    exit_code = gauntlet.main(
        [
            "--mode",
            "small",
            "--qasmbench-root",
            str(qasm_root),
            "--out",
            str(tmp_path / "gauntlet_out"),
            "--seeds",
            "1",
            "--jobs",
            "1",
        ]
    )

    assert exit_code == 0
    runs = sorted((tmp_path / "gauntlet_out").iterdir())
    assert runs, "timestamped gauntlet output was not created"
    combined = runs[-1] / "results_gauntlet_small.csv"
    assert combined.exists()
    df = pd.read_csv(combined)
    assert not df.empty
