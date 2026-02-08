from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from quantum_routing_rl.benchmarks.qasmbench_loader import (
    QasmCircuit,
    discover_qasm_files,
    load_qasm_file,
    load_qasmbench_tier,
    load_suite,
)
from scripts import fetch_qasmbench


@pytest.fixture(scope="module")
def qasm_root() -> Path:
    return Path("tests/fixtures/qasmbench").resolve()


def test_discover_qasm_files_recurses_and_sorts(qasm_root: Path) -> None:
    files = discover_qasm_files(qasm_root)
    assert len(files) == 20
    assert files == sorted(files)


def test_load_qasm_file_produces_circuit(qasm_root: Path) -> None:
    sample = qasm_root / "arithmetic" / "arithmetic_01.qasm"
    circuit = load_qasm_file(sample)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits >= 2


def test_dev_suite_round_robin_and_deterministic(qasm_root: Path) -> None:
    suite_a = load_suite(qasm_root, suite="dev", selection_seed=7)
    suite_b = load_suite(qasm_root, suite="dev", selection_seed=7)
    assert len(suite_a) == 20
    assert [c.circuit_id for c in suite_a] == [c.circuit_id for c in suite_b]
    assert all(isinstance(entry, QasmCircuit) for entry in suite_a)
    top_levels = {entry.circuit_id.split("/")[0] for entry in suite_a}
    assert len(top_levels) >= 3


def test_tier_small_filters_and_is_deterministic(qasm_root: Path) -> None:
    tier_a = load_qasmbench_tier(qasm_root, tier="small", limit=5, selection_seed=3, max_qubits=15)
    tier_b = load_qasmbench_tier(qasm_root, tier="small", limit=5, selection_seed=3, max_qubits=15)
    assert [c.circuit_id for c in tier_a] == [c.circuit_id for c in tier_b]
    assert all(c.circuit.num_qubits <= 15 for c in tier_a)
    assert len(tier_a) == 5


def test_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError):
        discover_qasm_files(Path("tests/fixtures/does-not-exist"))


def test_fetch_qasmbench_offline_noop(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "qasmbench_src"

    monkeypatch.setattr(fetch_qasmbench, "_network_available", lambda repo: False)
    exit_code = fetch_qasmbench.main(["--dest", str(dest), "--repo", "https://example.com/noop"])

    assert exit_code == 0
    assert not dest.exists()
