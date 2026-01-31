from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from quantum_routing_rl.benchmarks.qasmbench_loader import (
    QasmCircuit,
    discover_qasm_files,
    load_qasm_file,
    load_suite,
)


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
    suite = load_suite(qasm_root, suite="dev")
    assert len(suite) == 20
    assert all(isinstance(entry, QasmCircuit) for entry in suite)
    # Expect round-robin ordering across top-level folders.
    first_four = [entry.path.name for entry in suite[:4]]
    assert first_four == [
        "arithmetic_01.qasm",
        "crypto_01.qasm",
        "random_01.qasm",
        "small_01.qasm",
    ]
    # Deterministic circuit ids use relative paths without suffix.
    assert suite[0].circuit_id == "arithmetic/arithmetic_01"


def test_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError):
        discover_qasm_files(Path("tests/fixtures/does-not-exist"))
