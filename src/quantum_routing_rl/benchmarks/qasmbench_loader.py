"""Lightweight loader for local QASMBench circuits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class QasmCircuit:
    """Container tying a loaded circuit to its source path."""

    circuit_id: str
    path: Path
    circuit: QuantumCircuit


def discover_qasm_files(root: str | Path) -> list[Path]:
    """Recursively find ``.qasm`` files beneath ``root``.

    Raises:
        FileNotFoundError: when ``root`` does not exist or no files are found.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        msg = f"QASMBench root not found: {root_path}"
        raise FileNotFoundError(msg)

    qasm_files = sorted(p for p in root_path.rglob("*.qasm") if p.is_file())
    if not qasm_files:
        msg = f"No .qasm files found under {root_path}"
        raise FileNotFoundError(msg)
    return qasm_files


def load_qasm_file(path: str | Path) -> QuantumCircuit:
    """Load a single QASM file into a :class:`QuantumCircuit`."""
    file_path = Path(path).expanduser().resolve()
    try:
        return QuantumCircuit.from_qasm_file(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load QASM file: {file_path}"
        raise ValueError(msg) from exc


def load_suite(
    root: str | Path,
    suite: str = "dev",
    *,
    dev_limit: int = 20,
) -> list[QasmCircuit]:
    """Load a benchmark suite.

    Args:
        root: Local path to QASMBench directory.
        suite: Either ``\"dev\"`` (diverse ~20 circuits) or ``\"full\"`` (all).
        dev_limit: Maximum circuits for the dev suite.
    """
    all_files = discover_qasm_files(root)
    root_path = Path(root).expanduser().resolve()

    if suite == "dev":
        selected = _select_diverse_subset(all_files, root_path, limit=dev_limit)
    elif suite == "full":
        selected = all_files
    else:
        msg = f"Unknown suite '{suite}'. Expected 'dev' or 'full'."
        raise ValueError(msg)

    return [
        QasmCircuit(
            circuit_id=_circuit_id_for_path(path, root_path),
            path=path,
            circuit=load_qasm_file(path),
        )
        for path in selected
    ]


def _select_diverse_subset(paths: Iterable[Path], root: Path, *, limit: int) -> list[Path]:
    """Pick up to ``limit`` paths, round-robin across top-level folders."""
    buckets = _bucket_by_top_level(paths, root)
    ordered_keys = sorted(buckets)
    selected: list[Path] = []

    # Round-robin extraction to favor diversity.
    while len(selected) < limit and any(buckets.values()):
        for key in ordered_keys:
            bucket = buckets.get(key)
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _bucket_by_top_level(paths: Iterable[Path], root: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {}
    for path in sorted(paths):
        key = _top_level_dir(path, root)
        buckets.setdefault(key, []).append(path)
    return buckets


def _top_level_dir(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


def _circuit_id_for_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.with_suffix("").as_posix()


def iter_qasm_circuits(
    root: str | Path,
    suite: str = "dev",
    *,
    dev_limit: int = 20,
) -> Iterator[QasmCircuit]:
    """Yield circuits one-by-one (memory friendly wrapper over :func:`load_suite`)."""
    for entry in load_suite(root, suite=suite, dev_limit=dev_limit):
        yield entry
