"""Lightweight loader for local QASMBench circuits."""

from __future__ import annotations

import hashlib
import os
import warnings
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


def _resolve_root(root: str | Path | None) -> Path:
    """Prefer explicit root, fall back to env var."""
    if root is not None:
        return Path(root).expanduser().resolve()
    env_path = os.environ.get("QASMBENCH_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    msg = "QASMBench root not provided; set QASMBENCH_ROOT or pass --qasm-root pointing to the dataset."
    raise FileNotFoundError(msg)


def discover_qasm_files(root: str | Path | None) -> list[Path]:
    """Recursively find ``.qasm`` files beneath ``root``.

    Raises:
        FileNotFoundError: when ``root`` does not exist or no files are found.
    """
    root_path = _resolve_root(root)
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


def _stable_order(items: Iterable[QasmCircuit], seed: int) -> list[QasmCircuit]:
    keyed = []
    for circuit in items:
        digest = hashlib.sha1(f"{circuit.path}:{seed}".encode()).hexdigest()
        keyed.append((digest, circuit))
    keyed.sort(key=lambda x: x[0])
    return [c for _, c in keyed]


def _limit_qubits(
    circuit: QuantumCircuit, *, min_qubits: int | None, max_qubits: int | None
) -> bool:
    if min_qubits is not None and circuit.num_qubits < min_qubits:
        return False
    if max_qubits is not None and circuit.num_qubits > max_qubits:
        return False
    return True


def load_qasmbench_tier(
    root: str | Path | None,
    tier: str,
    *,
    limit: int | None = None,
    selection_seed: int = 0,
    min_qubits: int | None = None,
    max_qubits: int | None = None,
) -> list[QasmCircuit]:
    """Tiered loader that hashes paths + seed for deterministic selection."""
    tier = tier.lower()
    all_files = discover_qasm_files(root)
    root_path = _resolve_root(root)

    circuits: list[QasmCircuit] = []
    for path in all_files:
        try:
            qc = load_qasm_file(path)
        except ValueError as exc:  # pragma: no cover - defensive skip path
            warnings.warn(f"[qasmbench] skipping invalid QASM {path}: {exc}")
            continue
        if not _limit_qubits(qc, min_qubits=min_qubits, max_qubits=max_qubits):
            continue
        circuits.append(QasmCircuit(_circuit_id_for_path(path, root_path), path, qc))

    ordered = _stable_order(circuits, selection_seed)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def load_suite(
    root: str | Path | None,
    suite: str = "dev",
    *,
    dev_limit: int = 20,
    selection_seed: int = 0,
) -> list[QasmCircuit]:
    """Load a benchmark suite.

    Args:
        root: Local path to QASMBench directory (or env var).
        suite: Either ``dev`` (diverse ~20 circuits), ``full`` (all),
            or tiered values (``small``, ``medium``, ``hard``).
        dev_limit: Maximum circuits for the dev suite.
        selection_seed: Deterministic seed for subset ordering.
    """
    if suite in {"dev", "full"}:
        all_files = discover_qasm_files(root)
        root_path = _resolve_root(root)

        def _hash_key(path: Path) -> str:
            return hashlib.sha1(f"{path}:{selection_seed}".encode()).hexdigest()

        if suite == "dev":
            selected_paths = _select_diverse_subset(
                all_files, root_path, limit=dev_limit, selection_seed=selection_seed
            )
        else:
            selected_paths = all_files
        circuits: list[QasmCircuit] = []
        fallback_paths: list[Path] = []
        if suite == "dev":
            fallback_paths = sorted(
                [p for p in all_files if p not in selected_paths], key=_hash_key
            )

        def _try_add(path: Path) -> None:
            try:
                circuit = load_qasm_file(path)
            except ValueError as exc:  # pragma: no cover - defensive skip path
                warnings.warn(f"[qasmbench] skipping invalid QASM {path}: {exc}")
                return
            circuits.append(
                QasmCircuit(
                    circuit_id=_circuit_id_for_path(path, root_path),
                    path=path,
                    circuit=circuit,
                )
            )

        for path in selected_paths:
            _try_add(path)
            if suite == "dev" and len(circuits) >= dev_limit:
                break
        if suite == "dev" and len(circuits) < dev_limit:
            for path in fallback_paths:
                _try_add(path)
                if len(circuits) >= dev_limit:
                    break
        return circuits

    tier_cfg = {
        "small": {"max_qubits": 15, "limit": 40},
        "medium": {"min_qubits": 8, "max_qubits": 25, "limit": 60},
        "hard": {"min_qubits": 14, "max_qubits": 40, "limit": 80},
    }
    if suite not in tier_cfg:
        msg = f"Unknown suite '{suite}'. Expected dev/full/small/medium/hard."
        raise ValueError(msg)
    cfg = tier_cfg[suite]
    return load_qasmbench_tier(
        root,
        tier=suite,
        limit=cfg["limit"],
        selection_seed=selection_seed,
        min_qubits=cfg.get("min_qubits"),
        max_qubits=cfg.get("max_qubits"),
    )


def _select_diverse_subset(
    paths: Iterable[Path], root: Path, *, limit: int, selection_seed: int
) -> list[Path]:
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
    # Stable shuffle keyed by selection_seed for determinism.
    keyed = []
    for path in selected:
        digest = hashlib.sha1(f"{path}:{selection_seed}".encode()).hexdigest()
        keyed.append((digest, path))
    keyed.sort(key=lambda x: x[0])
    return [p for _, p in keyed]


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
    root: str | Path | None,
    suite: str = "dev",
    *,
    dev_limit: int = 20,
    selection_seed: int = 0,
) -> Iterator[QasmCircuit]:
    """Yield circuits one-by-one (memory friendly wrapper over :func:`load_suite`)."""
    for entry in load_suite(root, suite=suite, dev_limit=dev_limit, selection_seed=selection_seed):
        yield entry
