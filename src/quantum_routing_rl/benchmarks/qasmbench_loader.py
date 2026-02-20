"""Lightweight loader for local QASMBench circuits."""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from qiskit import QuantumCircuit

STRICT_QASM_ENV = "QRRL_REQUIRE_QASMBENCH"


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures" / "qasmbench"


def fixture_qasm_root() -> Path:
    """Public accessor for the bundled fixture dataset."""
    return _fixture_root()


def _strict_qasm_required() -> bool:
    val = os.environ.get(STRICT_QASM_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def strict_qasm_required() -> bool:
    """Check env toggle that forbids fixture QASMBench usage."""
    return _strict_qasm_required()


def _is_fixture_root(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path
    fixture = _fixture_root().resolve()
    return resolved == fixture or fixture in resolved.parents


def is_fixture_root(path: Path) -> bool:
    """Return True when ``path`` points inside the bundled fixtures."""
    return _is_fixture_root(path)


@dataclass(frozen=True)
class QasmCircuit:
    """Container tying a loaded circuit to its source path."""

    circuit_id: str
    path: Path
    circuit: QuantumCircuit


@dataclass
class _LoadStats:
    """Accumulates loader outcomes for concise reporting."""

    warn_limit: int = 5
    seen: int = 0
    loaded_ok: int = 0
    skipped_invalid: int = 0
    skipped_unsupported: int = 0
    warnings_emitted: int = 0

    @property
    def total_seen(self) -> int:
        return self.seen

    @property
    def suppressed(self) -> int:
        emitted_for_skips = min(
            self.warnings_emitted, self.skipped_invalid + self.skipped_unsupported
        )
        return (self.skipped_invalid + self.skipped_unsupported) - emitted_for_skips

    def _maybe_warn(self, message: str) -> None:
        if self.warnings_emitted < self.warn_limit:
            warnings.warn(message)
            self.warnings_emitted += 1

    def record_seen(self) -> None:
        self.seen += 1

    def record_invalid(self, path: Path, exc: Exception) -> None:
        self.skipped_invalid += 1
        self._maybe_warn(f"[qasmbench] skipping invalid QASM {path}: {exc}")

    def record_unsupported(self, path: Path, reason: str) -> None:
        self.skipped_unsupported += 1
        self._maybe_warn(f"[qasmbench] unsupported dialect for {path} ({reason}); skipping.")

    def record_loaded(self) -> None:
        self.loaded_ok += 1

    def summary_line(self) -> str:
        return (
            f"[qasmbench] loaded={self.loaded_ok} "
            f"skipped_invalid={self.skipped_invalid} "
            f"skipped_unsupported={self.skipped_unsupported} "
            f"total_seen={self.total_seen}"
        )


def _resolve_root(root: str | Path | None) -> Path:
    """Prefer explicit root, fall back to env var."""
    require_real = _strict_qasm_required()
    resolved: Path | None = None
    if root is not None:
        resolved = Path(root).expanduser().resolve()
    else:
        env_path = os.environ.get("QASMBENCH_ROOT")
        if env_path:
            resolved = Path(env_path).expanduser().resolve()
    if resolved is None:
        msg = (
            "QASMBench root not provided; set QASMBENCH_ROOT or pass --qasm-root pointing to "
            "the dataset."
        )
        raise FileNotFoundError(msg)
    if require_real and _is_fixture_root(resolved):
        msg = (
            f"Strict mode ({STRICT_QASM_ENV}=1) disallows fixture QASMBench at {resolved}; "
            "set QASMBENCH_ROOT to the full dataset."
        )
        raise FileNotFoundError(msg)
    if not resolved.exists():
        raise FileNotFoundError(f"QASMBench root not found: {resolved}")
    return resolved


def resolve_qasm_root(
    root: str | Path | None,
    *,
    allow_fixtures: bool = True,
    strict: bool | None = None,
) -> Path:
    """Resolve QASMBench root, optionally forbidding fixture fallbacks."""

    require_real = _strict_qasm_required() if strict is None else strict
    resolved: Path | None = None
    if root is not None:
        resolved = Path(root).expanduser()
    else:
        env_path = os.environ.get("QASMBENCH_ROOT")
        if env_path:
            resolved = Path(env_path).expanduser()
        elif require_real:
            msg = (
                "Strict mode enabled; set QASMBENCH_ROOT or pass --qasm-root pointing to the full "
                "QASMBench dataset."
            )
            raise FileNotFoundError(msg)
        else:
            resolved = _fixture_root()
    if (require_real or not allow_fixtures) and _is_fixture_root(resolved):
        msg = (
            f"Strict mode ({STRICT_QASM_ENV}=1) disallows fixture QASMBench at {resolved}; "
            "point QASMBENCH_ROOT to the full dataset."
        )
        raise FileNotFoundError(msg)
    resolved = resolved.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"QASMBench root not found: {resolved}")
    return resolved


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


def _unsupported_reason(path: Path) -> str | None:
    """Heuristic check for unsupported OpenQASM dialects."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    lowered = text.lower()
    if "openqasm 3" in lowered:
        return "OPENQASM 3"
    if "defcal" in lowered:
        return "OPENQASM 3 defcal"
    return None


def _load_with_stats(path: Path, stats: _LoadStats) -> QuantumCircuit | None:
    """Load a QASM file while updating summary statistics."""
    stats.record_seen()
    reason = _unsupported_reason(path)
    if reason:
        stats.record_unsupported(path, reason)
        return None

    try:
        qc = load_qasm_file(path)
    except ValueError as exc:  # pragma: no cover - defensive skip path
        stats.record_invalid(path, exc)
        return None
    return qc


def _log_summary(stats: _LoadStats) -> None:
    """Emit a concise summary and suppressed-warning count."""
    if stats.suppressed > 0:
        print(f"[qasmbench] suppressed {stats.suppressed} additional invalid/unsupported warnings.")
    print(stats.summary_line())


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
    stats = _LoadStats()

    circuits: list[QasmCircuit] = []
    for path in all_files:
        qc = _load_with_stats(path, stats)
        if qc is None:
            continue
        if not _limit_qubits(qc, min_qubits=min_qubits, max_qubits=max_qubits):
            continue
        circuits.append(QasmCircuit(_circuit_id_for_path(path, root_path), path, qc))
        stats.record_loaded()

    ordered = _stable_order(circuits, selection_seed)
    if limit is not None:
        ordered = ordered[:limit]
    _log_summary(stats)
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
        stats = _LoadStats()

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
            circuit = _load_with_stats(path, stats)
            if circuit is None:
                return
            circuits.append(
                QasmCircuit(
                    circuit_id=_circuit_id_for_path(path, root_path),
                    path=path,
                    circuit=circuit,
                )
            )
            stats.record_loaded()

        for path in selected_paths:
            _try_add(path)
            if suite == "dev" and len(circuits) >= dev_limit:
                break
        if suite == "dev" and len(circuits) < dev_limit:
            for path in fallback_paths:
                _try_add(path)
                if len(circuits) >= dev_limit:
                    break
        _log_summary(stats)
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
