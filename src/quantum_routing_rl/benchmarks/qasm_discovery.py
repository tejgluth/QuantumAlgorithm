"""Utilities for locating real QASMBench datasets on disk."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "torch_env",
    "__pycache__",
    "artifacts",
    "node_modules",
    ".mypy_cache",
}


def _is_fixture_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if {"tests", "fixtures"}.issubset(parts):
        return True
    return "fixtures" in parts and "test" in parts


def is_fixture_like(path: str | Path, qasm_count: int | None = None) -> bool:
    """Identify fixture-style datasets or trivially small samples."""

    candidate = Path(path)
    if _is_fixture_path(candidate):
        return True
    if qasm_count is not None and qasm_count < 50:
        return True
    return False


def _should_skip_dir(dirname: str, parent: Path, ignore_dirs: set[str]) -> bool:
    if dirname in ignore_dirs:
        return True
    joined = parent / dirname
    return _is_fixture_path(joined)


def _bounded_walk(root: Path, *, max_depth: int, ignore_dirs: set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_parts = current.relative_to(root).parts
        depth = len(rel_parts)
        dirnames[:] = [
            d
            for d in dirnames
            if depth < max_depth and not _should_skip_dir(d, current, ignore_dirs)
        ]
        for filename in filenames:
            if filename.endswith(".qasm"):
                yield Path(dirpath) / filename


def discover_qasm_root(
    general_root: str | Path,
    *,
    max_depth: int = 8,
    ignore_dirs: Iterable[str] | None = None,
    top_k: int = 5,
) -> dict:
    """Search ``general_root`` for QASMBench-style directories.

    Returns a dict with keys:
        selected_root (str|None), qasm_count (int), candidates (list[(str, int)]), notes (str)
    """

    ignore = set(ignore_dirs or []) | DEFAULT_IGNORE_DIRS
    root = Path(general_root).expanduser()
    if not root.exists():
        return {
            "selected_root": None,
            "qasm_count": 0,
            "candidates": [],
            "notes": f"{root} does not exist",
        }

    counts: defaultdict[Path, int] = defaultdict(int)
    qasm_paths = list(_bounded_walk(root, max_depth=max_depth, ignore_dirs=ignore))
    for qasm_path in qasm_paths:
        rel_ancestors = []
        for parent in qasm_path.parents:
            if parent == qasm_path:
                continue
            try:
                rel_depth = len(parent.relative_to(root).parts)
            except Exception:
                continue
            if rel_depth <= max_depth:
                rel_ancestors.append(parent)
            if parent == root:
                break
        for ancestor in set(rel_ancestors):
            try:
                depth = len(ancestor.relative_to(root).parts)
            except Exception:
                continue
            if depth <= max_depth:
                counts[ancestor] += 1

    if counts:
        sorted_candidates = sorted(
            counts.items(), key=lambda kv: (-kv[1], -len(kv[0].parts), str(kv[0]))
        )
        candidates = [(str(p), c) for p, c in sorted_candidates[:top_k]]
        selected_root, qasm_count = candidates[0]
    else:
        candidates = []
        selected_root, qasm_count = None, 0

    return {
        "selected_root": selected_root,
        "qasm_count": qasm_count,
        "candidates": candidates,
        "notes": f"scanned {len(qasm_paths)} qasm files under {root}",
    }


def pretty_print_discovery(result: dict) -> str:
    lines = ["[qasm-discovery] candidate roots:"]
    candidates = result.get("candidates", [])
    if not candidates:
        lines.append("  (no .qasm files found)")
    for path, count in candidates:
        marker = " *" if path == result.get("selected_root") else "  "
        lines.append(f"{marker} {path} ({count} qasm)")
    notes = result.get("notes")
    if notes:
        lines.append(f"  note: {notes}")
    return "\n".join(lines)
