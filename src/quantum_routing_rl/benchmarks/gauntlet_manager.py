"""Compose benchmark suites and topologies for gauntlet runs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.qasmbench_loader import (
    QasmCircuit,
    load_qasmbench_tier,
    load_suite,
)
from quantum_routing_rl.benchmarks.synthetic_generator import (
    pressure_suite,
    structured_hard_suite,
)
from quantum_routing_rl.benchmarks.topologies import tiered_topologies


@dataclass
class GauntletSuite:
    name: str
    circuits: list[QasmCircuit]
    topologies: dict[str, CouplingMap]
    metadata: dict[str, object] = field(default_factory=dict)
    status: str = "ready"
    skip_reason: str | None = None

    def is_ready(self) -> bool:
        return self.status == "ready" and bool(self.circuits)


def _max_qubits(topologies: dict[str, CouplingMap]) -> int:
    max_qb = 0
    for cmap in topologies.values():
        try:
            max_qb = max(max_qb, int(cmap.size()))
        except Exception:
            edges = cmap.get_edges()
            if edges:
                max_qb = max(max_qb, max(max(u, v) for u, v in edges) + 1)
    return max_qb


def _supermarq_small(selection_seed: int) -> tuple[list[QasmCircuit], str | None]:
    try:
        from supermarq.benchmarks import (
            BernsteinVazirani,
            GHZ,
            HiddenShift,
            QuantumVolume,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        return [], f"supermarq import failed: {exc}"

    rng = random.Random(selection_seed)
    benches = [
        BernsteinVazirani(6),
        GHZ(8),
        HiddenShift(6),
        QuantumVolume(4),
    ]
    rng.shuffle(benches)
    circuits: list[QasmCircuit] = []
    for bench in benches:
        qc = bench.circuit()
        name = f"supermarq_{bench.__class__.__name__.lower()}"
        circuits.append(QasmCircuit(name, Path(f"supermarq/{name}.qasm"), qc))
    return circuits, None


def build_suite(
    name: str,
    *,
    qasm_root: Path | None = None,
    selection_seed: int = 0,
    pressure_seed: int = 99,
) -> GauntletSuite:
    """Assemble gauntlet-ready circuit sets and topologies."""
    suite = name.lower()
    meta: dict[str, object] = {"selection_seed": selection_seed, "sources": []}

    if suite in {"dev", "pressure", "noise"}:
        topologies = tiered_topologies("small")
        circuits = []
        circuits.extend(
            load_suite(qasm_root, suite="dev", dev_limit=20, selection_seed=selection_seed)
        )
        meta["sources"].append("qasmbench_dev")
        if suite in {"pressure", "noise"}:
            circuits.extend(pressure_suite(seed=pressure_seed))
            meta["sources"].append("synthetic_pressure")
        return GauntletSuite(suite, circuits, topologies, metadata=meta)

    if suite.startswith("qasmbench_"):
        tier = suite.split("_", 1)[1]
        topologies = tiered_topologies(
            "small" if tier == "small" else "medium" if tier == "medium" else "industrial"
        )
        max_qb = _max_qubits(topologies)
        tier_limits = {"small": 24, "medium": 32, "hard": max_qb}
        limit = {"small": 40, "medium": 60, "hard": 80}.get(tier, 40)
        circuits = load_qasmbench_tier(
            qasm_root,
            tier=tier if tier in {"small", "medium", "hard"} else "small",
            limit=limit,
            selection_seed=selection_seed,
            max_qubits=min(max_qb, tier_limits.get(tier, max_qb)),
        )
        meta["sources"].append(f"qasmbench_{tier}")
        meta["topology_limit_qubits"] = max_qb
        return GauntletSuite(suite, circuits, topologies, metadata=meta)

    if suite.startswith("hard_"):
        tier = suite
        tier_key = "small"
        if tier == "hard_medium":
            tier_key = "medium"
        elif tier == "hard_large":
            tier_key = "industrial"
        topologies = tiered_topologies(tier_key)
        circuits = structured_hard_suite(tier, seed=selection_seed)
        meta["sources"].append("structured_hard_synthetic")
        meta["compile_only"] = tier == "hard_large"
        return GauntletSuite(suite, circuits, topologies, metadata=meta)

    if suite == "supermarq_small":
        topologies = tiered_topologies("small")
        circuits, err = _supermarq_small(selection_seed)
        status = "ready" if not err else "skipped"
        meta["sources"].append("supermarq")
        return GauntletSuite(
            suite, circuits, topologies, metadata=meta, status=status, skip_reason=err
        )

    msg = f"Unknown gauntlet suite '{name}'"
    raise ValueError(msg)


def available_suites() -> Iterable[str]:
    return [
        "dev",
        "pressure",
        "noise",
        "qasmbench_small",
        "qasmbench_medium",
        "qasmbench_hard",
        "hard_small",
        "hard_medium",
        "hard_large",
        "supermarq_small",
    ]
