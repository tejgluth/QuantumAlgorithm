"""Imitation learning: teacher trace recording + training."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import networkx as nx
import torch
import torch.nn.functional as F
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit, load_suite
from quantum_routing_rl.env.routing_env import RoutingEnv
from quantum_routing_rl.models.policy import SwapPolicy, candidate_features, route_with_policy


@dataclass(frozen=True)
class TraceSample:
    """Single teacher-labelled step."""

    features: torch.Tensor
    label: int
    meta: dict


def _default_qasm_root() -> Path:
    env_path = os.environ.get("QASMBENCH_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "qasmbench"


def _build_coupling_maps() -> dict[str, CouplingMap]:
    return {
        "line_3": CouplingMap([[0, 1], [1, 2]]),
        "square_4": CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]]),
    }


def _coupling_size(map_obj: CouplingMap) -> int:
    edges = map_obj.get_edges()
    return max(max(edge) for edge in edges) + 1


def _filter_coupling_maps(
    coupling_maps: dict[str, CouplingMap], circuit: QasmCircuit
) -> Iterable[tuple[str, CouplingMap]]:
    for name, cmap in coupling_maps.items():
        if circuit.circuit.num_qubits <= _coupling_size(cmap):
            yield name, cmap


def teacher_action(state, graph: nx.Graph) -> int:
    """Deterministic shortest-path teacher."""
    if not state.frontier or state.done:
        return 0
    logical_a, logical_b = state.frontier[0]
    phys_pair = (state.layout[logical_a], state.layout[logical_b])
    try:
        path = nx.shortest_path(graph, phys_pair[0], phys_pair[1])
        edge = (path[0], path[1])
    except nx.NetworkXNoPath:
        edge = state.candidate_swaps[0]

    if edge not in state.candidate_swaps and edge[::-1] in state.candidate_swaps:
        edge = edge[::-1]
    if edge not in state.candidate_swaps:
        edge = state.candidate_swaps[0]
    return state.candidate_swaps.index(edge)


def record_traces_for_circuit(
    circuit: QasmCircuit, coupling_map: CouplingMap, *, seed: int = 13, graph_id: str
) -> List[TraceSample]:
    """Run teacher policy in the environment and collect labelled steps."""
    env = RoutingEnv()
    state = env.reset(circuit.circuit, coupling_map, seed=seed)
    graph = nx.Graph(list(coupling_map.get_edges()))
    samples: List[TraceSample] = []
    step = 0
    while not state.done:
        feats = candidate_features(state, graph)
        action = teacher_action(state, graph)
        samples.append(
            TraceSample(
                features=feats.cpu(),
                label=action,
                meta={
                    "circuit_id": circuit.circuit_id,
                    "graph_id": graph_id,
                    "step": step,
                    "candidate_swaps": list(state.candidate_swaps),
                },
            )
        )
        state, _, _, _ = env.step(action)
        step += 1
    return samples


def collect_dataset(
    suite: list[QasmCircuit], coupling_maps: dict[str, CouplingMap], *, seed: int
) -> list[TraceSample]:
    """Collect traces for all applicable circuits."""
    all_samples: list[TraceSample] = []
    for circuit in suite:
        for graph_id, cmap in _filter_coupling_maps(coupling_maps, circuit):
            all_samples.extend(
                record_traces_for_circuit(circuit, cmap, seed=seed, graph_id=graph_id)
            )
    return all_samples


def save_dataset(samples: list[TraceSample], path: Path, *, seed: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_samples": len(samples),
        "seed": seed,
        "qiskit_version": qiskit_version,
        "created_at": time.time(),
        "path": str(path),
    }
    torch.save({"samples": samples, "metadata": metadata}, path)
    meta_json = path.with_suffix(".json")
    meta_json.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_dataset(path: Path) -> list[TraceSample]:
    payload = torch.load(path, map_location="cpu")
    return payload["samples"]


def train_model(
    samples: list[TraceSample], *, epochs: int, lr: float = 1e-3
) -> tuple[SwapPolicy, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwapPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        for sample in samples:
            feats = sample.features.to(device)
            label_tensor = torch.tensor([sample.label], dtype=torch.long, device=device)
            logits = model(feats).unsqueeze(0)
            loss = F.cross_entropy(logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg = total_loss / max(1, len(samples))
        history.append(avg)
    return model, history


def save_checkpoint(
    model: SwapPolicy, path: Path, *, epochs: int, seed: int, loss_history: list[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.net[0].in_features,
            "epochs": epochs,
            "seed": seed,
            "loss_history": loss_history,
        },
        path,
    )


def _update_metadata(out_dir: Path, updates: dict) -> None:
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
    else:
        data = {}
    data.update(updates)
    meta_path.write_text(json.dumps(data, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dev", "full"], default="dev")
    parser.add_argument("--out", type=Path, required=True, help="Artifacts directory root.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--regen-dataset", action="store_true", help="Force regeneration of teacher traces."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Path to QASMBench root (defaults to env QASMBENCH_ROOT or tests fixtures).",
    )
    parser.add_argument(
        "--dev-limit", type=int, default=20, help="Max circuits for dev dataset (matches eval)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "datasets" / "il_traces.pt"
    qasm_root = Path(args.qasm_root or _default_qasm_root()).expanduser()
    if args.regen_dataset or not dataset_path.exists():
        suite = load_suite(qasm_root, suite=args.suite, dev_limit=args.dev_limit)
        coupling_maps = _build_coupling_maps()
        samples = collect_dataset(suite, coupling_maps, seed=args.seed)
        save_dataset(samples, dataset_path, seed=args.seed)
    samples = load_dataset(dataset_path)

    model, history = train_model(samples, epochs=args.epochs, lr=args.lr)
    ckpt_path = out_dir / "checkpoints" / "il.pt"
    save_checkpoint(model, ckpt_path, epochs=args.epochs, seed=args.seed, loss_history=history)

    # Quick smoke rollout on first circuit (if any) to ensure checkpoint usable.
    if samples:
        suite = load_suite(qasm_root, suite=args.suite, dev_limit=1)
        first = suite[0]
        coupling_maps = _build_coupling_maps()
        try:
            graph_name, coupling_map = next(iter(_filter_coupling_maps(coupling_maps, first)))
            _ = route_with_policy(model, first.circuit, coupling_map, name="il_policy_smoke")
            _update_metadata(
                out_dir,
                {
                    "il_checkpoint": str(ckpt_path),
                    "il_dataset": str(dataset_path),
                    "il_seed": args.seed,
                    "il_epochs": args.epochs,
                    "il_lr": args.lr,
                    "il_qasm_root": str(qasm_root),
                    "il_graph": graph_name,
                },
            )
        except StopIteration:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
