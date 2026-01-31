"""Imitation learning: teacher trace recording + training."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import networkx as nx
import torch
import torch.nn.functional as F
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap
from torch.serialization import add_safe_globals

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit
from quantum_routing_rl.benchmarks.synthetic_generator import (
    SyntheticSpec,
    generate_pressure_circuit,
    pressure_suite,
)
from quantum_routing_rl.env.routing_env import RoutingEnv
from quantum_routing_rl.models.policy import (
    SwapPolicy,
    candidate_features,
    _match_feature_dim,
    route_with_policy,
    state_features,
)
from quantum_routing_rl.models.teacher import TeacherPolicy, TeacherWeights
from quantum_routing_rl.hardware.model import HardwareModel


FEATURE_VERSION = "il_soft_v1"


@dataclass(frozen=True)
class TraceSample:
    """Single teacher-labelled step."""

    candidate_features: torch.Tensor
    state_features: torch.Tensor
    teacher_scores: torch.Tensor
    action_mask: torch.Tensor
    label: int
    meta: dict


TraceSample.__module__ = "quantum_routing_rl.models.il_train"
PRESSURE_GRAPHS = ("ring_8", "grid_3x3", "heavy_hex_15")
GRAPH_SPECS = {
    "ring_8": {"n_qubits": 8, "twoq_layers": 18},
    "grid_3x3": {"n_qubits": 9, "twoq_layers": 18},
    "heavy_hex_15": {"n_qubits": 15, "twoq_layers": 22},
}


def _all_coupling_maps() -> dict[str, CouplingMap]:
    base = {
        "line_3": CouplingMap([[0, 1], [1, 2]]),
        "square_4": CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]]),
        "ring_8": CouplingMap([[i, (i + 1) % 8] for i in range(8)]),
        "grid_3x3": CouplingMap(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [0, 3],
                [3, 6],
                [1, 4],
                [4, 7],
                [2, 5],
                [5, 8],
            ]
        ),
        "heavy_hex_15": CouplingMap(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [2, 6],
                [4, 7],
                [5, 6],
                [6, 7],
                [5, 8],
                [6, 9],
                [7, 10],
                [8, 9],
                [9, 10],
                [8, 11],
                [9, 12],
                [10, 13],
                [11, 12],
                [12, 13],
                [11, 14],
                [13, 14],
            ]
        ),
    }
    return base


def _build_coupling_maps(graphs: Sequence[str]) -> dict[str, CouplingMap]:
    mapping = _all_coupling_maps()
    missing = [g for g in graphs if g not in mapping]
    if missing:
        msg = f"Unknown graphs requested for IL dataset: {missing}"
        raise ValueError(msg)
    return {g: mapping[g] for g in graphs}


def _synthetic_circuits(
    graphs: Sequence[str], *, num_seeds: int, base_seed: int
) -> dict[str, list[QasmCircuit]]:
    """Create synthetic pressure circuits per graph using different seeds."""

    suites: dict[str, list[QasmCircuit]] = {}
    for graph in graphs:
        spec = GRAPH_SPECS.get(graph)
        if spec is None:
            continue
        circuits: list[QasmCircuit] = []
        for idx in range(num_seeds):
            seed = base_seed + idx
            circ = generate_pressure_circuit(
                SyntheticSpec(
                    name=f"{graph}_seed{seed}",
                    n_qubits=spec["n_qubits"],
                    twoq_layers=spec["twoq_layers"],
                    seed=seed,
                )
            )
            circuits.append(circ)
        suites[graph] = circuits
    return suites


def _augment_with_pressure(
    suites: dict[str, list[QasmCircuit]], graphs: Sequence[str], seed: int
) -> dict[str, list[QasmCircuit]]:
    """Add pressure_suite circuits (seeded) to the training pool."""

    extras = pressure_suite(seed=seed)
    graph_map = {
        8: "ring_8",
        9: "grid_3x3",
        15: "heavy_hex_15",
    }
    for circ in extras:
        target = graph_map.get(circ.circuit.num_qubits)
        if target and target in graphs:
            suites.setdefault(target, []).append(circ)
    return suites


def _build_hardware_models(
    coupling_maps: dict[str, CouplingMap],
    seeds: Sequence[int],
    profile: str = "realistic",
) -> dict[str, list[tuple[int, HardwareModel]]]:
    models: dict[str, list[tuple[int, HardwareModel]]] = {}
    for graph_id, cmap in coupling_maps.items():
        edges = [tuple(edge) for edge in cmap.get_edges()]
        graph = nx.Graph()
        graph.add_edges_from(edges)
        models[graph_id] = [
            (seed, HardwareModel.synthetic(graph, seed=seed, profile=profile)) for seed in seeds
        ]
    return models


def record_traces_for_circuit(
    circuit: QasmCircuit,
    coupling_map: CouplingMap,
    *,
    seed: int = 13,
    graph_id: str,
    teacher: TeacherPolicy | None = None,
    hardware_model: HardwareModel | None = None,
    hardware_seed: int | None = None,
) -> List[TraceSample]:
    """Run teacher policy in the environment and collect labelled steps."""
    env = RoutingEnv()
    state = env.reset(circuit.circuit, coupling_map, seed=seed, hardware_model=hardware_model)
    graph = nx.Graph(list(coupling_map.get_edges()))
    hardware = env.hardware_model
    samples: List[TraceSample] = []
    teacher = teacher or TeacherPolicy()
    teacher.begin_episode(graph)
    step = 0
    while not state.done:
        cand_feats = candidate_features(state, graph, hardware)
        state_feats = state_features(state, graph)
        scores = torch.tensor(teacher.score_candidates(state, graph, hardware), dtype=torch.float32)
        mask = torch.tensor(state.action_mask, dtype=torch.bool)
        finite_scores = torch.where(mask, scores, torch.tensor(float("inf")))
        action = int(torch.argmin(finite_scores).item())
        samples.append(
            TraceSample(
                candidate_features=cand_feats.cpu(),
                state_features=state_feats.cpu(),
                teacher_scores=scores.cpu(),
                action_mask=mask.cpu(),
                label=action,
                meta={
                    "circuit_id": circuit.circuit_id,
                    "graph_id": graph_id,
                    "step": step,
                    "candidate_swaps": list(state.candidate_swaps),
                    "hardware_seed": hardware_seed,
                    "feature_version": FEATURE_VERSION,
                },
            )
        )
        swap_edge = state.candidate_swaps[action]
        state, _, _, _ = env.step(action)
        teacher.update_after_action(swap_edge, state.step_count)
        step += 1
    return samples


def collect_dataset(
    suite: dict[str, list[QasmCircuit]],
    coupling_maps: dict[str, CouplingMap],
    *,
    seed: int,
    hardware_models: dict[str, list[tuple[int, HardwareModel]]],
    teacher_weights: TeacherWeights | None = None,
) -> list[TraceSample]:
    """Collect traces for all applicable circuits across graphs/hardware."""

    teacher = TeacherPolicy(weights=teacher_weights)
    all_samples: list[TraceSample] = []
    for graph_id, circuits in suite.items():
        cmap = coupling_maps[graph_id]
        hw_list = hardware_models.get(graph_id, [(None, None)])
        for hw_seed, hw_model in hw_list:
            for circuit in circuits:
                all_samples.extend(
                    record_traces_for_circuit(
                        circuit,
                        cmap,
                        seed=seed,
                        graph_id=graph_id,
                        hardware_model=hw_model,
                        hardware_seed=hw_seed,
                        teacher=teacher,
                    )
                )
    return all_samples


def save_dataset(
    samples: list[TraceSample],
    path: Path,
    *,
    metadata: dict,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [
        {
            "candidate_features": sample.candidate_features,
            "state_features": sample.state_features,
            "teacher_scores": sample.teacher_scores,
            "action_mask": sample.action_mask,
            "label": sample.label,
            "meta": sample.meta,
        }
        for sample in samples
    ]
    meta = {
        "num_samples": len(samples),
        "qiskit_version": qiskit_version,
        "created_at": time.time(),
        "path": str(path),
        "feature_version": FEATURE_VERSION,
    }
    meta.update(metadata)
    torch.save({"samples": serialised, "metadata": meta}, path)
    meta_json = path.with_suffix(".json")
    meta_json.write_text(json.dumps(meta, indent=2))
    return meta


def load_dataset(path: Path, *, return_metadata: bool = False):
    try:  # torch >=2.6 tightened defaults; allow trusted TraceSample
        add_safe_globals([TraceSample])
        import importlib
        import sys

        main_mod = importlib.import_module("__main__")
        setattr(main_mod, "TraceSample", TraceSample)
        sys.modules["__main__"] = main_mod
        add_safe_globals([getattr(main_mod, "TraceSample")])
    except Exception:
        pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_samples = payload["samples"]
    if raw_samples and isinstance(raw_samples[0], dict):
        samples = [TraceSample(**item) for item in raw_samples]
    else:
        samples = raw_samples
    meta = payload.get("metadata", {})
    if return_metadata:
        return samples, meta
    return samples


def train_model(
    samples: list[TraceSample],
    *,
    epochs: int,
    lr: float = 1e-3,
    tau: float = 1.0,
    patience: int = 5,
    val_split: float = 0.1,
    seed: int = 0,
    hard_weight: float = 0.2,
) -> tuple[SwapPolicy, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = (
        samples[0].candidate_features.shape[1] + samples[0].state_features.shape[0]
        if samples
        else None
    )
    model = SwapPolicy(feature_dim=feature_dim or SwapPolicy().net[0].in_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    rng = random.Random(seed)
    torch.manual_seed(seed)
    hard_wt = max(0.0, min(1.0, hard_weight))

    def _assemble_features(sample: TraceSample) -> torch.Tensor:
        cand = sample.candidate_features.to(device)
        state_vec = sample.state_features.to(device)
        if cand.numel() == 0:
            return torch.zeros(
                (len(sample.candidate_features), cand.shape[1] + state_vec.shape[0]), device=device
            )
        tiled_state = state_vec.unsqueeze(0).repeat(cand.shape[0], 1)
        feats = torch.cat([cand, tiled_state], dim=1)
        return _match_feature_dim(feats, model.net[0].in_features)  # type: ignore[index]

    def _soft_targets(sample: TraceSample) -> torch.Tensor:
        scores = sample.teacher_scores.to(device)
        mask = sample.action_mask.to(device)
        masked = scores.clone()
        masked = torch.where(mask, masked, torch.tensor(float("inf"), device=device))
        # Stable softmax over allowed actions.
        scaled = -masked / max(tau, 1e-6)
        probs = torch.zeros_like(scaled, device=device)
        if mask.any():
            probs[mask] = torch.softmax(scaled[mask], dim=-1)
        if float(probs.sum().item()) == 0.0:
            probs = torch.ones_like(probs, device=device) / len(probs)
        return probs

    val_size = max(1, int(len(samples) * val_split)) if samples else 0
    rng.shuffle(samples)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:] or samples

    best_state = None
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        rng.shuffle(train_samples)
        total_loss = 0.0
        for sample in train_samples:
            feats = _assemble_features(sample)
            logits = model(feats)
            mask = sample.action_mask.to(device)
            logits = logits.masked_fill(~mask, -1e9)
            targets = _soft_targets(sample)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss_soft = F.kl_div(log_probs, targets, reduction="batchmean", log_target=False)
            label_tensor = torch.tensor([sample.label], dtype=torch.long, device=device)
            loss_hard = F.cross_entropy(logits.unsqueeze(0), label_tensor)
            loss = (1 - hard_wt) * loss_soft + hard_wt * loss_hard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        train_loss = total_loss / max(1, len(train_samples))

        # Validation
        with torch.no_grad():
            val_total = 0.0
            for sample in val_samples:
                feats = _assemble_features(sample)
                logits = model(feats)
                mask = sample.action_mask.to(device)
                logits = logits.masked_fill(~mask, -1e9)
                targets = _soft_targets(sample)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss_soft = F.kl_div(log_probs, targets, reduction="batchmean", log_target=False)
                label_tensor = torch.tensor([sample.label], dtype=torch.long, device=device)
                loss_hard = F.cross_entropy(logits.unsqueeze(0), label_tensor)
                loss = (1 - hard_wt) * loss_soft + hard_wt * loss_hard
                val_total += float(loss.item())
            val_loss = val_total / max(1, len(val_samples))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_checkpoint(
    model: SwapPolicy,
    path: Path,
    *,
    epochs: int,
    seed: int,
    loss_history: list[dict],
    tau: float,
    hard_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.net[0].in_features,
            "epochs": epochs,
            "seed": seed,
            "loss_history": loss_history,
            "use_value_head": bool(model.value_head is not None),
            "tau": tau,
            "hard_weight": hard_weight,
            "feature_version": FEATURE_VERSION,
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
    parser.add_argument("--out", type=Path, required=True, help="Artifacts directory root.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature.")
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=0.35,
        help="Weight for hard-label cross-entropy term (0-1).",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--graphs",
        type=str,
        default=",".join(PRESSURE_GRAPHS),
        help="Comma-separated coupling graph ids.",
    )
    parser.add_argument(
        "--num-circuit-seeds",
        type=int,
        default=50,
        help="Number of synthetic circuit seeds per graph.",
    )
    parser.add_argument(
        "--num-hardware-seeds",
        type=int,
        default=10,
        help="Number of hardware model seeds per graph.",
    )
    parser.add_argument(
        "--circuit-seed-base",
        type=int,
        default=0,
        help="Base seed for synthetic circuit generation.",
    )
    parser.add_argument(
        "--hardware-seed-base",
        type=int,
        default=101,
        help="Base seed for hardware model generation.",
    )
    parser.add_argument("--hardware-profile", type=str, default="realistic")
    parser.add_argument(
        "--pressure-seed",
        type=int,
        default=99,
        help="Seed for canonical pressure circuits to include in the dataset.",
    )
    parser.add_argument(
        "--regen-dataset", action="store_true", help="Force regeneration of teacher traces."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller dataset/training run for quick iteration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    if args.fast:
        args.num_circuit_seeds = max(4, args.num_circuit_seeds // 4)
        args.num_hardware_seeds = max(3, args.num_hardware_seeds // 3)
        args.epochs = max(5, args.epochs // 3)
        args.patience = min(args.patience, 3)

    dataset_path = out_dir / "datasets" / "il_soft_traces.pt"
    coupling_maps = _build_coupling_maps(graphs)

    circuit_seeds = list(
        range(args.circuit_seed_base, args.circuit_seed_base + args.num_circuit_seeds)
    )
    hardware_seeds = list(
        range(args.hardware_seed_base, args.hardware_seed_base + args.num_hardware_seeds)
    )
    synthetic_suite = _synthetic_circuits(
        graphs, num_seeds=len(circuit_seeds), base_seed=circuit_seeds[0]
    )
    synthetic_suite = _augment_with_pressure(synthetic_suite, graphs, args.pressure_seed)
    hardware_models = _build_hardware_models(
        coupling_maps, hardware_seeds, profile=args.hardware_profile
    )

    def _needs_regen(meta: dict) -> bool:
        if not meta:
            return True
        return (
            meta.get("feature_version") != FEATURE_VERSION
            or meta.get("graphs") != graphs
            or meta.get("num_circuit_seeds") != len(circuit_seeds)
            or meta.get("num_hardware_seeds") != len(hardware_seeds)
            or meta.get("hardware_profile") != args.hardware_profile
            or meta.get("pressure_seed") != args.pressure_seed
        )

    samples: list[TraceSample]
    dataset_meta: dict
    if args.regen_dataset or not dataset_path.exists():
        samples = collect_dataset(
            synthetic_suite,
            coupling_maps,
            seed=args.seed,
            hardware_models=hardware_models,
            teacher_weights=TeacherWeights(),
        )
        dataset_meta = save_dataset(
            samples,
            dataset_path,
            metadata={
                "graphs": graphs,
                "circuit_seeds": circuit_seeds,
                "hardware_seeds": hardware_seeds,
                "num_circuit_seeds": len(circuit_seeds),
                "num_hardware_seeds": len(hardware_seeds),
                "teacher_weights": asdict(TeacherWeights()),
                "seed": args.seed,
                "hardware_profile": args.hardware_profile,
                "pressure_seed": args.pressure_seed,
                "pressure_circuits": [
                    c.circuit_id for c in pressure_suite(seed=args.pressure_seed)
                ],
            },
        )
    else:
        samples, dataset_meta = load_dataset(dataset_path, return_metadata=True)
        if _needs_regen(dataset_meta):
            samples = collect_dataset(
                synthetic_suite,
                coupling_maps,
                seed=args.seed,
                hardware_models=hardware_models,
                teacher_weights=TeacherWeights(),
            )
            dataset_meta = save_dataset(
                samples,
                dataset_path,
                metadata={
                    "graphs": graphs,
                    "circuit_seeds": circuit_seeds,
                    "hardware_seeds": hardware_seeds,
                    "num_circuit_seeds": len(circuit_seeds),
                    "num_hardware_seeds": len(hardware_seeds),
                    "teacher_weights": asdict(TeacherWeights()),
                    "seed": args.seed,
                    "hardware_profile": args.hardware_profile,
                    "pressure_seed": args.pressure_seed,
                    "pressure_circuits": [
                        c.circuit_id for c in pressure_suite(seed=args.pressure_seed)
                    ],
                },
            )

    model, history = train_model(
        samples,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        patience=args.patience,
        val_split=args.val_split,
        seed=args.seed,
        hard_weight=args.hard_weight,
    )
    ckpt_path = out_dir / "checkpoints" / "il_soft.pt"
    save_checkpoint(
        model,
        ckpt_path,
        epochs=args.epochs,
        seed=args.seed,
        loss_history=history,
        tau=args.tau,
        hard_weight=args.hard_weight,
    )

    # Quick smoke rollout on first circuit (if any) to ensure checkpoint usable.
    if samples:
        first_graph = graphs[0]
        first_circuit = synthetic_suite[first_graph][0]
        coupling_map = coupling_maps[first_graph]
        _ = route_with_policy(
            model, first_circuit.circuit, coupling_map, name="il_policy_smoke", seed=args.seed
        )
        _update_metadata(
            out_dir,
            {
                "il_checkpoint": str(ckpt_path),
                "il_dataset": str(dataset_path),
                "il_seed": args.seed,
                "il_epochs": args.epochs,
                "il_lr": args.lr,
                "il_tau": args.tau,
                "il_hard_weight": args.hard_weight,
                "il_graphs": graphs,
                "il_circuit_seeds": circuit_seeds,
                "il_hardware_seeds": hardware_seeds,
                "il_feature_version": FEATURE_VERSION,
                "il_history_path": str(out_dir / "training" / "il_soft.json"),
            },
        )

    history_path = out_dir / "training" / "il_soft.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
