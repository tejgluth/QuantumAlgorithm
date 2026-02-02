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
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.models.policy import (
    SwapPolicy,
    _match_feature_dim,
    candidate_features,
    route_with_policy,
    state_features,
    _scores_to_logits,
)
from quantum_routing_rl.models.teacher import TeacherPolicy, TeacherWeights, _sabre_initial_layout
from quantum_routing_rl.hardware.model import HardwareModel


FEATURE_VERSION = "il_soft_v3"
DATASET_VERSION = "il_dataset_dagger_v3"
FRONTIER_SIZE = 4


@dataclass(frozen=True)
class TraceSample:
    """Single teacher-labelled step."""

    candidate_features: torch.Tensor
    state_features: torch.Tensor
    teacher_scores: torch.Tensor
    action_mask: torch.Tensor
    label: int
    meta: dict


TraceSample.__module__ = "experiments.legacy_rl.il_train"
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


def _reservoir_update(
    reservoir: list[TraceSample],
    new_samples: list[TraceSample],
    *,
    cap: int | None,
    rng: random.Random,
) -> list[TraceSample]:
    """Reservoir-sample new_samples into reservoir up to `cap` items."""

    if cap is None or cap <= 0:
        reservoir.extend(new_samples)
        return reservoir

    # Trim existing reservoir if it already exceeds the cap.
    if len(reservoir) > cap:
        reservoir = rng.sample(reservoir, cap)

    total_seen = len(reservoir)
    for sample in new_samples:
        total_seen += 1
        if len(reservoir) < cap:
            reservoir.append(sample)
            continue
        idx = rng.randrange(total_seen)
        if idx < cap:
            reservoir[idx] = sample
    return reservoir


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
    env = RoutingEnv(RoutingEnvConfig(frontier_size=FRONTIER_SIZE))
    initial_layout = _sabre_initial_layout(circuit.circuit, coupling_map, seed=seed)
    state = env.reset(
        circuit.circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
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


def rollout_policy_traces(
    model: SwapPolicy,
    circuit: QasmCircuit,
    coupling_map: CouplingMap,
    *,
    seed: int,
    graph_id: str,
    teacher: TeacherPolicy,
    hardware_model: HardwareModel | None,
    hardware_seed: int | None,
    dagger_round: int,
) -> list[TraceSample]:
    """Roll out a learned policy (unguarded) while labelling with teacher scores."""

    env = RoutingEnv(RoutingEnvConfig(frontier_size=FRONTIER_SIZE))
    initial_layout = _sabre_initial_layout(circuit.circuit, coupling_map, seed=seed)
    state = env.reset(
        circuit.circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
    graph = nx.Graph(list(coupling_map.get_edges()))
    hardware = env.hardware_model
    teacher.begin_episode(graph)

    samples: list[TraceSample] = []
    step = 0
    while not state.done:
        cand_feats = candidate_features(state, graph, hardware)
        state_feats = state_features(state, graph)
        teacher_scores = torch.tensor(
            teacher.score_candidates(state, graph, hardware), dtype=torch.float32
        )
        mask = torch.tensor(state.action_mask, dtype=torch.bool)
        finite_scores = teacher_scores.masked_fill(~mask, torch.tensor(float("inf")))
        teacher_action = int(torch.argmin(finite_scores).item())

        policy_feats = torch.cat(
            [cand_feats, state_feats.unsqueeze(0).repeat(cand_feats.shape[0], 1)], dim=1
        )
        policy_feats = _match_feature_dim(policy_feats, model.net[0].in_features)  # type: ignore[index]
        logits = _scores_to_logits(model(policy_feats), model)
        logits = logits.masked_fill(~mask, -1e9)
        if not mask.any():
            policy_action = teacher_action
        else:
            policy_action = int(torch.argmax(logits).item())

        samples.append(
            TraceSample(
                candidate_features=cand_feats.cpu(),
                state_features=state_feats.cpu(),
                teacher_scores=teacher_scores.cpu(),
                action_mask=mask.cpu(),
                label=teacher_action,
                meta={
                    "circuit_id": circuit.circuit_id,
                    "graph_id": graph_id,
                    "step": step,
                    "candidate_swaps": list(state.candidate_swaps),
                    "hardware_seed": hardware_seed,
                    "policy_action": policy_action,
                    "teacher_action": teacher_action,
                    "feature_version": FEATURE_VERSION,
                    "dagger_round": dagger_round,
                },
            )
        )

        swap_edge = state.candidate_swaps[policy_action]
        state, _, _, _ = env.step(policy_action)
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


def collect_dagger_dataset(
    model: SwapPolicy,
    suite: dict[str, list[QasmCircuit]],
    coupling_maps: dict[str, CouplingMap],
    *,
    seed: int,
    hardware_models: dict[str, list[tuple[int, HardwareModel]]],
    teacher_weights: TeacherWeights,
    dagger_round: int,
    rng: random.Random,
) -> list[TraceSample]:
    """Collect DAgger traces from policy rollouts labelled by the teacher."""

    teacher = TeacherPolicy(weights=teacher_weights)
    all_samples: list[TraceSample] = []
    for graph_id, circuits in suite.items():
        cmap = coupling_maps[graph_id]
        hw_list = hardware_models.get(graph_id, [(None, None)])
        circuits_iter = list(circuits)
        rng.shuffle(circuits_iter)
        for hw_seed, hw_model in hw_list:
            for circuit in circuits_iter:
                all_samples.extend(
                    rollout_policy_traces(
                        model,
                        circuit,
                        cmap,
                        seed=seed,
                        graph_id=graph_id,
                        teacher=teacher,
                        hardware_model=hw_model,
                        hardware_seed=hw_seed,
                        dagger_round=dagger_round,
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
    tau_start: float = 1.0,
    tau_end: float = 0.3,
    tau: float | None = None,
    patience: int = 5,
    val_split: float = 0.1,
    seed: int = 0,
    hard_weight: float = 0.05,
    max_train_samples: int | None = None,
    hidden_dim: int = 192,
    top_k: int | None = None,
    mse_weight: float = 1.0,
    rank_weight: float = 0.5,
    score_mode: str = "min",
) -> tuple[SwapPolicy, list[dict]]:
    if tau is not None:
        tau_start = tau_end = tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = (
        samples[0].candidate_features.shape[1] + samples[0].state_features.shape[0]
        if samples
        else None
    )
    model = SwapPolicy(
        feature_dim=feature_dim or SwapPolicy().net[0].in_features,
        hidden_dim=hidden_dim,
        score_mode=score_mode,
    ).to(device)
    model.score_mode = score_mode
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    rng = random.Random(seed)
    torch.manual_seed(seed)
    hard_wt = max(0.0, min(1.0, hard_weight))
    mse_wt = max(0.0, mse_weight)
    rank_wt = max(0.0, rank_weight)

    def _tau_for_epoch(epoch: int) -> float:
        if epochs <= 1:
            return tau_end
        progress = epoch / max(1, epochs - 1)
        return tau_start + (tau_end - tau_start) * progress

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

    def _compute_losses(
        sample: TraceSample, tau_value: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        mask = sample.action_mask.to(device)
        if not mask.any():
            return None
        feats = _assemble_features(sample)
        pred_scores = model(feats)
        teacher_scores = sample.teacher_scores.to(device)

        pred_valid = pred_scores[mask]
        teacher_valid = teacher_scores[mask]

        if top_k is not None and top_k > 0 and teacher_valid.numel() > top_k:
            # Use teacher ordering to select top-k actions and keep tensors aligned.
            _, top_idx = torch.topk(-teacher_valid, k=top_k)
            teacher_valid = teacher_valid[top_idx]
            pred_valid = pred_valid[top_idx]
        if teacher_valid.numel() == 0:
            return None

        # Center scores to stabilise regression and ranking.
        teacher_center = teacher_valid - teacher_valid.mean()
        pred_center = pred_valid - pred_valid.mean()

        mse_loss = F.mse_loss(pred_center, teacher_center)
        temp = max(tau_value, 1e-6)
        teacher_probs = torch.softmax(-teacher_center / temp, dim=-1)
        pred_probs = torch.softmax(-pred_center / temp, dim=-1)
        rank_loss = F.kl_div(
            torch.log(pred_probs + 1e-8), teacher_probs, reduction="batchmean", log_target=False
        )

        hard_loss = torch.tensor(0.0, device=device)
        if hard_wt > 0:
            best_idx = torch.argmin(teacher_center)
            hard_loss = F.cross_entropy((-pred_center).unsqueeze(0), best_idx.unsqueeze(0))
        return mse_loss, rank_loss, hard_loss

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
        mse_total = 0.0
        rank_total = 0.0
        hard_total = 0.0
        seen = 0
        epoch_samples = train_samples
        if max_train_samples is not None and len(epoch_samples) > max_train_samples:
            epoch_samples = epoch_samples[:max_train_samples]
        tau_value = _tau_for_epoch(epoch)
        for sample in epoch_samples:
            losses = _compute_losses(sample, tau_value)
            if losses is None:
                continue
            mse_loss, rank_loss, hard_loss = losses
            loss = mse_wt * mse_loss + rank_wt * rank_loss + hard_wt * hard_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            mse_total += float(mse_loss.item())
            rank_total += float(rank_loss.item())
            hard_total += float(hard_loss.item())
            seen += 1
        train_loss = total_loss / max(1, seen)

        # Validation
        with torch.no_grad():
            val_total = 0.0
            val_seen = 0
            for sample in val_samples:
                losses = _compute_losses(sample, tau_value)
                if losses is None:
                    continue
                mse_loss, rank_loss, hard_loss = losses
                loss = mse_wt * mse_loss + rank_wt * rank_loss + hard_wt * hard_loss
                val_total += float(loss.item())
                val_seen += 1
            val_loss = val_total / max(1, val_seen)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tau": tau_value,
                "mse": mse_total / max(1, seen),
                "rank": rank_total / max(1, seen),
                "hard": hard_total / max(1, seen),
            }
        )

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"[il] epoch {epoch + 1}/{epochs} tau={tau_value:.3f} "
                f"train={train_loss:.4f} val={val_loss:.4f}"
            )

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
    tau_start: float = 1.0,
    tau_end: float = 0.3,
    tau: float | None = None,
    top_k: int | None = None,
    hard_weight: float,
    mse_weight: float,
    rank_weight: float,
    max_train_samples: int | None = None,
    hidden_dim: int = 192,
    score_mode: str = "min",
) -> None:
    if tau is not None:
        tau_start = tau_end = tau

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.net[0].in_features,
            "epochs": epochs,
            "seed": seed,
            "loss_history": loss_history,
            "use_value_head": bool(model.value_head is not None),
            "tau_start": tau_start,
            "tau_end": tau_end,
            "top_k": top_k,
            "hard_weight": hard_weight,
            "mse_weight": mse_weight,
            "rank_weight": rank_weight,
            "feature_version": FEATURE_VERSION,
            "max_train_samples": max_train_samples,
            "hidden_dim": hidden_dim,
            "score_mode": score_mode,
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
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--tau-start", type=float, default=1.0, help="Softmax temperature at epoch 0."
    )
    parser.add_argument(
        "--tau-end", type=float, default=0.3, help="Softmax temperature at final epoch."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Deprecated fixed temperature; overrides tau-start/tau-end when set.",
    )
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=0.05,
        help="Weight for optional hard-label cross-entropy term (0-1).",
    )
    parser.add_argument(
        "--mse-weight",
        type=float,
        default=1.0,
        help="Weight for teacher score regression term.",
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.5,
        help="Weight for listwise ranking loss.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Whether lower (min) or higher (max) model outputs are preferred.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=192,
        help="Hidden dimension for the IL MLP (default 192 for stronger capacity).",
    )
    parser.add_argument(
        "--soft-top-k", type=int, default=8, help="Top-k teacher actions to distil."
    )
    parser.add_argument("--patience", type=int, default=12)
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
        default=220,
        help="Number of synthetic circuit seeds per graph.",
    )
    parser.add_argument(
        "--num-hardware-seeds",
        type=int,
        default=24,
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
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=120_000,
        help="Cap on number of training samples per epoch (after shuffling).",
    )
    parser.add_argument(
        "--dagger-rounds",
        type=int,
        default=5,
        help="Number of DAgger iterations after initial supervised fit.",
    )
    parser.add_argument(
        "--dagger-cap",
        type=int,
        default=450_000,
        help="Reservoir sampling cap for IL dataset size (0 = unlimited).",
    )
    parser.add_argument(
        "--dagger-seed",
        type=int,
        default=37,
        help="Random seed for DAgger shuffling/reservoir updates.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_start = args.tau if args.tau is not None else args.tau_start
    tau_end = args.tau if args.tau is not None else args.tau_end
    top_k = args.soft_top_k if args.soft_top_k and args.soft_top_k > 0 else None
    rng = random.Random(args.dagger_seed)

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    if args.fast:
        args.num_circuit_seeds = max(4, args.num_circuit_seeds // 4)
        args.num_hardware_seeds = max(3, args.num_hardware_seeds // 3)
        args.epochs = max(5, args.epochs // 3)
        args.patience = min(args.patience, 3)
        args.dagger_rounds = max(1, min(args.dagger_rounds, 2))
        args.dagger_cap = min(args.dagger_cap, 200_000) if args.dagger_cap else args.dagger_cap

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
    dagger_suite: dict[str, list[QasmCircuit]] = _augment_with_pressure(
        {}, graphs, args.pressure_seed
    )
    hardware_models = _build_hardware_models(
        coupling_maps, hardware_seeds, profile=args.hardware_profile
    )

    def _needs_regen(meta: dict) -> bool:
        if not meta:
            return True
        return (
            meta.get("dataset_version") != DATASET_VERSION
            or meta.get("feature_version") != FEATURE_VERSION
            or meta.get("frontier_size") != FRONTIER_SIZE
            or meta.get("graphs") != graphs
            or meta.get("num_circuit_seeds") != len(circuit_seeds)
            or meta.get("num_hardware_seeds") != len(hardware_seeds)
            or meta.get("hardware_profile") != args.hardware_profile
            or meta.get("pressure_seed") != args.pressure_seed
            or meta.get("circuit_seed_base") != args.circuit_seed_base
            or meta.get("hardware_seed_base") != args.hardware_seed_base
            or meta.get("score_mode") != args.score_mode
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
                "dataset_version": DATASET_VERSION,
                "teacher_weights": asdict(TeacherWeights()),
                "seed": args.seed,
                "hardware_profile": args.hardware_profile,
                "circuit_seed_base": args.circuit_seed_base,
                "hardware_seed_base": args.hardware_seed_base,
                "pressure_seed": args.pressure_seed,
                "pressure_circuits": [
                    c.circuit_id for c in pressure_suite(seed=args.pressure_seed)
                ],
                "frontier_size": FRONTIER_SIZE,
                "dagger_rounds": 0,
                "dagger_cap": args.dagger_cap,
                "dagger_seed": args.dagger_seed,
                "score_mode": args.score_mode,
                "mse_weight": args.mse_weight,
                "rank_weight": args.rank_weight,
                "hard_weight": args.hard_weight,
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
                    "dataset_version": DATASET_VERSION,
                    "teacher_weights": asdict(TeacherWeights()),
                    "seed": args.seed,
                    "hardware_profile": args.hardware_profile,
                    "circuit_seed_base": args.circuit_seed_base,
                    "hardware_seed_base": args.hardware_seed_base,
                    "pressure_seed": args.pressure_seed,
                    "pressure_circuits": [
                        c.circuit_id for c in pressure_suite(seed=args.pressure_seed)
                    ],
                    "frontier_size": FRONTIER_SIZE,
                    "dagger_rounds": 0,
                    "dagger_cap": args.dagger_cap,
                    "dagger_seed": args.dagger_seed,
                    "score_mode": args.score_mode,
                    "mse_weight": args.mse_weight,
                    "rank_weight": args.rank_weight,
                    "hard_weight": args.hard_weight,
                },
            )
        else:
            dataset_meta.setdefault("dagger_rounds", 0)
            dataset_meta["dagger_cap"] = args.dagger_cap
            dataset_meta["dagger_seed"] = args.dagger_seed
            dataset_meta["score_mode"] = args.score_mode
            dataset_meta.setdefault("frontier_size", FRONTIER_SIZE)
            dataset_meta.setdefault("mse_weight", args.mse_weight)
            dataset_meta.setdefault("rank_weight", args.rank_weight)
            dataset_meta.setdefault("hard_weight", args.hard_weight)

    # Enforce reservoir cap on pre-existing data.
    samples = _reservoir_update(samples, [], cap=args.dagger_cap, rng=rng)
    prev_num = dataset_meta.get("num_samples", len(samples))
    dataset_meta["num_samples"] = len(samples)
    if (
        len(samples) != prev_num
        and dataset_meta.get("path")
        and Path(dataset_meta["path"]).exists()
    ):
        # Refresh dataset on disk if we trimmed to the cap.
        dataset_meta = save_dataset(samples, dataset_path, metadata=dataset_meta)

    history_all: list[dict] = []

    def _train_round(round_idx: int, current_samples: list[TraceSample]) -> SwapPolicy:
        model, hist = train_model(
            current_samples,
            epochs=args.epochs,
            lr=args.lr,
            tau_start=tau_start,
            tau_end=tau_end,
            top_k=top_k,
            patience=args.patience,
            val_split=args.val_split,
            seed=args.seed + round_idx,
            hard_weight=args.hard_weight,
            max_train_samples=args.max_train_samples,
            hidden_dim=args.hidden_dim,
            mse_weight=args.mse_weight,
            rank_weight=args.rank_weight,
            score_mode=args.score_mode,
        )
        history_all.extend([{**rec, "round": round_idx} for rec in hist])
        return model

    model = _train_round(0, samples)

    for round_idx in range(1, args.dagger_rounds + 1):
        dagger_samples = collect_dagger_dataset(
            model,
            dagger_suite,
            coupling_maps,
            seed=args.seed + round_idx,
            hardware_models=hardware_models,
            teacher_weights=TeacherWeights(),
            dagger_round=round_idx,
            rng=rng,
        )
        samples = _reservoir_update(samples, dagger_samples, cap=args.dagger_cap, rng=rng)
        dataset_meta.update(
            {
                "dagger_rounds": round_idx,
                "dagger_cap": args.dagger_cap,
                "dagger_seed": args.dagger_seed,
                "dagger_added": dataset_meta.get("dagger_added", 0) + len(dagger_samples),
                "score_mode": args.score_mode,
                "mse_weight": args.mse_weight,
                "rank_weight": args.rank_weight,
                "hard_weight": args.hard_weight,
                "dataset_version": DATASET_VERSION,
            }
        )
        dataset_meta["num_samples"] = len(samples)
        dataset_meta = save_dataset(samples, dataset_path, metadata=dataset_meta)
        model = _train_round(round_idx, samples)

    ckpt_path = out_dir / "checkpoints" / "il_soft.pt"
    save_checkpoint(
        model,
        ckpt_path,
        epochs=args.epochs,
        seed=args.seed,
        loss_history=history_all,
        tau_start=tau_start,
        tau_end=tau_end,
        top_k=top_k,
        hard_weight=args.hard_weight,
        mse_weight=args.mse_weight,
        rank_weight=args.rank_weight,
        max_train_samples=args.max_train_samples,
        hidden_dim=args.hidden_dim,
        score_mode=args.score_mode,
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
                "il_tau": tau_end,
                "il_tau_start": tau_start,
                "il_tau_end": tau_end,
                "il_top_k": top_k,
                "il_hard_weight": args.hard_weight,
                "il_mse_weight": args.mse_weight,
                "il_rank_weight": args.rank_weight,
                "il_score_mode": args.score_mode,
                "il_graphs": graphs,
                "il_circuit_seeds": circuit_seeds,
                "il_hardware_seeds": hardware_seeds,
                "il_max_train_samples": args.max_train_samples,
                "il_hidden_dim": args.hidden_dim,
                "il_dataset_version": DATASET_VERSION,
                "il_feature_version": FEATURE_VERSION,
                "il_frontier_size": FRONTIER_SIZE,
                "il_history_path": str(out_dir / "training" / "il_soft.json"),
                "il_dagger_rounds": args.dagger_rounds,
                "il_dagger_cap": args.dagger_cap,
                "il_dagger_seed": args.dagger_seed,
            },
        )

    history_path = out_dir / "training" / "il_soft.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_all, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
