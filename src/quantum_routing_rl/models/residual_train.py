"""Supervised training script for the ResidualTopK scorer."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import torch
import torch.nn.functional as F
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.synthetic_generator import pressure_suite
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig, _swapped_layout
from quantum_routing_rl.env.state import RoutingState
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.policy import (
    SwapPolicy,
    candidate_features,
    logical_a,
    logical_b,
    load_swap_policy,
    model_features,
    _match_feature_dim,
)
from quantum_routing_rl.models.residual_policy import ResidualScorer, _candidate_context
from quantum_routing_rl.models.teacher import TeacherPolicy, _sabre_initial_layout


# --------------------------------------------------------------------------- CLI
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="Output root dir.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=6,
        metavar=("a", "b", "c", "d", "e", "f"),
        default=[1.0, 0.4, 1.0, 0.4, 0.2, 0.2],
        help="Utility weights (a=success, b=dist, c=error, d=time, e=duration, f=crosstalk).",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="teacher",
        choices=["teacher", "il_soft"],
        help="Rollout source for states (teacher or il_soft checkpoint).",
    )
    parser.add_argument(
        "--max-circuits",
        type=int,
        default=6,
        help="Limit number of training circuits (keeps runtime bounded).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Per-episode step cap during data collection.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- helpers
def _coupling_maps() -> dict[str, CouplingMap]:
    return {
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


def _hardware_models(
    coupling_maps: dict[str, CouplingMap],
    seeds: Iterable[int],
) -> dict[str, list[tuple[int, HardwareModel]]]:
    models: dict[str, list[tuple[int, HardwareModel]]] = {}
    for graph_id, cmap in coupling_maps.items():
        edges = [tuple(edge) for edge in cmap.get_edges()]
        graph = nx.Graph()
        graph.add_edges_from(edges)
        models[graph_id] = [
            (seed, HardwareModel.synthetic(graph, seed=seed, profile="realistic")) for seed in seeds
        ]
    return models


def _topk_indices(scores: torch.Tensor, mask: torch.Tensor, k: int) -> list[int]:
    scores = scores.clone()
    scores = scores.masked_fill(~mask, float("inf"))
    if not torch.isfinite(scores).any():
        return []
    order = torch.argsort(scores)
    idxs: list[int] = []
    for idx in order.tolist():
        if not mask[idx]:
            continue
        idxs.append(idx)
        if len(idxs) >= k:
            break
    return idxs


def _frontier_dist(state: RoutingState, graph: nx.Graph) -> float:
    try:
        return (
            float(nx.shortest_path_length(graph, *state.frontier_phys))
            if state.frontier_phys
            else 0.0
        )
    except Exception:
        return float(graph.number_of_nodes())


def _delta_dist_improvement(state: RoutingState, graph: nx.Graph, swap: tuple[int, int]) -> float:
    if not state.frontier:
        return 0.0
    current = _frontier_dist(state, graph)
    layout_after = _swapped_layout(state.layout, *swap)
    try:
        new = float(
            nx.shortest_path_length(
                graph, layout_after[logical_a(state)], layout_after[logical_b(state)]
            )
        )
    except Exception:
        new = float(graph.number_of_nodes())
    return current - new


def _lookahead_distance(state: RoutingState, graph: nx.Graph, layout: dict[int, int]) -> float:
    total = 0.0
    for gate in state.lookahead[:4]:
        try:
            total += float(nx.shortest_path_length(graph, layout[gate[0]], layout[gate[1]]))
        except Exception:
            total += graph.number_of_nodes()
    return total


def _success_proxy(
    state: RoutingState, graph: nx.Graph, hardware: HardwareModel | None, layout: dict[int, int]
) -> float:
    if not state.frontier:
        return 0.0
    phys = (layout[logical_a(state)], layout[logical_b(state)])
    if not graph.has_edge(*phys):
        return -0.5
    if hardware is None:
        return 0.1
    error, _ = hardware.edge_error_and_duration(phys[0], phys[1], directed=True)
    error = max(error, 1e-4)
    return float(torch.log(torch.tensor(1.0 - min(error, 0.9))).item())


def _crosstalk_proxy(
    graph: nx.Graph, hardware: HardwareModel | None, swap: tuple[int, int]
) -> float:
    if hardware is None or hardware.crosstalk_factor <= 0:
        return 0.0
    max_degree = max((deg for _, deg in graph.degree()), default=1)
    deg_u = graph.degree[swap[0]] if swap[0] in graph else 0
    deg_v = graph.degree[swap[1]] if swap[1] in graph else 0
    return hardware.crosstalk_factor * ((deg_u + deg_v) / max(1, max_degree))


def _utility_components(
    state: RoutingState,
    graph: nx.Graph,
    hardware: HardwareModel | None,
    swap: tuple[int, int],
) -> dict[str, float]:
    layout_after = _swapped_layout(state.layout, *swap)
    delta_dist = _delta_dist_improvement(state, graph, swap)
    lookahead_before = _lookahead_distance(state, graph, state.layout)
    lookahead_after = _lookahead_distance(state, graph, layout_after)
    delta_duration_proxy = lookahead_after - lookahead_before
    success_before = _success_proxy(state, graph, hardware, state.layout)
    success_after = _success_proxy(state, graph, hardware, layout_after)
    delta_success = success_after - success_before
    edge_error, edge_time_ns = (0.0, 0.0)
    if hardware is not None:
        edge_error, edge_time_ns = hardware.edge_error_and_duration(swap[0], swap[1], directed=True)
    crosstalk = _crosstalk_proxy(graph, hardware, swap)
    return {
        "delta_dist_improve": float(delta_dist),
        "edge_error_cost": float(edge_error),
        "edge_time_cost": float(edge_time_ns / 1000.0),
        "delta_duration_proxy": float(delta_duration_proxy / max(1.0, graph.number_of_nodes())),
        "delta_success_proxy": float(delta_success),
        "crosstalk_risk": float(crosstalk),
    }


def _compute_utility(components: dict[str, float], weights: Sequence[float]) -> float:
    a, b, c, d, e, f = weights
    return (
        a * components["delta_success_proxy"]
        + b * components["delta_dist_improve"]
        - c * components["edge_error_cost"]
        - d * components["edge_time_cost"]
        - e * components["delta_duration_proxy"]
        - f * components["crosstalk_risk"]
    )


@dataclass
class ResidualSample:
    cand_features: torch.Tensor
    context: torch.Tensor
    utilities: torch.Tensor


def _collect_samples_for_circuit(
    circuit,
    coupling_map: CouplingMap,
    *,
    seed: int,
    graph_id: str,
    teacher: TeacherPolicy,
    top_k: int,
    weights: Sequence[float],
    hardware_model: HardwareModel | None,
    max_steps: int,
    rollout_policy: SwapPolicy | None,
) -> list[ResidualSample]:
    env = RoutingEnv(RoutingEnvConfig(frontier_size=4))
    initial_layout = _sabre_initial_layout(circuit, coupling_map, seed=seed)
    state = env.reset(
        circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
    graph = nx.Graph(list(coupling_map.get_edges()))
    hardware = env.hardware_model
    teacher.begin_episode(graph)
    samples: list[ResidualSample] = []
    steps = 0
    while not state.done and steps < max_steps:
        cand_feats = candidate_features(state, graph, hardware)
        teacher_scores = torch.tensor(teacher.score_candidates(state, graph, hardware))
        mask = torch.tensor(state.action_mask, dtype=torch.bool)
        topk_idx = _topk_indices(teacher_scores, mask, top_k)
        if topk_idx:
            utilities: list[float] = []
            for idx in topk_idx:
                swap = state.candidate_swaps[idx]
                comps = _utility_components(state, graph, hardware, swap)
                utilities.append(_compute_utility(comps, weights))
            context = _candidate_context(state, graph, hardware, include_hardware=True)
            samples.append(
                ResidualSample(
                    cand_features=cand_feats[topk_idx],
                    context=context,
                    utilities=torch.tensor(utilities, dtype=torch.float32),
                )
            )

        # Rollout action for next state (teacher by default; optionally il_soft).
        if rollout_policy is not None and mask.any():
            feats = model_features(state, graph, hardware)
            feats = _match_feature_dim(feats, rollout_policy.net[0].in_features)
            logits = rollout_policy(feats)
            logits = logits.masked_fill(~mask, -1e9)
            action = int(torch.argmax(logits).item())
        else:
            action = teacher.select_action(state, graph, hardware)
        swap_edge = state.candidate_swaps[action]
        state, _, _, _ = env.step(action)
        teacher.update_after_action(swap_edge, state.step_count)
        steps += 1
    return samples


# ---------------------------------------------------------------------- train
def train_residual(
    samples: list[ResidualSample],
    *,
    epochs: int,
    hidden_dim: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[ResidualScorer, list[dict]]:
    if not samples:
        msg = "No samples available for training."
        raise RuntimeError(msg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = int(samples[0].cand_features.shape[1])
    context_dim = int(samples[0].context.numel())
    scorer = ResidualScorer(feature_dim=feature_dim, hidden_dim=hidden_dim, context_dim=context_dim)
    scorer.to(device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    history: list[dict] = []
    torch.manual_seed(seed)
    random.seed(seed)

    for epoch in range(epochs):
        total_loss = 0.0
        for sample in samples:
            cand = sample.cand_features.to(device)
            context = sample.context.to(device)
            utilities = sample.utilities.to(device)
            logits = scorer(cand, context)
            log_probs = F.log_softmax(logits, dim=-1)
            target = torch.argmax(utilities)
            ce_loss = F.nll_loss(log_probs.unsqueeze(0), target.unsqueeze(0))
            # Pairwise margin encouraging the best utility to exceed others.
            margin = 0.0
            for j in range(utilities.shape[0]):
                if j == int(target):
                    continue
                margin += F.relu(0.1 - (logits[int(target)] - logits[j]))
            margin = margin / max(1, utilities.shape[0] - 1)
            loss = ce_loss + margin
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg = total_loss / max(1, len(samples))
        history.append({"epoch": epoch, "loss": avg})
    scorer.eval()
    return scorer, history


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    train_dir = out_dir / "training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    coupling_maps = _coupling_maps()
    hardware_models = _hardware_models(coupling_maps, seeds=[args.seed + i for i in range(2)])
    circuits = pressure_suite(seed=args.seed)[: args.max_circuits]
    rollout_policy: SwapPolicy | None = None
    if args.data_source == "il_soft":
        ckpt = out_dir / "checkpoints" / "il_soft.pt"
        if ckpt.exists():
            rollout_policy = load_swap_policy(ckpt)

    teacher = TeacherPolicy()
    samples: list[ResidualSample] = []
    for circuit in circuits:
        graph_id = {8: "ring_8", 9: "grid_3x3", 15: "heavy_hex_15"}.get(circuit.circuit.num_qubits)
        if graph_id is None or graph_id not in coupling_maps:
            continue
        cmap = coupling_maps[graph_id]
        hw_list = hardware_models.get(graph_id, [(None, None)])
        rng.shuffle(hw_list)
        for hw_seed, hw_model in hw_list:
            samples.extend(
                _collect_samples_for_circuit(
                    circuit.circuit,
                    cmap,
                    seed=args.seed,
                    graph_id=graph_id,
                    teacher=teacher,
                    top_k=args.top_k,
                    weights=args.weights,
                    hardware_model=hw_model,
                    max_steps=args.max_steps,
                    rollout_policy=rollout_policy,
                )
            )

    scorer, history = train_residual(
        samples,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    ckpt_path = ckpt_dir / "residual_topk.pt"
    torch.save(
        {
            "scorer_state": scorer.state_dict(),
            "feature_dim": scorer.feature_dim,
            "hidden_dim": scorer.hidden_dim,
            "context_dim": scorer.context_dim,
            "top_k": args.top_k,
            "epochs": args.epochs,
            "seed": args.seed,
            "weights": args.weights,
        },
        ckpt_path,
    )
    log = {
        "epochs": args.epochs,
        "seed": args.seed,
        "top_k": args.top_k,
        "hidden_dim": args.hidden_dim,
        "num_samples": len(samples),
        "weights": args.weights,
        "history": history,
    }
    (train_dir / "residual_train.json").write_text(json.dumps(log, indent=2))
    print(f"[residual-train] wrote checkpoint to {ckpt_path}")
    print(f"[residual-train] samples={len(samples)} epochs={args.epochs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
