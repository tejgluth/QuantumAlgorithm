"""Reinforce fine-tuning on RoutingEnv."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import networkx as nx
import torch
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit, load_suite
from quantum_routing_rl.env.reward import RewardConfig
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.models.policy import (
    SwapPolicy,
    candidate_features,
    load_swap_policy,
    route_with_policy,
)


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
    if hasattr(map_obj, "size"):
        try:
            return int(map_obj.size())
        except Exception:
            pass
    edges = map_obj.get_edges()
    return max(max(edge) for edge in edges) + 1


def _filter_coupling_maps(
    coupling_maps: dict[str, CouplingMap], circuit: QasmCircuit
) -> Iterable[tuple[str, CouplingMap]]:
    for name, cmap in coupling_maps.items():
        if circuit.circuit.num_qubits <= _coupling_size(cmap):
            yield name, cmap


def _trainable_circuits(
    suite: list[QasmCircuit], coupling_maps: dict[str, CouplingMap]
) -> list[QasmCircuit]:
    eligible = []
    for circuit in suite:
        if any(True for _ in _filter_coupling_maps(coupling_maps, circuit)):
            eligible.append(circuit)
    if not eligible:
        msg = "No circuits fit provided coupling maps."
        raise ValueError(msg)
    return eligible


def discount_rewards(rewards: list[float], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def run_episode(
    model: SwapPolicy,
    circuit: QasmCircuit,
    coupling_map: CouplingMap,
    reward_config: RewardConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[float], list[torch.Tensor]]:
    env = RoutingEnv(RoutingEnvConfig(reward=reward_config))
    state = env.reset(circuit.circuit, coupling_map, seed=seed)
    graph = nx.Graph(list(coupling_map.get_edges()))

    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []

    while not state.done:
        feats = candidate_features(state, graph).to(device)
        logits = model(feats)
        mask = torch.tensor(state.action_mask, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, done, _info = env.step(int(action.item()))
        rewards.append(float(reward))
    return rewards, log_probs


def train_rl(
    circuits: list[QasmCircuit],
    coupling_maps: dict[str, CouplingMap],
    *,
    episodes: int,
    seed: int,
    lr: float,
    gamma: float,
    reward_config: RewardConfig,
    il_checkpoint: Path | None,
) -> tuple[SwapPolicy, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if il_checkpoint and Path(il_checkpoint).exists():
        model = load_swap_policy(il_checkpoint, device=device)
    else:
        model = SwapPolicy()
        model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    baseline = 0.0
    history: list[dict] = []
    trainable = _trainable_circuits(circuits, coupling_maps)

    for ep in range(episodes):
        circuit = trainable[ep % len(trainable)]
        graph_id, coupling_map = next(iter(_filter_coupling_maps(coupling_maps, circuit)))
        rewards, log_probs = run_episode(
            model,
            circuit,
            coupling_map,
            reward_config,
            seed=seed + ep,
            device=device,
        )
        if not rewards:
            continue

        returns = torch.tensor(discount_rewards(rewards, gamma), device=device)
        baseline = (
            returns.mean().item() if ep == 0 else 0.9 * baseline + 0.1 * returns.mean().item()
        )
        advantages = returns - baseline
        log_prob_tensor = torch.stack(log_probs)
        loss = -(log_prob_tensor * advantages).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "episode": ep,
                "reward": float(sum(rewards)),
                "length": len(rewards),
                "loss": float(loss.item()),
                "graph_id": graph_id,
                "circuit_id": circuit.circuit_id,
            }
        )

    model.eval()
    return model, history


def save_checkpoint(
    model: SwapPolicy,
    path: Path,
    *,
    history: list[dict],
    episodes: int,
    seed: int,
    gamma: float,
    reward_config: RewardConfig,
    il_checkpoint: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.net[0].in_features,
            "episodes": episodes,
            "seed": seed,
            "gamma": gamma,
            "history": history,
            "il_checkpoint": str(il_checkpoint) if il_checkpoint else None,
            "reward_config": reward_config.__dict__,
        },
        path,
    )


def _update_metadata(out_dir: Path, updates: dict) -> None:
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            data = {}
    else:
        data = {}
    data.update(updates)
    meta_path.write_text(json.dumps(data, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dev", "full"], default="dev")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.05)
    parser.add_argument("--noise-weight", type=float, default=0.01)
    parser.add_argument(
        "--il-checkpoint",
        type=Path,
        help="Path to IL checkpoint to initialise policy (defaults to artifacts/checkpoints/il.pt).",
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Path to QASMBench root (defaults to env QASMBENCH_ROOT or tests fixtures).",
    )
    parser.add_argument("--dev-limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    qasm_root = Path(args.qasm_root or _default_qasm_root()).expanduser()

    suite = load_suite(qasm_root, suite=args.suite, dev_limit=args.dev_limit)
    coupling_maps = _build_coupling_maps()

    il_ckpt = args.il_checkpoint
    if il_ckpt is None:
        default_il = out_dir / "checkpoints" / "il.pt"
        if default_il.exists():
            il_ckpt = default_il

    reward_cfg = RewardConfig(
        swap_weight=args.swap_weight,
        depth_weight=args.depth_weight,
        noise_weight=args.noise_weight,
    )
    model, history = train_rl(
        suite,
        coupling_maps,
        episodes=args.episodes,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        reward_config=reward_cfg,
        il_checkpoint=il_ckpt,
    )

    ckpt_path = out_dir / "checkpoints" / "rl.pt"
    save_checkpoint(
        model,
        ckpt_path,
        history=history,
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        reward_config=reward_cfg,
        il_checkpoint=il_ckpt,
    )

    # Smoke evaluation to ensure the policy can run.
    trainable = _trainable_circuits(suite, coupling_maps)
    circuit = trainable[0]
    graph_id, coupling_map = next(iter(_filter_coupling_maps(coupling_maps, circuit)))
    _ = route_with_policy(model, circuit.circuit, coupling_map, name="rl_policy_smoke")

    history_path = out_dir / "training" / "rl_train.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))

    _update_metadata(
        out_dir,
        {
            "rl_checkpoint": str(ckpt_path),
            "rl_episodes": args.episodes,
            "rl_seed": args.seed,
            "rl_lr": args.lr,
            "rl_gamma": args.gamma,
            "rl_reward": reward_cfg.__dict__,
            "rl_graph": graph_id,
            "rl_qasm_root": str(qasm_root),
            "qiskit_version": qiskit_version,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
