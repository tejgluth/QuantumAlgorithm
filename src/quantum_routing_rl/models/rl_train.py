"""PPO fine-tuning on RoutingEnv with curriculum and regression gating."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import networkx as nx
from qiskit import __version__ as qiskit_version
from qiskit.transpiler import CouplingMap

from quantum_routing_rl.eval.metrics import compute_metrics
from quantum_routing_rl.eval.run_eval import (
    _build_coupling_maps,
    _build_hardware_models,
    _load_benchmarks,
)
from quantum_routing_rl.env.reward import RewardConfig
from quantum_routing_rl.env.routing_env import RoutingEnv, RoutingEnvConfig
from quantum_routing_rl.hardware.model import HardwareModel
from quantum_routing_rl.models.policy import (
    SwapPolicy,
    _match_feature_dim,
    load_swap_policy,
    model_features,
    route_with_policy,
)
from quantum_routing_rl.models.teacher import (
    TeacherPolicy,
    _sabre_initial_layout,
    route_with_teacher,
)


@dataclass
class Rollout:
    features: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    rewards: list[float]
    values: list[torch.Tensor]
    masks: list[torch.Tensor]
    action_masks: list[torch.Tensor]
    entropies: list[torch.Tensor]
    infos: list[dict]
    episode_metrics: dict


def _default_qasm_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "qasmbench"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prep_policy(il_checkpoint: Path | None, device: torch.device) -> SwapPolicy:
    if il_checkpoint and il_checkpoint.exists():
        model = load_swap_policy(il_checkpoint, device=device)
        hidden_dim = model.net[0].weight.shape[0]  # type: ignore[attr-defined]
        if model.value_head is None:
            new_model = SwapPolicy(
                feature_dim=model.net[0].in_features,  # type: ignore[index]
                hidden_dim=hidden_dim,
                use_value_head=True,
            )
            new_model.net.load_state_dict(model.net.state_dict())
            model = new_model
    else:
        model = SwapPolicy(hidden_dim=256, use_value_head=True)
    model.to(device)
    model.train()
    return model


def run_episode(
    model: SwapPolicy,
    circuit,
    coupling_map: CouplingMap,
    reward_config: RewardConfig,
    *,
    seed: int,
    device: torch.device,
    hardware_model: HardwareModel | None = None,
) -> Rollout:
    env = RoutingEnv(RoutingEnvConfig(reward=reward_config))
    initial_layout = _sabre_initial_layout(circuit, coupling_map, seed=seed)
    state = env.reset(
        circuit,
        coupling_map,
        seed=seed,
        hardware_model=hardware_model,
        initial_layout=initial_layout,
    )
    if isinstance(coupling_map, CouplingMap):
        graph = nx.Graph(list(coupling_map.get_edges()))
    else:
        graph = nx.Graph(coupling_map)
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    infos: list[dict] = []

    teacher = TeacherPolicy()
    teacher.begin_episode(graph)

    while not state.done:
        feats = model_features(state, graph, hardware_model).to(device)
        feats = _match_feature_dim(feats, model.net[0].in_features)  # type: ignore[index]
        logits = model(feats)
        action_mask = torch.tensor(state.action_mask, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~action_mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if model.value_head is None:
            value = torch.tensor(0.0, device=device)
        else:
            value = model.state_value(feats).mean()

        chosen_edge = state.candidate_swaps[int(action.item())]
        state, reward, done, info = env.step(int(action.item()))
        teacher.update_after_action(chosen_edge, state.step_count)

        features.append(feats.detach())
        actions.append(action.detach())
        log_probs.append(log_prob.detach())
        entropies.append(entropy.detach())
        values.append(value.detach())
        rewards.append(float(reward))
        masks.append(torch.tensor(1.0 - float(done), device=device))
        action_masks.append(action_mask.detach())
        infos.append(info)

    epi_metrics = compute_metrics(env.routed_circuit, hardware_model=hardware_model)
    return Rollout(
        features=features,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        values=values,
        masks=masks,
        action_masks=action_masks,
        entropies=entropies,
        infos=infos,
        episode_metrics={
            "swaps": epi_metrics.swaps,
            "twoq_depth": epi_metrics.two_qubit_depth,
            "log_success_proxy": epi_metrics.log_success_proxy,
            "duration_proxy": epi_metrics.duration_proxy,
        },
    )


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rewards:
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
    device = values[0].device if values else _device()
    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    values_t = torch.stack(values + [torch.tensor(0.0, device=device)])
    masks_t = torch.stack(masks + [torch.tensor(0.0, device=device)])

    advantages: list[torch.Tensor] = []
    gae = torch.tensor(0.0, device=device)
    for t in reversed(range(len(rewards))):
        delta = rewards_t[t] + gamma * values_t[t + 1] * masks_t[t] - values_t[t]
        gae = delta + gamma * gae_lambda * masks_t[t] * gae
        advantages.insert(0, gae)
    advantages_t = torch.stack(advantages)
    returns_t = advantages_t + values_t[:-1]
    return advantages_t, returns_t


def _ppo_update(
    model: SwapPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    gamma: float,
    gae_lambda: float,
) -> dict:
    if not rollout.rewards:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    device = _device()
    advantages, returns = _compute_gae(
        rollout.rewards, rollout.values, rollout.masks, gamma=gamma, gae_lambda=gae_lambda
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_log_probs = torch.stack(rollout.log_probs).to(device)
    entropy_terms: list[torch.Tensor] = []

    new_log_prob_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []

    for feats, mask, action in zip(
        rollout.features, rollout.action_masks, rollout.actions, strict=False
    ):
        logits = model(feats)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_prob_list.append(dist.log_prob(action))
        entropy_terms.append(dist.entropy())
        if model.value_head is not None:
            values_list.append(model.state_value(feats).mean())
        else:
            values_list.append(torch.tensor(0.0, device=device))

    new_log_probs = torch.stack(new_log_prob_list)
    values_new = torch.stack(values_list)
    entropy = torch.stack(entropy_terms).mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values_new, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }


def _curriculum_reward(ep: int, args: argparse.Namespace) -> RewardConfig:
    if not args.curriculum:
        return RewardConfig(
            swap_weight=args.swap_weight,
            depth_weight=args.depth_weight,
            noise_weight=args.noise_weight,
            base_swap_penalty=args.base_swap,
            error_penalty_weight=args.error_penalty,
            time_penalty_weight=args.time_penalty,
            progress_weight=args.progress_weight,
            completion_bonus=args.completion_bonus,
            failure_penalty=args.failure_penalty,
        )
    if ep < args.curriculum_switch:
        return RewardConfig(
            swap_weight=args.swap_weight,
            depth_weight=0.0,
            noise_weight=0.0,
            base_swap_penalty=args.base_swap,
            error_penalty_weight=0.0,
            time_penalty_weight=0.0,
            progress_weight=args.progress_weight,
            completion_bonus=args.completion_bonus,
            failure_penalty=args.failure_penalty,
        )
    ramp = max(1, args.curriculum_ramp)
    frac = min(1.0, (ep - args.curriculum_switch) / ramp)
    return RewardConfig(
        swap_weight=args.swap_weight,
        depth_weight=args.depth_weight,
        noise_weight=args.noise_weight * frac,
        base_swap_penalty=args.base_swap,
        error_penalty_weight=args.error_penalty * frac,
        time_penalty_weight=args.time_penalty * frac,
        progress_weight=args.progress_weight,
        completion_bonus=args.completion_bonus,
        failure_penalty=args.failure_penalty,
    )


def _mini_eval(
    model: SwapPolicy,
    circuits: list,
    coupling_maps: dict[str, CouplingMap],
    hardware_models: dict[str, list[HardwareModel]],
    *,
    swap_guard: float,
    device: torch.device,
) -> dict:
    results: dict[str, list[float]] = {}
    for circuit in circuits:
        for graph_id, cmap in coupling_maps.items():
            if circuit.circuit.num_qubits > cmap.size():
                continue
            hw_list = hardware_models.get(graph_id, [None])
            hw_model = hw_list[0] if hw_list else None
            teacher = route_with_teacher(
                circuit.circuit,
                cmap,
                seed=13,
                hardware_model=hw_model,
            )
            policy_res = route_with_policy(
                model,
                circuit.circuit,
                cmap,
                name="rl_policy_val",
                seed=13,
                hardware_model=hw_model,
                teacher_swaps=teacher.metrics.swaps,
                swap_guard_ratio=swap_guard,
            )
            ratio = policy_res.metrics.swaps / max(1, teacher.metrics.swaps)
            results.setdefault(graph_id, []).append(ratio)
    return {k: sum(v) / len(v) for k, v in results.items() if v}


def train_rl(
    circuits: list,
    coupling_maps: dict[str, CouplingMap],
    hardware_models: dict[str, list[HardwareModel]],
    args: argparse.Namespace,
    *,
    il_checkpoint: Path | None,
) -> tuple[SwapPolicy, list[dict]]:
    device = _device()
    rng = random.Random(args.seed)
    attempt = 0
    while attempt <= args.max_restarts:
        current_lr = args.lr * (0.5**attempt)
        model = _prep_policy(il_checkpoint, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
        history: list[dict] = []
        gate_failed = False

        for ep in range(args.episodes):
            reward_cfg = _curriculum_reward(ep, args)
            circuit = circuits[rng.randrange(len(circuits))]
            applicable_maps = [
                (gid, cmap)
                for gid, cmap in coupling_maps.items()
                if circuit.circuit.num_qubits <= cmap.size()
            ]
            if not applicable_maps:
                continue
            graph_id, cmap = rng.choice(applicable_maps)
            hw_list = hardware_models.get(graph_id, [])
            hw_model = hw_list[rng.randrange(len(hw_list))] if hw_list else None

            rollout = run_episode(
                model,
                circuit.circuit,
                cmap,
                reward_cfg,
                seed=args.seed + ep + attempt * 1000,
                device=device,
                hardware_model=hw_model,
            )
            update_info = {}
            for _ in range(args.ppo_epochs):
                update_info = _ppo_update(
                    model,
                    optimizer,
                    rollout,
                    clip_eps=args.clip_eps,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    max_grad_norm=args.max_grad_norm,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
            history.append(
                {
                    "episode": ep,
                    "reward": float(sum(rollout.rewards)),
                    "length": len(rollout.rewards),
                    "swaps": rollout.episode_metrics.get("swaps"),
                    "twoq_depth": rollout.episode_metrics.get("twoq_depth"),
                    "log_success_proxy": rollout.episode_metrics.get("log_success_proxy"),
                    "duration_proxy": rollout.episode_metrics.get("duration_proxy"),
                    **update_info,
                }
            )

            if args.eval_interval and (ep + 1) % args.eval_interval == 0:
                val_ratios = _mini_eval(
                    model,
                    circuits[:3],
                    coupling_maps,
                    hardware_models,
                    swap_guard=args.swap_guard,
                    device=device,
                )
                gate_failed = any(val > 1.5 for val in val_ratios.values())
                history[-1]["val_ratios"] = val_ratios
                if gate_failed:
                    debug_path = Path("artifacts/debug/regression_fail.json")
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(
                        json.dumps({"attempt": attempt, "val_ratios": val_ratios}, indent=2)
                    )
                    break

        if not gate_failed:
            return model, history
        attempt += 1

    return model, history


def save_checkpoint(
    model: SwapPolicy,
    path: Path,
    *,
    history: list[dict],
    episodes: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
    il_checkpoint: Path | None,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    clip_eps: float,
    curriculum: bool,
    curriculum_switch: int,
    curriculum_ramp: int,
    ppo_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.net[0].in_features,
            "use_value_head": bool(model.value_head is not None),
            "episodes": episodes,
            "seed": seed,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "history": history,
            "il_checkpoint": str(il_checkpoint) if il_checkpoint else None,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "max_grad_norm": max_grad_norm,
            "clip_eps": clip_eps,
            "curriculum": curriculum,
            "curriculum_switch": curriculum_switch,
            "curriculum_ramp": curriculum_ramp,
            "ppo_epochs": ppo_epochs,
        },
        path,
    )


def _update_metadata(out_dir: Path, updates: dict) -> None:
    meta_path = out_dir / "metadata.json"
    data = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            data = {}
    data.update(updates)
    meta_path.write_text(json.dumps(data, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["dev", "pressure", "full"], default="pressure")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.05)
    parser.add_argument("--noise-weight", type=float, default=0.02)
    parser.add_argument("--base-swap", type=float, default=1.0)
    parser.add_argument("--error-penalty", type=float, default=50.0)
    parser.add_argument("--time-penalty", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=0.8)
    parser.add_argument("--completion-bonus", type=float, default=10.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Path to IL checkpoint (defaults to artifacts/checkpoints/il_soft.pt if present).",
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        help="Path to QASMBench root (defaults to env QASMBENCH_ROOT or tests fixtures).",
    )
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--pressure-seed", type=int, default=99)
    parser.add_argument("--pressure-qasm", type=int, default=6)
    parser.add_argument("--hardware-samples", type=int, default=3)
    parser.add_argument("--hardware-seed-base", type=int, default=131)
    parser.add_argument("--hardware-profile", type=str, default="realistic")
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--curriculum-switch", type=int, default=150)
    parser.add_argument("--curriculum-ramp", type=int, default=200)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--swap-guard", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    qasm_root = Path(args.qasm_root or _default_qasm_root()).expanduser()

    circuits = _load_benchmarks(
        args.suite,
        qasm_root,
        dev_limit=args.dev_limit,
        pressure_seed=args.pressure_seed,
        pressure_qasm=args.pressure_qasm,
    )
    coupling_maps = _build_coupling_maps(args.suite)
    hardware_seeds = [args.hardware_seed_base + i for i in range(max(1, args.hardware_samples))]
    hardware_models = _build_hardware_models(coupling_maps, hardware_seeds, args.hardware_profile)

    il_ckpt = args.init_checkpoint
    if il_ckpt is None:
        default_il = out_dir / "checkpoints" / "il_soft.pt"
        if default_il.exists():
            il_ckpt = default_il

    model, history = train_rl(
        circuits,
        coupling_maps,
        hardware_models,
        args,
        il_checkpoint=il_ckpt,
    )

    ckpt_path = out_dir / "checkpoints" / "rl_ppo.pt"
    save_checkpoint(
        model,
        ckpt_path,
        history=history,
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        il_checkpoint=il_ckpt,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        clip_eps=args.clip_eps,
        curriculum=args.curriculum,
        curriculum_switch=args.curriculum_switch,
        curriculum_ramp=args.curriculum_ramp,
        ppo_epochs=args.ppo_epochs,
    )

    history_path = out_dir / "training" / "rl_ppo.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))

    trainable = circuits[:1]
    circuit = trainable[0]
    coupling_map = None
    for _gid, cmap in _build_coupling_maps(args.suite).items():
        if circuit.circuit.num_qubits <= cmap.size():
            coupling_map = cmap
            break
    coupling_map = coupling_map or next(iter(_build_coupling_maps(args.suite).values()))
    _ = route_with_policy(
        model,
        circuit.circuit,
        coupling_map,
        name="rl_policy_smoke",
        seed=args.seed,
        swap_guard_ratio=args.swap_guard,
    )

    _update_metadata(
        out_dir,
        {
            "rl_checkpoint": str(ckpt_path),
            "rl_episodes": args.episodes,
            "rl_seed": args.seed,
            "rl_lr": args.lr,
            "rl_gamma": args.gamma,
            "rl_gae_lambda": args.gae_lambda,
            "rl_entropy_coef": args.entropy_coef,
            "rl_value_coef": args.value_coef,
            "rl_max_grad_norm": args.max_grad_norm,
            "rl_clip_eps": args.clip_eps,
            "rl_curriculum": args.curriculum,
            "rl_curriculum_switch": args.curriculum_switch,
            "rl_curriculum_ramp": args.curriculum_ramp,
            "rl_swap_guard": args.swap_guard,
            "rl_qasm_root": str(qasm_root),
            "qiskit_version": qiskit_version,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
