"""Train a fresh Stage 1 residual PPO policy for sustained recovery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config import balance as balance_config
from g1_control.config.recovery import (
    DEFAULT_RECOVERY_CRITERIA,
    DEFAULT_RECOVERY_STAGE,
    RECOVERY_ACTOR_LOG_STD,
    RECOVERY_NUM_UPDATES,
    RECOVERY_ROLLOUT_STEPS,
    RECOVERY_RUNS_DIR,
    get_recovery_stage,
)
from g1_control.envs.recovery_env import RecoveryEnv
from g1_control.evaluation.recovery_protocol import RecoveryTracker, measure_recovery_state
from g1_control.learning.buffer import RolloutBuffer
from g1_control.learning.networks import Actor, Critic
from g1_control.learning.ppo import compute_gae, ppo_update
from g1_control.training.recovery_curriculum import reset_recovery_env


def stack_buffer(buffer: RolloutBuffer):
    return (
        torch.stack(buffer.observations),
        torch.stack(buffer.actions),
        torch.tensor(buffer.rewards, dtype=torch.float32),
        torch.stack(buffer.values).reshape(-1),
        torch.stack(buffer.log_probs).reshape(-1),
        torch.tensor(buffer.dones, dtype=torch.float32),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory() -> Path:
    RECOVERY_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0
    while True:
        run_id = (
            f"recovery_{timestamp}"
            if suffix == 0
            else f"recovery_{timestamp}_{suffix:02d}"
        )
        run_dir = RECOVERY_RUNS_DIR / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return run_dir


def actor_distribution_config(actor: Actor) -> dict[str, object]:
    log_std = actor.log_std.detach().float().cpu()
    std = torch.exp(log_std)
    return {
        "log_std_mean": float(log_std.mean()),
        "std_mean": float(std.mean()),
        "log_std_trainable": bool(actor.log_std.requires_grad),
    }


def checkpoint_payload(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    update,
    total_env_steps,
    run_config,
    metrics,
):
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "update": update,
        "total_env_steps": total_env_steps,
        "obs_dim": balance_config.OBS_DIM,
        "action_dim": balance_config.ACTION_DIM,
        "task": "day10_recovery",
        "config": run_config,
        "training_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_RECOVERY_STAGE)
    parser.add_argument("--updates", type=int, default=RECOVERY_NUM_UPDATES)
    parser.add_argument("--rollout-steps", type=int, default=RECOVERY_ROLLOUT_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=balance_config.SEED)
    args = parser.parse_args()

    if args.updates <= 0 or args.rollout_steps <= 0:
        raise ValueError("--updates and --rollout-steps must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stage = get_recovery_stage(args.stage)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RecoveryEnv()
    actor = Actor(
        balance_config.OBS_DIM,
        balance_config.ACTION_DIM,
        log_std=RECOVERY_ACTOR_LOG_STD,
    ).to(device)
    critic = Critic(balance_config.OBS_DIM).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=balance_config.ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=balance_config.CRITIC_LR)
    run_dir = create_run_directory()

    source_paths = {
        "config/balance.py": PROJECT_ROOT / "src/g1_control/config/balance.py",
        "config/recovery.py": PROJECT_ROOT / "src/g1_control/config/recovery.py",
        "envs/balance_env.py": PROJECT_ROOT / "src/g1_control/envs/balance_env.py",
        "envs/recovery_env.py": PROJECT_ROOT / "src/g1_control/envs/recovery_env.py",
        "training/recovery_curriculum.py": PROJECT_ROOT / "src/g1_control/training/recovery_curriculum.py",
        "evaluation/recovery_protocol.py": PROJECT_ROOT / "src/g1_control/evaluation/recovery_protocol.py",
        "learning/networks.py": PROJECT_ROOT / "src/g1_control/learning/networks.py",
        "learning/ppo.py": PROJECT_ROOT / "src/g1_control/learning/ppo.py",
        "scripts/train_recovery.py": Path(__file__).resolve(),
    }
    run_config = {
        "schema_version": 1,
        "task": "day10_recovery",
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "obs_dim": balance_config.OBS_DIM,
        "action_dim": balance_config.ACTION_DIM,
        "frozen_pose": {
            "hip": balance_config.HIP_PITCH,
            "knee": balance_config.KNEE,
            "ankle": balance_config.ANKLE_PITCH,
        },
        "action_scale": balance_config.ACTION_SCALE,
        "stage": asdict(stage),
        "recovery_environment": env.recovery_config(),
        "rollout_steps": args.rollout_steps,
        "num_updates": args.updates,
        "ppo_epochs": balance_config.PPO_EPOCHS,
        "batch_size": balance_config.BATCH_SIZE,
        "gamma": balance_config.GAMMA,
        "gae_lambda": balance_config.GAE_LAMBDA,
        "clip_epsilon": balance_config.CLIP_EPSILON,
        "actor_lr": balance_config.ACTOR_LR,
        "critic_lr": balance_config.CRITIC_LR,
        "actor_log_std_config": RECOVERY_ACTOR_LOG_STD,
        "entropy_coef": balance_config.ENTROPY_COEF,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_selection": "none_during_training_use_strict_fixed_evaluation",
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        **actor_distribution_config(actor),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("=== DAY 10 RECOVERY PPO ===")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"stage: {stage.name}")
    print(f"run directory: {run_dir}")
    print(f"updates/rollout: {args.updates}/{args.rollout_steps}")
    print("checkpoint selection: deferred to strict fixed evaluation")
    print()

    reset_rng = np.random.default_rng(args.seed)
    reset_counts = {"nominal": 0, "randomized": 0}
    observation, reset_info = reset_recovery_env(env, reset_rng, stage)
    reset_counts[str(reset_info["mode"])] += 1
    tracker = RecoveryTracker(DEFAULT_RECOVERY_CRITERIA)
    current_episode_length = 0
    current_episode_return = 0.0
    total_env_steps = 0

    for update in range(1, args.updates + 1):
        buffer = RolloutBuffer()
        episode_lengths: list[int] = []
        episode_returns: list[float] = []
        episode_successes: list[bool] = []
        equilibrium_steps = 0
        reward_total = 0.0

        for _ in range(args.rollout_steps):
            obs_tensor = torch.from_numpy(observation).float().to(device)
            with torch.no_grad():
                distribution = actor(obs_tensor)
                raw_action = distribution.sample()
                log_prob = distribution.log_prob(raw_action).sum()
                value = critic(obs_tensor).squeeze(-1)
                env_action = torch.tanh(raw_action).cpu().numpy().astype(np.float32)

            next_observation, reward, terminated, truncated = env.step(env_action)
            episode_end = terminated or truncated
            buffer.add(obs_tensor, raw_action, reward, value, log_prob, episode_end)
            current_episode_length += 1
            current_episode_return += reward
            total_env_steps += 1
            reward_total += reward
            equilibrium_steps += int(env.last_reward_terms["equilibrium_bonus"] > 0.0)

            state = measure_recovery_state(
                observation=next_observation,
                height=float(env.data.qpos[2]),
                default_q=env.default_q,
            )
            tracker.update(current_episode_length - 1, state)
            observation = next_observation

            if episode_end:
                episode_lengths.append(current_episode_length)
                episode_returns.append(current_episode_return)
                episode_successes.append(tracker.succeeded(survived=not terminated))
                observation, reset_info = reset_recovery_env(env, reset_rng, stage)
                reset_counts[str(reset_info["mode"])] += 1
                tracker = RecoveryTracker(DEFAULT_RECOVERY_CRITERIA)
                current_episode_length = 0
                current_episode_return = 0.0

        observations, actions, rewards, values, old_log_probs, dones = stack_buffer(buffer)
        final_obs_tensor = torch.from_numpy(observation).float().to(device)
        with torch.no_grad():
            next_value = critic(final_obs_tensor).squeeze(-1).cpu()
        advantages, value_targets = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=balance_config.GAMMA,
            gae_lambda=balance_config.GAE_LAMBDA,
        )
        ppo_metrics = ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            value_targets=value_targets,
            epochs=balance_config.PPO_EPOCHS,
            batch_size=balance_config.BATCH_SIZE,
            clip_epsilon=balance_config.CLIP_EPSILON,
            entropy_coef=balance_config.ENTROPY_COEF,
        )

        metrics = {
            **{key: float(value) for key, value in ppo_metrics.items()},
            "episodes": len(episode_lengths),
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "strict_success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
            "equilibrium_fraction": equilibrium_steps / args.rollout_steps,
            "mean_step_reward": reward_total / args.rollout_steps,
            "reset_counts": dict(reset_counts),
        }
        print(
            f"Update {update:02d} | steps={total_env_steps:6d} | "
            f"episodes={metrics['episodes']:2d} | len={metrics['mean_episode_length']:6.1f} | "
            f"return={metrics['mean_episode_return']:8.2f} | "
            f"success={metrics['strict_success_rate']:.2f} | "
            f"equilibrium={metrics['equilibrium_fraction']:.2f} | "
            f"reward/step={metrics['mean_step_reward']:.3f} | "
            f"KL={ppo_metrics['approx_kl']:.4f} | clip={ppo_metrics['clip_fraction']:.3f} | "
            f"resets={reset_counts['nominal']}/{reset_counts['randomized']}"
        )

        payload = checkpoint_payload(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            update,
            total_env_steps,
            run_config,
            metrics,
        )
        torch.save(payload, run_dir / "last.pt")
        if update % args.checkpoint_every == 0:
            torch.save(payload, run_dir / f"update_{update:03d}.pt")

    print()
    print("Training finished.")
    print(f"Run directory: {run_dir}")
    print("No best checkpoint was selected during training.")


if __name__ == "__main__":
    main()
