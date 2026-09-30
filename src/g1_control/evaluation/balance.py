from pathlib import Path
import os
import json
import shutil

import numpy as np
import torch

from g1_control.envs.balance_env import G1Env
from g1_control.learning.networks import Actor
from g1_control.config import balance as config


OBS_DIM = config.OBS_DIM
ACTION_DIM = config.ACTION_DIM
SEED = config.SEED

CHECKPOINT_DIR = config.CHECKPOINT_DIR

# If True, copy the best deterministic checkpoint found in the sweep to:
# checkpoints/best_deterministic.pt
COPY_BEST_DETERMINISTIC = False


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(
    actor=None,
    device=None,
    policy="zero"
):
    """
    Run one deterministic evaluation episode.

    policy="zero":
        action = 0

    policy="actor":
        action = tanh(actor distribution mean)
    """

    env = G1Env()
    obs = env.reset()

    total_reward = 0.0

    max_abs_pitch = 0.0
    max_abs_pitch_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    termination_type = "max_steps"

    for step in range(
        env.max_episode_steps
    ):
        if policy == "zero":
            action = np.zeros(
                ACTION_DIM,
                dtype=np.float32
            )

        else:
            obs_tensor = (
                torch.from_numpy(obs)
                .float()
                .to(device)
            )

            with torch.no_grad():
                dist = actor(
                    obs_tensor
                )

                # Deterministic evaluation:
                # use Gaussian mean, not sampled action.
                raw_action = dist.mean

                action = (
                    torch.tanh(
                        raw_action
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            action
        )

        total_reward += reward

        pitch = float(
            obs[59]
        )

        pitch_rate = float(
            obs[62]
        )

        height = float(
            env.data.qpos[2]
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(pitch)
        )

        max_abs_pitch_rate = max(
            max_abs_pitch_rate,
            abs(pitch_rate)
        )

        min_height = min(
            min_height,
            height
        )

        if terminated:
            termination_type = "terminated"
            break

        if truncated:
            termination_type = "truncated"
            break

    return {
        "length": step + 1,
        "return": total_reward,
        "max_pitch": max_abs_pitch,
        "max_pitch_rate":
            max_abs_pitch_rate,
        "min_height": min_height,
        "termination":
            termination_type
    }


def print_result(
    name,
    result
):
    print(
        f"{name:14s} | "
        f"length={result['length']:4d} | "
        f"return={result['return']:8.2f} | "
        f"max_pitch={result['max_pitch']:.3f} | "
        f"max_rate={result['max_pitch_rate']:.3f} | "
        f"min_height={result['min_height']:.3f} | "
        f"end={result['termination']}"
    )


def load_actor_from_checkpoint(
    checkpoint_path,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    actor.load_state_dict(
        checkpoint["actor"]
    )

    actor.eval()

    return actor, checkpoint


def print_checkpoint_config(
    checkpoint_path,
    checkpoint
):
    print()
    print("-" * 72)
    print(
        "Checkpoint:",
        checkpoint_path.name
    )

    print(
        "  update:",
        checkpoint.get(
            "update"
        )
    )

    print(
        "  total_env_steps:",
        checkpoint.get(
            "total_env_steps"
        )
    )

    print(
        "  saved_stochastic_mean_length:",
        checkpoint.get(
            "mean_episode_length"
        )
    )

    print(
        "  saved_stochastic_mean_return:",
        checkpoint.get(
            "mean_episode_return"
        )
    )

    config = checkpoint.get(
        "config"
    )

    if config is None:
        print(
            "  config: <not stored in this checkpoint>"
        )
        return

    keys_to_print = [
        "actor_lr_config",
        "actor_lr_optimizer",
        "current_actor_lr",
        "critic_lr_config",
        "critic_lr_optimizer",
        "current_critic_lr",
        "entropy_coef",
        "log_std_mean",
        "current_log_std_mean",
        "std_mean",
        "current_std_mean",
        "log_std_trainable",
        "current_log_std_trainable",
        "clip_epsilon",
        "ppo_epochs",
        "batch_size",
        "rollout_steps",
        "num_updates",
        "seed",
    ]

    print(
        "  saved config:"
    )

    for key in keys_to_print:
        if key in config:
            print(
                f"    {key}={config[key]}"
            )


def main():
    set_seed()
    checkpoint_dir = Path(os.environ.get("G1_CHECKPOINT_DIR", CHECKPOINT_DIR))

    copy_best_deterministic = COPY_BEST_DETERMINISTIC or "G1_CHECKPOINT_DIR" in os.environ
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Evaluation device:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # ========================================================
    # Zero baseline
    # ========================================================

    set_seed()

    zero_result = run_episode(
        policy="zero"
    )

    # ========================================================
    # Fresh zero-init deterministic Actor
    # ========================================================

    set_seed()

    untrained_actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    untrained_actor.eval()

    untrained_result = run_episode(
        actor=untrained_actor,
        device=device,
        policy="actor"
    )

    # ========================================================
    # Find all update checkpoints
    # ========================================================

    checkpoint_paths = sorted(
        checkpoint_dir.glob(
            "update_*.pt"
        )
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No update_*.pt checkpoints found in "
            f"{checkpoint_dir}"
        )

    # ========================================================
    # Baselines
    # ========================================================

    print(
        "=== DETERMINISTIC BASELINES ==="
    )

    print_result(
        "zero",
        zero_result
    )

    print_result(
        "untrained",
        untrained_result
    )

    # ========================================================
    # Checkpoint sweep
    # ========================================================

    print()
    print(
        "=== DETERMINISTIC CHECKPOINT SWEEP ==="
    )

    sweep_results = []

    for checkpoint_path in checkpoint_paths:
        set_seed()

        actor, checkpoint = (
            load_actor_from_checkpoint(
                checkpoint_path,
                device
            )
        )

        result = run_episode(
            actor=actor,
            device=device,
            policy="actor"
        )

        update = checkpoint.get(
            "update"
        )

        sweep_results.append(
            {
                "path":
                    checkpoint_path,
                "checkpoint":
                    checkpoint,
                "result":
                    result,
                "update":
                    update,
            }
        )

        print_result(
            f"update_{int(update):03d}",
            result
        )

    # ========================================================
    # Select best deterministic checkpoint
    # ========================================================

    best_item = max(
        sweep_results,
        key=lambda item: (
            item["result"]["length"],
            item["result"]["return"]
        )
    )

    best_path = best_item["path"]
    best_checkpoint = (
        best_item["checkpoint"]
    )
    best_result = (
        best_item["result"]
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=== SUMMARY ==="
    )

    print_result(
        "zero",
        zero_result
    )

    print_result(
        "untrained",
        untrained_result
    )

    print_result(
        "best_trained",
        best_result
    )

    delta_vs_zero = (
        best_result["length"]
        - zero_result["length"]
    )

    print()
    print(
        "Best deterministic checkpoint:",
        best_path.name
    )

    print(
        "Best deterministic update:",
        best_checkpoint.get(
            "update"
        )
    )

    print(
        "Best deterministic length:",
        best_result["length"]
    )

    print(
        "Zero baseline length:",
        zero_result["length"]
    )

    print(
        "Delta vs zero:",
        f"{delta_vs_zero:+d} steps"
    )
    evaluation = {"checkpoint_dir": str(checkpoint_dir.resolve()), "selected_checkpoint": best_path.name, "selected_update": best_checkpoint.get("update"), "zero": zero_result, "untrained": untrained_result, "best_trained": best_result, "delta_vs_zero_steps": delta_vs_zero}
    result_path = checkpoint_dir / "deterministic_evaluation.json"
    result_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print("Saved deterministic evaluation:", result_path)

    # ========================================================
    # Print the exact saved training config of the winner
    # ========================================================

    print_checkpoint_config(
        best_path,
        best_checkpoint
    )

    # ========================================================
    # Copy deterministic-best checkpoint
    # ========================================================

    if copy_best_deterministic:
        destination = (
            checkpoint_dir
            / config.BEST_DETERMINISTIC_CHECKPOINT
        )

        shutil.copy2(
            best_path,
            destination
        )

        print()
        print(
            "Copied deterministic-best checkpoint:"
        )

        print(
            "  source:",
            best_path
        )

        print(
            "  destination:",
            destination
        )


if __name__ == "__main__":
    main()
