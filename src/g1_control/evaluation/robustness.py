from pathlib import Path
import os
import json
import math

import mujoco
import numpy as np
import torch

from g1_control.envs.balance_env import G1Env
from g1_control.learning.networks import Actor
from g1_control.config import balance as config


OBS_DIM = config.OBS_DIM
ACTION_DIM = config.ACTION_DIM

# ------------------------------------------------------------
# Robustness sweep configuration
# ------------------------------------------------------------

PITCH_VALUES = [
    -0.03,
    -0.02,
    -0.01,
     0.00,
     0.01,
     0.02,
     0.03,
]

PITCH_RATE_VALUES = [
    -0.30,
    -0.20,
    -0.10,
     0.00,
     0.10,
     0.20,
     0.30,
]

CHECKPOINT_NAME = config.BEST_DETERMINISTIC_CHECKPOINT


# ============================================================
# Initial-state perturbations
# ============================================================

def set_initial_pitch(
    env,
    pitch
):
    """
    Apply a pure base pitch rotation after env.reset().

    MuJoCo free-joint quaternion layout in qpos is:
        [w, x, y, z]

    A pure pitch rotation is around the y-axis:
        q = [cos(theta/2), 0, sin(theta/2), 0]
    """

    half = 0.5 * pitch

    env.data.qpos[3] = math.cos(half)
    env.data.qpos[4] = 0.0
    env.data.qpos[5] = math.sin(half)
    env.data.qpos[6] = 0.0

    mujoco.mj_forward(
        env.model,
        env.data
    )


def set_initial_pitch_rate(
    env,
    pitch_rate
):
    """
    qvel layout for the floating base:
        qvel[0:3] = base linear velocity
        qvel[3:6] = base angular velocity

    Pitch angular velocity is the y-axis component:
        qvel[4]
    """

    env.data.qvel[4] = pitch_rate

    mujoco.mj_forward(
        env.model,
        env.data
    )


# ============================================================
# Episode evaluation
# ============================================================

def run_episode(
    actor=None,
    device=None,
    policy="zero",
    initial_pitch=0.0,
    initial_pitch_rate=0.0
):
    env = G1Env()

    obs = env.reset()

    # Apply perturbation AFTER the environment's normal reset/settling.
    set_initial_pitch(
        env,
        initial_pitch
    )

    set_initial_pitch_rate(
        env,
        initial_pitch_rate
    )

    # Rebuild observation so the policy sees the perturbed state.
    #
    # Your G1Env already owns the observation-generation logic.
    # We try the common helper names first. If none exist, we take
    # one zero-time forward-consistent observation through the method
    # your environment exposes.
    if hasattr(env, "_get_obs"):
        obs = env._get_obs()
    elif hasattr(env, "get_obs"):
        obs = env.get_obs()
    elif hasattr(env, "_get_observation"):
        obs = env._get_observation()
    elif hasattr(env, "get_observation"):
        obs = env.get_observation()
    else:
        raise AttributeError(
            "G1Env has no recognized observation helper. "
            "Expected one of: _get_obs(), get_obs(), "
            "_get_observation()."
        )

    total_reward = 0.0

    max_abs_pitch = 0.0
    max_abs_pitch_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    termination_reason = "max_steps"

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
                # use the Gaussian mean only.
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

        # Observation layout:
        # 29 q + 29 dq + 3 RPY + 3 gyro = 64
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
            termination_reason = "terminated"
            break

        if truncated:
            termination_reason = "truncated"
            break

    return {
        "length": step + 1,
        "return": float(total_reward),
        "max_pitch": max_abs_pitch,
        "max_pitch_rate": max_abs_pitch_rate,
        "min_height": min_height,
        "end": termination_reason,
    }


# ============================================================
# Printing
# ============================================================

def print_pair(
    label,
    perturbation,
    zero_result,
    ppo_result
):
    delta_length = (
        ppo_result["length"]
        - zero_result["length"]
    )

    delta_return = (
        ppo_result["return"]
        - zero_result["return"]
    )

    print(
        f"{label:12s} "
        f"{perturbation:+.3f} | "
        f"zero_len={zero_result['length']:4d} | "
        f"ppo_len={ppo_result['length']:4d} | "
        f"delta={delta_length:+4d} | "
        f"zero_ret={zero_result['return']:8.2f} | "
        f"ppo_ret={ppo_result['return']:8.2f} | "
        f"dret={delta_return:+8.2f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    checkpoint_dir = Path(os.environ.get("G1_CHECKPOINT_DIR", config.CHECKPOINT_DIR))
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

    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Expected the deterministic-best checkpoint created "
            "from the checkpoint sweep."
        )

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

    print(
        "Loaded checkpoint:",
        checkpoint_path
    )

    print(
        "Checkpoint update:",
        checkpoint.get(
            "update"
        )
    )

    print(
        "Checkpoint steps:",
        checkpoint.get(
            "total_env_steps"
        )
    )

    checkpoint_config = checkpoint.get(
        "config"
    )

    if checkpoint_config is not None:
        print(
            "Actor LR:",
            checkpoint_config.get(
                "current_actor_lr",
                checkpoint_config.get(
                    "actor_lr_config"
                )
            )
        )

        print(
            "Critic LR:",
            checkpoint_config.get(
                "current_critic_lr",
                checkpoint_config.get(
                    "critic_lr_config"
                )
            )
        )

        print(
            "Entropy coef:",
            checkpoint_config.get(
                "entropy_coef"
            )
        )

        print(
            "log_std:",
            checkpoint_config.get(
                "current_log_std_mean",
                checkpoint_config.get(
                    "log_std_mean"
                )
            )
        )

        print(
            "std:",
            checkpoint_config.get(
                "current_std_mean",
                checkpoint_config.get(
                    "std_mean"
                )
            )
        )

    print()

    # ========================================================
    # Nominal deterministic baseline
    # ========================================================

    print(
        "=== NOMINAL ==="
    )

    zero_nominal = run_episode(
        policy="zero"
    )

    ppo_nominal = run_episode(
        actor=actor,
        device=device,
        policy="actor"
    )

    print_pair(
        "nominal",
        0.0,
        zero_nominal,
        ppo_nominal
    )

    print()

    # ========================================================
    # Initial pitch robustness
    # ========================================================

    print(
        "=== INITIAL PITCH SWEEP (rad) ==="
    )

    pitch_rows = []

    for pitch in PITCH_VALUES:
        zero_result = run_episode(
            policy="zero",
            initial_pitch=pitch,
            initial_pitch_rate=0.0
        )

        ppo_result = run_episode(
            actor=actor,
            device=device,
            policy="actor",
            initial_pitch=pitch,
            initial_pitch_rate=0.0
        )

        pitch_rows.append(
            (
                pitch,
                zero_result,
                ppo_result
            )
        )

        print_pair(
            "pitch",
            pitch,
            zero_result,
            ppo_result
        )

    print()

    # ========================================================
    # Initial pitch-rate robustness
    # ========================================================

    print(
        "=== INITIAL PITCH-RATE SWEEP (rad/s) ==="
    )

    rate_rows = []

    for pitch_rate in PITCH_RATE_VALUES:
        zero_result = run_episode(
            policy="zero",
            initial_pitch=0.0,
            initial_pitch_rate=pitch_rate
        )

        ppo_result = run_episode(
            actor=actor,
            device=device,
            policy="actor",
            initial_pitch=0.0,
            initial_pitch_rate=pitch_rate
        )

        rate_rows.append(
            (
                pitch_rate,
                zero_result,
                ppo_result
            )
        )

        print_pair(
            "pitch_rate",
            pitch_rate,
            zero_result,
            ppo_result
        )

    print()

    # ========================================================
    # Summary
    # ========================================================

    all_pairs = (
        [
            (
                f"pitch {value:+.3f}",
                zero_result,
                ppo_result
            )
            for (
                value,
                zero_result,
                ppo_result
            ) in pitch_rows
            if value != 0.0
        ]
        +
        [
            (
                f"rate {value:+.3f}",
                zero_result,
                ppo_result
            )
            for (
                value,
                zero_result,
                ppo_result
            ) in rate_rows
            if value != 0.0
        ]
    )

    wins = sum(
        1
        for (
            _,
            zero_result,
            ppo_result
        ) in all_pairs
        if (
            ppo_result["length"]
            >
            zero_result["length"]
        )
    )

    ties = sum(
        1
        for (
            _,
            zero_result,
            ppo_result
        ) in all_pairs
        if (
            ppo_result["length"]
            ==
            zero_result["length"]
        )
    )

    losses = (
        len(all_pairs)
        - wins
        - ties
    )

    mean_zero_length = float(
        np.mean(
            [
                zero_result["length"]
                for (
                    _,
                    zero_result,
                    _
                ) in all_pairs
            ]
        )
    )

    mean_ppo_length = float(
        np.mean(
            [
                ppo_result["length"]
                for (
                    _,
                    _,
                    ppo_result
                ) in all_pairs
            ]
        )
    )

    print(
        "=== ROBUSTNESS SUMMARY ==="
    )

    print(
        "Cases:",
        len(all_pairs)
    )

    print(
        "PPO wins:",
        wins
    )

    print(
        "Ties:",
        ties
    )

    print(
        "PPO losses:",
        losses
    )

    print(
        f"Mean zero length: "
        f"{mean_zero_length:.2f}"
    )

    print(
        f"Mean PPO length:  "
        f"{mean_ppo_length:.2f}"
    )

    print(
        f"Mean delta:       "
        f"{mean_ppo_length - mean_zero_length:+.2f}"
    )
    evaluation = {"checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_update": checkpoint.get("update"), "nominal": {"zero": zero_nominal, "ppo": ppo_nominal}, "pitch_rows": pitch_rows, "pitch_rate_rows": rate_rows, "summary": {"cases": len(all_pairs), "ppo_wins": wins, "ties": ties, "ppo_losses": losses, "mean_zero_length": mean_zero_length, "mean_ppo_length": mean_ppo_length}}
    result_path = checkpoint_dir / "robustness_evaluation.json"
    result_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print("Saved robustness evaluation:", result_path)


if __name__ == "__main__":
    main()
