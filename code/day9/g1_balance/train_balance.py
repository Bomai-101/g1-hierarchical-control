from pathlib import Path
import json

import numpy as np
import torch

from g1_env import G1Env
from networks import Actor, Critic
from buffer import RolloutBuffer
import experiment_config as config
from ppo import compute_gae, ppo_update
from provenance import build_run_manifest, create_run_directory
from reset_randomization import reset_training_env


# ============================================================
# Configuration
# ============================================================

SEED = config.SEED

OBS_DIM = config.OBS_DIM
ACTION_DIM = config.ACTION_DIM

ROLLOUT_STEPS = config.ROLLOUT_STEPS
NUM_UPDATES = config.NUM_UPDATES

PPO_EPOCHS = config.PPO_EPOCHS
BATCH_SIZE = config.BATCH_SIZE

GAMMA = config.GAMMA
GAE_LAMBDA = config.GAE_LAMBDA

CLIP_EPSILON = config.CLIP_EPSILON

# Current calibrated pilot configuration.
ACTOR_LR = config.ACTOR_LR
CRITIC_LR = config.CRITIC_LR

# We currently want no entropy bonus while using a small fixed std.
ENTROPY_COEF = config.ENTROPY_COEF

# Save every update during the short pilot so we can inspect
# deterministic performance at each update later if needed.
CHECKPOINT_EVERY = config.CHECKPOINT_EVERY


# ============================================================
# Helpers
# ============================================================

def stack_buffer(buffer):
    observations = torch.stack(
        buffer.observations
    )

    actions = torch.stack(
        buffer.actions
    )

    rewards = torch.tensor(
        buffer.rewards,
        dtype=torch.float32
    )

    values = torch.stack(
        buffer.values
    ).reshape(-1)

    old_log_probs = torch.stack(
        buffer.log_probs
    ).reshape(-1)

    dones = torch.tensor(
        buffer.dones,
        dtype=torch.float32
    )

    return (
        observations,
        actions,
        rewards,
        values,
        old_log_probs,
        dones
    )


def get_actor_distribution_config(actor):
    """
    Read the policy's actual current distribution settings from the
    instantiated Actor rather than trusting comments/constants.
    """
    if not hasattr(actor, "log_std"):
        return {
            "log_std_mean": None,
            "log_std_min": None,
            "log_std_max": None,
            "std_mean": None,
            "std_min": None,
            "std_max": None,
            "log_std_trainable": None,
        }

    log_std = actor.log_std.detach().float().cpu()
    std = torch.exp(log_std)

    return {
        "log_std_mean": float(log_std.mean()),
        "log_std_min": float(log_std.min()),
        "log_std_max": float(log_std.max()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "log_std_trainable": bool(actor.log_std.requires_grad),
    }


def get_actor_zero_obs_mean(actor, device):
    """
    Useful sanity check for residual RL:
    a zero-initialized output head should give mean ~= 0 at startup.
    """
    with torch.no_grad():
        probe_obs = torch.zeros(
            OBS_DIM,
            dtype=torch.float32,
            device=device
        )

        dist = actor(probe_obs)

        return (
            dist.mean
            .detach()
            .float()
            .cpu()
            .tolist()
        )


def build_run_config(
    actor,
    actor_optimizer,
    critic_optimizer,
    device
):
    dist_cfg = get_actor_distribution_config(actor)

    return {
        "seed": SEED,
        "device": str(device),
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "rollout_steps": ROLLOUT_STEPS,
        "num_updates": NUM_UPDATES,
        "ppo_epochs": PPO_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON,

        # Store BOTH configured LR and the LR actually held by Adam.
        "actor_lr_config": ACTOR_LR,
        "actor_lr_optimizer": float(
            actor_optimizer.param_groups[0]["lr"]
        ),
        "critic_lr_config": CRITIC_LR,
        "critic_lr_optimizer": float(
            critic_optimizer.param_groups[0]["lr"]
        ),

        "entropy_coef": ENTROPY_COEF,
        "checkpoint_every": CHECKPOINT_EVERY,
        "reset_strategy": "30_percent_nominal_70_percent_randomized",
        "reset_nominal_probability": config.RESET_NOMINAL_PROBABILITY,
        "reset_pitch_range": list(config.RESET_PITCH_RANGE),
        "reset_pitch_rate_range": list(config.RESET_PITCH_RATE_RANGE),

        **dist_cfg,

        # This lets us verify the fresh deterministic policy really starts
        # at the zero-residual baseline.
        "actor_mean_at_zero_obs": get_actor_zero_obs_mean(
            actor,
            device
        ),
    }


def print_run_config(config):
    print("=" * 72)
    print("RUN CONFIG")
    print("=" * 72)

    ordered_keys = [
        "seed",
        "device",
        "obs_dim",
        "action_dim",
        "rollout_steps",
        "num_updates",
        "ppo_epochs",
        "batch_size",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "actor_lr_config",
        "actor_lr_optimizer",
        "critic_lr_config",
        "critic_lr_optimizer",
        "entropy_coef",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "std_mean",
        "std_min",
        "std_max",
        "log_std_trainable",
        "checkpoint_every",
        "reset_strategy",
        "reset_nominal_probability",
        "reset_pitch_range",
        "reset_pitch_rate_range",
        "actor_mean_at_zero_obs",
    ]

    for key in ordered_keys:
        print(
            f"CONFIG | {key}={config.get(key)}"
        )

    print("=" * 72)
    print()


def make_checkpoint_payload(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    update,
    total_env_steps,
    mean_episode_length,
    mean_episode_return,
    run_config
):
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "update": update,
        "total_env_steps": total_env_steps,
        "mean_episode_length": mean_episode_length,
        "mean_episode_return": mean_episode_return,
        "action_dim": ACTION_DIM,
        "obs_dim": OBS_DIM,

        # Critical for experiment traceability.
        "config": run_config,
    }


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = G1Env()

    # --------------------------------------------------------
    # Networks
    # --------------------------------------------------------

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    critic = Critic(
        obs_dim=OBS_DIM
    ).to(device)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=ACTOR_LR
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=CRITIC_LR
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    checkpoint_dir = create_run_directory()

    # --------------------------------------------------------
    # Print + save exact run configuration
    # --------------------------------------------------------

    run_config = build_run_config(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device
    )
    run_config["run_id"] = checkpoint_dir.name
    run_config["run_dir"] = str(checkpoint_dir.resolve())
    run_manifest = build_run_manifest(env=env, run_dir=checkpoint_dir)
    run_manifest["run_config"] = run_config
    (checkpoint_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print_run_config(
        run_config
    )

    (
        checkpoint_dir
        / "run_config.json"
    ).write_text(
        json.dumps(
            run_config,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    total_env_steps = 0
    best_mean_episode_length = 0.0

    reset_rng = np.random.default_rng(SEED)
    reset_counts = {
        "nominal": 0,
        "randomized": 0,
    }

    obs, reset_info = reset_training_env(
        env,
        reset_rng
    )
    reset_counts[reset_info["mode"]] += 1

    # IMPORTANT:
    # Keep these OUTSIDE the PPO-update loop so an episode that crosses
    # a rollout boundary is not falsely restarted in the statistics.
    current_episode_length = 0
    current_episode_return = 0.0

    # ========================================================
    # PPO update loop
    # ========================================================

    for update in range(
        1,
        NUM_UPDATES + 1
    ):
        buffer = RolloutBuffer()

        episode_lengths = []
        episode_returns = []

        # ====================================================
        # Collect rollout
        # ====================================================

        for rollout_step in range(
            ROLLOUT_STEPS
        ):
            obs_tensor = (
                torch.from_numpy(obs)
                .float()
                .to(device)
            )

            with torch.no_grad():
                dist = actor(
                    obs_tensor
                )

                raw_action = (
                    dist.sample()
                )

                log_prob = (
                    dist
                    .log_prob(raw_action)
                    .sum()
                )

                value = (
                    critic(
                        obs_tensor
                    )
                    .squeeze(-1)
                )

                env_action_tensor = (
                    torch.tanh(
                        raw_action
                    )
                )

            env_action = (
                env_action_tensor
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            (
                next_obs,
                reward,
                terminated,
                truncated
            ) = env.step(
                env_action
            )

            episode_end = (
                terminated
                or truncated
            )

            buffer.add(
                observation=obs_tensor,
                action=raw_action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=episode_end
            )

            current_episode_length += 1
            current_episode_return += reward

            total_env_steps += 1
            obs = next_obs

            if episode_end:
                episode_lengths.append(
                    current_episode_length
                )

                episode_returns.append(
                    current_episode_return
                )

                obs, reset_info = reset_training_env(
                    env,
                    reset_rng
                )
                reset_counts[reset_info["mode"]] += 1

                current_episode_length = 0
                current_episode_return = 0.0

        # ====================================================
        # Stack rollout
        # ====================================================

        (
            observations,
            actions,
            rewards,
            values,
            old_log_probs,
            dones
        ) = stack_buffer(
            buffer
        )

        # ====================================================
        # Bootstrap final state
        # ====================================================

        obs_tensor = (
            torch.from_numpy(obs)
            .float()
            .to(device)
        )

        with torch.no_grad():
            next_value = (
                critic(
                    obs_tensor
                )
                .squeeze(-1)
                .cpu()
            )

        # ====================================================
        # GAE
        # ====================================================

        (
            advantages,
            value_targets
        ) = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA
        )

        # ====================================================
        # PPO
        # ====================================================

        metrics = ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            value_targets=value_targets,
            epochs=PPO_EPOCHS,
            batch_size=BATCH_SIZE,
            clip_epsilon=CLIP_EPSILON,
            entropy_coef=ENTROPY_COEF
        )

        # ====================================================
        # Statistics
        # ====================================================

        if len(
            episode_lengths
        ) > 0:
            mean_episode_length = float(
                np.mean(
                    episode_lengths
                )
            )

            mean_episode_return = float(
                np.mean(
                    episode_returns
                )
            )
        else:
            mean_episode_length = 0.0
            mean_episode_return = 0.0

        # Read ACTUAL values after this PPO update.
        current_actor_lr = float(
            actor_optimizer.param_groups[0]["lr"]
        )
        current_critic_lr = float(
            critic_optimizer.param_groups[0]["lr"]
        )

        current_dist_cfg = (
            get_actor_distribution_config(
                actor
            )
        )

        # ====================================================
        # Print
        # ====================================================

        print(
            f"Update {update:02d} | "
            f"steps={total_env_steps:6d} | "
            f"episodes={len(episode_lengths):2d} | "
            f"ep_len={mean_episode_length:7.2f} | "
            f"return={mean_episode_return:8.2f} | "
            f"actor={metrics['actor_loss']:+.4f} | "
            f"critic={metrics['critic_loss']:.2f} | "
            f"entropy={metrics['entropy']:.3f} | "
            f"KL={metrics['approx_kl']:.4f} | "
            f"clip={metrics['clip_fraction']:.3f} | "
            f"a_lr={current_actor_lr:.2e} | "
            f"c_lr={current_critic_lr:.2e} | "
            f"ent_coef={ENTROPY_COEF:g} | "
            f"log_std={current_dist_cfg['log_std_mean']:.3f} | "
            f"std={current_dist_cfg['std_mean']:.4f} | "
            f"std_trainable={current_dist_cfg['log_std_trainable']} | "
            f"resets={reset_counts['nominal']}/{reset_counts['randomized']}"
        )

        # Refresh config snapshot so checkpoints also contain the
        # distribution state observed after this update.
        checkpoint_config = {
            **run_config,
            "current_actor_lr": current_actor_lr,
            "current_critic_lr": current_critic_lr,
            "current_log_std_mean":
                current_dist_cfg["log_std_mean"],
            "current_std_mean":
                current_dist_cfg["std_mean"],
            "current_log_std_trainable":
                current_dist_cfg["log_std_trainable"],
            "reset_counts": dict(reset_counts),
        }

        payload = make_checkpoint_payload(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            update=update,
            total_env_steps=total_env_steps,
            mean_episode_length=mean_episode_length,
            mean_episode_return=mean_episode_return,
            run_config=checkpoint_config
        )

        # ====================================================
        # Save best checkpoint
        # ====================================================

        if (
            mean_episode_length
            >
            best_mean_episode_length
        ):
            best_mean_episode_length = (
                mean_episode_length
            )

            torch.save(
                payload,
                checkpoint_dir
                / "best.pt"
            )

        # Always keep the most recent policy.
        torch.save(
            payload,
            checkpoint_dir
            / "last.pt"
        )

        # During this short pilot, preserve every update so a noisy
        # stochastic "best" checkpoint does not hide a better
        # deterministic policy from another update.
        if (
            CHECKPOINT_EVERY > 0
            and update % CHECKPOINT_EVERY == 0
        ):
            torch.save(
                payload,
                checkpoint_dir
                / f"update_{update:03d}.pt"
            )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        "Training finished."
    )
    print(
        "Best mean episode length:",
        best_mean_episode_length
    )


if __name__ == "__main__":
    main()
