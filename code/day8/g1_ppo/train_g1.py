from pathlib import Path

import numpy as np
import torch

from g1_env import G1Env
from networks import Actor, Critic
from buffer import RolloutBuffer
from ppo import compute_gae, ppo_update


OBS_DIM = 64
ACTION_DIM = 29

ROLLOUT_STEPS = 2048
NUM_UPDATES = 20

GAMMA = 0.99
GAE_LAMBDA = 0.95

PPO_EPOCHS = 3
BATCH_SIZE = 128

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4

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


def main():

    torch.manual_seed(42)
    np.random.seed(42)

    # =========================
    # Device
    # =========================

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

    # =========================
    # Environment
    # =========================

    env = G1Env()

    # =========================
    # Networks
    # =========================

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    ).to(device)

    critic = Critic(
        obs_dim=OBS_DIM
    ).to(device)

    actor_optimizer = (
        torch.optim.Adam(
            actor.parameters(),
            lr=ACTOR_LR
        )
    )

    critic_optimizer = (
        torch.optim.Adam(
            critic.parameters(),
            lr=CRITIC_LR
        )
    )

    buffer = RolloutBuffer()

    # =========================
    # Checkpoints
    # =========================

    checkpoint_dir = (
        Path(__file__).parent
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    # =========================
    # Initial state
    # =========================

    obs = env.reset()

    episode_reward = 0.0
    episode_length = 0

    total_env_steps = 0

    best_mean_length = -1.0

    # =========================
    # PPO training
    # =========================

    for update in range(
        1,
        NUM_UPDATES + 1
    ):

        buffer.clear()

        completed_rewards = []
        completed_lengths = []

        # =====================
        # Collect rollout
        # =====================

        for _ in range(
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

                # Raw Gaussian action
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

                # Bounded action
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

            episode_reward += reward
            episode_length += 1

            total_env_steps += 1

            obs = next_obs

            if episode_end:

                completed_rewards.append(
                    episode_reward
                )

                completed_lengths.append(
                    episode_length
                )

                obs = env.reset()

                episode_reward = 0.0
                episode_length = 0

        # =====================
        # Stack rollout
        # =====================

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

        # =====================
        # Bootstrap
        # =====================

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

        # =====================
        # GAE
        # =====================

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

        # =====================
        # PPO Update
        # =====================

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
            clip_epsilon=0.2,
            entropy_coef=0.001
        )

        # =====================
        # Logging
        # =====================

        if completed_rewards:

            mean_reward = float(
                np.mean(
                    completed_rewards
                )
            )

            mean_length = float(
                np.mean(
                    completed_lengths
                )
            )

        else:

            mean_reward = float(
                "nan"
            )

            mean_length = float(
                "nan"
            )

        print(
            f"Update {update:03d} | "
            f"steps={total_env_steps:6d} | "
            f"episodes={len(completed_rewards):3d} | "
            f"return={mean_reward:8.2f} | "
            f"ep_len={mean_length:6.1f} | "
            f"actor={metrics['actor_loss']:7.4f} | "
            f"critic={metrics['critic_loss']:8.2f} | "
            f"entropy={metrics['entropy']:6.2f} | "
            f"KL={metrics['approx_kl']:.5f} | "
            f"clip={metrics['clip_fraction']:.3f}"
        )

        # =====================
        # Save best checkpoint
        # =====================

        if (
            completed_lengths
            and mean_length
            > best_mean_length
        ):

            best_mean_length = (
                mean_length
            )

            torch.save(
                {
                    "actor":
                        actor.state_dict(),

                    "critic":
                        critic.state_dict(),

                    "actor_optimizer":
                        actor_optimizer.state_dict(),

                    "critic_optimizer":
                        critic_optimizer.state_dict(),

                    "update":
                        update,

                    "total_env_steps":
                        total_env_steps,

                    "mean_episode_length":
                        mean_length,

                    "mean_episode_reward":
                        mean_reward,
                },
                checkpoint_dir
                / "best.pt"
            )

        # Periodic checkpoint
        if update % 10 == 0:

            torch.save(
                {
                    "actor":
                        actor.state_dict(),

                    "critic":
                        critic.state_dict(),

                    "update":
                        update,

                    "total_env_steps":
                        total_env_steps,
                },
                checkpoint_dir
                / f"update_{update:03d}.pt"
            )

    print()
    print("Training finished.")
    print(
        "Best mean episode length:",
        best_mean_length
    )


if __name__ == "__main__":
    main()