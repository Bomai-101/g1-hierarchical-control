import numpy as np
import torch

from g1_env import G1Env
from networks import Actor, Critic
from buffer import RolloutBuffer
from ppo import compute_gae, ppo_update


OBS_DIM = 64
ACTION_DIM = 29

ROLLOUT_STEPS = 256


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
    ).squeeze(-1)

    old_log_probs = torch.stack(
        buffer.log_probs
    )

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
    # Environment
    # =========================

    env = G1Env()

    # =========================
    # Networks
    # =========================

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    )

    critic = Critic(
        obs_dim=OBS_DIM
    )

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=3e-4
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=3e-4
    )

    # =========================
    # Buffer
    # =========================

    buffer = RolloutBuffer()

    obs = env.reset()

    episode_reward = 0.0
    episode_count = 0

    # =========================
    # Collect REAL rollout
    # =========================

    for step in range(ROLLOUT_STEPS):

        obs_tensor = torch.from_numpy(
            obs
        ).float()

        with torch.no_grad():

            dist = actor(
                obs_tensor
            )

            # Raw Gaussian action
            raw_action = dist.sample()

            log_prob = (
                dist
                .log_prob(raw_action)
                .sum()
            )

            value = critic(
                obs_tensor
            )

            # Actual bounded action
            env_action_tensor = torch.tanh(
                raw_action
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

        # Important:
        # Buffer stores RAW Gaussian action
        # because PPO log_prob is based on it.
        buffer.add(
            observation=obs_tensor,
            action=raw_action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=terminated
        )

        episode_reward += reward

        obs = next_obs

        # -------------------------
        # Episode reset
        # -------------------------

        if terminated or truncated:

            episode_count += 1

            print(
                f"Episode {episode_count} | "
                f"step={step} | "
                f"reward={episode_reward:.2f}"
            )

            obs = env.reset()

            episode_reward = 0.0

    print()

    print(
        "Collected transitions:",
        len(buffer)
    )

    # =========================
    # Convert Buffer
    # =========================

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

    print(
        "Observations:",
        observations.shape
    )

    print(
        "Raw actions:",
        actions.shape
    )

    print(
        "Rewards:",
        rewards.shape
    )

    # =========================
    # Bootstrap final state
    # =========================

    obs_tensor = torch.from_numpy(
        obs
    ).float()

    with torch.no_grad():

        next_value = critic(
            obs_tensor
        ).squeeze(-1)

    # =========================
    # GAE
    # =========================

    advantages, value_targets = (
        compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=0.99,
            gae_lambda=0.95
        )
    )

    print(
        "Advantages:",
        advantages.shape
    )

    print(
        "Advantage mean:",
        advantages.mean().item()
    )

    # =========================
    # Save parameter snapshot
    # =========================

    actor_before = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    # =========================
    # REAL PPO Update
    # =========================

    ppo_update(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        value_targets=value_targets,
        epochs=3,
        batch_size=64,
        clip_epsilon=0.2,
        entropy_coef=0.01
    )

    actor_after = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    print()

    print(
        "Actor changed:",
        not torch.equal(
            actor_before,
            actor_after
        )
    )

    print(
        "Mean actor parameter change:",
        (
            actor_after
            - actor_before
        )
        .abs()
        .mean()
        .item()
    )

    buffer.clear()

    print(
        "Buffer after clear:",
        len(buffer)
    )


if __name__ == "__main__":
    main()