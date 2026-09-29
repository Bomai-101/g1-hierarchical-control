import numpy as np
import torch

from g1_env import G1Env
from networks import Actor, Critic
from buffer import RolloutBuffer
from ppo import compute_gae, ppo_update
import experiment_config as config


OBS_DIM = config.OBS_DIM
ACTION_DIM = config.ACTION_DIM

ROLLOUT_STEPS = config.SMOKE_ROLLOUT_STEPS


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

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ========================================================
    # Device
    # ========================================================

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

    # ========================================================
    # Environment
    # ========================================================

    env = G1Env()

    print(
        "Environment action dim:",
        env.ACTION_DIM
    )

    # ========================================================
    # Networks
    # ========================================================

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
            lr=config.ACTOR_LR
        )
    )

    critic_optimizer = (
        torch.optim.Adam(
            critic.parameters(),
            lr=config.CRITIC_LR
        )
    )

    buffer = RolloutBuffer()

    obs = env.reset()

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    # ========================================================
    # Collect real 6D G1 rollout
    # ========================================================

    for step in range(
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

            # Raw Gaussian action:
            # shape = [6]
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

            # Real action sent to environment
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

        obs = next_obs

        if episode_end:

            episode_count += 1

            print(
                f"Episode {episode_count:2d} | "
                f"rollout_step={step:3d} | "
                f"length={episode_length:3d} | "
                f"return={episode_reward:7.2f}"
            )

            obs = env.reset()

            episode_reward = 0.0
            episode_length = 0

    # ========================================================
    # Stack buffer
    # ========================================================

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

    print()
    print(
        "Collected transitions:",
        len(buffer)
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

    # ========================================================
    # Bootstrap
    # ========================================================

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

    # ========================================================
    # GAE
    # ========================================================

    (
        advantages,
        value_targets
    ) = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA
    )

    print(
        "Advantages:",
        advantages.shape
    )

    print(
        "Advantage mean:",
        advantages.mean().item()
    )

    # ========================================================
    # PPO update
    # ========================================================

    actor_before = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

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
        epochs=config.PPO_EPOCHS,
        batch_size=config.BATCH_SIZE,
        clip_epsilon=config.CLIP_EPSILON,
        entropy_coef=config.ENTROPY_COEF
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
        "Actor loss:",
        metrics["actor_loss"]
    )

    print(
        "Critic loss:",
        metrics["critic_loss"]
    )

    print(
        "Entropy:",
        metrics["entropy"]
    )

    print(
        "Approx KL:",
        metrics["approx_kl"]
    )

    print(
        "Clip fraction:",
        metrics[
            "clip_fraction"
        ]
    )

    buffer.clear()

    print(
        "Buffer after clear:",
        len(buffer)
    )


if __name__ == "__main__":
    main()