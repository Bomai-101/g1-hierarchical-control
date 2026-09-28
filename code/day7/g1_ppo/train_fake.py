import torch

from networks import Actor, Critic
from buffer import RolloutBuffer
from ppo import compute_gae, ppo_update


OBS_DIM = 64
ACTION_DIM = 29
ROLLOUT_STEPS = 128


def fake_env_step(action):
    """
    Fake environment:
    - not real MuJoCo
    - only used to produce next_obs / reward / done
    """

    next_obs = torch.randn(OBS_DIM)

    # fake reward:
    # action don't need to be so high, just increase reward a bit
    reward = 1.0 - 0.01 * action.pow(2).sum().item()

    # not let episode end so fast
    done = False

    return next_obs, reward, done


def stack_buffer(buffer):
    """
    change Python list to PyTorch Tensor
    """

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

    # -------------------------
    # 1. Create Actor / Critic
    # -------------------------

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

    # -------------------------
    # 2. Create Buffer
    # -------------------------

    buffer = RolloutBuffer()

    # -------------------------
    # 3. Initial observation
    # -------------------------

    obs = torch.randn(OBS_DIM)

    # -------------------------
    # 4. Collect fake rollout
    # -------------------------

    for t in range(ROLLOUT_STEPS):

        with torch.no_grad():

            dist = actor(obs)

            action = dist.sample()

            log_prob = (
                dist
                .log_prob(action)
                .sum()
            )

            value = critic(obs)

        next_obs, reward, done = \
            fake_env_step(action)

        buffer.add(
            observation=obs,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )

        obs = next_obs

    print(
        "Collected transitions:",
        len(buffer)
    )

    # -------------------------
    # 5. Convert buffer → Tensor
    # -------------------------

    (
        observations,
        actions,
        rewards,
        values,
        old_log_probs,
        dones
    ) = stack_buffer(buffer)

    print(
        "Observations shape:",
        observations.shape
    )

    print(
        "Actions shape:",
        actions.shape
    )

    print(
        "Rewards shape:",
        rewards.shape
    )

    print(
        "Values shape:",
        values.shape
    )

    # -------------------------
    # 6. Bootstrap last state
    # -------------------------

    with torch.no_grad():
        next_value = (
            critic(obs)
            .squeeze(-1)
        )

    # -------------------------
    # 7. Compute GAE
    # -------------------------

    advantages, value_targets = \
        compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=0.99,
            gae_lambda=0.95
        )

    print(
        "Advantages shape:",
        advantages.shape
    )

    print(
        "Value targets shape:",
        value_targets.shape
    )

    print(
        "Advantage mean:",
        advantages.mean().item()
    )

    # -------------------------
    # 8. Save Actor parameter
    # -------------------------

    actor_before = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    # -------------------------
    # 9. PPO Update
    # -------------------------

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
        batch_size=32,
        clip_epsilon=0.2,
        entropy_coef=0.01
    )

    # -------------------------
    # 10. Check parameter change
    # -------------------------

    actor_after = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    actor_changed = not torch.equal(
        actor_before,
        actor_after
    )

    print(
        "Actor changed after PPO:",
        actor_changed
    )

    # -------------------------
    # 11. Clear rollout
    # -------------------------

    buffer.clear()

    print(
        "Buffer length after clear:",
        len(buffer)
    )


if __name__ == "__main__":
    main()