import torch
from networks import Actor, Critic

def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    gae_lambda=0.95
):
    T = len(rewards)

    advantages = torch.zeros(T)

    gae = 0.0

    for t in reversed(range(T)):

        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        nonterminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_val * nonterminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * gae
        )

        advantages[t] = gae

    value_targets = advantages + values

    return advantages, value_targets

def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    observations,
    actions,
    old_log_probs,
    advantages,
    value_targets,
    epochs=5,
    batch_size=64,
    clip_epsilon=0.2,
    entropy_coef=0.01
):
    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std() + 1e-8
    )

    num_samples = observations.shape[0]

    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        for start in range(
            0,
            num_samples,
            batch_size
        ):
            end = start + batch_size
            batch_indices = indices[start:end]
        
            obs_batch = observations[batch_indices]
            action_batch  = actions[batch_indices]
            
            old_log_prob_batch = old_log_probs[batch_indices]

            advantage_batch = advantages[batch_indices]

            target_batch = value_targets[batch_indices]

            # -----------------
            # Actor
            # -----------------

            dist = actor(obs_batch)

            new_log_prob = (dist.log_prob(action_batch).sum(dim=-1))

            ratio = torch.exp(new_log_prob - old_log_prob_batch)

            if epoch == 0 and start == 0:
                print(
                    "Mean ratio first batch:",
                    ratio.mean().item()
                )

            surrogate1 = (ratio * advantage_batch)

            clipped_ratio = torch.clamp(
                ratio,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon
            )

            surrogate2 = (
                clipped_ratio
                * advantage_batch
            )

            actor_loss = -torch.min(
                surrogate1,
                surrogate2
            ).mean()

            entropy = (
                dist
                .entropy()
                .sum(dim=-1)
                .mean()
            )

            actor_total_loss = (
                actor_loss
                - entropy_coef * entropy
            )
            
            actor_optimizer.zero_grad()

            actor_total_loss.backward()

            actor_optimizer.step()

            # -----------------
            # Critic
            # -----------------

            predicted_values = (
                critic(obs_batch).squeeze(-1)
            )

            critic_loss = (
                predicted_values
                - target_batch
            ).pow(2).mean() 

            critic_optimizer.zero_grad()

            critic_loss.backward()

            critic_optimizer.step()

        print(
            f"Epoch {epoch}: "
            f"actor_loss={actor_loss.item():.4f}, "
            f"critic_loss={critic_loss.item():.4f}, "
            f"entropy={entropy.item():.4f}"
        )   


if __name__ == "__main__":

    torch.manual_seed(42)

    # -------------------------
    # 1. Create networks
    # -------------------------

    actor = Actor(
        obs_dim=64,
        action_dim=29
    )

    critic = Critic(
        obs_dim=64
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
    # 2. Fake rollout data
    # -------------------------

    num_samples = 128

    observations = torch.randn(
        num_samples,
        64
    )

    # Use OLD actor to generate actions
    with torch.no_grad():

        old_dist = actor(observations)

        actions = old_dist.sample()

        old_log_probs = (
            old_dist
            .log_prob(actions)
            .sum(dim=-1)
        )

    # Fake advantages
    advantages = torch.randn(
        num_samples
    )

    # Fake critic targets
    value_targets = torch.randn(
        num_samples
    )

    # -------------------------
    # 3. Save parameters BEFORE
    # -------------------------

    actor_weight_before = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    critic_weight_before = (
        critic.net[0]
        .weight
        .detach()
        .clone()
    )

    log_std_before = (
        actor.log_std
        .detach()
        .clone()
    )

    # -------------------------
    # 4. PPO update
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
    # 5. Save parameters AFTER
    # -------------------------

    actor_weight_after = (
        actor.net[0]
        .weight
        .detach()
        .clone()
    )

    critic_weight_after = (
        critic.net[0]
        .weight
        .detach()
        .clone()
    )

    log_std_after = (
        actor.log_std
        .detach()
        .clone()
    )

    # -------------------------
    # 6. Compare
    # -------------------------

    actor_changed = not torch.equal(
        actor_weight_before,
        actor_weight_after
    )

    critic_changed = not torch.equal(
        critic_weight_before,
        critic_weight_after
    )

    log_std_changed = not torch.equal(
        log_std_before,
        log_std_after
    )

    print("Actor weight changed:", actor_changed)
    print("Critic weight changed:", critic_changed)
    print("log_std changed:", log_std_changed)

    print()

    print(
        "Actor mean parameter change:",
        (
            actor_weight_after
            - actor_weight_before
        ).abs().mean().item()
    )

    print(
        "Critic mean parameter change:",
        (
            critic_weight_after
            - critic_weight_before
        ).abs().mean().item()
    )

    print(
        "log_std before:",
        log_std_before
    )

    print(
        "log_std after:",
        log_std_after
    )