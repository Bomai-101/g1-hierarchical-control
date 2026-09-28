import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Generalized Advantage Estimation
# ============================================================

def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    gae_lambda=0.95
):
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards:
        Tensor shape [T]

    values:
        Critic predictions V(s_t)
        Tensor shape [T]

    dones:
        Episode boundary mask.
        1.0 means the episode ended after this transition.
        Tensor shape [T]

    next_value:
        V(s_{T}) used to bootstrap the final rollout state.

    gamma:
        Discount factor.

    gae_lambda:
        GAE lambda.

    Returns
    -------
    advantages:
        Tensor shape [T]

    value_targets:
        Tensor shape [T]
        Used to train the Critic.
    """

    rewards = rewards.float()
    values = values.float()
    dones = dones.float()

    # Same device as rollout tensors.
    advantages = torch.zeros_like(
        rewards
    )

    gae = torch.tensor(
        0.0,
        dtype=rewards.dtype,
        device=rewards.device
    )

    next_value = torch.as_tensor(
        next_value,
        dtype=rewards.dtype,
        device=rewards.device
    )

    T = len(rewards)

    # --------------------------------------------------------
    # Compute backwards through the rollout
    # --------------------------------------------------------

    for t in reversed(
        range(T)
    ):

        if t == T - 1:

            # Final transition:
            # bootstrap from state after the rollout.
            next_val = next_value

        else:

            # Otherwise V(s_{t+1})
            next_val = values[t + 1]

        # If episode terminated here:
        # nonterminal = 0
        #
        # Therefore no value / GAE is propagated
        # across the episode boundary.
        nonterminal = (
            1.0
            - dones[t]
        )

        # TD error:
        #
        # delta_t =
        # r_t
        # + gamma * V(s_{t+1})
        # - V(s_t)
        delta = (
            rewards[t]
            + gamma
            * next_val
            * nonterminal
            - values[t]
        )

        # GAE:
        #
        # A_t =
        # delta_t
        # + gamma * lambda
        #   * A_{t+1}
        gae = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * gae
        )

        advantages[t] = gae

    # --------------------------------------------------------
    # Critic regression target
    # --------------------------------------------------------

    value_targets = (
        advantages
        + values
    )

    return (
        advantages,
        value_targets
    )


# ============================================================
# PPO Update
# ============================================================

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
    batch_size=128,
    clip_epsilon=0.2,
    entropy_coef=0.01
):
    """
    Perform PPO optimization on one rollout.

    IMPORTANT:
    ---------
    `actions` contains RAW Gaussian actions u.

        u ~ Normal(mu, sigma)

    The environment actually executes:

        a = tanh(u)

    We keep the raw Gaussian action in the rollout buffer
    because PPO probability ratios are evaluated using
    the same sampled raw action.

    Parameters
    ----------
    actor:
        Actor neural network.

    critic:
        Critic neural network.

    actor_optimizer:
        Optimizer for Actor.

    critic_optimizer:
        Optimizer for Critic.

    observations:
        [T, obs_dim]

    actions:
        Raw Gaussian actions.
        [T, action_dim]

    old_log_probs:
        Log probability under rollout policy.
        [T]

    advantages:
        GAE advantages.
        [T]

    value_targets:
        Critic targets.
        [T]
    """

    # ========================================================
    # Find network device
    # ========================================================

    device = next(
        actor.parameters()
    ).device

    # ========================================================
    # Advantage normalization
    # ========================================================

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std()
        + 1e-8
    )

    num_samples = (
        observations.shape[0]
    )

    # ========================================================
    # Metrics
    # ========================================================

    metric_actor_loss = []
    metric_critic_loss = []
    metric_entropy = []
    metric_kl = []
    metric_clip_fraction = []

    # ========================================================
    # Multiple PPO epochs
    # ========================================================

    for epoch in range(
        epochs
    ):

        # Random permutation of rollout samples.
        indices = torch.randperm(
            num_samples
        )

        # ====================================================
        # Mini-batches
        # ====================================================

        for start in range(
            0,
            num_samples,
            batch_size
        ):

            end = (
                start
                + batch_size
            )

            batch_idx = (
                indices[start:end]
            )

            # ------------------------------------------------
            # CPU rollout buffer
            #       ↓
            # GPU mini-batch
            # ------------------------------------------------

            obs_batch = (
                observations[
                    batch_idx
                ]
                .to(device)
            )

            action_batch = (
                actions[
                    batch_idx
                ]
                .to(device)
            )

            old_log_prob_batch = (
                old_log_probs[
                    batch_idx
                ]
                .to(device)
            )

            advantage_batch = (
                advantages[
                    batch_idx
                ]
                .to(device)
            )

            value_target_batch = (
                value_targets[
                    batch_idx
                ]
                .to(device)
            )

            # =================================================
            # ACTOR
            # =================================================

            dist = actor(
                obs_batch
            )

            # -------------------------------------------------
            # Probability of SAME raw actions
            # under current policy.
            #
            # 29 independent Gaussian action dimensions:
            #
            # log π(a|s)
            # =
            # sum_i log π(a_i|s)
            # -------------------------------------------------

            new_log_prob = (
                dist
                .log_prob(
                    action_batch
                )
                .sum(
                    dim=-1
                )
            )

            # -------------------------------------------------
            # PPO probability ratio
            #
            # ratio =
            # π_new(a|s)
            # /
            # π_old(a|s)
            #
            # Using log probabilities:
            #
            # exp(
            #   log π_new
            #   -
            #   log π_old
            # )
            # -------------------------------------------------

            ratio = torch.exp(
                new_log_prob
                - old_log_prob_batch
            )

            # Debug sanity check:
            # First mini-batch before any update
            # should be approximately 1.
            if (
                epoch == 0
                and start == 0
            ):

                print(
                    "Mean ratio first batch:",
                    ratio.mean().item()
                )

            # -------------------------------------------------
            # PPO surrogate objective
            # -------------------------------------------------

            surrogate_1 = (
                ratio
                * advantage_batch
            )

            clipped_ratio = (
                torch.clamp(
                    ratio,
                    1.0
                    - clip_epsilon,
                    1.0
                    + clip_epsilon
                )
            )

            surrogate_2 = (
                clipped_ratio
                * advantage_batch
            )

            # PPO maximizes:
            #
            # min(
            #   ratio * A,
            #   clipped_ratio * A
            # )
            #
            # PyTorch optimizer minimizes,
            # therefore negative sign.
            actor_loss = -(
                torch.min(
                    surrogate_1,
                    surrogate_2
                )
                .mean()
            )

            # -------------------------------------------------
            # Entropy bonus
            #
            # Current implementation uses entropy of the
            # raw Gaussian policy.
            # -------------------------------------------------

            entropy = (
                dist
                .entropy()
                .sum(
                    dim=-1
                )
                .mean()
            )

            actor_total_loss = (
                actor_loss
                - entropy_coef
                * entropy
            )

            # -------------------------------------------------
            # Actor gradient update
            # -------------------------------------------------

            actor_optimizer.zero_grad()

            actor_total_loss.backward()

            actor_optimizer.step()

            # =================================================
            # PPO Diagnostics
            # =================================================

            with torch.no_grad():

                # Approximate KL divergence.
                #
                # Small value:
                # policy has not moved much.
                approx_kl = (
                    old_log_prob_batch
                    - new_log_prob
                ).mean()

                # Fraction of samples whose ratio moved
                # outside clipping interval.
                clip_fraction = (
                    (
                        torch.abs(
                            ratio
                            - 1.0
                        )
                        > clip_epsilon
                    )
                    .float()
                    .mean()
                )

            # =================================================
            # CRITIC
            # =================================================

            predicted_value = (
                critic(
                    obs_batch
                )
                .squeeze(-1)
            )

            # MSE:
            #
            # predicted V(s)
            # vs
            # GAE-derived value target.
            critic_loss = (
                F.mse_loss(
                    predicted_value,
                    value_target_batch
                )
            )

            # -------------------------------------------------
            # Critic gradient update
            # -------------------------------------------------

            critic_optimizer.zero_grad()

            critic_loss.backward()

            critic_optimizer.step()

            # =================================================
            # Record metrics
            # =================================================

            metric_actor_loss.append(
                actor_loss.item()
            )

            metric_critic_loss.append(
                critic_loss.item()
            )

            metric_entropy.append(
                entropy.item()
            )

            metric_kl.append(
                approx_kl.item()
            )

            metric_clip_fraction.append(
                clip_fraction.item()
            )

    # ========================================================
    # Mean metrics over ALL epochs / mini-batches
    # ========================================================

    metrics = {
        "actor_loss":
            float(
                np.mean(
                    metric_actor_loss
                )
            ),

        "critic_loss":
            float(
                np.mean(
                    metric_critic_loss
                )
            ),

        "entropy":
            float(
                np.mean(
                    metric_entropy
                )
            ),

        "approx_kl":
            float(
                np.mean(
                    metric_kl
                )
            ),

        "clip_fraction":
            float(
                np.mean(
                    metric_clip_fraction
                )
            ),
    }

    return metrics