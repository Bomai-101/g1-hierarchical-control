# Day 6 — Reinforcement Learning and PPO Foundations

## Goal

Extend the existing G1 observation-policy-action architecture from a
hand-written scripted policy toward the foundations required for a
learned reinforcement learning policy.

Topics covered:

- environment, state, observation, and action
- reward and reward design
- return and discount factor
- value functions
- Bellman recursion
- bootstrapping
- TD error
- advantage
- actor-critic
- GAE
- policy gradient
- PPO clipping
- rollout, batch, mini-batch, and epoch
- mapping PPO back to the G1 + MuJoCo architecture

---

## 1. Reinforcement Learning Interaction Loop

The basic RL interaction loop is:

Observation_t
      ↓
Policy
      ↓
Action_t
      ↓
Environment
      ↓
Reward_t
+
Observation_t+1

For the current project:
Environment
=
MuJoCo
+
G1 model
+
gravity
+
contacts
+
simulation dynamics

## 2. State and Observation

State represents the complete physical state of the environment.
Observation represents the information actually provided to the policy.
The current project uses a 64-dimensional observation:
29 joint positions q
29 joint velocities dq
3 RPY values
3 gyroscope values
----------------------
64 dimensions

The policy does not necessarily observe the complete simulator state.

## 3. Action

The policy produces:
Action[29]

In the current architecture, the action represents relative
joint-position offsets.
q_target = reference_q + action

An RL action space could alternatively represent:
- joint positions
- joint-position offsets
- joint velocities
- torques
- task-space commands
The meaning of the action space is a controller-design choice.

## 4. Reward

Reward is a scalar evaluation of the result of an interaction step.
The policy selects the action.
The environment and reward function determine the reward.
Example standing reward terms may include:
upright posture
desired height
low body velocity
low angular velocity
reasonable joint posture
low energy consumption
fall penalty

A reward function may be expressed conceptually as:
reward =
  upright_reward
+ height_reward
+ stability_reward
- energy_penalty
- fall_penalty

Reward design must be handled carefully because an agent optimizes the
implemented reward, not the designer's unstated intention.
This can lead to reward hacking or specification gaming.

## 5. Dense and Sparse Rewards

Sparse reward:
success -> reward
failure -> no reward

provides limited learning information.
Dense reward provides feedback during intermediate steps.
Dense rewards can improve learning signals but poorly designed terms may
conflict with each other.

## 6. Return and Discount Factor

Immediate reward:
r_t

Return represents discounted future reward:
G_t =
r_t
+ gamma r_t+1
+ gamma^2 r_t+2
+ ...

where:
0 <= gamma <= 1

The discount factor gamma controls how strongly distant future rewards
affect the current return.
A larger gamma generally places more importance on long-term outcomes.

## 7. State Value

The state-value function is:
V^pi(s)

and represents the expected return when starting in state s and then
following policy pi.
V^pi(s)
=
E_pi[G_t | S_t = s]

It can also be written as the policy-weighted expectation over action
values:
V^pi(s)
=
E_{a ~ pi}[Q^pi(s,a)]

## 8. Action Value

The action-value function is:
Q^pi(s,a)

It represents the expected return when:
1. the current state is s,
2. the current action is fixed to a,
3. subsequent actions follow policy pi.
Therefore:
Q^pi(s,a)
=
E[
  r_t + gamma V^pi(s_t+1)
]

or equivalently:
Q^pi(s_t,a_t)
=
E[
  r_t
  + gamma Q^pi(s_t+1,a_t+1)
]

with:
a_t+1 ~ pi(. | s_t+1)

## 9. Advantage

Advantage measures whether an action performs better or worse than the
policy's expected performance in the same state.
A^pi(s,a)
=
Q^pi(s,a) - V^pi(s)

Interpretation:
A > 0
-> action is better than the policy baseline

A < 0
-> action is worse than the policy baseline

Advantage is useful for policy updates because it compares an action
against the expected performance of the current policy.
10. Bellman Recursion
Return satisfies:
G_t
=
r_t + gamma G_t+1

This gives the Bellman recursion:
V^pi(s_t)
=
E[
  r_t
  + gamma V^pi(s_t+1)
]

The future value term compresses information about future rewards into a
single value estimate.

## 11. Bootstrapping

In practice, the true value function is unknown.
A critic estimates it:
V_phi(s)

A one-step target can be constructed using:
target_t
=
r_t
+ gamma V_phi(s_t+1)

Using the estimated value of the next state to help train the current
value estimate is called bootstrapping.
This avoids requiring every update to wait until the entire episode has
finished.

## 12. TD Error
The one-step temporal-difference error is:
delta_t
=
r_t
+ gamma V(s_t+1)
- V(s_t)

It measures the difference between:
current value prediction

and:
one-step Bellman target

A positive TD error means the sampled transition produced better
one-step evidence than the critic predicted.
A negative TD error means it produced worse evidence.

## 13. Actor-Critic

The Actor represents the policy:
pi_theta(a | s)

and selects actions.
The Critic estimates value:
V_phi(s)

The critic does not directly modify the actor.
Instead:
Critic estimates value
        ↓
Advantage is estimated
        ↓
Actor loss uses advantage
        ↓
Backpropagation updates actor parameters theta

Actor and Critic may therefore have separate optimization objectives.

## 14. Continuous Action Policy

For continuous humanoid control, the Actor may output parameters of an
action distribution.
For example:
mu(s)
sigma(s)

for a Gaussian distribution:
a_t ~ Normal(mu(s_t), sigma(s_t))

The sampled result becomes the 29-dimensional action vector.

## 15. Generalized Advantage Estimation

One-step TD error can be noisy.
Generalized Advantage Estimation combines multiple future TD errors:
A_hat_t =
delta_t
+ gamma lambda delta_t+1
+ (gamma lambda)^2 delta_t+2
+ ...

gamma controls future reward discounting.
lambda controls how strongly multi-step TD information contributes to
the advantage estimate.
A smaller lambda relies more strongly on short-horizon TD estimates.
A larger lambda incorporates longer-horizon information.
GAE provides a bias-variance tradeoff.

## 16. Policy Gradient

Policy gradient updates the Actor so that actions with positive
advantage become more likely and actions with negative advantage become
less likely.
Conceptually:
A > 0
-> increase tendency toward sampled action

A < 0
-> decrease tendency toward sampled action

A common policy-gradient expression is:
grad J(theta)
≈
E[
  A_t grad log pi_theta(a_t | s_t)
]

## 17. PPO

PPO stands for:
Proximal Policy Optimization

The key idea is to improve the policy without allowing a single update
to move too far away from the policy that generated the rollout.
PPO compares the new and old action probabilities:
ratio_t =
pi_theta(a_t | s_t)
/
pi_theta_old(a_t | s_t)

In implementation this is commonly computed from log probabilities:
ratio =
exp(new_log_prob - old_log_prob)

## 18. PPO Clipping

A common PPO clipped objective is:
L_CLIP =
E[
  min(
    ratio_t A_t,
    clip(ratio_t, 1-epsilon, 1+epsilon) A_t
  )
]

For positive advantage, clipping prevents the optimization objective from
continuing to reward excessively large increases in action probability.
For negative advantage, it prevents excessively aggressive decreases.
Proximal refers to keeping the updated policy relatively close to the
old policy.

## 19. Transition, Episode, Rollout, Batch, and Epoch

Transition
One environment interaction:
state / observation
action
reward
next state / observation
done

Episode
One environment trajectory:
reset
↓
environment interaction
↓
termination

A fall may terminate an episode.
Rollout
A collection of experience gathered by PPO before a training phase.
A rollout may contain multiple episodes.
Batch
The complete collected set of rollout samples used for an update.
Mini-batch
A smaller subset of the rollout batch used for one optimization step.
Epoch
One complete training pass over the entire rollout batch.
If:
batch size = 2048
mini-batch size = 256

then:
8 mini-batches per epoch

If PPO trains for four epochs, the same rollout dataset is completely
traversed four times.

## 20. PPO Training Cycle

Old Policy
    ↓

Collect rollout from G1 environments
    ↓

Store:
observation
action
reward
value
old log probability
done
    ↓

Compute:
returns
GAE advantages
    ↓

Shuffle rollout batch
    ↓

Split into mini-batches
    ↓

For several epochs:

  Actor PPO loss
      ↓
  backpropagation
      ↓
  update theta

  Critic value loss
      ↓
  backpropagation
      ↓
  update phi

    ↓

New Policy
    ↓

Collect new rollout
    ↺

Backpropagation and optimizer updates normally occur for each
mini-batch, not only after all epochs have finished.

## 21. Mapping PPO to the G1 Architecture

The current scripted architecture is:
LowState
   ↓
Observation[64]
   ↓
ScriptedPolicy
   ↓
Action[29]
   ↓
reference_q + action
   ↓
LowCmd
   ↓
PD
   ↓
MuJoCo

A learned version becomes:
Observation[64]
      ↓
     Actor
      ↓
Action Distribution
      ↓
sample Action[29]
      ↓
reference_q + action
      ↓
LowCmd
      ↓
PD
      ↓
MuJoCo
      ↓
Reward + New Observation

The Critic operates in parallel:
Observation
    ↓
Critic
    ↓
Value estimate

The rollout data is then used to calculate GAE and train both Actor and
Critic.

## Key Takeaways

By the end of Day 6:
- Understood the RL interaction loop
- Distinguished reward, return, value, Q-value, and advantage
- Understood Bellman recursion
- Understood bootstrapping
- Understood TD error
- Understood Actor-Critic architecture
- Understood GAE and the role of lambda
- Understood policy-gradient intuition
- Understood PPO probability ratios and clipping
- Distinguished transition, episode, rollout, batch, mini-batch, and epoch
- Connected PPO directly to the existing G1 + MuJoCo control pipeline

Day 6 establishes the reinforcement-learning foundation required before
implementing and training a learned humanoid policy.
