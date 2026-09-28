# Day 7 — PPO Core Implementation

## Goal

Build and understand a minimal Proximal Policy Optimization (PPO) pipeline before integrating reinforcement learning with the Unitree G1 MuJoCo environment.

---

## 1. PPO Concepts Covered

### Transition

A single environment transition is represented as:

\[
(s_t, a_t, r_t, s_{t+1}, done_t)
\]

A rollout consists of multiple sequential transitions collected using the current policy.

### Actor

The Actor receives the current observation and defines a Gaussian action distribution.

Current planned G1 dimensions:

- Observation dimension: 64
- Action dimension: 29

Observation:

\[
29q + 29\dot q + 3RPY + 3gyro = 64
\]

The Actor architecture is:

```text
64
↓
256
↓
256
↓
29-dimensional mean μ
```

A learnable `log_std` parameter is maintained separately from the MLP layers.

\[
\sigma = e^{log\_std}
\]

The policy distribution is:

\[
a_t \sim \mathcal{N}(\mu(s_t), \sigma)
\]

Both the neural-network parameters and `log_std` are optimized through backpropagation.

### Critic

The Critic estimates:

\[
V(s_t)
\]

Architecture:

```text
64
↓
256
↓
256
↓
1 scalar value
```

---

## 2. Advantage and TD Error

TD error:

\[
\delta_t =
r_t +
\gamma V(s_{t+1})
-
V(s_t)
\]

The true advantage is:

\[
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
\]

Because the exact \(Q^\pi(s,a)\) requires an expectation over many possible future trajectories, it is expensive and high-variance to estimate directly.

TD error therefore provides a one-step estimate of advantage.

---

## 3. Generalized Advantage Estimation

GAE combines multiple future TD errors:

\[
\hat A_t
=
\delta_t
+
\gamma\lambda\hat A_{t+1}
\]

The implementation calculates GAE backwards through the rollout.

Episode termination is handled using a `done` mask so that advantages from a new episode are not propagated into the previous episode.

Each transition receives its own advantage estimate.

Value targets for Critic training are calculated as:

\[
V^{target}_t
=
\hat A_t + V_{old}(s_t)
\]

---

## 4. Rollout Buffer

Implemented a `RolloutBuffer` storing:

```text
observation
action
reward
value
old_log_prob
done
```

Tensor values from the old policy are detached before storage.

`.detach()` preserves the numerical value while removing the old autograd computation graph.

This:

- avoids retaining unnecessary computation graphs,
- reduces memory usage,
- prevents gradients from flowing into the old rollout.

After PPO finishes using a rollout, the buffer is cleared and new data is collected using the updated policy.

---

## 5. PPO Probability Ratio

For each stored action:

\[
ratio_t
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{old}(a_t|s_t)
}
\]

In practice:

```python
ratio = torch.exp(
    new_log_prob - old_log_prob
)
```

Log probabilities are used for numerical stability, especially for the 29-dimensional G1 action vector.

The joint log probability is calculated by summing the individual action-dimension log probabilities.

---

## 6. PPO Clipping

The PPO clipped objective is:

\[
L^{CLIP}
=
\mathbb{E}
[
\min(
ratio_t A_t,
clip(ratio_t, 1-\epsilon, 1+\epsilon)A_t
)
]
\]

Current clipping parameter:

\[
\epsilon = 0.2
\]

The clipping mechanism prevents the new policy from moving too far away from the policy that generated the rollout.

The old policy remains fixed as the reference throughout all mini-batches and epochs for the same rollout.

---

## 7. PPO Losses

### Actor Loss

```python
actor_loss = -torch.min(
    surrogate1,
    surrogate2
).mean()
```

### Critic Loss

Mean-squared error between predicted value and value target:

\[
L_{critic}
=
(V_\phi(s)-V^{target})^2
\]

### Entropy Bonus

Entropy encourages exploration and prevents the Gaussian policy from becoming deterministic too quickly.

The Actor objective includes:

```text
actor loss
-
entropy coefficient × entropy
```

---

## 8. Mini-Batch and Epoch Training

Temporal order is preserved while collecting the rollout and calculating GAE.

Only after all advantages and value targets have been calculated is the rollout shuffled into mini-batches.

Example:

```text
2048 transitions
↓
calculate TD errors and GAE in temporal order
↓
shuffle samples
↓
64 samples / mini-batch
↓
32 mini-batches / epoch
↓
multiple PPO epochs
```

This is valid for the current feed-forward MLP policy because each transition already contains its calculated advantage and value target.

---

## 9. Implementation Files

Day 7 PPO implementation:

```text
code/day7/g1_ppo/

networks.py
    Actor
    Critic
    Gaussian policy
    learnable log_std

buffer.py
    RolloutBuffer
    detach and clear operations

ppo.py
    compute_gae()
    ppo_update()
    PPO clipping
    Actor/Critic optimization

train_fake.py
    end-to-end fake PPO training pipeline
```

---

## 10. Sanity-Test Results

The fake PPO update test successfully verified the complete gradient pipeline.

Initial probability ratio:

```text
Mean ratio first batch:
1.0000001192092896
```

This is effectively:

\[
ratio \approx 1
\]

as expected because the current and old policies are initially identical.

Parameter update checks:

```text
Actor weight changed: True
Critic weight changed: True
log_std changed: True
```

Mean parameter changes:

```text
Actor:  ~0.00108
Critic: ~0.00120
```

The initial 29-dimensional Gaussian entropy was approximately:

```text
41.16
```

which is consistent with 29 independent standard-normal action dimensions.

These tests confirm that:

```text
forward pass
→ log probability
→ PPO ratio
→ clipping
→ Actor/Critic losses
→ backward()
→ optimizer.step()
```

works correctly.

---

## 11. Current Limitation

The PPO implementation currently operates with fake observations and rewards.

It is not yet connected to:

- MuJoCo physics,
- G1 joint states,
- PD control,
- G1 reward functions,
- termination conditions.

GPU acceleration also still needs to be configured because the current WSL PyTorch installation reports a CUDA-driver compatibility warning.

---

## 12. Next Step — Day 8

Integrate the PPO skeleton with the Unitree G1 MuJoCo model.

Planned pipeline:

```text
G1 MuJoCo state
↓
64D observation
↓
Actor
↓
29D action
↓
joint target scaling
↓
PD controller
↓
torque
↓
MuJoCo physics
↓
reward + next observation
↓
RolloutBuffer
↓
GAE
↓
PPO update
```

Before training, inspect the G1 MuJoCo model to determine the exact:

- `qpos` layout,
- `qvel` layout,
- joint addresses,
- actuator mapping,
- floating-base representation,
- simulation timestep.