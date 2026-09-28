# G1 Hierarchical Control Preparation

A hands-on preparation project for humanoid robot control using Unitree G1, MuJoCo, Unitree SDK2, PD control, and reinforcement learning.

The project progressively builds the control stack from low-level robot state interfaces to whole-body hierarchical control and PPO-based learning.

## Current Progress

### Day 1 — Environment Setup ✅  
`notes/day1_environment_setup.md`

- Unitree SDK2
- MuJoCo
- `unitree_mujoco`
- G1 29-DoF simulation
- WSL2 robotics development environment

---

### Day 2 — State / Action Interface ✅  
`notes/day2_state_action_interface.md`

- `q` / `dq`
- `LowState`
- actuator and sensor mapping
- Unitree SDK2 bridge
- G1 joint indexing
- floating-base state
- quaternion representation
- MuJoCo ↔ Unitree state mapping

---

### Day 3 — PD Joint Tracking ✅  
`notes/day3_pd_joint_tracking.md`

- `LowState` subscriber
- `LowCmd` publisher
- 500 Hz control loop
- left-knee PD tracking
- fixed `+0.1 rad` target
- `Kp` comparison
- CSV logging
- quantitative tracking plots
- feedback-control analysis

---

### Day 4 — Whole-Body Observation / Action ✅  
`notes/day4_whole_body_policy_interface.md`

- 64D whole-body observation
- joint state + IMU
- 29D relative action vector
- multi-joint scripted policy
- policy → target → PD hierarchy
- whole-body policy interface

Current observation design:

```text
29 joint positions
+ 29 joint velocities
+ 3 RPY
+ 3 angular velocities
= 64D
```

---

### Day 5 — Coordinated Hip–Knee–Ankle Control ✅  
`notes/day5_hierarchical_whole_body_control.md`

- coordinated hip–knee–ankle motion
- joint space vs task space
- forward kinematics
- inverse kinematics
- Jacobian
- task-space velocity control
- singularity and pseudoinverse
- redundancy and null space
- task-priority hierarchical control

---

### Day 6 — Reinforcement Learning Foundations ✅  
`notes/day6_reinforcement_learning_foundations.md`

- reward
- return
- state value \(V(s)\)
- action value \(Q(s,a)\)
- advantage
- Bellman recursion
- bootstrapping
- TD error
- Actor–Critic
- Generalized Advantage Estimation
- Policy Gradient
- PPO clipping
- rollout / batch / mini-batch / epoch
- on-policy learning
- mapping PPO to G1 + MuJoCo

---

### Day 7 — PPO Core Implementation ✅  
`notes/day7_PPO_core_implementation.md`

- Actor network
- Critic network
- 29D Gaussian policy
- learnable `log_std`
- RolloutBuffer
- `.detach()` and rollout storage
- TD error
- GAE recursion
- value targets
- PPO probability ratio
- PPO clipping
- Actor loss
- Critic loss
- entropy bonus
- mini-batch training
- multiple PPO epochs
- fake PPO update sanity test
- end-to-end fake PPO training pipeline

Day 7 established the algorithmic PPO layer independently from the real G1 simulator.

Core pipeline:

```text
Fake Observation
      ↓
Actor / Critic
      ↓
Gaussian Action
      ↓
RolloutBuffer
      ↓
GAE
      ↓
PPO Clipped Update
      ↓
Updated Actor / Critic
```

---

### Day 8 — G1 / MuJoCo PPO Integration ✅  
`notes/day8_g1_ppo_integration.md`

- inspected real G1 MuJoCo model
- verified:
  - `nq = 36`
  - `nv = 35`
  - `nu = 29`
  - physics timestep = `0.002 s`
- real 64D G1 observation
- quaternion → RPY conversion
- 29D PPO action interface
- `tanh`-bounded actions
- standing reference pose
- joint-specific PD control
- action → `q_target` → PD → torque
- 500 Hz physics / 50 Hz policy
- environment `reset()`
- reward function
- termination / truncation
- zero-action PD baseline
- Actor → real G1 integration
- MuJoCo Viewer
- real RolloutBuffer collection
- real GAE calculation
- shuffled mini-batch PPO
- GPU Actor / Critic training
- repeated rollout → PPO update loop
- checkpoint saving
- deterministic policy evaluation
- baseline comparison

Complete learning pipeline:

```text
G1 MuJoCo State
        ↓
64D Observation
        ↓
PPO Actor
        ↓
Gaussian Raw Action
        ↓
tanh
        ↓
29D Bounded Action
        ↓
Joint Target
        ↓
PD Controller
        ↓
Torque
        ↓
MuJoCo Physics
        ↓
Reward / Next State
        ↓
RolloutBuffer
        ↓
GAE
        ↓
Mini-batch PPO Update
```

First repeated PPO experiment:

```text
40,960 environment transitions
Rollout size: 2048
Mini-batch size: 128
PPO epochs: 3

Actor LR: 1e-4
Critic LR: 3e-4
Initial log_std: -1.0
Entropy coefficient: 0.001
```

Deterministic evaluation:

| Policy | Episode Length | Return | Max |Pitch| | Min Height |
| --- | ---: | ---: | ---: | ---: |
| Zero-action PD | 52 | 98.54 | 0.814 | 0.599 |
| Untrained Actor | 52 | 98.48 | 0.814 | 0.598 |
| Trained Actor | 52 | 98.49 | 0.813 | 0.599 |

Current finding:

> The PPO–G1 training pipeline is functional, but the current task formulation does not yet learn an effective standing-balance correction policy.

This suggests that the next bottleneck is no longer PPO implementation, but control/task design.

---

## Next — Day 9

Focus:

- action authority experiments
- action scale comparison
- reward shaping
- angular-velocity-aware balance reward
- fall penalty
- lower-dimensional action-space experiments
- lower-body + waist control
- PPO retraining
- PD-only vs PPO+PD comparison

Main question:

> What observation, action, reward, and hierarchical-control formulation enables PPO to learn effective humanoid standing balance?

---

## Current Control Architecture

```text
High-Level PPO Policy
        ↓
Joint Target Corrections
        ↓
Low-Level PD Controller
        ↓
Joint Torque
        ↓
MuJoCo G1
        ↓
Whole-Body Feedback
        └────────────→ PPO
```

The long-term goal is to extend this system toward robust humanoid control, disturbance recovery, and Embodied AI research.