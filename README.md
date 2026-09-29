# G1 Hierarchical Control Preparation

A hands-on preparation project for robust hierarchical humanoid control using Unitree G1, MuJoCo, Unitree SDK2, PD control, and reinforcement learning. It supports preparation for the USYD VRI ECE11 project, *Hierarchical Control of Humanoid Robots with Embodied AI*.

The project progressively builds the control stack from low-level robot state interfaces to whole-body hierarchical control and PPO-based learning.

## Current Progress

### Day 1 — Environment Setup ✅  
`notes/day01_environment.md`

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

### Day 9 — Residual PPO Balance and Robustness ✅
`notes/day9_residual_ppo_robustness.md`

- calibrated a standing pose and joint-specific PD baseline
- reduced the learned action from 29D whole-body exploration to a 6D
  lower-body residual
- zero-initialized the Actor output head so training begins at the PD baseline
- fixed `log_std = -3` and separated Actor/Critic learning rates
- added deterministic checkpoint selection and experiment provenance
- evaluated nominal, coarse, and near-zero pitch/pitch-rate disturbances
- trained with 30% nominal and 70% randomized initial states
- swept all 30 checkpoints over 12 nonzero near-zero perturbations

Headline deterministic results:

| Policy / checkpoint | Nominal length | Delta vs zero | Robustness interpretation |
| --- | ---: | ---: | --- |
| Optimized PD zero policy | 215 | — | fixed baseline |
| Original residual PPO, update 1 | 241 | +26 | nominal winner |
| Randomized PPO, update 4 | 232 | +17 | no nonzero mean gain |
| Randomized PPO, update 19 | 200 | -15 | least-negative worst case (-1) |
| Randomized PPO, update 24 | 150 | -65 | directional specialization, not uniform robustness |

Day 9 conclusion:

> Residual PPO improved nominal standing and later learned direction-specific
> corrective behavior, but no checkpoint demonstrated consistent symmetric
> robustness across all nonzero perturbation cases.

The compact data and figures are in `results/day9/`. Full checkpoints and
per-step traces remain local experiment artifacts rather than being committed
in bulk.

---

## Current Control Architecture

```text
64D whole-body state
        ↓
6D residual PPO policy
        ↓
standing pose + bounded joint-target corrections
        ↓
29D target_q
        ↓
joint-space PD controller
        ↓
torque-clipped MuJoCo G1
        ↓
whole-body feedback
```

## Next — Hierarchical Safety and Fallback Control

Day 9 closes the open-ended low-level PPO optimization branch at a documented
research boundary. The next stage restores the missing middle layer of the
original project plan:

```text
high-level command / skill
        ↓
controller interface
        ↓
PD, residual PPO, or future compatible skill executor
        ↓
watchdog / failure detection
        ↓
recovery / fallback
        ↓
G1
```

The next quantitative experiments focus on latency, observation dropout,
disturbance handling, safety intervention, and recovery success. A compatible
pretrained G1 policy may later be added behind the same controller interface;
no pretrained asset is selected yet.

The long-term direction is Physical / Embodied AI with Safety and Reliability
throughout: control → hierarchy → language/vision/memory → planning/world
models → sim-to-real. High-level AI produces skills, goals, poses, velocities,
or recovery requests; it does not directly produce actuator torque.
