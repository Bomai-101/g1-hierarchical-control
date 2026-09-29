# Day 8 — G1 PPO Integration and First Real Training

## Goal

The goal of Day 8 was to move from the fake PPO environment developed on Day 7 to a real Unitree G1 MuJoCo reinforcement-learning environment.

The main objective was not necessarily to solve standing balance immediately.

The objective was to build and validate the complete pipeline:

```text
G1 MuJoCo
→ real observation
→ PPO Actor
→ bounded action
→ PD control
→ torque
→ physics
→ reward
→ rollout
→ GAE
→ PPO update
```

---

## 1. G1 MuJoCo Model Inspection

The 29-DoF G1 model was inspected directly.

Observed dimensions:

```text
nq = 36
nv = 35
nu = 29
timestep = 0.002 s
```

Therefore:

\[
36 = 7 + 29
\]

where the first seven `qpos` values represent the floating base.

The state layout is:

```text
qpos[0:3]
→ base XYZ position

qpos[3:7]
→ base quaternion
   [w, x, y, z]

qpos[7:36]
→ 29 joint positions
```

Velocity layout:

```text
qvel[0:3]
→ base linear velocity

qvel[3:6]
→ base angular velocity

qvel[6:35]
→ 29 joint velocities
```

All 29 actuators were verified to correspond to the 29 actuated joints.

The simulation timestep is:

\[
0.002s
\]

which corresponds to:

\[
500Hz
\]

physics simulation.

---

## 2. Real Observation Interface

A real 64-dimensional G1 observation was implemented:

\[
obs =
[q,\dot q,RPY,\omega]
\]

Dimensions:

```text
29 joint positions
+
29 joint velocities
+
3 RPY
+
3 angular velocities

= 64
```

The quaternion stored by MuJoCo is converted to roll, pitch, and yaw before being passed to the Actor.

The final neural-network observation layout is:

```text
obs[0:29]
→ joint positions

obs[29:58]
→ joint velocities

obs[58:61]
→ roll, pitch, yaw

obs[61:64]
→ base angular velocity
```

This ordering is designed by the RL environment and is not the original MuJoCo storage order.

---

## 3. PPO Action to PD Control

The control architecture implemented on Day 8 is:

```text
PPO
↓
29D action
↓
joint target offset
↓
target joint position
↓
PD controller
↓
torque
↓
MuJoCo actuator
```

The target position is calculated using:

\[
q_{target}
=
q_{default}
+
action\_scale \cdot action
\]

Current:

\[
action\_scale = 0.10
\]

The low-level PD controller is:

\[
\tau
=
K_p(q_{target}-q)
-
K_d\dot q
\]

Torque is clipped according to MuJoCo actuator control limits before being written to:

```python
data.ctrl
```

---

## 4. Standing Reference and PD Baseline

The original MuJoCo initial joint state was approximately zero for most joints.

Using this directly as the standing target caused the humanoid to fall rapidly.

The standing reference was therefore changed to a bent-leg configuration:

```text
Left leg:
hip pitch   -0.1
hip roll     0.0
hip yaw      0.0
knee         0.3
ankle pitch -0.2
ankle roll   0.0

Right leg:
same configuration
```

Leg PD gains were also increased from the initial uniform temporary gains.

The zero-action test represents:

```text
PPO action = 0
↓
q_target = standing reference
↓
PD-only controller
```

The robot still falls forward after approximately one second.

This is an important observation:

> Joint-space PD tracking alone does not provide whole-body balance control.

---

## 5. Control Frequency

The MuJoCo simulation runs at:

\[
500Hz
\]

The policy uses:

```text
decimation = 10
```

Therefore one PPO action remains active for ten MuJoCo simulation steps.

The policy frequency is:

\[
500/10=50Hz
\]

---

## 6. Reward and Episode Termination

The environment was upgraded from a simple MuJoCo wrapper into an RL environment.

The API became:

```python
obs = env.reset()

next_obs, reward, terminated, truncated = (
    env.step(action)
)
```

The current reward contains:

```text
alive reward
upright reward
height reward
joint velocity penalty
action penalty
```

The episode terminates when the robot falls, based on criteria such as:

```text
base height too low
roll too large
pitch too large
```

A separate timeout is represented using `truncated`.

---

## 7. Actor Integration

The Day 7 Actor was connected to the real G1 environment.

The Actor outputs a Gaussian distribution:

\[
u \sim
\mathcal N(\mu,\sigma)
\]

The raw Gaussian action is transformed using:

\[
a=\tanh(u)
\]

so the executed action satisfies:

\[
-1<a<1
\]

The raw Gaussian action is stored in the PPO rollout buffer for probability-ratio calculation.

The bounded action is sent to the environment.

Therefore:

```text
Buffer:
raw action u

Environment:
tanh(u)
```

---

## 8. Why Tanh Instead of Direct Clipping

Originally, the action was directly clipped:

```text
2.0 → 1.0
5.0 → 1.0
100 → 1.0
```

This destroys information around the action boundary.

The updated approach uses:

```text
raw action
↓
tanh
↓
bounded action
```

Examples:

```text
1.0 → 0.762
2.0 → 0.964
3.0 → 0.995
```

The environment still contains a final `clip(-1, 1)` as a defensive safety check, but normal actions should already satisfy the range constraint after `tanh`.

---

## 9. Real PPO Smoke Test

The first real G1 PPO smoke test successfully completed:

```text
Real G1 rollout
↓
RolloutBuffer
↓
GAE
↓
PPO ratio
↓
Actor loss
↓
Critic loss
↓
backpropagation
```

A 256-step smoke test produced:

```text
Observations:
[256, 64]

Raw actions:
[256, 29]

Rewards:
[256]

Mean initial PPO ratio:
≈ 1.0
```

Both Actor and Critic parameters changed after the update.

This verified that real G1 experience could successfully train the PPO networks.

---

## 10. GPU Configuration

The initial environment had:

```text
NVIDIA driver:
572.16

Driver CUDA support:
12.8

PyTorch:
2.14.0 + CUDA 13.0
```

This caused:

```text
CUDA available: False
```

The project PyTorch environment was changed to a CUDA 12.8-compatible build.

After correction:

```text
CUDA available: True

GPU:
NVIDIA GeForce RTX 4060 Laptop GPU
```

MuJoCo physics remains CPU-based while Actor/Critic inference and PPO backpropagation use the GPU.

---

## 11. First Repeated PPO Training

The first training configuration was too aggressive:

```text
initial sigma ≈ 1.0
entropy coefficient = 0.01
Actor LR = 3e-4
PPO epochs = 5
```

Observed:

```text
entropy ≈ 41
clip fraction ≈ 0.50
KL ≈ 0.04–0.07
```

The policy did not improve.

The configuration was then changed to:

```text
log_std = -1
sigma ≈ 0.368

Actor LR = 1e-4
Critic LR = 3e-4

entropy coefficient = 0.001

PPO epochs = 3

rollout size = 2048

mini-batch size = 128
```

This produced healthier PPO optimization:

```text
entropy ≈ 12.15

KL ≈ 0.02

clip fraction ≈ 0.30
```

---

## 12. 40,960-Step Training Result

The second experiment ran for:

\[
20\times2048
=
40,960
\]

environment transitions.

Episode length remained approximately:

```text
52 steps
```

and episode return remained approximately:

```text
98–99
```

No clear upward learning trend appeared.

This means the PPO optimizer itself became more stable, but the current control/task formulation still did not produce effective standing behavior.

---

## 13. Shuffle and Mini-Batch Training

Each rollout contains:

```text
2048 transitions
```

The rollout is shuffled using:

```python
torch.randperm(...)
```

Mini-batch size:

```text
128
```

Therefore:

\[
2048/128=16
\]

mini-batches per epoch.

With:

```text
3 PPO epochs
```

each rollout produces:

\[
16\times3=48
\]

gradient-update mini-batches.

After the PPO update, the rollout is discarded and a new rollout is collected using the updated policy.

---

## 14. Deterministic Evaluation

Three policies were compared without random sampling:

```text
1. Zero-action PD baseline
2. Untrained Actor
3. Trained Actor
```

The Actor evaluation uses:

```text
raw_action = Gaussian mean

action = tanh(raw_action)
```

Results:

```text
Zero:
length      = 52
return      = 98.54
max pitch   = 0.814
min height  = 0.599

Untrained:
length      = 52
return      = 98.48
max pitch   = 0.814
min height  = 0.598

Trained:
length      = 52
return      = 98.49
max pitch   = 0.813
min height  = 0.599
```

The trained policy is therefore effectively identical to the baseline.

The lack of improvement is not caused only by exploration noise.

---

## 15. Viewer Evaluation

MuJoCo passive viewer support was added for qualitative evaluation.

Viewer experiments confirmed the quantitative result:

```text
Zero PD
≈
Untrained Actor
≈
Trained Actor
```

All three configurations fall forward in a visually similar way.

A WSL/WSLg viewer cleanup issue was initially observed when interrupting the process using keyboard signals.

The evaluation script was modified so that the Viewer is closed directly through the GUI window instead.

---

## 16. Main Day 8 Finding

The main Day 8 conclusion is:

> The complete PPO–G1 integration pipeline works, but the current balance task formulation does not yet provide an effective learning signal.

This is an important distinction.

The current limitation is no longer:

```text
PPO implementation
MuJoCo integration
GPU support
RolloutBuffer
GAE
Mini-batch training
Checkpointing
Evaluation
```

These components are working.

The remaining problem is now primarily:

```text
reward design
action authority
action-space dimensionality
balance-control formulation
```

---

## 17. Day 9 Questions

The next experiments should answer:

### Action Authority

Current maximum correction:

\[
\pm0.10 rad
\]

Possible comparison:

```text
0.10
0.20
0.25 rad
```

Question:

> Does the policy have enough authority to recover from forward falling?

---

### Reward Shaping

Current reward may provide insufficient information before the robot is already falling.

Potential improvements:

```text
stronger continuous upright reward
angular velocity penalty
explicit terminal penalty
better height shaping
```

---

### Action-Space Reduction

Current Actor controls all 29 joints.

For standing balance, many wrist and arm joints may be unnecessary.

Potential initial policy:

```text
legs
+
waist
```

with upper-body joints held near their reference pose.

This may substantially reduce the exploration/search space.

---

## Day 8 Status

Day 8 is complete.

Completed:

```text
G1 model inspection
64D observation
29D action interface
PD controller
tanh action bounding
reward / termination
real rollout
GAE
PPO update
GPU training
shuffle
mini-batches
checkpoints
headless evaluation
MuJoCo viewer evaluation
```

The next phase moves from system integration to reinforcement-learning task design and humanoid balance learning.