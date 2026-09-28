# G1 Hierarchical Control Preparation

A hands-on preparation project for humanoid robot control using
Unitree G1, MuJoCo and Unitree SDK2.

## Current Progress

### Day 1 — Environment Setup ✅ (notes/day1_environment_setup.md) 
- Unitree SDK2
- MuJoCo
- unitree_mujoco
- G1 29-DOF simulation

### Day 2 — State / Action Interface ✅ (notes/day2_state_action_interface.md)
- q / dq
- LowState
- actuator and sensor mapping
- Unitree SDK2 bridge
- G1 joint indexing

### Day 3 — PD Joint Tracking ✅ (notes/day3_pd_joint_tracking.md)
- LowState subscriber
- LowCmd publisher
- 500 Hz control loop
- left-knee PD tracking
- fixed +0.1 rad target
- Kp comparison
- CSV logging and quantitative plots

### Day 4 — Whole-Body Observation / Action ✅ (notes/day4_whole_body_policy_interface.md)
  - 64D whole-body observation
  - Joint state + IMU
  - 29D relative action vector
  - Multi-joint scripted policy
  - Policy → target → PD hierarchy

### Day 5 — Coordinated Hip–Knee–Ankle Control ✅ (notes/day5_hierarchical_whole_body_control.md)
  - Coordinated hip–knee–ankle motion
  - Joint space vs task space
  - Forward / inverse kinematics
  - Jacobian and task-space velocity control
  - Singularity and pseudoinverse
  - Redundancy and null space
  - Task-priority hierarchical control

### Day 6 — Reinforcement Learning Foundations ✅ (notes/day6_reinforcement_learning_foundations.md)
  - Reward, return, value, Q-value, and advantage
  - Bellman recursion and bootstrapping
  - TD error and Actor-Critic
  - Generalized Advantage Estimation
  - Policy Gradient and PPO clipping
  - Rollout / batch / mini-batch / epoch
  - Mapping PPO to G1 + MuJoCo

### Day 7 — PPO Policy Implementation ✅ (notes/day7_PPO_core_implementation.md)
- Actor network
- Critic network
- Gaussian policy
- learnable log_std
- RolloutBuffer
- detach / rollout storage
- TD error
- GAE recursion
- value targets
- PPO ratio
- PPO clipping
- actor loss
- critic loss
- entropy bonus
- mini-batch / epochs
- fake PPO update sanity test
- end-to-end fake PPO training skeleton

### Day 7 — G1 / MuJoCo Integration - In progress