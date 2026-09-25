# G1 Hierarchical Control Preparation

A hands-on preparation project for humanoid robot control using
Unitree G1, MuJoCo and Unitree SDK2.

## Current Progress

### Day 1 — Environment Setup ✅
- Unitree SDK2
- MuJoCo
- unitree_mujoco
- G1 29-DOF simulation

### Day 2 — State / Action Interface ✅
- q / dq
- LowState
- actuator and sensor mapping
- Unitree SDK2 bridge
- G1 joint indexing

### Day 3 — PD Joint Tracking ✅
- LowState subscriber
- LowCmd publisher
- 500 Hz control loop
- left-knee PD tracking
- fixed +0.1 rad target
- Kp comparison
- CSV logging and quantitative plots

### Day 4 — Whole-Body Observation / Action ✅
  - 64D whole-body observation
  - Joint state + IMU
  - 29D relative action vector
  - Multi-joint scripted policy
  - Policy → target → PD hierarchy

### Day 5 — Coordinated Hip–Knee–Ankle Control 
  - in progress
  