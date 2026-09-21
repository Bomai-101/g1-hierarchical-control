# Day 01 — Environment Setup

## System

- OS: Windows + WSL2
- WSL: WSL2
- Ubuntu version: Ubuntu 24.04.3 LTS

## Simulation Stack

- MuJoCo: 3.3.6
- Unitree SDK2 commit:   63096d0
- unitree_mujoco commit: 1eb6642
- Python environment: `.venv-g1`

## Robot

- Robot: Unitree G1
- Simulator: Unitree MuJoCo
- Simulation scene: `scene.xml`

## Environment Setup Completed

- Built Unitree SDK2 successfully.
- Installed Unitree SDK2 to `/opt/unitree_robotics`.
- Installed MuJoCo 3.3.6.
- Built the official `unitree_mujoco` simulator.
- Successfully launched the Unitree G1 model in MuJoCo.
- Created a separate Python virtual environment for the G1 project.
- Created a Git repository for the project.

## Initial Robot Interfaces Observed

### qpos
Generalised position of the robot, including joint positions and the floating-base pose.

### qvel
Generalised velocity of the robot, including joint velocities and floating-base velocity.

### Joints
The articulated degrees of freedom of the Unitree G1 humanoid.

### Actuators
The simulated motors that apply control inputs to the robot joints.

### IMU
Provides information related to the robot body's orientation and motion, such as angular velocity and acceleration.

## Problems Encountered

1. Unitree SDK2 initially failed to compile because Eigen3 was missing.
   - Solution: installed `libeigen3-dev`.

2. `unitree_mujoco` joystick test failed because `jstest.cc` did not include `<cstdint>`.
   - Solution: added:
     ```cpp
     #include <cstdint>
     ```

3. Initial Git repository was accidentally created inside the MuJoCo build directory.
   - Solution: removed it and created the project repository correctly at:
     `/home/omai/robotics/projects/g1-hierarchical-control`

## Day 01 Result

The official Unitree G1 model was successfully launched in a physics-based MuJoCo simulation environment.

This establishes the simulation foundation for implementing feedback control, hierarchical behaviours, safety monitoring, and later embodied-AI integration.
