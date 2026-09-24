# Day 3 — G1 PD Joint Tracking

## Goal
Implement and validate closed-loop PD joint tracking on the Unitree G1 in MuJoCo.

## Control Pipeline

Target
→ Position / Velocity Error
→ PD Controller
→ LowCmd
→ Unitree SDK2 Bridge
→ MuJoCo Physics
→ LowState Feedback

## Controlled Joint

- Robot: Unitree G1 29-DOF
- Joint: Left Knee
- Motor index: 3

## PD Interface

tau = tau_ff
    + Kp(q_target - q)
    + Kd(dq_target - dq)

For this experiment:

- dq_target = 0
- tau_ff = 0
- target offset = +0.1 rad
- control loop ≈ 500 Hz

## Experiments

### Experiment A
Kp = 20
Kd = 2

Steady-state tracking error ≈ 0.044 rad

### Experiment B
Kp = 40
Kd = 2

Steady-state tracking error ≈ 0.017 rad

## Observation

Increasing Kp produced stronger position tracking and reduced the
steady-state error.

The residual error exists because the P term requires a non-zero
position error to generate the torque needed to balance persistent
external loads.

Higher Kp is not always better because excessive stiffness may increase
overshoot, oscillation and instability risk.

## Output

See:

results/day3/knee_tracking_normalized_comparison.png
