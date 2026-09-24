# Day 2 — G1 State / Action Interface

## Goal

Understand how the Unitree G1 state and command interfaces connect the controller to the MuJoCo simulator.

The main goal of Day 2 was to understand the control loop:

```text
Target
  ↓
Error
  ↓
Controller
  ↓
Action / Torque
  ↓
MuJoCo Physics
  ↓
G1 State
  ↓
Feedback
  └────────→ Controller

1. Core State Variables
Joint Position — q

q represents the current joint position.

Example:

Left knee q = 0.154 rad

For the G1 29-DOF model, each motor has one corresponding joint position.

Joint Velocity — dq

dq represents the current joint angular velocity.

dq > 0  → joint moving in the positive direction
dq < 0  → joint moving in the negative direction
dq ≈ 0  → joint approximately stationary

Small values such as:

1e-5
1e-6

may appear due to numerical precision, contact dynamics, or simulation noise.

2. MuJoCo Runtime State

MuJoCo separates the robot model and runtime state.

mjModel

Contains mostly static information:

body definitions
joints
actuators
sensors
mass and inertia
geometry
simulation parameters
mjData

Contains runtime simulation state:

qpos
qvel
ctrl
sensordata
contact and force information

Conceptually:

mjModel
= What the robot is

mjData
= What the robot is doing now
3. Unitree SDK State Interface

The Unitree SDK exposes motor state using:

motor_state[i].q()
motor_state[i].dq()
motor_state[i].tau_est()

These provide:

q       → joint position
dq      → joint velocity
tau_est → estimated joint torque

The controller therefore does not need to directly access MuJoCo's internal arrays.

4. Unitree SDK Command Interface

Motor commands are sent using:

motor_cmd[i].q()
motor_cmd[i].dq()
motor_cmd[i].kp()
motor_cmd[i].kd()
motor_cmd[i].tau()

These correspond to:

q      → desired joint position
dq     → desired joint velocity
kp     → proportional gain
kd     → derivative gain
tau    → feed-forward torque
5. Unitree SDK2 Bridge

The unitree_sdk2_bridge connects the Unitree SDK interface to MuJoCo.

The bridge reads MuJoCo sensor data and converts it into Unitree LowState.

Conceptually:

MuJoCo sensordata
        ↓
Unitree SDK2 Bridge
        ↓
LowState
        ↓
Controller

The command path works in the opposite direction:

Controller
        ↓
LowCmd
        ↓
Unitree SDK2 Bridge
        ↓
MuJoCo ctrl

The bridge therefore creates an interface-level abstraction between simulation and a real Unitree control interface.

6. PD Command Mapping in the Bridge

The simulator bridge applies the motor command approximately as:

torque =
tau_ff
+ Kp × (q_target - q)
+ Kd × (dq_target - dq)

This means the controller can provide:

q_target
dq_target
Kp
Kd
tau_ff

while the bridge continuously computes the actual actuator torque using the latest simulated state.

7. G1 Joint Index Mapping

The simulation uses the G1 29-DOF model.

Important joint indices include:

0  Left Hip Pitch
1  Left Hip Roll
2  Left Hip Yaw
3  Left Knee
4  Left Ankle Pitch
5  Left Ankle Roll

6  Right Hip Pitch
7  Right Hip Roll
8  Right Hip Yaw
9  Right Knee
10 Right Ankle Pitch
11 Right Ankle Roll

12 Waist Yaw
13 Waist Roll
14 Waist Pitch
...
28 Right Wrist Yaw

The main Day 3 experiment therefore uses:

LEFT_KNEE = 3;
8. Sensor Ordering

For the current G1 MuJoCo model, sensor data is arranged in blocks.

Conceptually:

sensordata
│
├── all joint positions
├── all joint velocities
└── all estimated torques

The Unitree bridge converts these arrays into per-motor state objects:

motor_state[i].q
motor_state[i].dq
motor_state[i].tau_est
9. DDS Communication

The simulator and controller communicate using DDS topics.

State Topic
rt/lowstate

Used for:

Simulator → Controller
Command Topic
rt/lowcmd

Used for:

Controller → Simulator

The current simulator configuration uses:

DDS domain: 1
Network interface: lo

The lo interface is the local loopback interface.

A warning such as:

selected interface "lo" is not multicast-capable

does not prevent local DDS communication in the current setup.

10. Control Frequency vs Logging Frequency

Several frequencies exist independently.

Physics Frequency

MuJoCo advances physics using its simulation timestep.

For the current G1 setup, the effective timestep is approximately:

0.002 s
≈ 500 Hz
Control Frequency

The Unitree example uses:

2 ms
≈ 500 Hz
State / DDS Frequency

State messages may be published at their own frequency.

Logging Frequency

Console output can be reduced independently:

if (++counter % 500 == 0)

This only reduces printed output.

It does not reduce the control or sensor update frequency.

11. Minimal LowState Reader

A minimal subscriber was implemented to read the left knee state.

The program successfully received:

motor_state[3].q()
motor_state[3].dq()

Example output:

Left knee q ≈ 0.154 rad
dq ≈ 0

This confirmed that the complete state path works:

MuJoCo
  ↓
Unitree SDK2 Bridge
  ↓
DDS LowState
  ↓
Custom C++ Subscriber
Key Takeaways

By the end of Day 2:

Understood q and dq
Understood the difference between mjModel and mjData
Identified the G1 29-DOF joint ordering
Identified the left knee as motor index 3
Understood LowState and LowCmd
Understood how the Unitree SDK2 bridge maps MuJoCo state and commands
Successfully received real-time G1 joint state through DDS
Understood the difference between physics, control, sensor, and logging frequencies

Day 2 established the state/action interface needed for the Day 3 closed-loop PD controller.
