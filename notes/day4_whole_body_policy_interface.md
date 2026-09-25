# Day 4 Main Workflow + Content Oral Defense — Whole-Body Observation and Policy Interface

## 1. Main Workflow

Observation[64]
= 29 q
+ 29 dq
+ 3 RPY
+ 3 gyro

        ↓

ScriptedPolicy:
body motion large → knee offset 0.03
joint moving fast → knee offset 0.05
otherwise         → knee offset 0.10
        ↓

Action[29]
= relative joint-position offsets

        ↓

q_target[i]
= reference_q[i] + action[i]

        ↓

LowCmd / PD


## 2. Content Oral Defense

1. What is the difference between State and Observation?

State is the complete real physical state of the system.

Observation is the information that the policy can actually receive, usually through sensors or processed sensor data.

Observation may be incomplete or noisy, so it is not necessarily identical to the true state.


2. Why was the original observation 58 dimensions, and why is it now 64 dimensions?

The original observation contained:

29 joint positions q
+
29 joint velocities dq

Therefore:

29 + 29 = 58 dimensions

We then added:

3 RPY values:
roll, pitch, yaw

+
3 gyroscope values:
angular velocity around x, y, z

Therefore:

29 + 29 + 3 + 3 = 64 dimensions.


3. What does Action[29] represent in the current design?

Action[29] is a 29-dimensional vector.

Each element represents the relative joint-position offset for one G1 joint.

For joint i:

q_target[i] = reference_q[i] + action[i]

Therefore, the current action is not torque.

It is a relative position command used to generate the joint target.


4. What is the relationship between reference_q, action, and q_target?

The relationship is:

q_target_t = reference_q + action_t

In the current implementation:

reference_q is fixed and is copied from the first valid robot state.

action_t can change every policy cycle.

q_target_t can therefore also change every policy cycle.

In future architectures, reference_q may also become time-dependent.


5. Why is q_target recalculated every loop but does not accumulate infinitely?

Because every target is recalculated from the fixed reference:

q_target = reference_q + action

It is NOT calculated as:

q_target += action

For example:

reference_q = 0.15

If action = 0.10:

q_target = 0.25

On the next loop, if action is still 0.10:

q_target is still 0.25

It does not become 0.35.


6. What is the difference between a Scripted Policy and a Neural Policy?

Both can use the same interface:

Observation -> Policy -> Action

A Scripted Policy uses human-written rules, such as if/else conditions and manually selected parameters.

A Neural Policy uses learned parameters theta to learn the mapping from observation to action.

For example:

Observation[64]
-> Neural Network Policy
-> Action[29]

The major difference is how the policy mapping is produced, not the input/output interface.


7. Why can a humanoid policy not rely only on joint q and dq?

Joint q and dq mainly describe local joint states.

They do not fully describe the global motion and orientation of the robot body.

For example, all knee joint states may look normal while the torso is rapidly falling forward.

Therefore, whole-body information such as:

body orientation
body angular velocity
IMU measurements

is required.

Combining joint state and body state allows the policy to make coordinated whole-body decisions.


8. What are the different responsibilities of the Policy and the PD Controller?

Policy:

Observation -> Action / Target

The policy decides where the robot should move.

It uses information such as joint state and IMU state to generate the action vector.

PD Controller:

Target + Current q/dq -> Corrective Control

The PD controller continuously tracks the latest target using feedback.

In the current Unitree MuJoCo architecture, the program sends:

q_target
dq_target
Kp
Kd
tau_ff

through LowCmd.

The Unitree SDK2 bridge then applies the PD relationship to generate the actuator control.

Therefore:

Policy = decides where to go

PD = determines how to track that target using feedback