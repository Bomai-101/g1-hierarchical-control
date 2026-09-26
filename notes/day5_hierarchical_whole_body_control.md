# Day 5 — Hierarchical Whole-Body Control Foundations

## Goal

Extend the Day 4 policy-control architecture from simple multi-joint
actions toward the foundations of hierarchical whole-body humanoid
control.

The main topics covered were:

- coordinated joint motion
- motion primitives and motion scaling
- joint space and task space
- forward and inverse kinematics
- Jacobian-based control
- task-space velocity control
- singularities
- redundancy
- null-space control
- task-priority hierarchical control

---

## 1. Coordinated Joint Motion

A humanoid motion is generally not produced by one isolated joint.

For example, a leg movement may require coordinated changes in:

- hip pitch
- knee
- ankle pitch

A coordinated motion means that multiple joints change together to
achieve one common motion objective.

Controlling only the knee may change:

- foot position
- body height
- center of mass
- balance

without compensating through the hip or ankle.

---

## 2. Motion Primitive and Motion Scale

A simple coordinated motion can be represented by a base joint pattern.

Example:

Hip   = +0.04
Knee  = +0.10
Ankle = -0.02

A policy can then select a motion scale:
Normal motion        -> scale = 1.0
Fast leg motion      -> scale = 0.5
Large body motion    -> scale = 0.3

The resulting action is:
action = motion_scale × base_motion_pattern

or mathematically:
a_t = s_t × a_base

In the current scripted design:
- the base motion pattern is hand-designed
- the policy selects the scale
- the final joint actions are generated from both

A neural policy could instead:
- learn the scale
- learn a residual correction
- or directly output the complete Action[29] vector

## 3. Joint Space vs Task Space

Joint Space
Joint space represents the robot using joint variables.
Examples:
Hip angle
Knee angle
Ankle angle

A joint-space command may be:
Knee target = +0.10 rad

Task Space
Task space represents the physical task that should be achieved.
Examples:
Move the foot forward by 8 cm
Move the hand upward
Keep the torso upright
Move the center of mass forward

The task-space target does not directly specify the individual joint
angles needed to achieve the task.

## 4. Forward Kinematics and Inverse Kinematics

Forward Kinematics (FK) maps joint configuration to task-space pose:
Joint Space
    ↓
   FK
    ↓
Task Space

Conceptually:
x = f(q)

Given the joint angles, FK determines where the foot, hand, or another
robot body point is located.
Inverse Kinematics (IK) performs the reverse mapping:
Task-Space Target
       ↓
      IK
       ↓
Joint Targets

Conceptually:
q_target = IK(x_target)

The same task-space target may have multiple valid joint-space
solutions.
The controller may therefore also consider:
- joint limits
- balance
- previous posture
- collision avoidance
- comfortable configurations

## 5. Jacobian

The Jacobian describes how small joint-space motion maps to task-space
motion.
For small position changes:
Δx ≈ J(q) Δq

For velocity:
x_dot = J(q) q_dot

This means that joint velocities can be mapped into task-space
velocities.
For example:
joint velocity
    ↓
Jacobian
    ↓
foot velocity

##6. Jacobian Pseudoinverse

When the desired task-space velocity is known, joint velocity can be
estimated using:
q_dot = J⁺ x_dot

where:
J⁺

is the Jacobian pseudoinverse.
The pseudoinverse is not simply another notation for the ordinary
matrix inverse.
It is useful because robot Jacobians may be:
- non-square
- redundant
- rank deficient
- close to singular configurations

## 7. Singularity

A singularity occurs when the Jacobian loses rank.
At a singular configuration, the robot loses the ability to generate
independent motion in one or more task-space directions.
Near a singularity, a very small desired task-space motion may require
a very large joint-space command.
Conceptually:
Small task-space command
        ↓
Jacobian near singularity
        ↓
Very large joint correction

This occurs because some motion directions become poorly conditioned.
Practical controllers may therefore use techniques such as:
- pseudoinverse methods
- damped least squares
- singularity avoidance
instead of relying on a simple matrix inverse.

## 8. Redundancy

A robot is redundant with respect to a task when it has more available
degrees of freedom than are independently required by the task.
Example:
3 joint DOF
2D foot-position task

If the Jacobian has rank 2:
null-space dimension = 3 - 2 = 1

More generally:
dim(Null Space) = n - rank(J)

Redundancy provides additional freedom that can be used for secondary
objectives.

## 9. Null Space

A null-space joint motion satisfies:
J q_dot_null = 0

This means the joints can move while producing no change in the
primary task-space quantity.
The null space can therefore be used for secondary objectives without
disturbing the primary task.
Examples include:
- avoiding joint limits
- improving posture
- maintaining torso orientation
- reducing unnecessary motion
- improving balance
A common null-space projector is:
N = I - J⁺J

## 10. Hierarchical Task Priority

Hierarchical whole-body control assigns different priorities to
multiple robot tasks.
Example:
Priority 1:
Maintain foot contact

Priority 2:
Keep torso upright

Priority 3:
Avoid joint limits

The lower-priority task should not destroy the result of a
higher-priority task.
Conceptually:
Primary task
    ↓
Solve first

Remaining null-space freedom
    ↓
Secondary task

Remaining freedom
    ↓
Lower-priority task

A simplified two-task structure is:
q_dot =
J1⁺ x1_dot
+
N1 q2_dot

with:
N1 = I - J1⁺J1

The first term performs the highest-priority task.
The second term performs a secondary task only within motion directions
that do not interfere with the first task.

## 11. COM Target

COM means Center of Mass.
It represents the combined mass distribution of the complete robot,
not simply the torso position.
A COM target specifies where the controller would like the robot's
center of mass to move.
Humanoid control may simultaneously consider:
- foot targets
- torso orientation
- COM target
- joint limits
- contact state

## 12. Position and Velocity Control

Position control specifies:
Where should the robot go?

Example:
q_target = 0.5 rad

Velocity control specifies:
How fast and in which direction should it move now?

Example:
dq_target = 0.3 rad/s

Continuous motion does not require velocity control exclusively.
A continuously changing position trajectory can also produce
continuous motion.
A PD controller can use both:
tau =
Kp(q_target - q)
+
Kd(dq_target - dq)

## 13. Overall Hierarchical Control Pipeline

The control architecture can now be understood as:
High-Level Task / Policy
        ↓
Task-Space Target
        ↓
IK / Jacobian / Whole-Body Controller
        ↓
Joint Targets
        ↓
Low-Level PD Controller
        ↓
Torque
        ↓
Robot / MuJoCo
        ↓
New Observation
        └──────────────→

A learned system may also combine classical and neural methods:
Observation
    ↓
Neural Policy
    ↓
Task-Space Goal
    ↓
Hierarchical Whole-Body Controller
    ↓
Joint Targets
    ↓
PD

or:
Observation
    ↓
Neural Policy
    ↓
Joint Action[29]
    ↓
PD

## Key Takeaways

By the end of Day 5:
- Understood coordinated multi-joint motion
- Understood motion primitives and motion scaling
- Distinguished joint space from task space
- Understood FK and IK
- Understood Jacobian position and velocity mappings
- Understood why pseudoinverse is used
- Understood singularities and rank loss
- Distinguished redundancy from null space
- Understood null-space projection
- Understood strict task priority
- Connected these concepts to hierarchical whole-body humanoid control

Day 5 establishes the classical whole-body control foundation required
before introducing reinforcement learning and learned policies.
