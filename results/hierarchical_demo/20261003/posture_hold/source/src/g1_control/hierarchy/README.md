# Task-to-skill demonstration interface

This is a simulation-based locomotion task-layer prototype. It is separate from
`g1_control.safety`, which contains historical 64D standing experiments.

`TaskRequest(task_id, sequence, issued_s)` renews a preconfigured task lease.
`RobotSample` carries finite scalar world position and **unwrapped** Euler
heading. `TaskSequencer` emits immutable `Decision` objects containing a task
stage, progress, message age and a 3D velocity command. `VelocityCommand`
validates the existing actor command contract. The runner supplies it to
observation slots 9:12 of the fixed 123D/37D actor, then preserves the original
position-target rule and actuator interface.

Forward stages complete using displacement projected onto their entry
heading, not a timer or path length. Turn stages complete using relative
heading. Turning is an arc with positive forward velocity, not an in-place
turn. Timeouts are independent limits. Robot tracking quality is not inferred
from task completion.

The freshness watchdog rejects wrong task IDs, nonfinite/future/expired
issue times, replayed or reordered sequence numbers, and reversed issue
stamps. Expiry blocks new stages. Two fresh accepted messages are required
before resuming stage admission. Stage timeouts keep running during dropout.
A missing initial request and persistent stale data eventually abort the task.
Use a new sequencer after an explicit reset or terminal result.

The **last velocity command persists during stale data**. This is an interface
fault demonstration, not a safe stop or fallback. Terminal task results end
the simulation experiment; they do not specify a robot stopping action.
`switch_authorized` concerns controller switching, and remains false; changing
parameters of the same locomotion actor at a task-stage handoff is distinct.
`fallback_authorized` remains false. No action arrays or simulator handles
enter this module. No existing anomaly thresholds gain action authority.

All task/watchdog times are simulation times in the current runner. The
separate timing CSV measures wall-clock task, observation/actor, physics with
state hashing, and total headless policy frames. These measurements do not
include ROS, networking, cameras or hardware, and do not guarantee real-time
execution. Fault injection drops task heartbeats; robot-state observation
continues. It is not a tested camera or VLA perception failure.

See [the tested demonstration and video](../../../results/hierarchical_demo/20261003/README.md).


Independent posture probes in `experimental_hold.py` use the current123D/37D
affine position-target interface and smooth handoff. They are not exported
as deployable skills or registered with the watchdog. All six evaluated
captured/default-pose holds failed height checks within0.70–0.78s; compatibility
does not imply body balance. See [results](../../../results/hierarchical_demo/20261003/posture_hold/README.md).
