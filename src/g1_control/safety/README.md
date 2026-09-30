# Day 10 safety supervisor

The safety layer sits above controller selection and below any future skill
planner.  It receives the 64D observation plus base height and returns a
classification; it does not alter MuJoCo, reward, PPO, PD gains, or torque.

At a warning it records a degraded state but preserves the selected controller.
At fallback it latches `pd_stand` for the remainder of the episode.  The latch
prevents rapid controller switching while the robot is near failure.

Current thresholds are deliberately inside the existing Day 9 termination
boundary: warning at height below 0.65 m or tilt above 0.35 rad; fallback at
height below 0.55 m or tilt above 0.55 rad.  These are initial engineering
thresholds, not learned values; Day 10 experiments will calibrate them.
