# Day 10: environment, instability, and early-recovery diagnosis

## Research question

The G1 is initialized near an upright standing pose. Day 10 asks why it
becomes unstable, whether the frozen 6D residual joint-target interface has
authority to redirect an early failure, and whether any such correction
continues to hold under small changes in initial state. This is not a floor
get-up task or a claim that robust standing has been solved.

## Completed checks

1. **Environment Reference V1.** The loaded 29-DOF Unitree scene runs at a
   0.002 s physics step with a 0.020 s policy period. Contact, friction,
   actuator limits, and hidden-support checks found no gross explanation for
   the nominal fall. The zero-residual PD reference reproduced 215 policy
   steps. This is a coherent simulation reference, not hardware validation.
   See [the environment audit](environment_reference_v1.md).
2. **Matched failure trajectories.** Starting from the same post-settling
   simulator state, zero PD fell at step 215 and the Day 9 deterministic PPO
   checkpoint at step 241. Sustained pitch deviation began at steps 140 and
   165, respectively; clear outward acceleration began at steps 166 and 192.
   Both progressed toward forward toppling without observed torque clipping.
   See [trajectory analysis](instability_trajectory_phase1.md).
3. **Exact-state branches.** Replaying saved full integration states produced
   matching first transitions. The earlier-window search found one
   four-second success among 228 finite candidate branches: from the PPO
   trajectory at step 145, a ten-step normalized +0.5 knee-pair pulse followed
   by zero-residual PD. It was a scripted intervention, not a learned PPO
   recovery. See [branch protocol](instability_branch_phase2.md).
4. **Local state slice.** Under the fixed first-pulse action, only the exact
   center passed the four-second criterion on each tested 7x7 pitch/angular-y
   grid. The same center fell after 347 post-branch steps in the longer run;
   neither the coarse nor fine grid demonstrated a robust region. See
   [the state-map study](instability_basin_phase3.md).
5. **Continuation and second correction.** Existing PPO, a fixed feedback
   candidate, and simple post-pulse handoffs did not produce a ten-second
   uninterrupted hold. A second timed knee pulse extended the exact-center
   branch from 347 to 373 steps, but the four nearest tested states fell at
   153 or 192 steps, before that pulse's scheduled step 220. See
   [the public evidence package](../results/day10/README.md),
   [handoff test](post_pulse_handoff_phase4.md), and
   [drift diagnosis](post_pulse_drift_diagnosis.md).

## Interpretation and boundary

Day 9's 215-to-241 nominal improvement did not establish symmetric
disturbance robustness. Day 10 demonstrates a reproducible, finite-horizon
effect of bounded scripted control at one exact early-instability state. The
tested neighboring states do not yet show sustained recovery, and the best
center trace ultimately falls. A later standing-controller version should be
judged by braking, recovery-envelope re-entry, and sustained hold on a fixed
set of neighboring states, not by nominal episode length alone.

The watchdog and controller interface can switch modes, but switching to the
existing PD controller is not proof of a safe fallback. The failure-envelope
data should inform later controller-arbitration thresholds; no universal safe
threshold follows from this thin local slice.

## Immediate next experiment

Use the same five saved branch states to compare an **early state-triggered**
bounded correction against zero-residual PD, beginning before the neighboring
trajectories terminate. Preserve the simulator and controller reference.
Measure braking, entry, continuous hold, contact slip, torque clipping, and
adverse correction in the opposite pitch direction. Do not promote a candidate
until it improves held-out neighboring states under the same fixed gate.
