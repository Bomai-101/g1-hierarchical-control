# Post-pulse controller handoff experiment

Status: fixed-candidate diagnostic completed. The Day 9 pose, PD gains,
observation/action definitions, action scale, reward, termination, checkpoint,
and MuJoCo physics were unchanged. No PPO training was run.

## Question and protocol

Phase 3 found one short-horizon success: at the saved PPO trajectory's step
145, a normalized `+0.5` residual on both knees for 10 policy steps followed
by zero-residual PD passed the 2 s recovery + 2 s hold criterion. It then fell
at step 347 after branching. This experiment asks whether a different
**post-pulse controller** can hold that standing state longer and work at
nearby states.

The source states are exactly the five saved full MuJoCo integration states
from the Phase 3 fine grid: center; pitch offsets `±0.0005 rad`; and body-local
angular-y offsets `±0.002 rad/s`. Every candidate starts from the same state
within a given offset. The pulse is identical for every pulse candidate;
only the controller after policy step 10 changes.

Thirteen candidates were evaluated for up to 500 policy steps (10 s):

- no-pulse PD and no-pulse deterministic PPO controls;
- pulse then PD; pulse then original deterministic PPO;
- pulse then the existing `RecoveryStandController` at output scales 0.25,
  0.5, and 1.0;
- pulse then that controller at scales 0.5 or 1.0 for 25, 50, or 100
  steps, then PD.

Feedback scales here are **evaluation candidates** multiplying the existing
controller output, not changes to frozen Day 9 gains. All candidates pass
through the unchanged 6D residual-to-target-to-PD path. The results record
the original 4 s criterion, whether the final 2 s of a 10 s run meet the hold
criterion, and a stricter flag requiring uninterrupted hold from 2 s through
10 s. Surviving 10 s would still be finite-horizon evidence, not an indefinite
standing guarantee.

## Results

- Center: pulse → PD lasted **347 steps** and was the only candidate to pass
  the original 4 s criterion. Pulse → PPO lasted 237 steps. The best
  continuous feedback candidate, scale 0.25, lasted 181 steps. Higher
  feedback scale performed worse: scale 0.5 lasted 169 and scale 1.0 lasted
  151 steps. No-pulse PD and PPO both fell after 96 steps.
- At pitch offset `-0.0005 rad`, pulse → PD led at 153 steps. At `+0.0005
  rad`, scaled feedback 0.5 for 25 steps then PD led at 197 steps; pulse → PD
  lasted 153. At angular-y offset `-0.002 rad/s`, pulse → PD led at 192.
  At `+0.002 rad/s`, continuous scaled feedback 0.25 led at 195; pulse → PD
  lasted 192.
- Across all 65 paired continuations, only the original center pulse → PD
  passed 4 s. **None** satisfied the 10 s final-hold or uninterrupted-hold
  criterion. No raw PD torque command was clipped or reached 95% of its limit.
  Feedback scale 1.0 did reach the normalized residual-action bound, so
  stronger action alone is not supported by these results.

The existing fixed feedback controller and simple timed handoffs do not solve
the recovery-to-hold problem. A little feedback sometimes lengthens survival
in a neighboring state, but its behavior is state-sensitive and below even
the four-second criterion. The 347-step center run is a trajectory-local
delay, not a robust standing policy.

The next research step should diagnose *why the post-pulse state drifts*: log
pitch/rate, six controlled joint positions/velocities, contact/COM state, and
action over the hold and pre-fall intervals. Then test a small number of
explicit hypotheses for continued correction, with the same five states and
the 10 s hold gate. The evidence does not yet justify increasing PPO updates,
expanding to large perturbations, or declaring a recovery skill learned.

Follow-up: [the exact-state drift diagnosis](post_pulse_drift_diagnosis.md)
logs controlled joints, torques, COM/contact positions and foot forces. It
locates early pitch-rate departures before hold-envelope exit or the measured
COM/contact crossing.

## Reproduce

```bash
cd ~/robotics/projects/g1-hierarchical-control
source .venv/bin/activate
python3 scripts/evaluate_post_pulse_handoffs.py --center-only
python3 scripts/evaluate_post_pulse_handoffs.py
```

Local-only output under the ignored checkpoint tree:

- Center validation: `code/day9/g1_balance/checkpoints/handoff_evaluations/phase4_20260930T012148Z/`
- Center and four neighbors: `code/day9/g1_balance/checkpoints/handoff_evaluations/phase4_20260930T012210Z/`

Each directory contains `results.csv` and `results.json` with full traces,
candidate definitions, source state-grid hash, and checkpoint hash.
