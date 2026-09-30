# Post-pulse drift diagnosis from exact center-state replay

Status: completed under Environment Reference V1. This is a measurement pass:
the frozen Day 9 pose, PD, action scale, observation, reward, termination,
checkpoint, and MuJoCo physics remain unchanged.

## Protocol

The source is the saved full MuJoCo integration state at deterministic PPO
trajectory step 145. Four Phase 4 controllers were replayed from those exact
bytes: no-pulse PD, knee-pair `+0.5` for 10 policy steps then PD, the same
pulse then original PPO, and the same pulse then the existing feedback
controller at scale 0.25. Final episode lengths, termination flags, pitch,
and angular-y matched the Phase 4 evaluation results within `1e-6`.

The new recorder logs every 50 Hz policy boundary and all 10 underlying
500 Hz applied PD torques. Its timelines include base roll/pitch and body-local
angular-y velocity, full qpos/qvel/integration state, six controlled joint
positions/velocities/targets/torques/actions, left/right foot contact force,
occupancy and contact-point speed, and COM/contact positions. The x interval
between active foot-floor contacts is used only as a *static COM projection
indicator*: it is not ZMP or a dynamic stability certificate.

## Event sequence from the center state

- **Pulse → PD (347 steps):** after the pulse, pitch settled near `-0.0065
  rad` and body-local angular-y stayed near zero through about step 180.
  Forward rate exceeded `+0.005 rad/s` by step 230; pitch exceeded `+0.02
  rad` at step 284; the full-body hold envelope was left at step 306.
  The COM projection crossed the front of the measured contact x interval at
  step 323, and termination occurred at step 347. At steps 150, 240, and 300,
  the front contact margin was about `0.127`, `0.125`, and `0.089 m`.
- **Pulse → PPO (237 steps):** the post-pulse state drifted backward. Angular-y
  fell below `-0.005 rad/s` by step 120, `|pitch|` exceeded `0.02 rad` at
  step 159, and the COM projection crossed the rear contact x interval at
  step 192, also the first exit from the hold envelope. Termination was step
  237. PPO residual action remained small through the initial drift (action
  norm about `0.0038` at step 100), but its accumulated effect was not neutral.
- **Pulse → existing feedback ×0.25 (181 steps):** backward rate passed
  `-0.005 rad/s` by step 65; `|pitch|` passed `0.02 rad` by step 103; the COM
  projection crossed the rear contact x interval at step 134. Hold-envelope
  exit was step 137 and termination was step 181. This fixed feedback
  worsened the backward drift in this starting state.
- **No pulse → PD (96 steps):** it drifted forward immediately, left the hold
  envelope at step 55, crossed the front contact x interval at step 72, and
  terminated at step 96.

These event times are descriptive thresholds on this one deterministic
trajectory. They are not universal recoverability limits. The early
pitch-rate departures happen *before* the corresponding large pitch changes
or COM-contact crossings, so the data identify a useful early observation
window for a later intervention experiment.

## What the mechanics rule in and out

For pulse → PD, both feet had contact through every recorded policy step;
left/right normal contact forces were near 172 N each during the quiet phase.
At step 240 the foot-contact speed p95 was about `0.00011 m/s` per side, and
even as pitch grew at step 300 it was below `0.001 m/s`. No raw PD torque
was clipped and no applied torque reached 95% of its limit in any of the four
replays. This run is consistent with a slow sagittal instability under fixed
pose tracking, followed by forward toppling. It does not support slip or
actuator saturation as the initiating failure.

The PD ankle target stayed at `-0.07 rad`; the measured left ankle moved
from about `-0.035 rad` at step 150 to `-0.038 rad` at step 240 and `-0.087
rad` at step 300. The controller applied changing torque as the joint moved,
but it had no direct base-pitch feedback after the initial pulse. These
observations do not establish which joint is causally responsible for the
drift. The opposite drift under PPO and the fixed recovery feedback shows
that simply handing control to either existing policy is insufficient.

## Next falsifiable test

Use the saved pulse → PD trajectory to branch *before* the observed forward
rate crosses `+0.005 rad/s` (for example around local steps 200–220). From the
same full state compare zero residual against a small, time-limited sagittal
correction and a rate-triggered correction. Require the same five initial
states from Phase 3 and at least the 10-second uninterrupted-hold gate.
Record whether the intervention reduces outward rate **without** creating a
backward overshoot. A delay alone should not be labelled recovery.

## Reproduce and inspect

```bash
cd ~/robotics/projects/g1-hierarchical-control
source .venv/bin/activate
python3 scripts/record_post_pulse_drift.py
```

Local-only data:
`code/day9/g1_balance/checkpoints/drift_diagnostics/phase4_drift_20260930T014151Z/`

The directory contains `summary.json`, four `*_timeline.csv` files, four
`*.npz` files with full state/torque sequences, and `drift_comparison.svg`.
The SVG has explicit colored line samples in its legend; the CSV contains
unclipped values for quantitative analysis.
