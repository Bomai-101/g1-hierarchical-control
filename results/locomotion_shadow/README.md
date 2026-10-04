# Locomotion shadow monitor

This is a passive diagnostic layer for the reference-aligned 123D/37D
locomotion pipeline. It is independent of the historical 64D standing
watchdog. It cannot change actions, switch controllers or recover the robot.

## Actual command/rough-terrain rollouts completed on 2 October 2026

[Physical command matrix](physical_commands/README.md): 80 cases / 160 off/on
rollouts, all exact non-interference checks PASS. Plane cases complete 30 s,
but 54 of 60 rough cases terminate; six remaining rough cases barely move.
Forty fall cases end before 1 s yaw windows can become eligible. Early scalar
screens and accumulated heading deviation need separate calibration; no
switching/recovery authority is added. New recorder captures model/scene/
initial-state provenance before the first action and validates actual policy
commands against the schedule.

## Signal-level calibration completed on 2 October 2026

[Calibration](yaw_calibration/README.md) compares 192 labeled synthetic signals
using independent development/heldout seeds and 27 window/threshold/dwell
settings. No setting satisfies every predeclared development limit, so no
parameter is promoted. Longer windows reduce slow-oscillation candidates but
increase known-onset delay. These are signal experiments, not physical G1
normal/fault labels. A [supervisor/fallback interface draft](../../notes/locomotion_supervisor_interface.md)
keeps diagnostic evidence separate from switching authority and defines the
next real command-schedule/rough-terrain calibration protocol.

## Offline yaw-window diagnostics completed on 2 October 2026

[Window analysis](yaw_window/README.md) confirms why the local zero/+0.2 yaw
runs have no legacy yaw candidate: their uninterrupted post-startup threshold
exceedances last at most 0.16 s, below the required 0.2 s dwell. A 1 s signed
error window with 2 s startup exclusion finds one persistent candidate in all
six saved runs, and quaternion-based Euler heading supports the same bias
directions. No simulator steps were rerun and the recorder was unchanged.

Synthetic controls also expose false alarms on zero-mean 0.5 Hz oscillation.
Window parameters remain exploratory; neither detection reliability nor
fallback authority is established. See the [six-panel diagnostic figure](yaw_window/figures/yaw_windows.svg).

## Six turning pairs verified on 2 October 2026

The completed [turning archive](turning/README.md) contains all six original
comparison CSVs, their SHA-256 manifest and a compact comparison table.
Both the supplied actor and the locally trained actor were also replayed with
monitoring off/on for yaw commands -0.2, 0 and +0.2 rad/s, 20 seconds each.
All six pairs passed exact action, sampled-state, per-physics-step digest,
trajectory CSV and metric equality checks. Each pair recorded 1,000 actions,
1,001 states and 20,000 physics steps; every monitored state has a signal row.

Fresh replay metrics differ from the historical results. They are archived
separately, with per-metric deltas, and are not substituted for the original
six comparisons. The cause is unresolved; see the archive's runtime notes.
The checks establish passive non-interference in these conditions, not
heading-tracking quality or reliable detection of directional bias.

## Earlier straight-command verification

On 2026-10-01, the local exported policy was evaluated twice on plane terrain,
friction 0.8, seed 42, velocity command [1, 0, 0], with 1 ms physics and
50 Hz policy updates. Both runs completed 20 seconds without falling.

With monitoring off/on, the 1,000 policy actions and 1,001 sampled states
were exactly equal. The digest of qpos, qvel and ctrl after each of the
20,000 physical steps was identical. Trajectory CSV bytes and evaluation
metrics were identical, excluding the monitor-enabled flag.
See [verification.json](verification.json).

These checks cover this single non-rendered rollout, not every environment
or rendering mode. Both runs use the same recorder instrumentation. Their
evaluation metrics also match the earlier unmonitored evaluator result.

## Signals and limitations

[Signals](shadow_signals.csv) record tilt, angular-speed magnitude, signed
body-frame forward speed and angular z, and command errors. Body angular z
matches the reference evaluator's yaw-rate signal; it is not the Euler
heading derivative when roll/pitch are nonzero.

The default exploratory screens are tilt >15 degrees, angular-speed magnitude
>1 rad/s, absolute forward error >0.3 m/s and absolute yaw error >0.3 rad/s,
each sustained for 0.2 seconds. These are configurable through
`ShadowThresholds`, not validated safety boundaries. Clearing a screen does
not establish safety. No warm-up exclusion is applied.

The sampled mean body angular z was -0.40417 rad/s. The monitor recorded
one forward-error candidate at 0.2 seconds during startup, but no yaw-error
candidate. A continuous-exceedance screen can miss persistent directional
bias when gait oscillations repeatedly bring instantaneous errors below
threshold. Therefore this result verifies non-interference, not an effective
heading-drift detector. A windowed diagnostic should be tested before any
switching authority is introduced.

## Reproduction

Use the MuJoCo environment and the same exported local policy. Replace the
reference and output paths with your installation paths. Output directories
must not already exist.

```bash
python scripts/record_locomotion_sim2sim.py \
  --reference-root /path/to/g1_walk_isaaclab_mujoco \
  --policy /path/to/policy_actor.npz \
  --output-dir /path/to/new_off --duration 20
python scripts/record_locomotion_sim2sim.py \
  --reference-root /path/to/g1_walk_isaaclab_mujoco \
  --policy /path/to/policy_actor.npz \
  --output-dir /path/to/new_on --duration 20 --shadow
python scripts/verify_locomotion_shadow.py \
  --off /path/to/new_off --on /path/to/new_on \
  --output /path/to/new_verification.json
python tests/test_locomotion_shadow.py
```

Full traces and the preceding six turning comparisons and three timestep
comparisons are retained in the ignored local checkpoint directory under
`locomotion_shadow/20261001/`. No weights or reference control code were changed.
