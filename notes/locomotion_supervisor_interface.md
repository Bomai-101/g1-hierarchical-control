# Locomotion supervisor observation contract (draft, no switching authority)

Date: 2 October 2026. Active locomotion interface: 123D observation / 37D
policy action. Historical 64D standing diagnostics are not a fallback for this
interface. This document specifies a future integration boundary; it does
not implement a supervisor, change an actor, or authorize recovery.

## Diagnostic message

A recorder/observer should publish an immutable message after a policy sample:

| Field | Meaning |
| --- | --- |
| `rollout_id`, `reset_id`, `sample_index` | Identify ordering and prevent windows crossing reset |
| `simulation_time_s`, `sample_interval_s` | Simulation clock, not wall-clock latency |
| `signal_definition` | Explicit `body_angular_z` or `world_zyx_euler_heading_rate` |
| `command_yaw_rps`, `command_epoch` | Current command and schedule version |
| `window_s`, `window_start_s`, `window_end_s` | Exact trailing interval |
| `eligible`, `quality_reason` | Complete post-startup window with valid, ordered inputs |
| `signed_error_mean_rps`, `error_rmse_rps` | Command subtracted at each sample before integration |
| `candidate_direction`, `candidate_onset_s` | Exploratory same-sign evidence, never a recovery decision |
| `policy_hash`, `metadata_hash`, `model_hash`, `runtime_config_hash` | Input/version provenance captured before simulation |
| `calibration_version`, `calibration_scope` | Distinguish signal experiments from physical validation |

Use separate fields for 1/2/4 s windows; do not silently replace one signal
with another. Startup, insufficient history, reset and invalid inputs should
emit explicit unavailable evidence rather than a zero error. In a future
online implementation, invalidate a window across a data gap or reset and
start a fresh dwell. Threshold tuning does not validate an invalid signal.

Time-varying commands require integrating `(measured_rate-command_rate)` at
each timestamp. Subtracting the most recent command from a historical rate
mean creates artificial bias at a command change. Command steps also introduce
legitimate tracking transients; record command epochs and evaluate those
transients separately before choosing a holdoff rule. Current offline heading
analysis handles constant commands only and must not be reused unchanged for
variable-command evaluation.

## Supervisor assessment

The next prototype may consume diagnostic messages and emit only:
`observation_valid`, `evidence_state`, `candidate_reasons`, and diagnostic
history. An unavailable or stale message must not be interpreted as evidence
of nominal tracking. Candidate evidence may request more logging or flag a
review, but no controller output is connected in the present phase.

`switch_authorized` remains false and `fallback_request` remains absent.
Keep decision authorization separate from the diagnostic message, so a
screening threshold cannot automatically become an action gate. No action
array, simulator object, model handle or controller callback belongs in the
observer API.

## Fallback compatibility requirements

A future fallback proposal must explicitly specify compatible 123D/37D
observations and joint mapping, action scale, PD target/torque convention,
timing, state continuity at handoff, termination/reset behavior and an
independently evaluated operating envelope. Neither the old 64D standing
actor nor zeroing the 37D action is assumed to provide safe recovery.
Those candidates need separate exact-state experiments before an interface
can authorize their use.

## Physical calibration protocol to implement next

1. Preserve the reference actor and control path. Capture model XML, generated
   scene, mesh, actor and metadata hashes *before* each new rollout, plus
   Python/NumPy/MuJoCo versions, command schedule and complete initial state.
2. Build command-schedule recording separately from diagnostic authority:
   forward commands 0.5 and 1.0 m/s; steady yaw 0 and +/-0.1 rad/s; a slow yaw
   sinusoid and a +/-0.2 step schedule with logged transition times. Policy
   inputs and per-sample recorded commands must use the same schedule.
3. Use plane/friction 0.8 as a deterministic check, then rough/friction 0.8
   with seeds 42/43/44 for actual terrain variation. In the current reference
   `configure_runtime_terrain`, seed affects rough heightfields only. Changing
   plane seed alone does not create independent physical conditions.
4. Replay both exported actors without automatic reset, logging actions,
   qpos/qvel, heading, body angular rates, tilt, height and per-step controls.
   Completed episodes or lack of a fall flag do not label normal tracking.
5. Establish labels independently of candidate thresholds: reference heading
   from the command integral, externally defined heading-error duration and
   review of transient/steady segments. Neither a detector's own window mean
   nor a non-fall label is independent ground truth for directional bias.
6. Separate real dynamics perturbations from measurement-only bias injection.
   Injected signal offsets can test known-onset detection timing but cannot
   prove the detector recognizes a physical G1 fault or that a fallback works.
7. Freeze development thresholds and scoring limits before new holdout cases.
   Report family-specific false candidates, time active, pre-onset alarms,
   missed/late/wrong-direction events and small-sample uncertainty. Do not
   retune using the holdout set and then quote it as validation.
8. If connecting a new online observer or command-schedule recorder, repeat
   exact monitor-off/on action/state/physics-digest/trajectory checks on that
   new implementation. Existing proofs cover the earlier recorder only.

The current signal calibration finds no development setting that satisfies
all exploratory limits. That outcome blocks promotion of window candidates
into action authority; it does not stop passive data collection or interface
review. Real dynamic-command/rough-terrain validation remains future work.
