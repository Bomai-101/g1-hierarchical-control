# Day 10 early-intervention evidence

This directory contains a compact, reproducible summary of a deterministic
five-state MuJoCo branch experiment. It is evidence of **trajectory-local
control authority**, not a trained recovery policy or robust standing.

## Experimental reference

- Model: Unitree G1 29-DOF scene under the frozen Environment Reference V1.
- Controller: Day 9 standing pose, PD gains, 6D residual mapping, and action
  scale remained unchanged.
- Branch: full MuJoCo integration state from the deterministic PPO trajectory
  at policy step 145. Each controller comparison starts from the same state
  within a given pitch/body-local angular-y offset.
- First intervention: normalized +0.5 residual on both knees for 10 policy
  steps, followed by zero-residual PD.
- Second candidate: the same first intervention, then a normalized +0.25
  knee-pair pulse for 5 steps scheduled at post-branch step 220. This is a
  timed script, not PPO inference.
- Five tested states: exact center; pitch offsets +/-0.0005 rad; body-local
  angular-y offsets +/-0.002 rad/s. The body-local angular-y component is a
  local pitch-rate proxy, not a global Euler pitch derivative.
- Policy period: 0.020 s. The sweep horizon was 500 post-branch policy steps.

The four-second criterion requires recovery-envelope entry within 2 s,
survival to 4 s, and a sustained final 2 s in the hold envelope. The stricter
ten-second flag requires uninterrupted hold after the first 2 s through the
full 10 s. Episode duration by itself does not satisfy either criterion.

## Result

At the exact center, first-pulse-then-PD lasted 347 post-branch steps and the
scheduled second pulse lasted 373. Both passed the original four-second test,
but neither passed uninterrupted ten-second hold. The common 145-step PPO
prefix makes these 492 and 518 steps from reset. The original PPO-only
trajectory lasted 241 steps from reset; it is a different control sequence.

At both nearest pitch offsets, each plotted candidate fell after 153
post-branch steps. At both nearest angular-y offsets, each fell after 192.
All four neighbors fell before the second pulse's scheduled step 220, so the
second pulse **did not execute** there. None passed the four-second or
uninterrupted ten-second criterion. No plotted case clipped raw PD torque.
The full 17-candidate by five-state sweep had no uninterrupted ten-second
success. These are deterministic tested states, not success probabilities.

![Five-state survival and hold comparison](figures/early_intervention_five_states.svg)

## Files and reproduction

- `source_candidates.csv`: all 85 compact candidate summaries from the local
  sweep, without high-volume per-step traces or checkpoints.
- `five_state_summary.csv`: the ten rows plotted in the figure, including
  whether the scheduled second pulse actually executed.
- `figures/early_intervention_five_states.svg`: a standard-library-generated
  figure with labeled axes, colored legend, and explicit limitations.

From the repository root, run:

```bash
python3 scripts/generate_day10_evidence.py
```

The source file and generated products are tracked so the figure can be
regenerated without private checkpoint files. The generation script checks
all 85 source cases, exact state labels, the center reference, and neighbor
timing before writing outputs.

Source identifiers from the local sweep:

```text
branch step: 145
checkpoint SHA-256: 1c05d33ab182335ca96ab5aa871e6808ed67f6c37fe99fa2727829273a9417f8
Phase 3 fine-grid results SHA-256: d986dc977b0f9b4df45f23587776ab7dd83ecf1ffec26ce2fc26919d42cbeaed
source_candidates.csv SHA-256: 19dff6ac2a14128a32bf208697757ca0cf91f5d61c39e03101df83c42c318ccc
full local sweep JSON SHA-256: b9a2209a9f9ad09f3e366b19feb4122940f54b202019631446bcdf8712fe800a
```

The full integration-state archives and complete step traces remain outside
Git in the ignored `code/day9/g1_balance/checkpoints/` tree. The public data
supports the plotted claim; exact physics replay additionally requires the
matching Unitree model, checkpoint, and local archives.

## Videos

- [First pulse versus second pulse](../../media/videos/day10_pulse_baseline_comparison.mp4): direct center-state comparison, 492 versus 518 total steps.
- [Historical PPO versus scripted intervention](../../media/videos/day10_ppo_vs_scripted_intervention.mp4): distinct control sequences, 241 versus 518 total steps.

Each 10.38 s side-by-side recording uses saved qpos/qvel for offline playback.
After the shorter run terminates, its last frame remains visible while the
longer one continues. Playback is not a new simulation or a recovery test.
