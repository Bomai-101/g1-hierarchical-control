# Phase 2 — exact-state recovery branching

Status: initial finite search completed under Environment Reference V1. The
Day 9 pose, PD, 6D action mapping and scale, reward, termination, and MuJoCo
physics were unchanged. This is an action-authority diagnosis, not PPO training.

## Protocol

`scripts/evaluate_trajectory_branches.py` restores `mjSTATE_INTEGRATION` from
the Phase 1 zero-PD and deterministic-PPO trajectories. Each candidate begins
from the same full snapshot at its branch point. For every stage, the original
controller's first transition was compared against the recorded qpos/qvel to
`1e-10` tolerance; its remaining episode length matched the original fall.

The fixed physical stages are A (sustained deviation), B (between A and C),
C (clear outward acceleration), and D (20 policy steps before termination).
An exploratory earlier search also uses P-40 and P-20, respectively 40 and 20
steps before A. These are **offsets from an observed marker**, not new physical
events. Both policy trajectories were tested separately.

Candidates: original controller; zero residual PD; and symmetric hip, knee,
or ankle pair pulses in both directions. Normalized amplitudes were 0.5 and
1.0; the first sweep used 10-step pulses at A–D, and the early-window sweep
used 5-, 10-, and 20-step pulses at P-40, P-20, and A. Each pulse is followed
by **zero-residual PD**, even when its branch state came from the PPO trajectory.
The `original` candidate continues the source trajectory's controller. The frozen action scale of 0.15 rad
means a normalized 0.5 pulse changes each selected target by 0.075 rad.

Success requires a 10-step entry into the existing full-body recovery envelope
within the first 100 steps (2 s), survival for 200 steps (4 s), and all final
100 steps (2 s) in the wider hold envelope. `braking_step` means outward pitch
rate became nonpositive for three samples; it alone is not recovery and may
also occur before overshoot. Initial envelope membership is recorded so an
already-near-upright branch is not mistaken for a large-angle recovery.

## Results

- A–D sweep: **0/112** candidate branches met the full criterion. Some pulses
  delayed a fall. From zero-PD A, the original lasted 75 steps after the branch;
  the best 10-step ankle +1.0 pulse lasted 124 steps, then fell backward.
- Earlier-window sweep: **1/228** candidate branches met the full criterion.
  From the PPO trajectory at step **145** (P-20, 2.90 s after the settled
  reset), knee-pair +0.5 for 10 policy steps, then zero-residual PD,
  survived the 200-step horizon. The unmodified PPO from the identical state
  fell after 96 steps.
- The successful branch started inside the entry envelope, left it on local
  steps 1–4 and 11–14, re-entered for the required 10 steps starting at local
  index 14, and was inside the hold envelope continuously from step 14 onward.
  Its pitch stayed in [-0.0349, -0.0002] rad and height stayed at or above
  0.7798 m. Pitch-rate transient ranged from -0.519 to +0.540 rad/s, so this
  is a controlled early transient, not an effortless static hold.
- In that branch, no raw PD torque command was clipped and no applied torque
  reached 95% of its limit. Contact-point speed p95 was 0.00077 m/s, maximum
  0.00308 m/s. The maximum joint-speed magnitude was 1.276 rad/s.

This deterministic single-state result demonstrates that a bounded 6D pulse
followed by PD can redirect at least one early instability trajectory into a
4-second stable hold. It does not establish a robust recovery basin, an
indefinitely stable stand, or a working recovery policy across initial
conditions. No A–D state was recovered by this finite pulse grid. The next
test should repeat the successful action against nearby *physical* states
(pitch and pitch rate), separate perturbations in the two coordinates, and
map success/failure under the same 2+2-second criterion before changing
training or control parameters.

Follow-up: [Phase 3](instability_basin_phase3.md) found success only at the
exact recorded state among the tested nearby states; the same center branch
fell after 347 policy steps when observed beyond the original four seconds.

## Reproduce

```bash
cd ~/robotics/projects/g1-hierarchical-control
source .venv/bin/activate
python3 scripts/evaluate_trajectory_branches.py
python3 scripts/evaluate_trajectory_branches.py \
  --stages P_pre_40 P_pre_20 A_deviation --pulse-steps 5 10 20
```

Local-only outputs, under the ignored checkpoint tree:

- `code/day9/g1_balance/checkpoints/branch_recovery/phase2_20260930T005035Z/`
- `code/day9/g1_balance/checkpoints/branch_recovery/phase2_20260930T005253Z/`

Each output has the selected full branch states (`branch_states.npz`), branch
metadata and hashes, a compact results CSV, and complete per-case traces in
`results.json`. The Phase 1 source and its summary hash are recorded there.
