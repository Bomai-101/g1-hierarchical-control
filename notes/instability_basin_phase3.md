# Phase 3 — local recoverability map around PPO step 145

Status: fixed-action, deterministic slice completed under Environment Reference
V1. No pose, PD, action mapping/scale, reward, termination, PPO parameter, or
MuJoCo physics value was changed.

## Corrected Phase 2 premise

The successful Phase 2 candidate was **knee-pair +0.5 for 10 policy steps,
then zero-residual PD**. It did not return to PPO after the pulse. The Phase 2
report and script documentation were corrected without changing their action
semantics. Original PPO and pure PD are separate comparison controllers here.

## State construction and criterion

The source is the saved `mjSTATE_INTEGRATION` state of the deterministic PPO
trajectory at policy step 145. Its initial Euler pitch is `-0.00072833 rad`;
the observation's angular-y value (`qvel[4]`) is `+0.01360237 rad/s`.
All joints, base position, other base orientation components, and other
velocities are held fixed. Only Euler pitch and `qvel[4]` vary on the grid.
For altered states, the old solver warmstart is cleared and `mj_forward`
recomputes contact dynamics before saving the new full integration state. A
warmstart-only center control still meets the original success criterion, so
the changed outcome is not explained by that bookkeeping step alone.

`qvel[4]` is the **body-local angular-y component**, which approximates Euler
pitch rate while the robot is nearly upright. It is not an exact global pitch
derivative away from this local slice. MuJoCo documents the free-joint
rotational velocity convention in its
[overview](https://mujoco.readthedocs.io/en/3.2.7/overview.html) and the
integration-state/warmstart definitions in its
[simulation guide](https://mujoco.readthedocs.io/en/latest/programming/simulation.html).

At every grid state, the same three controllers run: pulse then PD, pure PD,
and original deterministic PPO. Success means entry into the existing
full-body recovery envelope within the first 100 policy steps (2 s), survival
for 200 steps (4 s), and continuous residence in the hold envelope for the
last 100 steps (2 s). Each grid cell is one deterministic simulation, so the
counts below describe tested states, not probabilities.

## Results

- Coarse grid: seven pitch offsets from `-0.020` to `+0.020 rad` and seven
  angular-y offsets from `-0.100` to `+0.100 rad/s`. Pulse then PD passed at
  **1/49 states**: the exact center. Pure PD and original PPO passed at 0/49.
- Fine grid: seven pitch offsets from `-0.002` to `+0.002 rad` and seven
  angular-y offsets from `-0.010` to `+0.010 rad/s`, including `±0.0005 rad`
  and `±0.002 rad/s` near center. Again, pulse then PD passed only at the
  exact center (1/49); both comparisons passed at 0/49. The nearest tested
  angular-y neighbors at `±0.002 rad/s` fell after 192 steps, only eight
  steps short of the four-second cutoff. The nearest `±0.0005 rad` pitch
  neighbors with unchanged angular-y fell after 153 steps.
- Both feet retained four floor-contact geoms at every fine-grid initial
  state. Initial contact penetration ranged from `0.000741` to `0.000779 m`.
  Coarse-grid states had either two or four contacts per foot, so larger
  artificial pitch offsets also changed the support geometry. Across both
  grids, no tested run clipped raw PD torque or reached 95% of its limit.
- A 10-second center stress test showed that the pulse-plus-PD trajectory
  **eventually fell at 347 policy steps after branching** (6.94 s). Pure PD
  and PPO both fell after 96 steps from the same saved state.

The pulse has genuine short-horizon control effect, but the tested action is
not a robust recovery policy or an indefinitely stable stand. The result is a
thin, horizon-dependent success point in this fixed full-body slice. It does
not establish a global recoverability boundary: action timing, pulse shape,
other joints, contact states, and longer control horizons were not mapped.

The next decision should focus on **recovery-to-hold control**, particularly
what happens after the short corrective pulse. Before changing PPO reward or
training length, compare explicit post-pulse feedback/handoff strategies on
this same state and nearby `(pitch, angular-y)` states, and require a longer
hold. The current map does not justify expanding to large-angle disturbances.

Follow-up: the [post-pulse handoff experiment](post_pulse_handoff_phase4.md)
compared existing feedback, PPO, PD, and timed handoffs on the center and four
nearest states. None met a 10-second continuous-hold criterion.

## Reproduce and inspect

```bash
cd ~/robotics/projects/g1-hierarchical-control
source .venv/bin/activate
python3 scripts/map_recoverability_basin.py
python3 scripts/map_recoverability_basin.py \
  --pitch-offsets -0.002 -0.001 -0.0005 0 0.0005 0.001 0.002 \
  --rate-offsets -0.010 -0.005 -0.002 0 0.002 0.005 0.010
python3 scripts/map_recoverability_basin.py \
  --pitch-offsets 0 --rate-offsets 0 --horizon-steps 500
```

Local-only output directories under the ignored checkpoint tree:

- Coarse: `code/day9/g1_balance/checkpoints/recoverability_maps/phase3_20260930T010734Z/`
- Fine: `code/day9/g1_balance/checkpoints/recoverability_maps/phase3_20260930T010926Z/`
- Extended center: `code/day9/g1_balance/checkpoints/recoverability_maps/phase3_20260930T011141Z/`

Each directory contains `initial_states.npz`, state metadata/hashes,
`results.csv`, complete `results.json`, and the labelled vector plot
`recoverability_map.svg`. The result metadata records Phase 1/2/checkpoint
hashes and the exact perturbation definition.
