# Phase 1 — matched instability trajectories

Status: **complete** under [Environment Reference V1](environment_reference_v1.md).
No controller, reward, physics, pose, PD, or termination parameters were changed.

## Reproduce

```bash
python3 scripts/record_instability_trajectories.py
python3 scripts/verify_instability_restore.py \
  code/day9/g1_balance/checkpoints/instability_trajectories/<run-directory>
```

The recorder requires the Phase 0 audit to match the current scene, robot XML,
and MuJoCo version. It loads the historical `checkpoints/best_deterministic.pt`
checkpoint and verifies its dimensions, update number and fixed `log_std=-3`.
Both policies start from the **identical** post-settling MuJoCo integration state.

Local-only data for this run:

`code/day9/g1_balance/checkpoints/instability_trajectories/phase1_20260930T002805Z/`

Each policy has:

- `*.npz`: full MuJoCo integration state at every policy boundary, 36 qpos,
  35 qvel, 64D observation, 6D action, 29D target pose and raw/applied PD
  torques at all 10 physics steps within each policy step;
- `*.json`: step-level base state, joint/control/contact summaries, foot and COM
  positions, rewards and termination flags;
- `*_timeline.csv`: compact time series for plotting or spreadsheet inspection.

Array index `0` is the state immediately after reset settling. Index `k` in
`integration_state`, `qpos`, `qvel`, and `observation` is after policy step `k`.
Action/target/torque index `k-1` produced state `k`. States use MuJoCo's
`mjSTATE_INTEGRATION`, which includes physics, user inputs and solver warm-start.
One-step replay from five saved boundaries per policy yielded **zero qpos and
qvel difference** against the original trajectory.

## Matched results

| Measure | Zero PD | Best deterministic PPO |
|---|---:|---:|
| Termination | step 215 / 4.30 s | step 241 / 4.82 s |
| Return | 444.01 | 501.17 |
| First sustained deviation | step 140 / 2.80 s | step 165 / 3.30 s |
| First clear outward acceleration | step 166 / 3.32 s | step 192 / 3.84 s |
| Maximum residual action norm | 0 | 0.00964 |
| Raw PD torque commands clipped | 0 | 0 |
| Physics steps at ≥95% torque limit | 0 | 0 |

Both trajectories maintained floor contact under both feet through nearly all
physics steps (`~99.95%` occupancy per foot). The highest per-step 95th
percentile contact-point horizontal speed was about `0.021 m/s` in each run.
The two trajectories reached almost identical pitch and pitch rate at the
event thresholds; PPO mainly **delayed the same forward-divergence pattern** by
about 25–26 policy steps. This observation does not establish the mechanism
of the delay or show that the PPO policy can recover after divergence starts.

## Event definitions and interpretation

The quiet pitch reference is the median over policy steps 20–80, separately
for each trajectory (about `-0.0064` and `-0.0068 rad`).

- `T_deviation`: at least 10 consecutive steps with pitch error ≥`0.015 rad`
  in magnitude and pitch error × pitch rate positive. The marker is the first
  step in that run, confirmed only after ten steps.
- `T_acceleration`: at least five consecutive steps with pitch error
  ≥`0.03 rad`, outward pitch rate ≥`0.15 rad/s`, and five-step mean outward
  angular acceleration ≥`0.20 rad/s²`. Marker is the first step in the run.
- `T_termination`: first step satisfying the unchanged environment fall rule.

These are operational markers for selecting branch states. They are **not**
physical recoverability limits. In this nominal run, a fixed two-second
training window would end before sustained deviation begins. The next phase
should therefore choose branch points from the observed physical stages and
test braking, return to standing, and sustained hold from the same restored
state. The two-second recovery budget begins at the branch point, not at
episode reset.

MuJoCo state semantics: [Simulation / integration state](https://mujoco.readthedocs.io/en/latest/programming/simulation.html).
