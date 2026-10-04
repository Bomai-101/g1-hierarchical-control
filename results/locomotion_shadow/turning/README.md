# Six turning comparisons and passive-monitor verification

Original comparisons: 1 October 2026. Monitor grid rechecked: 2 October 2026.

## Original results

[comparison.csv](comparison.csv) preserves every evaluator metric from the
six original single-rollout CSVs, normalizing only model/metadata paths and
policy labels. The [raw directory](raw/) preserves the original files byte
for byte, including paths and line endings. [manifest.json](manifest.json)
records their SHA-256 hashes and sizes. No weights or external source/assets
are bundled here.

Forward/lateral/yaw velocity commands are [1, 0, yaw] in m/s, m/s, rad/s.
All runs use plane terrain, friction 0.8 and 20 seconds without reset.
The intended setup is seed 42, 1 ms physics and 50 Hz policy updates; these
settings were captured explicitly in the fresh paired replay. Original CSVs
do not independently record seed, timestep or runtime version.

| Actor | Yaw command (rad/s) | Body-forward mean (m/s) | Forward RMSE (m/s) | Yaw RMSE (rad/s) | Net world X (m) | Absolute net world Y (m) | Fell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supplied | -0.2000 | 1.0319 | 0.1971 | 0.7710 | 1.2418 | 4.1613 | 0 |
| Supplied | 0.0000 | 1.0662 | 0.2174 | 0.7488 | 1.2197 | 0.3505 | 0 |
| Supplied | 0.2000 | 1.0707 | 0.2181 | 0.6791 | 1.3897 | 2.2561 | 0 |
| Locally trained | -0.2000 | 1.1422 | 0.2426 | 0.5184 | 1.6637 | 0.3376 | 0 |
| Locally trained | 0.0000 | 0.9845 | 0.1471 | 0.4536 | 2.9134 | 3.2724 | 0 |
| Locally trained | 0.2000 | 0.9182 | 0.1537 | 0.4491 | -1.2796 | 6.0104 | 0 |

Yaw RMSE measures body-frame angular z against the yaw command. It does not
establish the direction of Euler-heading change. Net world X is displacement,
not path length, and world Y is absolute end-to-start displacement, not total
lateral travel. Both actors completed each run without the evaluator's
base-height fall flag, yet yaw tracking remained poor. These six runs are one
condition per actor/command, not a multi-seed robustness evaluation.

## Exact monitoring off/on checks

[verification_grid.json](verification_grid.json) reports **PASS for all six
pairs**: two exported actors × yaw commands [-0.2, 0, +0.2] rad/s. Each pair
contains two fresh independent 20-second non-rendered rollouts with seed 42,
plane/friction 0.8, 1 ms physics, 20 ms policy updates and the same CPU setup.
Each records 1,000 actions, 1,001 sampled states and 20,000 physics steps.

For every pair, actions and qpos/qvel samples are exactly equal; the digest
of qpos, qvel and ctrl after every physics step is equal; trajectory CSV bytes
and all metrics except `shadow_enabled` are equal. Signal timestamps/counts
match the monitored trajectory. The monitor receives only copied scalar
signals. It cannot write actions, PD targets, simulator state or switch a
controller; its returned diagnostics are used only for log output.

Full off/on CSVs, control traces, scalar signals, candidate events and logs
remain local under the ignored checkpoint directory:
`code/day9/g1_balance/checkpoints/locomotion_shadow/20261002/turning_validation/`.
The original off/on proof remains under the `20261001` checkpoint directory.

## Runtime and historical reproduction limits

The fresh six-pair sweep used NumPy 2.5.3 and MuJoCo 3.14.0. All six paired
trajectories agree exactly, but their numerical metrics differ from the
historical CSVs. [replay_comparison.csv](replay_comparison.csv) contains the
fresh replay metrics; each pair's JSON contains the exact per-metric delta
from the original. `original_numeric_metrics_exactly_reproduced` is false
and is independent of the non-interference verdict.

A further straight-command check with the existing MuJoCo 3.13.0 environment
also differed from the historical metrics, so simulator version alone is
not established as the cause. Historical CSVs lack a complete runtime/model
snapshot. Policy and metadata hashes identify current replay inputs, not an
immutable snapshot of all original inputs. The reference HEAD remains
`19ce3c3561e3024055bf250ddd851b325175a7eb`; the local actor hash is
`ae0ca6b9677a19ffbc09325733ce3673d6ef7732547d96662a85016d84de31f0`.
Do not describe the old and new trajectories as identical.

The continuous-exceedance screens remain exploratory. The locally trained
zero-command replay has mean signed body angular z about -0.4043 rad/s but
only a startup forward-error candidate, and no yaw-error candidate; gait
oscillations interrupt the yaw-error dwell. The +0.2 command also produces
negative mean body angular z, with no yaw-error candidate. Thus passive
non-interference is established here, while directional-bias detection and
turning quality remain unresolved. No recovery/switching authority is added.

## Reproduce or recheck

From the repository root, with the existing MuJoCo virtual environment and
matching external actor/model assets, choose new output directories:

```bash
.venv/bin/python scripts/check_locomotion_shadow_grid.py \
  --reference-root /home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco \
  --archive-source code/day9/g1_balance/checkpoints/locomotion_shadow/20261001/g1-turning-comparison-7W7p2P4L \
  --output-dir code/day9/g1_balance/checkpoints/locomotion_shadow/NEW_RUN/turning_validation \
  --summary-dir results/locomotion_shadow/NEW_ARCHIVE
```

Use `--verify-existing` with a saved rollout directory to recheck all pairs
and archive their summaries without rerunning physics. Summary directories
must always be new. The runner sets OpenBLAS/OMP threads to one consistently
for both sides and never alters source result files.

```bash
.venv/bin/python tests/test_locomotion_shadow.py
.venv/bin/python tests/test_locomotion_shadow_verification.py
```

Three temporal-screen tests and six evidence-verification tests pass. The
verification tests also pass under `python -O`; missing signals, mismatched
timestamps and empty traces remain errors in optimized execution.
