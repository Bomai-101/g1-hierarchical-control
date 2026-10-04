# Held-out reset-seed and entry-time replication; experimental demo frozen

Frozen local walking1499 + independently trained teacher-handoff hold499. Direct actor switching was selected from the prior seed42 comparison **before** this retest. No network retraining, threshold tuning or default-controller promotion.

Isaac flat,0.5m/s; new reset seeds101/202/303; stop/drop bases3.22/6.22/9.22s with four0.02s-staggered copies each; yawprelude -0.2/0/+0.2.27batch runs,324environment cases. No prior seed42 cases pooled into these results. Seeds vary initial world root x/y/yaw; initial joint poses and velocities remain fixed. Parallel copies within a seed are not independent statistical trials. No noise, external forces or rough terrain.

| Scenario | New cases | Requested task pass | Physical failures before case end |
|---|---:|---:|---:|
| Normal: walk → hold → walk |108|108|0|
| Temporary1s task-heartbeat loss: stop → resume |108|108|0|
| Persistent heartbeat loss: hold actor observed12s; no resume |108|108|0|

Each seed/time stratum passes12/12 per scenario. The persistent observation includes stopping; median confirmed-hold segment is10.16s, not12s of already stationary motion. End-of-observation ABORT terminates scoring and is not a physical emergency stop. There were zero returns-to-walk while freshness was absent.

## Measured distances and timing

Medians across all108available cases per scenario; full n/p90/min/max in `summary.json`. Net displacement and accumulated root path are distinct.

| Measurement | Normal | Temporary loss | Persistent loss |
|---|---:|---:|---:|
| Request/drop → stopped confirmation: root xy net |8.83cm|20.64cm|20.64cm|
| Same interval: accumulated root xy path |14.84cm|26.74cm|26.74cm|
| Confirmed stop → hold exit: root xy net drift |0.903cm|0.889cm|0.911cm|
| Stop decision → stopping confirmation |1.84s|1.84s|1.84s|
| Stop decision → permission to resume walking |3.84s|3.84s|N/A|
| Resume actor → sustained speed-tracking confirmation |1.72s|1.72s|N/A|

Watchdog detection delay after delivery interruption:0.24–0.30s,median0.27s. After delivery resumes, two-message freshness confirmation delay0.12–0.18s,median0.15s. Physical stopping time is separate from communication detection. The stop-decision-to-resume delay includes1s confirmation and2s retained readiness. Tracking confirmation after resume uses a full0.5s window plus1s dwell, and is separate from surviving the full10s resumed horizon. Final2s forward-speed RMSE median~0.0752m/s.

Maximum request-to-stop net displacement11.49cm(normal)/23.58cm(loss); maximum post-confirmation hold net drift1.79/1.59/1.67cm. These are observed values, not safety boundaries. Net drift may point backwards, so net magnitudes across phases do not add. Supplementary metrics exclude invalid/reset and post-terminal states; missing stop confirmation would be unavailable rather than zero.

## Verification and freeze

Independent optimized-Python reconstruction:324cases,318455first-episode action rows,max actor proposal error3.34e-6,216matched pre-request prefixes exact. Check state-derived123D observations,37D actor outputs,previous applied actions,command schedule,gates,watchdog freshness/permission,metrics and pass predicates; explicitly verify completed10s resume and12s persistent observation horizons. Three new seeds produce distinct initial root-pose arrays. Five metric tests pass, including reset exclusion,causal first confirmation and full1s resume dwell.

`frozen_demo.json` records experimental version `g1_hierarchical_isaac_low_speed_demo_v1_20261004`, immutable policy/model/source/interface hashes, prospective configuration,validated results and reproduction arguments. This freezes a bounded **experimental demo**, not a production controller or the outstanding MuJoCo directional baseline.

- `summary.json`, `strata.csv`: numerical results with complete denominators and availability.
- `verification/`, `final_verification/`: reconstruction evidence and trace SHA-256.
- `comparison.png`: per-entry-time pass rates and stop-vs-hold displacement.
- `initial_pose_variation.json`: actual reset-pose variation by seed.
- `evidence/`: final source snapshots (authoritative `final_analysis_and_source_hashes.json`; earlier analysis hash records retained as history),, plans, input hashes, case results and environment inventory.
- Full traces,weights/logs: local Git-ignored `code/day9/g1_balance/checkpoints/skills_sprint/20261004/sequence_retest/`; not uploaded to GitHub.
- Previous labeled visual replay: [walk–hold–walk video](../../20261003/skill_sequence/video/walk_hold_walk.mp4). Isaac recorded states rendered with MuJoCo geometry, zero physics steps; not MuJoCo dynamics validation.

Original standalone10s hold position/heading acceptance is unchanged and distinct. Actual perception-pipeline dropout,high-speed entry,disturbance/rough terrain,MuJoCo transfer,fallen-robot recovery and hardware safety remain unvalidated.20ms simulation control period is not measured real-time compute deadline compliance. No retraining,commit/push or full application package assembly.
