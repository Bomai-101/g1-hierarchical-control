# Frozen-policy skill sequencing: walk → hold → walk

This experiment uses **two independently trained actors**, local walking1499 and teacher-handoff hold499, under an explicit state-based sequencer. The hold actor is not the walking actor with a zero command. Neither network is retrained here. Flat Isaac, vx0.5m/s, pre-entry yaw -0.2/0/+0.2, seed42,16 staggered handoffs per yaw; parallel copies are not independent random seeds.

| Handoff protocol | Cases | Reached readiness / resumed | Completed10s resumed walking | Sequence pass | Physical failures before case end | Median hold-to-resume time* |
|---|---:|---:|---:|---:|---:|---:|
| Direct actor switch |48|48|48|48|0|3.86s|
| 0.5s fixed-entry action smoothstep |48|47|43|43|5|4.44s|
| 1.5s walking-command braking + smoothstep |48|37|37|37|11|4.60s|

*Time medians condition on cases that resumed. Braking time is additional and not included in hold-to-resume time. Resume tail forward-speed RMSE medians among completed cases are0.0754/0.0747/0.0748m/s.

**Result:** fixed blending/braking did not improve this specific teacher-trained low-speed sequence. Direct switching is the stronger candidate for the following bounded watchdog experiment; no universal conclusion about switching or high-speed entry follows. Both protocols change the physical state fed to the actors. The failure mechanism has not been isolated.

Readiness: full causal0.5s window, mean horizontal body speed<=0.05m/s, actual Euler heading-rate RMS<=0.05rad/s, instantaneous roll/pitch<=20deg and both feet contact>1N. Conditions must persist1s confirmation+2s retention, minimum hold2s, timeout12s. Resume walking command[0.5,0,0] for10s. Sequence pass additionally requires resumed max tilt<=20deg, tail2s forward-speed RMSE<=0.2m/s and yaw-rate RMS<=0.15rad/s. These prospective exploratory criteria are **separate from the original standalone10s hold acceptance**, which remains unchanged. Variable dwell cannot demonstrate every case survives10s holding, precise stop position or target heading.

Independent verification under `python -O`:144cases,137057first-episode action rows, actor max error2.86e-6,96paired pre-decision prefixes exact. Reconstructed state-derived123D observations, applied-action history, frozen actor proposals, blending, causal readiness, metric and pass predicates. Raw post-terminal simulation samples are excluded from case scoring; a recorded later physical failure is not counted as a sequence failure.

- `verification/summary.json`, `verification/cases.csv`: numerical results and check counts.
- `comparison.png`: complete denominators and conditional readiness latency.
- `video/walk_hold_walk.mp4`: preselected direct/yaw0/env0,19.16s, stages shown. **Isaac recorded states rendered on MuJoCo geometry, zero physics steps; not a MuJoCo dynamics validation.**
- `evidence/`: plans, input hashes, case results, environment inventory and frozen source snapshots.
- Raw full-state traces/models/logs: locally Git-ignored `code/day9/g1_balance/checkpoints/skills_sprint/20261003/skill_sequence/`. No default controller change, commit or push.

## Task-heartbeat watchdog follow-up: completed

Use the direct protocol, heartbeat10Hz, TTL0.3s, require two delivered messages to recover freshness. Per environment, messages are dropped at5.22+0.02*j seconds, either for1s or persistently. Actual stale detection triggers the hold actor; physical readiness and fresh requests must both permit return-to-walk. This is a **task-message fault**, not a perception-pipeline dropout.

| Fault | Cases | Stop → return-to-walk sequence pass |12s hold observation pass| Unauthorized return while stale | Physical failures before case end |
|---|---:|---:|---:|---:|---:|
| Temporary1s dropout |48|48|N/A|0|0|
| Persistent dropout |48|N/A|48|0|0|

Persistent cases intentionally do not resume; their0 resumed-walk completions are not failures of the requested stop task. Hold observation pass requires12s first-episode survival, retained settling through the end and maximum tilt<=20deg. This differs from the original standalone hold position/heading acceptance. End-of-observation `abort` denotes experiment termination, not a physical emergency stop or an implemented recovery action. No indefinite holding claim.

Independent `python -O` verification:96cases,89197action rows,max actor error3.34e-6,48paired pre-detection prefixes exact. Message schedules/stale and freshness events reconstructed; persistent resumption prohibited; gate,actor/action and metric checks passed. See `watchdog_verification/` and `evidence/watchdog/`. Actual dropout-to-detection delay0.22–0.30s; temporary median hold-to-resume3.94s includes physical settling and3s readiness confirmation/retention. Communication latency and physical transition time are reported separately.

This establishes a bounded multi-skill supervision demonstration in Isaac. It does not establish MuJoCo transfer, high-speed switching, disturbance robustness, actual perception-dropout recovery or hardware safety. Work started2026-10-03 and final archive completed2026-10-04 Australia/Sydney; date-based directories preserve the original experiment identity.
