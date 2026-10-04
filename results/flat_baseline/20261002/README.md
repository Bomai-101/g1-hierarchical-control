# Flat baseline commands and PPO continuation — 2026-10-02

Status: Isaac24 + MuJoCo24 pre-training cases complete. Isolated500-update
continuation is COMPLETE (500 updates, process exit0). No completed trained
candidate or accepted baseline is claimed yet.

The reference flat task includes target-heading feedback and yaw-rate
tracking. The policy sees velocity commands, not target heading. The current
suite compares explicit constant rates and the original heading feedback
separately at vx0.5/1.0. Zero-heading cases start at0.5rad to test correction.

MuJoCo24 cases complete30s without the evaluator's height fallflag. Original
actors and PD are preserved. Local final heading errors0.573–0.862rad;
supplied1.333–1.430rad. Same feedback therefore does not eliminate transfer
bias. This does not establish successful straight walking or safe posture.

Local exported NPZ8arrays exactly equal the checkpoint's actor parameters.
Static joint effort limits differ for37/37mapped joints; implicit versus
explicit PD, model/collision assets, timestep and friction also differ.
These are audit observations, not proven causes or authorization to enlarge
model force limits.

The continuation is an isolated candidate initialized from local model1499,
4096envs,seed42,500additional PPOupdates,49,152,000transitions. Original flat
reward/command/PD configuration and optimizer are retained. A completed
training run alone cannot promote a candidate or demonstrate transfer repair.

See notes/flat_baseline_execution_plan.md for scope and command semantics.
Full traces/logs: ignored code/day9/g1_balance/checkpoints/flat_baseline/20261002/.
No rough training, recovery switching, commit or push.

## Completed Isaac comparison

All24 cases complete30s without physical termination or timeout. All384
environment-case episodes survive; parallel copies are not independent seeds.
Independent stored-trace checks confirm1500samples/case, no reset contamination,
initial heading and per-step actual command law. Detailed ranges appear in
baseline_summary.json. Original command/source/checkpoint hashes were verified.

This supports flat task capability in Isaac. Fixed zero rate still permits
accumulated heading drift; target-heading feedback is evaluated separately.
This does not establish reliable MuJoCo transfer or robust terrain capability.

## Training status

The learner completed all500updates through iteration1999 and exited0.
Final model1999 SHA-256 matches complete.json; original inputs were unchanged.
See training_complete.json and training_status_snapshot.json.
Candidate export and matched evaluation are pending; no improvement is claimed.

## Post-training evaluation completed

See [model1499 versus model1999](posttraining/README.md). Candidate improves
some Isaac tracking metrics but falls in9/12matched MuJoCo cases; it is not
promoted. Original1499remains the experimental comparison.
