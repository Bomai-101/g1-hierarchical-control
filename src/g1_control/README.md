# Control code and experiment interfaces

The latest bounded walk/hold prototype is documented in the [research overview](../../README.md). Its online experimental sequence/watchdog protocols live in `scripts/skill_sequence_protocol.py` and `scripts/skill_watchdog_protocol.py`; Isaac integration lives in the evaluation scripts. The earlier `hierarchy/task_sequence.py` has a different single-policy command-retention protocol and is not the validated two-actor stop/resume implementation.

`src/g1_control/` contains the reusable interfaces, metrics and historical learning components.  It is grouped by
responsibility rather than by the date on which a learning exercise happened.

```text
config/       frozen experiment configuration
envs/         MuJoCo state, observation, action-to-PD, and physics path
learning/     Actor, Critic, rollout buffer, GAE, and PPO update
training/     reset curriculum and run provenance
evaluation/   deterministic, disturbance, and checkpoint evaluations
monitoring/   passive locomotion signal recording and temporal screens
hierarchy/    measured-goal task sequencing, skill commands and message watchdog
controllers/  PD and residual-PPO controller interface for Day 10
```

The Day 9 checkpoint directory, results, media, and notes deliberately remain
where they were recorded.  `code/day9/g1_balance/` retains compatibility
wrappers, so prior commands keep working while new work uses `scripts/` and
this package.

From the repository root, current commands are:

```bash
python3 scripts/run_controller_smoke.py --mode pd_stand --steps 100
python3 scripts/run_controller_smoke.py --mode ppo_stand \
  --checkpoint code/day9/g1_balance/checkpoints/best_deterministic.pt --steps 100
python3 scripts/evaluate_robustness.py
```

The [locomotion hierarchy MVP](hierarchy/README.md) uses 123D/37D locomotion
and stays separate from historical 64D standing safety experiments.
