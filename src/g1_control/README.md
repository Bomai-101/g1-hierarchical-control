# Active control code

`src/g1_control/` is the canonical, reusable implementation.  It is grouped by
responsibility rather than by the date on which a learning exercise happened.

```text
config/       frozen experiment configuration
envs/         MuJoCo state, observation, action-to-PD, and physics path
learning/     Actor, Critic, rollout buffer, GAE, and PPO update
training/     reset curriculum and run provenance
evaluation/   deterministic, disturbance, and checkpoint evaluations
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
