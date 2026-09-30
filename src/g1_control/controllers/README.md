# Day 10 — Hierarchical controller interface

This package is the first Day 10 boundary above the frozen Day 9 controller.
It provides a common residual-action interface for:

- `pd_stand`: the calibrated zero-residual PD standing baseline;
- `ppo_stand`: a deterministic residual PPO checkpoint.

Both controllers output the same bounded 6D residual action. `G1Env` still
owns the unchanged Day 9 path from action to target joint positions, PD torque,
and MuJoCo physics.

Run the PD interface smoke test from `code/day9/g1_balance`:

```bash
python3 -m hierarchy.run_controller_smoke --mode pd_stand --steps 100
```

To test a local PPO checkpoint:

```bash
python3 -m hierarchy.run_controller_smoke \
  --mode ppo_stand \
  --checkpoint checkpoints/best_deterministic.pt \
  --steps 100
```

The next Day 10 step adds a watchdog above this registry. It will decide when
the currently selected controller may continue and when the system must switch
to the always-available PD fallback; it will not directly generate torque.
