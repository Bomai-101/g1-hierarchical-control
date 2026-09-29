# Day 9 tools

These scripts support calibration, diagnostics, and report generation. They
are not imported by the training or evaluation entrypoints in the parent
directory.

Run current tools from `code/day9/g1_balance` with module syntax, for example:

```bash
python3 -m tools.test_balance_env
python3 -m tools.test_action_direction
python3 -m tools.sweep_joint_pd
python3 -m tools.sweep_standing_pose
```

The fine, micro, and ridge pose sweeps reuse `sweep_standing_pose` and must
also be run with `python3 -m tools.<name>`.

`legacy/` contains historical diagnostics that no longer match the current
6D action interface. They are retained only to document the experiment path
and should not be treated as current runnable tools.

`generate_day9_artifacts` converts the local checkpoint-sweep JSON into the
small CSV, JSON, and SVG artifacts committed under `results/day9/`. It uses
only the Python standard library.
