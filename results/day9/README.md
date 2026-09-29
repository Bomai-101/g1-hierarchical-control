# Day 9 results

This directory contains compact, reviewable artifacts for the Day 9 residual
PPO robustness milestone.

- `day9_results_summary.json` records the headline baseline, selected
  checkpoints, protocol, conclusion, and source hashes.
- `checkpoint_robustness_summary.csv` contains one row per update from the
  30-update randomized-reset run.
- `figures/` contains dependency-free SVG charts generated from the local
  checkpoint-sweep JSON.

The full `.pt` checkpoints and per-step trace JSON files remain local under
`code/day9/g1_balance/checkpoints/` and are intentionally not committed in
bulk. The figures can be regenerated with:

```bash
cd code/day9/g1_balance
python3 -m tools.generate_day9_artifacts \
  --checkpoint-sweep-json /path/to/checkpoint_sweep/summary.json \
  --original-robustness-json checkpoints/robustness_evaluation.json \
  --output-dir ../../../results/day9
```
