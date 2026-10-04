# Flat baseline execution plan — 2026-10-02

User authorized execution after reviewing the plan. Priority is stable flat
walking and correct turning. No commit/push, rough training or supervisor
switching is part of this stage.

## Reference task and reproduction boundary

The reference flat task samples vx in [0,1], vy in [-0.5,0.5], and world
heading in [-pi,pi] every 10 s. Its original command manager converts heading
error to a rate: clip(0.5*wrapped_error,-1,1). The actor sees three velocity
commands, not heading. Local saved training env.yaml confirms these settings.
The 2% standing probability is a training setting; directed evaluations
disable standing sampling so requested walking is actually tested.

Reference play/eval overrides clamp yaw bounds to the CLI value while
retaining heading mode. Equal lower/upper bounds make the output a constant
rate. Default zero therefore removes heading correction. Fixed-rate and
heading-feedback tasks must be scored separately. Reference training reward
uses world omega_z, while the stock evaluator reports body omega_z; both are
recorded in this stage. Neither is automatically identical to Euler yaw rate.

## Execution order

1. Preserve existing model_1499 and all older result packages. Check WSL
   execution, GPU and immutable input hashes.
2. Evaluate supplied/local checkpoint in Isaac with 16 fixed-start parallel
   environments, seed42, plane, static/dynamic friction0.8/0.6, 30s. Disable
   observation corruption, random standing and mid-run command resampling.
   At vx0.5/1 evaluate constant yaw0/-0.2/+0.2 and target heading0/-pi/2/+pi/2.
   The heading0 task starts at+0.5rad to verify correction. Actor observation
   command slices and the original heading feedback law are checked each step.
   Score first episode only, excluding post-reset samples; separate physical
   terminations from timeouts. Parallel copies are not independent seeds.
3. Match these command modes in MuJoCo using the original reference actor,
   joint mapping, action scaling and PD. Friction0.8, policy20ms, physics1ms.
   Restore the same upstream heading law, without changing actor/PD. The
   engines/assets/contact models differ, so this is a transfer comparison,
   not an assertion of identical physics.
4. Decide training from this comparison. If an isolated continuation is
   warranted, initialize from the local checkpoint and retain the original
   flat task/rewards/commands/PD/optimizer as a controlled experiment: 4096
   environments, 500 additional PPO updates, seed42, 49,152,000 transitions.
   Save a separate candidate under the project's ignored checkpoint directory;
   never overwrite the original. Training return alone cannot promote it.
5. Evaluate any candidate with the same commands and compare stability,
   speed error, signed yaw bias, heading convergence and posture. If Isaac is
   already satisfactory and transfer remains poor, prioritize interface/model
   audit; extra PPO iterations are not evidence of a migration fix.
6. Freeze a reviewed flat baseline only after these tasks meet a separately
   stated acceptance contract. Then branch rough-terrain PPO fine-tuning from
   that checkpoint and retain flat regression tests. Hierarchical diagnostics
   and recovery remain a later stage.

## Runtime caveat discovered during preflight

The first process lacked the WSL CUDA library path. PhysX failed to load CUDA
and fell back to software; it produced no usable evaluation and was stopped.
`scripts/run_flat_isaac.sh` supplies `/usr/lib/wsl/lib` to this process only.
The subsequent CUDA smoke completed. Graphics/Vulkan shutdown errors remain
distinct from usable headless GPU physics; graphical support is not claimed.
An initial smoke also exposed that changing the template config after env
construction does not update reset yaw. The full evaluator changes the live
event-manager config and asserts the actual initial heading before scoring.
Both preliminary smoke directories are retained and excluded from results.

## Output locations

- Full traces/logs: ignored
  `code/day9/g1_balance/checkpoints/flat_baseline/20261002/`.
- Planned reviewed summaries: `results/flat_baseline/20261002/`.
- Execution state and subsequent decision: `G1_PROGRESS.md`.

No accepted baseline, successful transfer or improved trained candidate is
claimed solely by this plan.
