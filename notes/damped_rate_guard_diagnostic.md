# Day 10: bounded rate-feedback diagnostic

## Question and frozen comparison

The previous one-shot knee correction improved survival in some exact branch states but reversed the body-local angular-y rate sharply. This experiment asks whether a much smaller, rate-proportional correction avoids that overshoot while retaining the survival benefit. It is a diagnostic, not a new learned policy or a tuned recovery controller.

All cases restore the same saved MuJoCo integration states at reset step 145, apply the same normalized `+0.5` paired-knee pulse for 10 policy steps, and then compare three continuations under the frozen Day 9 model and PD settings:

1. Zero residual / PD only.
2. The prior one-shot `+0.25` paired-knee rate guard for five steps.
3. A predeclared one-shot bounded rate-feedback guard for five steps: paired-knee normalized action `clip(2.0 × observed body-local angular-y, ±0.02)`, recomputed at each policy step.

The two guards use the same trigger: monitoring from local step 30; `|angular-y| ≥ 0.003 rad/s` for three consecutive policy decisions while `|pitch| ≤ 0.02 rad`. They do not retrigger. The `0.02` cap is 12.5 times smaller than the previous `0.25` pulse. The sign and gain were fixed before the nine-state run; they have not been optimized on these outcomes.

The five original states are the center and offsets `pitch ±0.0005 rad`, `angular-y ±0.002 rad/s`. Four additional states were selected from the previously generated Phase 3 grid: `pitch ±0.001 rad`, `angular-y ±0.005 rad/s`. They were not used to choose the prior guard trigger, but are not an independent distribution or a held-out experiment in a strict statistical sense.

## Recorded outcome

The full run contains 27 paired cases (nine states × three continuations). Episode lengths below are policy steps **after** the saved branch state; add 145 for the corresponding count from reset. The experimental horizon is 500 post-branch steps.

| Continuation | Mean post-branch length (9 states) | Four-second criterion | Hold from 2 s through 10 s | Pulse-end rate magnitude reduced | Pulse-end rate reversed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pulse then PD | 184.1 | 1/9 | 0/9 | n/a | n/a |
| Prior `+0.25` guard | 233.9 | 2/9 | 0/9 | 0/9 | 9/9 |
| Bounded rate feedback | 184.4 | 1/9 | 0/9 | 4/9 | 0/9 |

The bounded guard matched the PD-only step count exactly in six states and gained just one step in three others; it did not preserve the strong guard's survival benefit. Its center state remained at 347 post-branch steps, compared with 378 for the strong guard. In the `angular-y = -0.002 rad/s` design state, the strong guard reached 426 steps and passed the four-second criterion; the bounded guard reached 193 steps and failed it. None of the four extra states passed the four-second criterion under any continuation.

The bounded action reached at most normalized magnitude `0.02`. No case recorded torque clipping or near-limit events. The rate comparison refers only to the instant at the end of the second five-step intervention; it is **not** a sustained braking or recovery measure. In particular, a small reduction in instantaneous rate did not imply stable standing.

## Interpretation and next decision

There is a local tradeoff, not a recovery solution: the large pulse changes trajectories and sometimes delays falling, but produces rate reversal; the predeclared small feedback avoids reversal while failing to improve hold or survival meaningfully. A simple `gain × angular-y` knee rule is not sufficient on this branch-state set. This does not prove that the 6D residual action space lacks recovery authority; action direction, timing, duration, pose, contact state, and closed-loop policy structure remain confounded.

Do not add this guard to the production watchdog or claim a robust recovery region. The next useful diagnostic is a **fixed-state pulse-response identification** over short, signed knee/hip interventions, measuring pitch and angular-y response before searching further feedback rules. Keep the existing PD and residual PPO baselines unchanged.

## Reproduction and provenance

Run from the repository root with its existing virtual environment:

```bash
python3 scripts/evaluate_damped_rate_guard.py
```

The script verifies source trajectory, checkpoint, Phase 3 results, and saved integration-state hashes before evaluation. It writes `results.csv` and `results.json` under the locally ignored `code/day9/g1_balance/checkpoints/damped_rate_guard/`. The run summarized here is `damped_guard_20260930T085326Z`; raw outputs are local and are not part of the public repository.
