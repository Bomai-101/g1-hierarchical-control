# Day 10: continuous bounded hip–knee feedback diagnostic

## Question and fixed protocol

Earlier fixed five-step pulses sometimes delayed falling but overshot angular-y rate and never produced sustained standing. This experiment compares two predeclared, continuously recomputed 50 Hz feedback rules against the same zero-residual PD continuation and the prior strong knee guard. It is **scripted diagnostic control**, not PPO training or a learned policy.

All cases restore the same saved MuJoCo integration state at reset step 145 and share the previously validated normalized `+0.5` paired-knee action for the first 10 policy steps. A common trigger begins monitoring at post-branch step 30 and fires after three consecutive decisions with `|body-local angular-y| ≥ 0.003 rad/s` and `|pitch| ≤ 0.020 rad`. Once triggered, the feedback rules remain active until the episode ends or the 500-step horizon; they output zero inside their specified deadbands. They do not change the frozen Day 9 action scale, 64D observation, pose, PD, physics, reward, termination, or PPO weights.

The four continuations are:

1. Zero residual/PD after the shared pulse.
2. Prior one-shot knee guard: `sign(trigger rate) × 0.25` on both knees for five steps, then PD.
3. Rate feedback: on both hips and both knees, `clip(2 × current angular-y, ±0.04)` every policy step; zero when `|angular-y| ≤ 0.001 rad/s`.
4. Phase-space feedback: on those same four joints, `clip(2 × (current angular-y + 0.5 s⁻¹ × pitch), ±0.04)` every step; zero when both `|angular-y| ≤ 0.001 rad/s` and `|pitch| ≤ 0.002 rad`. This corresponds to a local target angular-y rate of `-0.5 s⁻¹ × pitch`.

The gain, limits, deadbands, and state set were fixed before the full run. The 13 states are the earlier center/four near states, four prior extra Phase 3 states, and four additional grid states at `pitch ±0.002 rad` or angular-y offset `±0.01 rad/s`. The latter four were not used to choose this rule, but all 13 still come from the same Phase 3 grid; they are not an independent distribution or evidence of statistical generalization.

## Outcome

The full run contains 52 deterministic paired cases. Lengths are policy steps **after** the saved branch state; add 145 for counts from reset. The 500-step post-branch horizon is 10 s.

| Continuation | Mean post-branch length (13 states) | Four-second entry/hold pass | Hold from 2 s through 10 s | Rate reversed at five steps after trigger |
| --- | ---: | ---: | ---: | ---: |
| Shared pulse, then PD | 170.5 | 1/13 | 0/13 | n/a |
| Strong knee guard | 214.2 | 2/13 | 0/13 | 13/13 |
| Rate feedback | 173.4 | 1/13 | 0/13 | 8/13 |
| Phase-space feedback | 173.8 | 1/13 | 0/13 | 8/13 |

The center state lasted 347, 378, 355, and 353 post-branch steps respectively. None of the four additional grid states passed the four-second criterion under any continuation. At five steps after trigger, the rate-feedback and phase-space rules reduced the absolute rate in only 6/13 and 4/13 cases respectively. Their small mean survival increments are not meaningful recovery evidence.

Both continuous rules spent much of their remaining episode at the normalized action cap: rate feedback averaged 119 active steps and 91 saturated steps per case; phase-space feedback averaged 120 active and 93 saturated steps. The rate-only deadband was reached an average of 0.5 steps per case; the two-variable deadband was never reached. Saturation counts are over the **whole post-trigger trajectory**, including late failure; they do not establish that saturation caused the fall. No case reached PD torque clipping or 95% of a configured torque limit. Maximum measured foot-contact speed across cases was about `0.032 m/s`.

## Interpretation and stop/go decision

Replacing a one-shot knee pulse with these two continuously recomputed, bounded hip–knee rules did not create a sustained recoverability region. It reduced the prevalence of immediate five-step rate reversal relative to the strong knee guard but did not satisfy the recovery/hold criterion. This does **not** prove that feedback control, the 6D action interface, or PPO recovery is impossible. The shared large initial pulse, one narrow family of exact states, hand-chosen linear gains, no foot-placement freedom, and limited observation remain important constraints.

Do not promote either feedback rule to the watchdog or train PPO on an assumed “recoverable basin” from this result. Further blind gain changes on the same 13 states would confound attribution. The next research decision should be whether to (a) investigate *why* the shared large pulse leaves the system on a divergent trajectory, including action/torque/contact time series and the phase-space flow, or (b) establish a new standing/recovery task version with a more informative initial-state and action interface. This decision should be made explicitly before another controller sweep. Previous-action and projected-gravity observations remain separate Standing V2 ablations, not part of this frozen diagnostic.

## Reproduction

```bash
PYTHONPATH=src:scripts python3 tests/test_continuous_feedback.py
python3 scripts/evaluate_continuous_feedback.py
```

The summarized local ignored output is `code/day9/g1_balance/checkpoints/continuous_feedback/continuous_20260930T094435Z/`. Its CSV/JSON preserve the state/checkpoint provenance hashes, per-step state traces, trigger rates, hold outcomes, contact speeds, and action-cap counts. The script reuses `evaluate_damped_rate_guard.load_states` for validated Phase 3 state loading.
