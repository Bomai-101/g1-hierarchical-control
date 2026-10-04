# Day 10: predeclared multi-joint correction and counter-brake

## Question

Can a coordinated hip–knee correction, with a small conditional second phase, reduce the overshoot of the previous knee-only guard and turn local survival gains into sustained standing? This is a scripted, exact-state diagnostic; it does **not** train or evaluate a new PPO recovery policy.

All five continuations restore each of the same nine saved Phase 3 integration states at reset step 145. They share the previously verified `+0.5` paired-knee residual for 10 policy steps, then use zero-residual PD until a common trigger: beginning at post-branch step 30, `|body-local angular-y| ≥ 0.003 rad/s` on three consecutive 50 Hz decisions while `|pitch| ≤ 0.020 rad`. No trigger retriggers.

The predeclared continuations are:

1. Pulse, then zero-residual PD.
2. Prior strong knee guard: normalized `sign(rate) × 0.25` on both knees for five steps.
3. Hip–knee blend: normalized `sign(rate) × 0.05` on both hips and both knees for five steps.
4. The same blend, then up to five steps of `-sign(trigger rate) × 0.05` on both ankles **only while current rate has reversed**.
5. The same blend, then up to five steps of `-sign(trigger rate) × 0.03` on both hips and knees under the same reversal condition.

All actions remain within the unchanged 6D residual interface, action scale `0.15`, PD gains, pose, MuJoCo physics, reward, termination, and 64D observation. The blend/counter values were fixed before the nine-state run; they were not tuned after inspecting its outcomes. The five original states are design states, and the four further grid offsets are additional comparisons, not a statistically independent distribution.

## Result

The full run contains 45 deterministic paired cases. Episode lengths are policy steps **after** the branch; add 145 for counts from reset. The 500-step horizon is 10 s after branching.

| Continuation | Mean length over 9 states | Four-second entry/hold pass | Hold from 2 s through 10 s | Rate reversed at first phase end |
| --- | ---: | ---: | ---: | ---: |
| Pulse then PD | 184.1 | 1/9 | 0/9 | n/a |
| Strong knee guard | 233.9 | 2/9 | 0/9 | 9/9 |
| Hip–knee blend | 191.6 | 1/9 | 0/9 | 9/9 |
| Blend then ankle | 190.2 | 1/9 | 0/9 | 9/9 |
| Blend then opposite hip–knee | 191.8 | 1/9 | 0/9 | 9/9 |

The center state reached 347, 378, 356, 354, and 356 post-branch steps respectively. The strong knee guard still has the largest mean survival gain, driven partly by a negative angular-y offset state that reaches 426 steps, but none of the five continuations holds through the full horizon.

All three first-phase intervention types reverse the very small pre-trigger rate at the end of five steps. In both counter-brake variants the rate crosses back during the next step, so the conditional rule fires **exactly once per state**, not for the allowed five steps. At the end of the nominal second-phase window, none of the 36 triggered intervention trials has a lower absolute angular-y rate than at trigger. Thus the counter-phase experiment tests a one-step conditional response in practice; it does **not** establish that a properly timed sustained ankle or hip–knee brake is ineffective.

No case recorded PD torque clipping or reaching 95% of configured torque limits. The largest measured foot-contact speed across continuations was about `0.032 m/s`; this is a diagnostic cost signal, not a physical safety certification. The observed survival increments in the blend cases are small and are not evidence of an enlarged recovery region.

## Interpretation and next decision

Coordinating hip and knee at these predeclared magnitudes did not solve the overshoot/hold problem. The trigger occurs at a very small angular rate, and even the smaller blended five-step pulse reverses its sign. The one-step conditional counter action is too brief, by its own switching rule, to support a conclusion about multi-joint recovery capacity.

Do **not** promote any of these scripted guards to the watchdog or a learned-policy baseline. The next diagnostic should examine a continuous, magnitude-limited feedback/ramp with a neutral deadband and a fixed phase-space braking target, measuring both pitch and angular-y over the full trajectory. Its candidates and hold-out states must be defined before looking at their results. Observation changes such as previous action or projected gravity belong to a separate, versioned Standing V2 ablation; mixing them into this frozen Day 10 comparison would confound attribution.

## Reproduction

```bash
PYTHONPATH=src:scripts python3 tests/test_multijoint_brake.py
python3 scripts/evaluate_multijoint_brake.py
```

The summarized run is local and ignored by Git: `code/day9/g1_balance/checkpoints/multijoint_brake/multijoint_20260930T093217Z/`. Its CSV/JSON record state provenance, per-step traces, trigger and phase-end rates, hold outcomes, torque counts, and foot-contact speed. State loading depends on the existing `evaluate_damped_rate_guard.load_states` diagnostic helper.
