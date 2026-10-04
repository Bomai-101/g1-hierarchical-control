# Day 10: early rate-guard diagnostic

This is an exploratory, deterministic controller comparison under the frozen
Environment Reference V1. No Day 9 pose, PD gain, action scale, observation,
reward, termination, PPO parameter, or MuJoCo physics setting was changed.
The guard is a hand-designed diagnostic action, not a trained policy.

## Protocol

All cases start from the same five full MuJoCo integration states used in the
Day 10 local comparison: the exact PPO-trajectory state at policy step 145,
plus pitch offsets of +/-0.0005 rad or body-local angular-y offsets of
+/-0.002 rad/s. Every case first applies the already verified normalized
knee-pair action +0.5 for 10 policy steps. Thereafter, three continuations
are compared on each identical state:

1. Zero residual action with the frozen PD controller.
2. A one-shot rate guard: begin monitoring at post-branch step 30; if
   `|angular-y| >= 0.003 rad/s` for three consecutive 50 Hz policy decisions
   while `|pitch| <= 0.020 rad`, apply normalized knee-pair action
   `sign(angular-y) * 0.25` for five steps, then return to zero residual.
3. The same trigger and duration with the knee action sign reversed, as a
   direction control.

The threshold and action are fixed across all five states. The five states
were previously examined when choosing this early window, so they are **not
an independent validation set**. Each case is one deterministic replay, not a
success-probability estimate. The horizon is 500 post-branch policy steps.
The existing four-second entry/hold criterion and stricter uninterrupted
ten-second hold criterion are unchanged.

## Results

| State offset | Pulse then PD | Early guard | Reversed guard | Guard trigger |
| --- | ---: | ---: | ---: | ---: |
| Exact center | 347 | 378 | 327 | 221 |
| Pitch -0.0005 rad | 153 | 188 | 136 | 33 |
| Pitch +0.0005 rad | 153 | 214 | 137 | 32 |
| Angular-y -0.002 rad/s | 192 | 426 | 175 | 70 |
| Angular-y +0.002 rad/s | 192 | 199 | 168 | 59 |

Lengths and trigger times are post-branch policy steps; the common 145-step
PPO prefix is not included. The early guard increased episode length versus
the paired PD continuation in all five tested states, while the reversed
action shortened all five. The four-second criterion passed in two of five
guard cases (center and negative angular-y), versus one of five baseline
cases. **No case achieved uninterrupted ten-second hold.** No raw PD torque
was clipped in the 15 cases.

The survival improvements are not evidence of smooth braking. At the end of
the five-step guard pulse, the signed angular-y velocity had reversed in all
five guard cases, while its absolute magnitude was **16.5-27.6 times** the
small pre-trigger magnitude (approximately 0.003-0.005 rad/s before the
pulse, 0.070-0.110 rad/s afterward). This is a substantial opposite-direction
overshoot. The controller altered the failure trajectory and sometimes
delayed termination; it did not demonstrate sustained stabilization or a
reliable recovery region. Contact-speed and torque-cost measurements are in
the full local output and should be considered before any safety claim.

## Reproduction and next gate

From the repository root with the matching local model, checkpoint, and saved
Phase 1/3 states:

```bash
PYTHONPATH=src:scripts python3 tests/test_early_rate_guard.py
python3 scripts/evaluate_early_rate_guard.py
```

The latest full output is under the ignored local directory
`code/day9/g1_balance/checkpoints/early_rate_guard/early_guard_20260930T082618Z/`.
Its `results.json` contains the full policy-step traces, provenance hashes,
trigger measurements, hold outcomes, torque counts, and contact-speed costs.

The next controlled hypothesis is that the fixed five-step pulse is too
strong once a small rate departure is detected. A smaller or rate-proportional
bounded correction should be evaluated with an explicit pulse-end overshoot
limit and on additional, previously unused Phase 3 states. Do not tune a
new controller on these same five states and call them held out.
