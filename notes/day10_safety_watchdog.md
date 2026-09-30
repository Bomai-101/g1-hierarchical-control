# Day 10 — Watchdog and PD fallback

## Phase 1 objective

Add an explicit safety boundary above the frozen Day 9 residual controllers.
The watchdog observes 64D state-derived signals and selects a controller mode;
it does not change PPO, PD gains, reward, termination, action scale, or torque.

## Initial design

```text
requested skill (PD or PPO)
        ↓
watchdog: height, roll/pitch, gyro norm, finite data
        ↓
nominal / warning / fallback
        ↓
latched PD stand fallback when necessary
        ↓
existing Day 9 action → target q → PD → torque → MuJoCo path
```

The fallback decision is deliberately inside the environment fall boundary:

| Signal | Warning | Fallback | Existing termination |
| --- | ---: | ---: | ---: |
| Height | below 0.65 m | below 0.55 m | below 0.45 m |
| Absolute roll/pitch | above 0.35 rad | above 0.55 rad | above 0.80 rad |
| Gyro norm | above 2.5 rad/s | above 4.0 rad/s | none |

## Initial smoke findings

- Nominal PD stand: 100 steps, no warning, no safety event, no termination.
- PPO with an explicit +0.60 rad pitch injection at step 20: immediate
  `ppo_stand → pd_stand` safety event, then termination at step 31.
- PPO with +0.40 rad injection: warning first, escalation to fallback at step
  31, then termination.

These are interface checks, not a claim of recovery success.  The observed
failure after fallback establishes the next experiment: measure the safe
intervention/recovery boundary across controlled disturbance magnitudes.
