# Day 10: matched first-pulse trajectory mechanism

## Purpose and protocol

Earlier branch experiments established that a fixed knee pulse can extend survival from one exact state, but did not show *how* it changes the trajectory. This diagnostic aligns four continuations from each of five identical saved Phase 3 MuJoCo integration states at reset step 145:

1. Zero-residual PD, with no pulse.
2. The frozen deterministic Day 9 PPO checkpoint, with no pulse.
3. Normalized `+0.5` on both knees for 10 policy steps, then zero-residual PD.
4. The same first pulse, then the previously declared rate-triggered `+0.25` knee guard for five steps, then PD.

The first pulse changes **knee target position** from `0.230` to `0.305 rad` through the frozen `0.15` action scale; it is not a direct external force. At every 50 Hz policy step the replay records 6D action, six selected joint targets/actual positions/velocities, applied PD torque, pitch, body-local angular-y, height, and left/right floor contact occupancy, normal force, and contact-point speed. The original model, checkpoint, reward, termination, pose, PD, physics, and observation are unchanged. Source/model/checkpoint and saved-state hashes are verified before replay.

## Exact-state result

Episode lengths below are policy steps **after** the common saved state; add 145 for counts from reset. Each case was replayed once deterministically. The two no-pulse cases on the center state both reproduce 96 post-branch steps, corresponding to the historical PPO total of 241 steps from reset.

| Saved-state offset | No-pulse PD | No-pulse PPO | First pulse → PD | First pulse → strong guard |
| --- | ---: | ---: | ---: | ---: |
| Center | 96 | 96 | 347 | 378 |
| Pitch −0.0005 rad | 97 | 97 | 153 | 188 |
| Pitch +0.0005 rad | 95 | 95 | 153 | 214 |
| Angular-y −0.002 rad/s | 96 | 96 | 192 | 426 |
| Angular-y +0.002 rad/s | 96 | 96 | 192 | 199 |

The first pulse strongly delays the center-state failure but does not produce a comparably long interval in the four nearby states. For the center, no-pulse PD and PPO also have very similar pitch trajectories: the maximum absolute pitch difference over their common 96 steps is about `0.00054 rad`. This does not mean the PPO policy is identically zero or ineffective in every situation; it is a local, matched-state observation.

## What the time series shows

From the center state, no-pulse PD reaches persistent outward pitch (`|pitch| ≥ 0.02 rad` with outward rate for five steps) at local step 33 and `|angular-y| ≥ 0.15 rad/s` at step 47, then terminates at 96. Both feet retain floor contact until termination; no torque command clips. PPO without a pulse follows almost the same sequence.

The first pulse immediately redirects motion: at local step 5 the pulsed state has pitch `−0.0324 rad` and angular-y `−0.1856 rad/s`; the no-pulse PD state is at `+0.0008 rad` and `+0.0171 rad/s`. The first pulse also changes the mean knee torque and foot normal forces, but does not cause gross foot sliding or torque clipping in the measured run. Its knee target returns to `0.230 rad` at local step 11. By local steps 50–200, the center pulsed trajectory is near pitch `−0.007` to `−0.006 rad` with small angular-y, both feet contacting. Later it again drifts forward: first persistent outward pitch occurs at step 284, `|angular-y| ≥ 0.15 rad/s` at step 298, and it terminates at 347. This is a long **transient near-standing interval**, not sustained recovery.

The strong second knee guard instead moves the center trajectory toward negative pitch: at local step 300 it has pitch `−0.0201 rad` and angular-y `−0.0324 rad/s`, whereas first-pulse-then-PD is already at `+0.0575 rad` and `+0.1731 rad/s`. The guarded trajectory finally terminates at step 378 with pitch `−0.834 rad` and angular-y `−2.251 rad/s`. The 31-step gain over first-pulse-then-PD therefore ends in **opposite-direction failure**, not stable hold.

All five states and four continuations retain both foot contacts through the sampled policy steps before termination; no raw PD torque clipping was recorded. These checks do not establish hardware safety. The large asymmetry between the exact center and tiny neighboring perturbations is the most important limitation: the pulse appears to place one trajectory into a narrow long-lived transient, not a demonstrated neighborhood of recoverable states.

## Decision

The first pulse is not merely delaying the same immediate forward fall: it creates a distinct short backward excursion and a long near-upright transient for the exact center. Yet this transient eventually drifts forward and is fragile to tiny initial offsets; the second strong guard can redirect it into backward failure. Thus the present evidence supports **trajectory shaping and local authority**, not a reliable recovery controller or an expanded basin.

Stop adding gains or observations to this frozen branch experiment as if one number were the sole problem. The next decision should compare the *entry state of the long transient* with the original standing reference and determine whether a versioned standing baseline or a new, clearly specified recovery task is needed. This should be separated from later Standing V2 observation ablations (previous action, projected gravity, etc.).

## Artifacts and reproduction

Run from the repository root with the existing virtual environment:

```bash
PYTHONPATH=src:scripts python3 tests/test_first_pulse_trajectories.py
python3 scripts/compare_first_pulse_trajectories.py
```

The verified full run is local and ignored by Git at `code/day9/g1_balance/checkpoints/first_pulse_timelines/first_pulse_20260930T100200Z/`. It contains one per-step CSV for each state/continuation, `summary.json`, `center_timeline.svg`, and a rendered `center_timeline.png`. The figure plots pitch, body-local angular-y, total foot normal force, and mean knee residual action from the same exact center state. The script depends on the existing `evaluate_damped_rate_guard.load_states` verified-state loader.
