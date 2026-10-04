# Day 10: exact-state joint pulse-response probe

## Purpose

Before designing a multi-joint recovery rule or training a new PPO policy, identify the short-horizon response of each available residual-action channel. This is **system identification**, not a recovery test or a learned policy. It does not claim that a locally helpful pitch-rate change will produce stable standing.

The probe replays nine saved MuJoCo integration states from the Day 10 Phase 3 grid at reset step 145: the center, four near offsets, and four additional grid offsets. Each state receives 19 deterministic trials: zero residual; positive and negative actions on each of the six individual sagittal joint channels; and positive and negative actions on the left/right hip, knee, and ankle pairs. The six single channels are tested separately to reveal lateral coupling; paired channels test the symmetric sagittal interface used by earlier experiments.

Every nonzero trial applies normalized residual action `±0.10` for five 50 Hz policy steps (0.10 s), then zero residual/PD for 15 more steps (0.30 s). With the frozen action scale `0.15`, the joint-target offset is `±0.015 rad` on each selected joint. All deltas below are relative to the **zero-residual trial from the identical saved state**, not relative to an upright ideal state. There is no PPO action after branching, and no reward optimization. Source trajectory, checkpoint, Phase 3 results, and integration-state hashes are checked before running.

## Initial result

The full local run contains 171 trials (nine states × 19 actions). Means below summarize the nine states; these are deterministic finite-difference responses within one nearby state grid, not population estimates.

| Paired channel | Signed action | Mean angular-y change at 0.10 s (rad/s) | Mean pitch change at 0.10 s (rad) | Mean angular-y change at 0.40 s (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Hip | -0.10 | +0.02305 | +0.01122 | +0.00428 |
| Hip | +0.10 | -0.02491 | -0.01082 | -0.00115 |
| Knee | -0.10 | +0.03804 | +0.00487 | +0.00924 |
| Knee | +0.10 | -0.02217 | -0.00470 | +0.00026 |
| Ankle | -0.10 | +0.00226 | -0.00004 | +0.00854 |
| Ankle | +0.10 | -0.00129 | +0.00026 | -0.00692 |

The paired hip and knee actions have clear immediate pitch-rate authority at this amplitude; paired ankle action has a smaller immediate response but a larger delayed rate effect by 0.40 s. The two signs are not perfectly symmetric, and a pulse's immediate response may fade or change sign after the action returns to zero. Thus it would be premature to assign a fixed “braking joint” using only the 0.10 s sample.

For individual-joint pulses, the left/right hip responses are similar in pitch rate but have opposite small roll effects. Single-knee pulses produce larger opposing roll effects (roughly `0.0017–0.0019 rad` at the 0.10 s sample), while symmetric knee-pair pulses largely cancel them. A sagittal-only score would miss this lateral coupling.

Across all 171 short trials, no episode terminated, no torque command clipped or reached 95% of its limit, the maximum applied torque fraction was about `0.23`, and the maximum measured foot-contact speed was about `0.0034 m/s`. These checks apply only to this small-pulse, 0.40 s window. They are **not** a safety or long-horizon recovery guarantee.

## Interpretation and next gate

The six-channel action interface can measurably alter pitch and angular-y rate around these early states. This establishes local *response*, not a recoverability boundary. It also explains why a large fixed knee pulse can be misleading: joint effects depend on sign, horizon, and coupling to the other joints and contacts.

The next controlled experiment should predeclare a small set of **multi-joint pulse-and-brake sequences** based on these measured responses, replay them from identical states, and assess rate reversal, re-entry into the standing envelope, sustained hold, torque, and foot contact. Only after a repeatable recovery region is demonstrated should its states and perturbations become a training distribution for recovery PPO. The existing PD and residual PPO baselines remain unchanged.

## Reproduction

From the repository root with its existing virtual environment:

```bash
PYTHONPATH=src:scripts python3 tests/test_joint_pulse_response.py
python3 scripts/evaluate_joint_pulse_response.py
```

The run summarized here is the local ignored output `code/day9/g1_balance/checkpoints/joint_pulse_response/joint_response_20260930T091217Z/`, containing `results.csv` and `results.json`. The script uses `evaluate_damped_rate_guard.load_states` for verified Phase 3 state loading; both diagnostic scripts are needed to reproduce this result.
