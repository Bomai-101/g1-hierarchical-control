# Day 10: second-correction probe and MuJoCo video

This is an exploratory, deterministic branch experiment, not a newly trained PPO policy. The Day 9 pose, PD gains, action scale, observation/action spaces, reward, termination, and PPO hyperparameters were not changed.

## Provenance

- Branch state: original PPO checkpoint trajectory at policy step 145, from `instability_trajectories/phase1_20260930T002805Z`.
- First intervention: the previously tested paired-knee pulse (+0.5 normalized action for 10 policy steps), then zero residual action / PD control.
- Second intervention candidates: fixed-time paired-knee pulses at local step 180, 200, or 220, or one-shot pitch-rate-triggered pulses. See `scripts/evaluate_early_correction.py` for the exact candidate definitions and checks.
- Center pilot: `correction_experiments/correction_20260930T015345Z`.
- Five-state comparison: `correction_experiments/correction_20260930T015410Z`, with the exact center plus small ±pitch and ±pitch-rate offsets. Results and state arrays are stored in that run directory, under ignored checkpoints.

## Outcome

The center-only best fixed second pulse, starting at local step 220 with normalized knee-pair amplitude +0.25 for 5 steps, lasted **373 policy steps after the branch**, versus **347** for the first-pulse→PD baseline. With the common 145-step PPO prefix, these trajectories total **518** and **492** steps from reset. The historical PPO-only trajectory lasted **241** steps from reset. These are trajectory lengths under different action sequences; they must not be described as PPO improvement.

None of the 17 candidates achieved an uninterrupted 10-second hold. None of the four neighboring branch states passed the 4-second criterion. The candidate ranked best across five states was a triggered knee pulse, but its center lasted only 334 post-branch steps, shorter than the 347-step first-pulse→PD center baseline. There was no torque clipping in the 85 full-sweep cases. The result is a narrow center-trajectory improvement, **not** demonstrated robust recovery.

## Video

`scripts/render_correction_comparison.py` uses saved qpos/qvel trajectories for offline side-by-side MuJoCo rendering; it does not re-run or alter the experiments. The generated MP4s live in the ignored five-state run directory:

- `historical_comparison.mp4`: left historical PPO (241 steps), right fixed second-pulse center trajectory (518 steps).
- `pulse_baseline_comparison.mp4` if rendered: left first-pulse→PD (492 steps), right fixed second-pulse center (518 steps).

After a shorter trajectory falls, its last frame is held while the longer trajectory continues. Both panes use the same original 145-step prefix and the same camera. The frame labels report steps from reset, not seconds or generalization performance.

## Next decision

Do not promote this to a recovery policy or widen perturbations based on center survival alone. Diagnose why tiny neighboring branch states fail, then define a feedback controller or training objective that improves the held-out neighbors and uninterrupted hold, with a fixed evaluation set.
