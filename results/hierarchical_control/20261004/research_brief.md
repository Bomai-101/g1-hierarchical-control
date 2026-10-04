# Hierarchical G1 Control in Simulation

**Learned skills, state-based handoffs and task-message watchdogs**  
**Evidence reviewed: 4 October 2026, Australia/Sydney**

## Research question and implemented system

Can separately parameterized locomotion and stop/hold policies be coordinated through explicit interfaces, state-based handoff conditions and a task-message watchdog?

The prototype uses a rule-based supervisor above two separately parameterized PPO actors: a locally trained walking policy and a separately fine-tuned stop/hold policy. The supervisor selects the actor using task state, message freshness and measured motion. Both actors consume a 123-dimensional observation and produce 37-dimensional joint actions. State feedback gates return-to-walk after stopping. This is a hierarchical control architecture; the supervisor itself is not learned hierarchical reinforcement learning. The current system does not include high-level language reasoning, a perception pipeline or fallen-robot recovery.

## Contributions and attribution

- Reproduced the G1 training workflow from `yezzzzye/g1_walk_isaaclab_mujoco`: 4096 parallel environments, 1500 PPO updates and 147,456,000 transitions on an 8 GB RTX 4060 Laptop GPU. The local walking actor was trained from random weights. The reference supplies robot assets, task formulation, rewards and baseline PPO settings; no new locomotion algorithm is claimed.
- Implemented and evaluated simulator interfaces, passive monitoring, skill adapters, teacher-state handoff training, supervisor sequencing, watchdog logic and independent trajectory reconstruction in the local project.
- Preserved unsuccessful configurations and explained why they were not adopted. Additional PPO updates improved some Isaac metrics but caused 9/12 MuJoCo conditions to terminate. Fixed action blending/braking was weaker than direct switching in the tested low-speed sequence.

## Main quantitative results

The handoff comparison used fixed weights, seed 42 and 48 cases per protocol: direct switching 48/48 sequence passes, 0.5 s action blending 43/48, and 1.5 s command braking plus blending 37/48. Direct switching and all thresholds were fixed before the subsequent retest.

| New retest scenario | Environment cases | Requested task pass |
|---|---:|---:|
| Walk → stop/hold → return-to-walk |108|108|
| Temporary 1 s task-message loss → stop → resume after freshness recovery |108|108|
| Persistent message loss → 12 s hold-actor observation; no return-to-walk |108|108|

The retest used three new reset seeds 101/202/303, stop/drop onset bases 3.22/6.22/9.22 s, four staggered copies per onset, and initial yaw commands −0.2/0/+0.2 rad/s. All 324 cases were retained. Seed 42 cases were not pooled with this retest. Seeds vary initial world position and yaw; joint poses and initial velocities remain fixed. Copies within a seed are not independent statistical trials.

Median stop-decision-to-stopping-confirmation time was 1.84 s. Request-to-confirmation net displacement was 8.83 cm with normal requests and 20.64 cm with message loss, including watchdog delay; accumulated root path was 14.84/26.74 cm respectively. Post-confirmation hold net drift medians were 0.903/0.889/0.911 cm across the three scenarios. Detection delay was 0.24–0.30 s. Return-to-walk permission occurred 3.84 s after the stop decision; sustained speed-tracking confirmation followed actor resumption by 1.72 s. The persistent 12 s observation includes stopping, with a median 10.16 s confirmed-hold segment.

Independent optimized-Python verification reconstructed 318,455 first-episode action rows, checked 216 matched prefixes and verified observations, actor outputs, applied-action history, commands, gates, watchdog events, metrics and explicit observation horizons. Maximum actor reconstruction error was 3.34×10⁻⁶. This validates the recorded implementation/evidence, not general control robustness.

## Evaluation protocol and limits

Flat Isaac simulation at a commanded 0.5 m/s, physics 5 ms/control 20 ms, no observation noise or external forces. Readiness uses a causal 0.5 s window, mean horizontal speed≤0.05 m/s, Euler heading-rate RMS≤0.05 rad/s, tilt≤20° and both feet contact. It requires 1 s confirmation plus 2 s retained readiness. Holding is bounded at 12 s; resumed walking is observed for 10 s. These are exploratory protocol values, not safety-certified limits. Original standalone skill position/heading acceptance is separate and unchanged.

The watchdog observes simulated task-message freshness: 10 Hz heartbeat, 0.3 s TTL and two messages for freshness recovery. This is not an actual perception dropout. The simulated control period does not establish real-time compute deadline compliance. MuJoCo directional transfer and stationary skill performance remain unresolved; marching shows rhythmic motion but has not passed its original full in-place criteria. High-speed switching, disturbances, rough terrain, hardware validation and fallen-robot recovery are not established. Persistent observation termination is not a physical emergency-stop implementation.

## Research relevance and next questions

The work studies hierarchical layer interfaces, skill sequencing, timing analysis, watchdog handling and quantitative simulation-first evaluation. The next research step would connect genuine perception/task interfaces, profile compute timing and jitter, and calibrate recovery under a broader state and disturbance distribution before any supervised hardware work.

## Evidence and reproducibility

`evidence_chain.md` links the research stages, including negative results. `claims.csv` maps material claims to source reports. `frozen_demo.json` pins weights, model/interface/source hashes, protocol and reproduction arguments. This directory contains the architecture, comparison plots, labeled saved-state demo video and verified retest tables. Raw traces, training logs and weights remain in local Git-ignored storage and are not included in this compact brief. The video displays Isaac recorded states on MuJoCo geometry with zero physics steps; it is not a MuJoCo dynamics validation. Historical baseline metrics have incomplete training-time provenance; later hashes do not repair that gap.
