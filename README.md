# Hierarchical G1 Control in Simulation

A simulation-first research prototype combining learned humanoid skills, explicit skill handoffs and task-message watchdogs. A rule-based supervisor coordinates separately parameterized walking and stop/hold PPO actors through a 123D observation / 37D action interface.

**Start here:** [Research brief (PDF)](results/hierarchical_control/20261004/research_brief.pdf) · [Three-scenario demonstration](results/hierarchical_control/20261004/three_scenarios.mp4) · [Research decisions and evidence](results/hierarchical_control/20261004/evidence_chain.md)

![Two-policy architecture](results/hierarchical_control/20261004/architecture.png)

## Results and scope

| Experiment | Result | Scope |
|---|---|---|
| Walking reproduction |4,096 environments; 1,500 PPO updates; 147,456,000 transitions|Local actor trained from random weights using an attributed reference workflow|
| Fixed-policy handoff comparison |Direct 48/48; blending 43/48; braking + blending 37/48|Prior seed42 low-speed sequence protocol; direct chosen before retest|
| New sequence/watchdog retest |Normal 108/108; temporary loss 108/108; persistent loss 108/108|324 retained cases; flat Isaac, 0.5 m/s, root-pose reset variation|
| Independent evidence verification |318,455 first-episode actions; 216 exact matched prefixes|Recorded observation/action/gate/event reconstruction, distinct from robustness|

The retest varies three reset seeds, entry times and yaw preludes. Parallel copies within a seed are not independent statistical trials. The persistent 12 s holding-actor observation includes stopping; the median confirmed-hold segment is 10.16 s. Median stopping confirmation takes 1.84 s after the stop decision.

The synchronized video is **Isaac saved-state replay rendered on MuJoCo geometry, with zero MuJoCo physics steps**. It is not MuJoCo dynamics validation. Faults interrupt simulated task messages; robot-state feedback remains available. The supervisor is rule-based, not a learned hierarchical RL manager.

MuJoCo directional/stationary transfer, marching acceptance, real perception faults, high-speed/rough/disturbed operation and fallen-robot recovery remain open. These results are not hardware safety guarantees or safety-certified thresholds.

## Research progression

1. **Reproduce:** train a local walking actor using the reference task and PPO configuration.
2. **Diagnose:** examine simulator interfaces, heading drift and contact response; preserve unsuccessful variants.
3. **Train for incoming states:** fine-tune a separate hold actor on states collected from the frozen walking actor.
4. **Integrate:** test state-based handoffs and compare direct switching, blending and braking.
5. **Retest and freeze:** retain fixed actors/thresholds, evaluate new reset/entry-time cases and independently reconstruct evidence.

[Detailed research chain](results/hierarchical_control/20261004/evidence_chain.md) · [Negative results and limitations](results/hierarchical_control/20261004/research_brief.md) · [Frozen configuration](results/hierarchical_control/20261004/frozen_demo.json)

## Code and reproducibility

- [Reproduction tiers and environment prerequisites](docs/REPRODUCIBILITY.md)
- [Source map](src/g1_control/README.md): reusable interfaces, monitoring, metrics and historical learning components.
- [Sequence protocol](scripts/skill_sequence_protocol.py) and [watchdog protocol](scripts/skill_watchdog_protocol.py): current bounded two-policy experiment.
- [Isaac evaluator](scripts/evaluate_skill_sequence_retest_isaac.py), [independent verifier](scripts/verify_skill_sequence_retest.py) and [published retest](results/skills_sprint/20261004/sequence_retest/README.md).
- [Passive shadow monitor](results/locomotion_shadow/README.md): monitor-off/on trajectory equivalence and detection limitations.

Lightweight contract checks require Python and NumPy, not Isaac or model weights:

```bash
python scripts/check_core_protocols.py
```

Full simulator replay requires the external Isaac/reference installations and locally retained weights/raw traces. The repository publishes reports, source, configuration and checksums; it does not bundle those dependencies. See the reproduction guide before running simulator commands.

## Repository layout

| Directory | Role |
|---|---|
| `src/g1_control/` |Reusable control interfaces, monitoring, metrics and historical PPO components|
| `scripts/` |Training/evaluation/diagnostics and evidence verification|
| `tests/` |Contract and evidence tests|
| `results/hierarchical_control/` |Latest technical brief, architecture and synchronized demo|
| `results/skills_sprint/` |Skill training, negative ablations, sequence/watchdog and retest evidence|
| `results/flat_baseline/`, `results/locomotion_shadow/` |Sim2Sim and passive-monitor diagnostics|
| `notes/`, `docs/HISTORY.md`, `code/day*/` |Research notes and preserved historical standing/PD experiments|

Earlier 29DoF / 64D standing experiments and the single-policy watchdog demo are **historical investigations**, separate from the current 123D/37D sequence. They are retained for research provenance, not registered as validated fallback controllers. [Historical overview](docs/HISTORY.md) · [Evidence index](notes/research_evidence_index.md).

## Attribution

The walking foundation uses [yezzzzye/g1_walk_isaaclab_mujoco](https://github.com/yezzzzye/g1_walk_isaaclab_mujoco): robot assets, task/reward formulation and baseline PPO settings. Local contributions include training/evaluation, transfer diagnostics, handoff-state fine-tuning, explicit supervision and independent evidence verification. No new locomotion algorithm or general superiority over the reference is claimed.

External models/checkpoints remain in separate local storage. See [third-party notices](THIRD_PARTY_NOTICES.md). No new license grant is introduced by this repository cleanup.
