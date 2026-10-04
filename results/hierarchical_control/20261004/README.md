# Two-policy hierarchical control: bounded simulation demonstration

A rule-based supervisor coordinates a locally trained walking1499 actor and a separately fine-tuned hold499 actor, using compatible 123D observations / 37D actions, physical readiness gates and a task-message watchdog.

Start with the [two-page research brief](research_brief.pdf), [English report](research_brief.md), [synchronized three-scenario video](three_scenarios.mp4) and [11-stage evidence chain](evidence_chain.md).

![Architecture](architecture.png)

| Scenario | New cases | Task criterion passed |
|---|---:|---:|
| Normal walk → hold → walk |108|108|
| Temporary 1 s message interruption → stop → resume |108|108|
| Persistent interruption → bounded hold observation, no resume |108|108|

These 324 cases use fixed actors/thresholds, three new reset seeds, expanded handoff times and three yaw preludes. Scope: flat Isaac at a commanded 0.5 m/s; seeds vary root position/yaw. Parallel copies are not independent statistical trials. The persistent 12 s actor observation includes stopping. No disturbance/hardware/perception-dropout robustness is established.

The video preserves 436 frames / 25 fps / 17.44 s from fixed seed101, yaw0, env0 traces. It displays **recorded Isaac states using MuJoCo geometry; zero MuJoCo physics steps**. Explicit freeze markers preserve each scenario's observation endpoint. `source_render_manifest.json` records source trace hashes; `video_provenance.json` records the padded video origin. The public video is a byte-identical copy of that padded render, as recorded in `publication_manifest.json`.

Full [retest protocol and independent verification](../../skills_sprint/20261004/sequence_retest/README.md), [handoff ablations](../../skills_sprint/20261003/skill_sequence/README.md) and [claim-source table](claims.csv) remain available. Runtime source/weight/model hashes are retained in [frozen_demo.json](frozen_demo.json). Full raw traces/weights are local Git-ignored artifacts, not downloadable from this repository.

See [reproduction tiers and dependencies](../../../docs/REPRODUCIBILITY.md). No model assets, training weights, CV, transcript or application recordings are bundled. This is a research prototype, not an accepted MuJoCo baseline or general recovery controller.
