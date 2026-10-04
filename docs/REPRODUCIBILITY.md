# Reproduction tiers and dependencies

## 1. Read the published evidence (no simulator required)

Start with the [technical showcase](../results/hierarchical_control/20261004/README.md), [324-case report](../results/skills_sprint/20261004/sequence_retest/README.md), per-case tables, source/weight/model hashes and independent verification summaries. The showcase's `SHA256SUMS` verifies its published files. Full raw traces and actor weights are not bundled.

```bash
cd results/hierarchical_control/20261004
sha256sum -c SHA256SUMS
```

The video is saved Isaac state replay displayed through MuJoCo geometry, with no MuJoCo physics integration. It is visualization evidence, not a second-engine control test.

## 2. Run contract and metric checks (Python + NumPy)

The existing contract suite can run without Isaac, CUDA or model files:

```bash
python scripts/check_core_protocols.py
```

It covers skill interfaces, task-message validation, passive monitoring, rejection of altered evidence, physical readiness, watchdog admission and causal metric boundaries. Run normally; its original tests include Python `assert` statements. Full independent trace verifiers separately use explicit checks and were also tested with `python -O`.

The local lightweight environment used Python 3.12 and NumPy 2.5.3. This is an observed environment, not a claim that every package/version combination has been validated. A fresh environment needs NumPy; pytest is not required by this runner.

## 3. Reconstruct retained raw evaluations (requires local artifacts)

The complete 27 retest batches are locally retained in:

```text
code/day9/g1_balance/checkpoints/skills_sprint/20261004/sequence_retest/evaluation/
```

They contain `plan.json`, input hashes, complete case results and first-episode trace arrays. Weight inputs and metadata must match the recorded SHA-256 values. The current verifier resolves the original absolute input paths and reference metadata; moving files requires an explicit path-remapping change followed by new verification. Historical manifests are retained unchanged rather than pretending this is a portable model bundle.

With the original local artifacts available, use a **new, non-existing output directory**:

```bash
python -O scripts/verify_skill_sequence_retest.py \
  --root code/day9/g1_balance/checkpoints/skills_sprint/20261004/sequence_retest/evaluation \
  --output /tmp/g1_sequence_reconstruction_new
```

The previously archived verifier reconstructed 318,455 action rows and 216 matched pre-request prefixes. Unit tests and raw reconstruction are distinct from running new physical simulations. Downloading this repository alone does not supply the weight/trace inputs.

## 4. Run a new simulator evaluation (external environment and actors required)

The recorded Isaac environment used Isaac Sim 5.1, Python 3.11, PyTorch 2.7/CUDA 12.8 and NumPy 1.26. The lightweight replay/render environment used MuJoCo 3.14 with the corresponding local MJB model. Version/environment evidence is in the experiment archives; do not mix the MJB with an unverified older engine.

Required external dependencies include the local IsaacLab installation, the [reference checkout](https://github.com/yezzzzye/g1_walk_isaaclab_mujoco), its G1 assets and metadata, and the locally trained walking/hold checkpoints plus actor exports. These assets and weights are not redistributed here.

`scripts/run_flat_isaac.sh` is a launcher for the author's installed environment, with absolute IsaacLab/Conda paths. It is **not a general installer**. Before using it elsewhere, adapt those paths to your own compatible installations. The evaluator additionally requires the reference checkout's `isaac_sim` package import path and metadata.

The prospective evaluator accepts explicit inputs:

```bash
# Replace every placeholder with an existing, compatible local path.
bash scripts/run_flat_isaac.sh scripts/evaluate_skill_sequence_retest_isaac.py \
  --headless \
  --reference-root /path/to/g1_walk_isaaclab_mujoco \
  --walk-checkpoint /path/to/walk/assets/model_1499.pt \
  --skill-checkpoint /path/to/hold/train_hold/model_499.pt \
  --num-envs 12 --seeds 101 202 303 \
  --output /path/to/new_evaluation_directory
```

The walking directory must also contain `policy_actor.npz`; the hold directory needs `export/policy_actor.npz`. The output directory must not exist. Do not overwrite the frozen archival run. `--smoke` produces an explicitly partial run and is not a replacement for 324-case replication.

## Limits and historical provenance

No new simulation or retraining was performed during publication cleanup. Baseline historical metrics sometimes lack complete training-time environment snapshots; present-day hashes cannot repair those gaps. The original 29DoF/64D standing and single-policy watchdog experiments remain historical, separate from current 123D/37D two-actor sequencing. Full fallback/hardware/perception robustness is not established.
