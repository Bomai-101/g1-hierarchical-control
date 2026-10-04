# Manual publication after local cleanup

The repository was prepared locally; **no commit or push was performed**. Existing experiment data and uncommitted work are preserved. Read the [current overview](../README.md), [reproduction guide](REPRODUCIBILITY.md) and [publication checks](PUBLICATION_CHECKS.json).

## Included

Current control/monitoring interfaces; evaluation/verification scripts and tests; reports, source snapshots, experiment configuration and hashes; latest neutral technical brief and synchronized video; negative results and the preserved historical overview.

## Kept local

`results/vri_preparation/` and its application-only helper scripts are ignored. They retain scripts/subtitles/CV wording locally and are not part of the public technical showcase. Actual CV/transcript files are outside this repository. Full raw traces, checkpoints, MJB assets and training logs stay in the existing ignored checkpoint directory. Private data is not made public by the cleanup.

Documentation was made repository-relative where appropriate. Runtime `evidence/` snapshots remain byte-identical to their recorded hashes; they can contain old machine paths or relative links that are meaningful only in the original run. Use current report entry points for navigation. The retest README's editorial revision is recorded separately; no experiment metrics or runtime source hashes were rewritten.

## Review and publish manually

From the repository root:

```bash
python scripts/check_core_protocols.py
python scripts/check_publication.py

git status --short
git diff --check
git diff -- README.md .gitignore src/g1_control/README.md

# Stage the reviewed public changes; ignored application/raw artifacts stay local.
git add -- README.md .gitignore THIRD_PARTY_NOTICES.md docs/ src/ scripts/ tests/ notes/ results/
git diff --cached --stat
git diff --cached --check

# Run only after reviewing the staged file list.
git commit -m "Present and verify two-policy G1 hierarchical control experiments"
git push origin HEAD
```

The last commands are instructions for the owner, not actions performed by the assistant. Re-run publication checks if files change before staging. No force push is needed.

After pushing, open the repository without relying on local file access. Confirm that the README, PDF, video download and evidence reports are accessible. Full weights/raw replay inputs are not public, so do not describe a clone as a self-contained training or replay bundle. No license for third-party assets is newly granted.
