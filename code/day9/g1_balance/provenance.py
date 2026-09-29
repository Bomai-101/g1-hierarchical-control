"""Run-directory creation and reproducibility metadata helpers."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import experiment_config as config


SOURCE_FILES = (
    "experiment_config.py",
    "g1_env.py",
    "networks.py",
    "ppo.py",
    "buffer.py",
    "reset_randomization.py",
    "train_balance.py",
    "provenance.py",
    "evaluate_balance.py",
    "evaluate_robustness.py",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_run_directory():
    """Create one unique output directory without reusing an existing run."""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0

    while True:
        run_id = f"run_{timestamp}" if suffix == 0 else f"run_{timestamp}_{suffix:02d}"
        run_dir = config.RUNS_DIR / run_id

        try:
            run_dir.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return run_dir


def build_run_manifest(env, run_dir):
    """Capture source and environment identity before training begins."""
    base_dir = config.BASE_DIR
    source_hashes = {
        name: sha256_file(base_dir / name)
        for name in SOURCE_FILES
    }

    scene_path = getattr(env, "scene_path", None)

    return {
        "schema_version": 1,
        "run_id": Path(run_dir).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(Path(run_dir).resolve()),
        "source_sha256": source_hashes,
        "scene": {
            "path": str(scene_path) if scene_path else None,
            "sha256": sha256_file(scene_path) if scene_path else None,
        },
        "environment": {
            "obs_dim": env.OBS_DIM,
            "action_dim": env.ACTION_DIM,
            "policy_joint_indices": env.POLICY_JOINT_INDICES.tolist(),
            "action_scale": env.action_scale,
            "max_episode_steps": env.max_episode_steps,
            "decimation": env.decimation,
        },
    }
