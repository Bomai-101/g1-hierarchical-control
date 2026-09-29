"""Sweep every saved PPO update over the fixed near-zero robustness cases.

This is an evaluation-only diagnostic. It never changes training state or
overwrites checkpoints. Results are written to a unique directory beneath
the selected run's ``evaluations`` directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import torch

import experiment_config as config
from evaluate_local_robustness import episode
from networks import Actor


CASES = [
    ("nominal", 0.0, 0.0),
    ("pitch", -0.005, 0.0),
    ("pitch", -0.002, 0.0),
    ("pitch", -0.001, 0.0),
    ("pitch", +0.001, 0.0),
    ("pitch", +0.002, 0.0),
    ("pitch", +0.005, 0.0),
    ("pitch_rate", 0.0, -0.050),
    ("pitch_rate", 0.0, -0.020),
    ("pitch_rate", 0.0, -0.010),
    ("pitch_rate", 0.0, +0.010),
    ("pitch_rate", 0.0, +0.020),
    ("pitch_rate", 0.0, +0.050),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_episode(result: dict) -> dict:
    return {
        "length": int(result["length"]),
        "return": float(result["return"]),
        "action_rms": float(result["action_rms"]),
        "action_max_abs": float(result["action_max_abs"]),
    }


def checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected checkpoint name: {path.name}") from exc


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def pareto_front(rows: list[dict]) -> list[dict]:
    """Return rows not dominated on nominal length and nonzero mean delta."""
    front = []
    for candidate in rows:
        candidate_nominal = candidate["nominal"]["ppo"]["length"]
        candidate_robust = candidate["nonzero_summary"]["mean_delta_steps"]
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_nominal = other["nominal"]["ppo"]["length"]
            other_robust = other["nonzero_summary"]["mean_delta_steps"]
            if (
                other_nominal >= candidate_nominal
                and other_robust >= candidate_robust
                and (
                    other_nominal > candidate_nominal
                    or other_robust > candidate_robust
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda row: (
            -row["nonzero_summary"]["mean_delta_steps"],
            -row["nominal"]["ppo"]["length"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Run directory containing update_*.pt checkpoints.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N checkpoints (for a smoke test).",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)

    checkpoints = sorted(
        checkpoint_dir.glob("update_*.pt"),
        key=checkpoint_sort_key,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise FileNotFoundError(
            f"No update_*.pt checkpoints found in {checkpoint_dir}"
        )

    device = choose_device(args.device)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Evaluation device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Cases per checkpoint: {len(CASES)} (1 nominal + 12 nonzero)")
    print()

    # The environment and evaluation policy are deterministic, so each zero
    # baseline case only needs to be evaluated once for the entire sweep.
    print("Computing paired zero-policy baselines...")
    zero_cases = {}
    for index, (kind, pitch, rate) in enumerate(CASES):
        full = episode(None, device, pitch, rate)
        zero_cases[index] = {
            "initial_state": full["initial_state"],
            "metrics": compact_episode(full),
        }

    rows = []
    for position, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if checkpoint.get("obs_dim", config.OBS_DIM) != config.OBS_DIM:
            raise ValueError(f"obs_dim mismatch in {checkpoint_path}")
        if checkpoint.get("action_dim", config.ACTION_DIM) != config.ACTION_DIM:
            raise ValueError(f"action_dim mismatch in {checkpoint_path}")

        actor = Actor(config.OBS_DIM, config.ACTION_DIM).to(device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        case_rows = []
        for index, (kind, pitch, rate) in enumerate(CASES):
            full = episode(actor, device, pitch, rate)
            if full["initial_state"] != zero_cases[index]["initial_state"]:
                raise AssertionError(
                    f"Unpaired initial state for {checkpoint_path.name}, case {index}"
                )
            ppo = compact_episode(full)
            zero = zero_cases[index]["metrics"]
            case_rows.append(
                {
                    "case_index": index,
                    "kind": kind,
                    "pitch_delta": pitch,
                    "pitch_rate_delta": rate,
                    "zero": zero,
                    "ppo": ppo,
                    "delta_steps": ppo["length"] - zero["length"],
                    "delta_return": ppo["return"] - zero["return"],
                }
            )

        nominal = case_rows[0]
        nonzero = case_rows[1:]
        deltas = [case["delta_steps"] for case in nonzero]
        row = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "update": int(checkpoint.get("update", checkpoint_sort_key(checkpoint_path))),
            "total_env_steps": int(checkpoint.get("total_env_steps", 0)),
            "saved_training_mean_length": float(
                checkpoint.get("mean_episode_length", 0.0)
            ),
            "saved_training_mean_return": float(
                checkpoint.get("mean_episode_return", 0.0)
            ),
            "nominal": nominal,
            "nonzero_summary": {
                "cases": len(nonzero),
                "wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta < 0 for delta in deltas),
                "mean_zero_length": float(
                    np.mean([case["zero"]["length"] for case in nonzero])
                ),
                "mean_ppo_length": float(
                    np.mean([case["ppo"]["length"] for case in nonzero])
                ),
                "mean_delta_steps": float(np.mean(deltas)),
                "worst_delta_steps": int(min(deltas)),
                "best_delta_steps": int(max(deltas)),
                "mean_action_rms": float(
                    np.mean([case["ppo"]["action_rms"] for case in nonzero])
                ),
            },
            "cases": case_rows,
        }
        rows.append(row)

        summary = row["nonzero_summary"]
        print(
            f"[{position:02d}/{len(checkpoints):02d}] "
            f"update={row['update']:03d} "
            f"nominal={nominal['ppo']['length']:3d} "
            f"nominal_delta={nominal['delta_steps']:+4d} "
            f"nonzero_mean_delta={summary['mean_delta_steps']:+6.2f} "
            f"worst={summary['worst_delta_steps']:+3d} "
            f"W/T/L={summary['wins']}/{summary['ties']}/{summary['losses']}",
            flush=True,
        )

    best_nominal = max(
        rows,
        key=lambda row: row["nominal"]["ppo"]["length"],
    )
    best_mean_robustness = max(
        rows,
        key=lambda row: (
            row["nonzero_summary"]["mean_delta_steps"],
            row["nonzero_summary"]["worst_delta_steps"],
            row["nominal"]["ppo"]["length"],
        ),
    )
    best_worst_case = max(
        rows,
        key=lambda row: (
            row["nonzero_summary"]["worst_delta_steps"],
            row["nonzero_summary"]["mean_delta_steps"],
            row["nominal"]["ppo"]["length"],
        ),
    )
    front = pareto_front(rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        checkpoint_dir
        / "evaluations"
        / f"checkpoint_robustness_sweep_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_dir = Path(__file__).resolve().parent
    source_names = [
        Path(__file__).name,
        "evaluate_local_robustness.py",
        "g1_env.py",
        "networks.py",
        "experiment_config.py",
    ]
    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "seed": config.SEED,
        "protocol": (
            "deterministic actor mean; relative base-local pitch rotation and "
            "additive qvel[4] after reset; fixed 1 nominal + 12 near-zero cases"
        ),
        "source_sha256": {
            name: sha256(source_dir / name) for name in source_names
        },
        "selection": {
            "best_nominal": best_nominal["checkpoint"],
            "best_nonzero_mean_delta": best_mean_robustness["checkpoint"],
            "best_nonzero_worst_case": best_worst_case["checkpoint"],
            "pareto_front": [row["checkpoint"] for row in front],
        },
        "results": rows,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)

    csv_path = output_dir / "checkpoint_summary.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "checkpoint",
                "update",
                "total_env_steps",
                "saved_training_mean_length",
                "nominal_zero_length",
                "nominal_ppo_length",
                "nominal_delta_steps",
                "nonzero_mean_zero_length",
                "nonzero_mean_ppo_length",
                "nonzero_mean_delta_steps",
                "nonzero_worst_delta_steps",
                "nonzero_best_delta_steps",
                "wins",
                "ties",
                "losses",
                "nonzero_mean_action_rms",
            ],
        )
        writer.writeheader()
        for row in rows:
            nominal = row["nominal"]
            robust = row["nonzero_summary"]
            writer.writerow(
                {
                    "checkpoint": row["checkpoint"],
                    "update": row["update"],
                    "total_env_steps": row["total_env_steps"],
                    "saved_training_mean_length": row[
                        "saved_training_mean_length"
                    ],
                    "nominal_zero_length": nominal["zero"]["length"],
                    "nominal_ppo_length": nominal["ppo"]["length"],
                    "nominal_delta_steps": nominal["delta_steps"],
                    "nonzero_mean_zero_length": robust["mean_zero_length"],
                    "nonzero_mean_ppo_length": robust["mean_ppo_length"],
                    "nonzero_mean_delta_steps": robust["mean_delta_steps"],
                    "nonzero_worst_delta_steps": robust[
                        "worst_delta_steps"
                    ],
                    "nonzero_best_delta_steps": robust["best_delta_steps"],
                    "wins": robust["wins"],
                    "ties": robust["ties"],
                    "losses": robust["losses"],
                    "nonzero_mean_action_rms": robust["mean_action_rms"],
                }
            )

    print()
    print("=== SELECTION SUMMARY ===")
    print(f"Best nominal:             {best_nominal['checkpoint']}")
    print(
        "Best nonzero mean delta:  "
        f"{best_mean_robustness['checkpoint']} "
        f"({best_mean_robustness['nonzero_summary']['mean_delta_steps']:+.2f})"
    )
    print(
        "Best nonzero worst case:  "
        f"{best_worst_case['checkpoint']} "
        f"({best_worst_case['nonzero_summary']['worst_delta_steps']:+d})"
    )
    print(
        "Pareto front:             "
        + ", ".join(row["checkpoint"] for row in front)
    )
    print(f"Saved JSON: {summary_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
