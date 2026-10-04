"""Verify matched flat rollouts and summarize a continuation candidate.

Requires completed Isaac and MuJoCo suites. Failed/truncated episodes are
reported as stability outcomes, never as successful 30 s tracking scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proofs, paired, groups, inputs = [], [], [], {}
    for backend in ("isaac", "mujoco"):
        run = args.run_root / f"{backend}_posttraining"
        previous = args.run_root / f"{backend}_flat_commands"
        complete = json.loads((run / "complete.json").read_text())
        if complete["cases"] != 24:
            raise RuntimeError("Incomplete evaluation")
        inputs[str(run / "complete.json")] = digest(run / "complete.json")
        current = rows(run / "summary.csv")
        prior = rows(previous / "summary.csv")
        inputs[str(previous / "summary.csv")] = digest(previous / "summary.csv")
        inputs[str(run / "summary.csv")] = digest(run / "summary.csv")
        if len(current) != 24 or len({(r["actor"], r["case_id"]) for r in current}) != 24:
            raise RuntimeError("Duplicate/missing cases")
        for row in current:
            actor, case_id = row["actor"], int(row["case_id"])
            if backend == "isaac":
                path = run / f"{actor}_{case_id:02d}.npz"
                arrays = np.load(path)
                t, data = arrays["time"], arrays["trace"]
                if data.shape[1:] != (16, 25) or not np.allclose(np.diff(t), .02):
                    raise RuntimeError("Invalid Isaac trace dimensions/times")
                if not np.allclose(data[0, :, 20], float(row["start_yaw"]), atol=1e-4):
                    raise RuntimeError("Wrong initial heading")
                w, x, y, z = (data[:, :, i] for i in (4, 5, 6, 7))
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                delta = np.arctan2(np.sin(yaw - data[:, :, 20]), np.cos(yaw - data[:, :, 20]))
                if np.max(np.abs(delta)) > 1e-5:
                    raise RuntimeError("Recorded heading differs from root quaternion")
                if row["mode"] == "heading":
                    error = np.arctan2(np.sin(float(row["target"]) - data[:, :, 20]),
                                       np.cos(float(row["target"]) - data[:, :, 20]))
                    expected = np.clip(.5 * error, -1, 1)
                else:
                    expected = float(row["target"])
                if not np.allclose(data[:, :, 24], expected, atol=1e-6):
                    raise RuntimeError("Wrong actor command")
                valid = data[:, :, 0].astype(bool)
                if np.any(np.diff(valid.astype(int), axis=0) > 0):
                    raise RuntimeError("Reset included in first-episode scoring")
                mask = valid & (t[:, None] >= 2)
                if not mask.any():
                    raise RuntimeError("No eligible tracking samples")
                forward = float(np.sqrt(np.mean((data[:, :, 11] - data[:, :, 22])[mask] ** 2)))
                yaw = float(np.sqrt(np.mean((data[:, :, 16] - data[:, :, 24])[mask] ** 2)))
                if not np.isclose(forward, float(row["forward_rmse"]), atol=1e-6) or not np.isclose(yaw, float(row["world_yaw_rmse"]), atol=1e-6):
                    raise RuntimeError("Recorded metrics do not match trace")
                is_full = int(row["completed"]) == 16 and int(row["falls"]) == 0 and int(row["timeouts"]) == 0 and len(t) == 1500
            else:
                path = run / f"{actor}_{case_id:02d}.csv"
                trace = rows(path)
                if abs(float(trace[0]["heading"]) - float(row["start_yaw"])) > 1e-6:
                    raise RuntimeError("Wrong MuJoCo initial heading")
                for sample in trace:
                    expected = np.clip(.5 * float(sample["heading_error"]), -1, 1) if row["mode"] == "heading" else float(row["target"])
                    if abs(float(sample["command_wz"]) - expected) > 1e-8:
                        raise RuntimeError("Wrong MuJoCo heading/rate command")
                is_full = int(row["fall"]) == 0 and float(row["elapsed_s"]) == 30
            inputs[str(path)] = digest(path)
            proof = dict(backend=backend, actor=actor, case_id=case_id,
                         trace_commands_verified=True, full_30s_task=is_full)
            if actor == "baseline1499":
                earlier = next(r for r in prior if r["actor"] == "local" and int(r["case_id"]) == case_id)
                proof["previous_metrics_exact"] = all(row[k] == earlier[k] for k in row if k != "actor")
                old_path = previous / f"local_{case_id:02d}{'.npz' if backend == 'isaac' else '.csv'}"
                inputs[str(old_path)] = digest(old_path)
                if backend == "isaac":
                    old_arrays = np.load(old_path)
                    proof["previous_trace_exact"] = all(np.array_equal(arrays[k], old_arrays[k]) for k in arrays.files)
                else:
                    proof["previous_trace_exact"] = path.read_bytes() == old_path.read_bytes()
                if not proof["previous_metrics_exact"] or not proof["previous_trace_exact"]:
                    raise RuntimeError("Old baseline does not replay exactly")
            proofs.append(proof)
        for case_id in range(12):
            old = next(r for r in current if r["actor"] == "baseline1499" and int(r["case_id"]) == case_id)
            new = next(r for r in current if r["actor"] == "candidate1999" and int(r["case_id"]) == case_id)
            old_proof = next(r for r in proofs if r["backend"] == backend and r["actor"] == "baseline1499" and r["case_id"] == case_id)
            new_proof = next(r for r in proofs if r["backend"] == backend and r["actor"] == "candidate1999" and r["case_id"] == case_id)
            both_full = old_proof["full_30s_task"] and new_proof["full_30s_task"]
            item = dict(backend=backend, case_id=case_id, mode=old["mode"], speed=float(old["speed"]), target=float(old["target"]),
                        baseline_full=old_proof["full_30s_task"], candidate_full=new_proof["full_30s_task"],
                        tracking_deltas_comparable=both_full,
                        baseline_falls=int(old.get("falls", old.get("fall"))), candidate_falls=int(new.get("falls", new.get("fall"))),
                        baseline_elapsed_s=float(old.get("elapsed_s", old.get("observed_s"))),
                        candidate_elapsed_s=float(new.get("elapsed_s", new.get("observed_s"))))
            for metric in ("forward_rmse", "world_yaw_rmse", "final_heading_error_abs"):
                item[f"baseline_{metric}"] = float(old[metric]) if old[metric] else None
                item[f"candidate_{metric}"] = float(new[metric]) if new[metric] else None
                item[f"delta_{metric}"] = item[f"candidate_{metric}"] - item[f"baseline_{metric}"] if both_full and old[metric] and new[metric] else None
            paired.append(item)
        for actor in ("baseline1499", "candidate1999"):
            selected = [r for r in current if r["actor"] == actor]
            full_ids = {p["case_id"] for p in proofs if p["backend"] == backend and p["actor"] == actor and p["full_30s_task"]}
            full = [r for r in selected if int(r["case_id"]) in full_ids]
            heads = [r for r in full if r["mode"] == "heading"]
            groups.append(dict(backend=backend, actor=actor, cases=len(selected), full_30s_cases=len(full),
                physical_termination_count=sum(int(r.get("falls", r.get("fall"))) for r in selected),
                forward_rmse_mean_full_cases=float(np.mean([float(r["forward_rmse"]) for r in full])) if full else None,
                world_yaw_rmse_mean_full_cases=float(np.mean([float(r["world_yaw_rmse"]) for r in full])) if full else None,
                heading_endpoint_error_mean_full_cases=float(np.mean([float(r["final_heading_error_abs"]) for r in heads])) if heads else None))
    with (args.output_dir / "comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    regressed = any(p["candidate_falls"] > p["baseline_falls"] or (p["baseline_full"] and not p["candidate_full"]) for p in paired)
    summary = dict(groups=groups, candidate_promoted=False,
                   decision="reject model1999 for flat sim2sim baseline: stability regression" if regressed else "no stability regression; acceptance and tracking review still required",
                   limitations="single training seed; fixed starts; parallel envs not independent trials; no safety claims",
                   proofs=proofs)
    (args.output_dir / "analysis.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "analysis_provenance.json").write_text(json.dumps(dict(inputs_sha256=inputs,
         script_sha256=digest(Path(__file__)), python=sys.version, numpy=np.__version__), indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
