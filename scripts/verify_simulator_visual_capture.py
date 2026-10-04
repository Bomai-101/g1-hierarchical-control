"""Check visual captures against existing physics evidence and encoded media."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for engine in ("isaac", "mujoco"):
        capture = args.run_root / f"visual_{engine}"
        for file, expected in json.loads((capture / "inputs.json").read_text()).items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Input changed: {file}")
    for actor in ("baseline1499", "candidate1999"):
        isaac = np.load(args.run_root / "visual_isaac" / f"{actor}.npz")
        previous = np.load(args.run_root / "isaac_posttraining" / f"{actor}_06.npz")["trace"][:401, 0]
        previous_root = np.c_[previous[:, 1:11], previous[:, 14:17]]
        if not np.array_equal(previous_root, isaac["root"]):
            raise RuntimeError("Isaac root state differs from existing evidence")
        if isaac["body_positions"].shape != (401, 44, 3):
            raise RuntimeError("Unexpected Isaac body sample count")
        for engine in ("isaac", "mujoco"):
            trace = np.load(args.run_root / f"visual_{engine}" / f"{actor}.npz")
            if not np.array_equal(trace["commands"], np.tile([1, 0, 0], (len(trace["commands"]), 1))):
                raise RuntimeError("Commands differ")
            if not np.allclose(np.diff(trace["time"]), .02, atol=1e-10):
                raise RuntimeError("Sampling interval differs")
        metrics = json.loads((args.run_root / "visual_mujoco" / f"{actor}.json").read_text())
        previous_metrics = json.loads((args.run_root / "mujoco_posttraining" / f"{actor}_06.json").read_text())["reference_metrics"]
        if actor == "candidate1999":
            excluded = {"duration_s", "survival_ratio"}
            for key, value in metrics.items():
                if key not in excluded and previous_metrics[key] != value:
                    raise RuntimeError(f"Candidate MuJoCo metric changed: {key}")
        media = {}
        for suffix in ("isaac", "mujoco", "comparison"):
            path = args.media / f"{actor}_{suffix}.mp4"
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate", "-show_entries", "format=duration", "-of", "json", str(path)]))
            stream = info["streams"][0]
            if stream["nb_frames"] != "200" or info["format"]["duration"] != "8.000000" or stream["r_frame_rate"] != "25/1":
                raise RuntimeError("Encoded timeline differs")
            media[suffix] = info
        report[actor] = dict(isaac_root_13d_exact_previous_401_samples=True, fixed_commands_pass=True,
            mujoco_candidate_metrics_exact_previous=actor == "candidate1999", mujoco_real_duration=metrics["elapsed_s"],
            encoded_media=media)
    (args.media / "verification.json").write_text(json.dumps(report, indent=2))
    print("PASS: fixed commands, recorded state provenance, previous evidence, six video timelines")


if __name__ == "__main__":
    main()
