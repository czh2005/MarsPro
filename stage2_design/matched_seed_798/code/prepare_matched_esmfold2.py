from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(method: str, input_csv: Path, workers: int, output: Path):
    root = output / method
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, root / "input.csv")
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2394:
        raise ValueError(f"{method}: expected 2394 rows, got {len(rows)}")
    keys = [(row["backbone_id"], row["replicate_seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{method}: duplicate backbone-seed keys")
    order = sorted(range(len(rows)), key=lambda i: (-len(rows[i]["sequence"]), i))
    shards = [[] for _ in range(workers)]
    loads = [0] * workers
    for index in order:
        target = min(range(workers), key=lambda i: loads[i])
        shards[target].append(index)
        loads[target] += len(rows[index]["sequence"])
    for idx, shard in enumerate(shards):
        (root / "jobs" / f"worker_{idx:02d}.json").write_text(json.dumps(shard))
    return {
        "method": method,
        "rows": len(rows),
        "workers": workers,
        "worker_rows": [len(shard) for shard in shards],
        "worker_residues": loads,
        "input_sha256": sha256(input_csv),
        "parameters": dict(num_loops=3, num_sampling_steps=200, num_diffusion_samples=1, seed=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--joint-csv", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--compat-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.worker_script, args.output / "esmfold2_matched_worker.py")
    shutil.copy2(args.compat_script, args.output / "batch_compat.py")
    manifest = {
        "base": prepare("base", args.base_csv, 7, args.output),
        "joint_lora": prepare("joint_lora", args.joint_csv, 8, args.output),
        "gpu_plan": {
            "ibsgpu02": {"method": "base", "gpu_indices": [0, 1, 3, 4, 5, 6, 7], "excluded": [2]},
            "ibsgpu03": {"method": "joint_lora", "gpu_indices": list(range(8)), "excluded": []},
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
