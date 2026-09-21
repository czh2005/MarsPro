#!/usr/bin/env python3
"""Resumable sharded ESMFold inference for fixed-test challenge backbones."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import esm


AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def validate_pdb(text: str, expected: str) -> tuple[float, str]:
    seen = []
    b_factors = []
    last = None
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        key = (line[21:22], line[22:27].strip())
        if key == last:
            continue
        last = key
        seen.append(AA3.get(line[17:20].strip(), "X"))
        try:
            b_factors.append(float(line[60:66]))
        except ValueError:
            pass
    recovered = "".join(seen)
    if recovered != expected:
        raise ValueError(f"PDB sequence mismatch: expected {len(expected)}, recovered {len(recovered)}")
    if not b_factors or not np.isfinite(b_factors).all():
        raise ValueError("Missing or invalid PDB confidence values")
    return float(np.mean(b_factors)), recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(2)

    with (args.work / "inputs/fixed_test_selected_1000.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))[args.shard :: args.shards]
    structures = args.work / "structures"
    status_dir = args.work / "fold_status"
    structures.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    journal = status_dir / f"shard_{args.shard}.jsonl"
    summary_path = status_dir / f"shard_{args.shard}_summary.json"

    model = esm.pretrained.esmfold_v1().eval().cuda().requires_grad_(False)
    model.set_chunk_size(64)
    completed = 0
    failed = 0
    started = time.time()
    with journal.open("a", encoding="utf-8", buffering=1) as log:
        for offset, row in enumerate(rows, 1):
            sid = row["challenge_id"]
            seq = row["sequence"]
            destination = structures / f"{sid}.pdb"
            if destination.exists():
                try:
                    validate_pdb(destination.read_text(), seq)
                    completed += 1
                    continue
                except Exception:
                    destination.rename(destination.with_suffix(".pdb.invalid"))
            record = {
                "challenge_id": sid,
                "sequence_id": row["sequence_id"],
                "length": len(seq),
                "shard": args.shard,
                "started_at": time.time(),
            }
            try:
                output = None
                last_error = None
                for chunk_size in (64, 32, 16):
                    try:
                        model.set_chunk_size(chunk_size)
                        with torch.inference_mode():
                            output = model.infer_pdb(seq)
                        break
                    except torch.cuda.OutOfMemoryError as exc:
                        last_error = repr(exc)
                        torch.cuda.empty_cache()
                if output is None:
                    raise RuntimeError(last_error or "ESMFold returned no output")
                mean_plddt, _ = validate_pdb(output, seq)
                tmp = destination.with_suffix(f".pdb.tmp.{os.getpid()}")
                tmp.write_text(output)
                tmp.replace(destination)
                record.update(
                    status="complete",
                    mean_plddt=mean_plddt,
                    pdb_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    elapsed_seconds=time.time() - record["started_at"],
                )
                completed += 1
            except Exception as exc:
                record.update(
                    status="failed",
                    error=repr(exc),
                    elapsed_seconds=time.time() - record["started_at"],
                )
                failed += 1
            log.write(json.dumps(record) + "\n")
            atomic_text(
                summary_path,
                json.dumps(
                    {
                        "state": "running",
                        "shard": args.shard,
                        "completed": completed,
                        "failed": failed,
                        "assigned": len(rows),
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                ),
            )
            torch.cuda.empty_cache()
            print(f"[{args.shard}] {offset}/{len(rows)} {sid} {record['status']}", flush=True)

    atomic_text(
        summary_path,
        json.dumps(
            {
                "state": "complete" if failed == 0 else "needs_review",
                "shard": args.shard,
                "completed": completed,
                "failed": failed,
                "assigned": len(rows),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
