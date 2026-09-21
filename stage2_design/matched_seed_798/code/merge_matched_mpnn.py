from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/media/data/panting/novlight/Biodata/matched_seed_generation_20260918")
SEEDS = (42, 43, 44)
METHODS = ("base", "joint_lora")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


out = ROOT / "final_tables"
out.mkdir(parents=True, exist_ok=True)
all_rows = {method: [] for method in METHODS}

for method in METHODS:
    for seed in SEEDS:
        path = ROOT / "outputs" / f"seed{seed}" / f"{method}_seed{seed}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 798:
            raise ValueError(f"{path}: expected 798 rows, got {len(rows)}")
        all_rows[method].extend(rows)

summary = {}
for method, rows in all_rows.items():
    keys = [(row["backbone_id"], int(row["replicate_seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate backbone-seed key in {method}")
    expected = {(f"{i:04d}", seed) for i in range(1, 10000) for seed in ()}
    backbone_counts = Counter(row["backbone_id"] for row in rows)
    seed_counts = Counter(int(row["replicate_seed"]) for row in rows)
    invalid = [row["backbone_id"] for row in rows if set(row["sequence"]) - VALID_AA]
    length_bad = [row["backbone_id"] for row in rows if len(row["sequence"]) != int(row["generated_length"])]
    if invalid or length_bad or set(seed_counts) != set(SEEDS) or set(seed_counts.values()) != {798}:
        raise ValueError(f"QC failure for {method}: invalid={len(invalid)} length={len(length_bad)} seeds={seed_counts}")
    if set(backbone_counts.values()) != {3} or len(backbone_counts) != 798:
        raise ValueError(f"backbone coverage failure for {method}")

    csv_path = out / f"{method}_matched_seeds_42_43_44.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fasta_path = out / f"{method}_matched_seeds_42_43_44.fasta"
    fasta_path.write_text(
        "".join(
            f">{row['backbone_id']}|{method}|seed={row['replicate_seed']}|sample_seed={row['sample_seed']}\n{row['sequence']}\n"
            for row in rows
        ),
        encoding="ascii",
    )
    summary[method] = {
        "rows": len(rows),
        "unique_backbones": len(backbone_counts),
        "seed_counts": dict(sorted(seed_counts.items())),
        "unique_sequences": len({row["sequence"] for row in rows}),
        "csv_sha256": digest(csv_path),
        "fasta_sha256": digest(fasta_path),
    }

base_map = {(r["backbone_id"], r["replicate_seed"]): r for r in all_rows["base"]}
joint_map = {(r["backbone_id"], r["replicate_seed"]): r for r in all_rows["joint_lora"]}
if set(base_map) != set(joint_map):
    raise ValueError("Base and joint LoRA pairing keys differ")
length_mismatch = sum(
    int(base_map[key]["generated_length"]) != int(joint_map[key]["generated_length"])
    for key in base_map
)
if length_mismatch:
    raise ValueError(f"paired length mismatches: {length_mismatch}")
summary["paired_keys"] = len(base_map)
summary["paired_length_mismatches"] = length_mismatch
summary["seeds"] = list(SEEDS)
summary["temperature"] = 0.1
summary["qc_status"] = "PASS"
(out / "QC_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
