#!/usr/bin/env python3
"""Validate folded backbones and create the ProteinMPNN challenge table."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


work = Path(sys.argv[1])
source = work / "inputs/fixed_test_selected_1000.csv"
with source.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

records = []
missing = []
for row in rows:
    pdb = work / row["structure_file"]
    if not pdb.is_file() or pdb.stat().st_size < 1000:
        missing.append(row["challenge_id"])
        continue
    row = dict(row)
    row["prediction_id"] = row["challenge_id"]
    row["pdb_sha256"] = hashlib.sha256(pdb.read_bytes()).hexdigest()
    records.append(row)

if missing:
    raise RuntimeError(f"Missing/invalid structures: {len(missing)}; first={missing[:10]}")
destination = work / "inputs/mpnn_test.csv"
with destination.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(work / "inputs/mpnn_input_qc.json").write_text(
    json.dumps(
        {
            "rows": len(records),
            "unique_sequence_ids": len({r["sequence_id"] for r in records}),
            "unique_pdb_sha256": len({r["pdb_sha256"] for r in records}),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "status": "PASS",
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"PASS: {len(records)} folded backbones")
