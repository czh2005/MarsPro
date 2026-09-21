"""Build the accession-first MarsPro 40k release without changing labels or splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing release directory: {output}")
    output.mkdir(parents=True)

    data = source / "stage1_prediction" / "data"
    provenance = data / "provenance"
    full = output / "full_training_tables"
    public = output / "public_accession_only"
    split_rows: dict[str, list[dict[str, str]]] = {}

    for split in ("train", "validation", "test"):
        fields, rows = read_csv(data / f"{split}.csv")
        if "sequence" not in fields or "sequence_id" not in fields:
            raise SystemExit(f"Unexpected split schema: {split}")
        split_rows[split] = rows
        copy_file(data / f"{split}.csv", full / f"{split}.csv")
        write_csv(public / f"{split}.csv", [f for f in fields if f != "sequence"], rows)

    side_fields, side_rows = read_csv(data / "metadata_sidecar_40k.csv")
    copy_file(data / "metadata_sidecar_40k.csv", full / "metadata_sidecar_40k.csv")
    write_csv(public / "metadata_sidecar_40k.csv", side_fields, side_rows)

    for name in ("field_dictionary.csv", "source_registry.csv", "source_inventory_40k.csv"):
        copy_file(provenance / name, output / "provenance" / name)
    for name in ("release_audit.json", "metadata_build_summary.json"):
        copy_file(provenance / name, output / "provenance" / name)
    copy_file(
        source / "docs" / "SEQUENCE_REDISTRIBUTION_AUDIT_20260922.md",
        output / "provenance" / "SEQUENCE_REDISTRIBUTION_AUDIT_20260922.md",
    )

    accessions = []
    for row in side_rows:
        raw = (row.get("legacy_uniprot_accessions") or "").strip()
        accessions.append({
            "sequence_id": row.get("sequence_id", ""),
            "sequence_sha256": row.get("sequence_sha256", ""),
            "sequence_length": row.get("sequence_length", ""),
            "legacy_uniprot_accessions": raw,
            "scientific_source_ids": row.get("scientific_source_ids", ""),
            "retrieval_status": "accession_recorded" if raw else "source_mapping_required",
        })
    write_csv(public / "sequence_retrieval_manifest.csv", list(accessions[0]), accessions)

    counts = {split: len(rows) for split, rows in split_rows.items()}
    counts["total"] = sum(counts.values())
    expected = {"train": 36000, "validation": 2000, "test": 2000, "total": 40000}
    if counts != expected:
        raise SystemExit(f"Unexpected split counts: {counts}")

    readme = f"""# MarsPro 40k accession-first release

Build date: 2026-09-22

This is a derived release of the frozen MarsPro Stage 1 dataset. Labels, splits,
ID50 clusters, and sequence hashes are copied from the existing checked dataset;
no rows were re-split or relabeled.

## Contents

- `full_training_tables/`: exact 36,000/2,000/2,000 tables, including sequence strings, for authorized internal training.
- `public_accession_only/`: the same rows without the raw `sequence` column, plus sequence hashes and retrieval fields.
- `provenance/`: field dictionary, source registry, source inventory, and build audits.
- `MANIFEST.sha256`: file integrity manifest.

The accession-only view is the default public handoff. The full sequence view is
not a blanket relicensing of third-party sequence records; use and redistribution
remain subject to the upstream database and source terms in the provenance audit.

## Fixed split

| split | rows |
|---|---:|
| train | {counts['train']} |
| validation | {counts['validation']} |
| test | {counts['test']} |
| total | {counts['total']} |

Six labels are `cold`, `desiccation`, `oxidative`, `perchlorate`, `radiation`, and
`salt`. Unknown values and their masks are retained; unknown is not rewritten as
a biological negative.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            manifest_rows.append({
                "path": path.relative_to(output).as_posix(),
                "bytes": str(path.stat().st_size),
                "sha256": sha256(path),
            })
    write_csv(output / "MANIFEST.sha256", ["path", "bytes", "sha256"], manifest_rows)
    print(json.dumps({"output": str(output), "counts": counts, "files": len(manifest_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
