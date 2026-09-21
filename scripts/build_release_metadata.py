#!/usr/bin/env python3
"""Build auditable release metadata from the packaged MarsPro tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_source_inventory(data_dir: Path) -> dict:
    registry_path = data_dir / "provenance" / "source_registry.csv"
    registry = {row["source_id"]: row for row in rows(registry_path)}
    counts: dict[str, Counter] = defaultdict(Counter)
    task_scopes: dict[str, set[str]] = defaultdict(set)
    title_values: dict[str, set[str]] = defaultdict(set)
    sidecar = data_dir / "metadata_sidecar_40k.csv"
    sidecar_rows = 0
    for row in rows(sidecar):
        sidecar_rows += 1
        ids = [value.strip() for value in row.get("scientific_source_ids", "").split("|") if value.strip()]
        titles = [value.strip() for value in row.get("scientific_source_titles", "").split("|") if value.strip()]
        scopes = [value.strip() for value in row.get("task_scopes", "").split("|") if value.strip()]
        for source_id in ids:
            counts[source_id]["rows"] += 1
            counts[source_id][f"split_{row['split']}"] += 1
            task_scopes[source_id].update(scopes)
            title_values[source_id].update(titles)

    inventory = []
    for source_id in sorted(counts):
        item = registry.get(source_id, {})
        license_status = item.get("license_public_redistribution_status", "").strip()
        inventory.append(
            {
                "source_id": source_id,
                "rows": counts[source_id]["rows"],
                "train_rows": counts[source_id]["split_train"],
                "validation_rows": counts[source_id]["split_validation"],
                "test_rows": counts[source_id]["split_test"],
                "top_label": item.get("top_label", ""),
                "task_scopes": "|".join(sorted(task_scopes[source_id])),
                "title": item.get("title", "") or "|".join(sorted(title_values[source_id])),
                "source_type": item.get("source_type", ""),
                "url": item.get("url", ""),
                "access_status": item.get("access_status", "") or "not_recorded",
                "validation_status": item.get("validation_status", "") or "not_recorded",
                "license_public_redistribution_status": license_status or "not_verified_for_redistribution",
                "source_label_meaning": item.get("source_label_meaning", "") or "source_or_collection_membership; not per-protein functional proof",
            }
        )
    fields = list(inventory[0])
    write_csv(data_dir / "provenance" / "source_inventory_40k.csv", inventory, fields)
    return {
        "sidecar_rows": sidecar_rows,
        "scientific_source_count": len(inventory),
        "sources_with_unverified_redistribution": sum(
            row["license_public_redistribution_status"] == "not_verified_for_redistribution"
            for row in inventory
        ),
    }


def build_dataset_audit(root: Path, source_summary: dict) -> dict:
    data_dir = root / "stage1_prediction" / "data"
    state_counts = {task: Counter() for task in TASKS}
    split_counts = Counter()
    cluster_split: dict[str, set[str]] = defaultdict(set)
    sequence_ids = set()
    total = 0
    for split_name in ("train", "validation", "test"):
        for row in rows(data_dir / f"{split_name}.csv"):
            total += 1
            split_counts[split_name] += 1
            sequence_ids.add(row["sequence_id"])
            cluster_split[row["id50_cluster_id"]].add(split_name)
            for task in TASKS:
                state_counts[task][row[task]] += 1
    sidecar_ids = {row["sequence_id"] for row in rows(data_dir / "metadata_sidecar_40k.csv")}
    cross_split_clusters = sorted(cluster for cluster, splits in cluster_split.items() if len(splits) > 1)
    audit = {
        "schema_version": "marslikepro_public_release_audit_v1_20260918",
        "rows": total,
        "unique_sequence_ids": len(sequence_ids),
        "split_counts": dict(split_counts),
        "label_state_counts": {task: dict(state_counts[task]) for task in TASKS},
        "known_label_cells": sum(
            counts.get("0", 0) + counts.get("1", 0) for counts in state_counts.values()
        ),
        "unknown_label_cells": sum(counts.get("unknown", 0) for counts in state_counts.values()),
        "unique_id50_clusters": len(cluster_split),
        "cross_split_id50_cluster_count": len(cross_split_clusters),
        "cross_split_id50_clusters": cross_split_clusters,
        "sidecar_rows": len(sidecar_ids),
        "sidecar_sequence_id_match": sidecar_ids == sequence_ids,
        **source_summary,
    }
    (data_dir / "provenance" / "release_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if total != 40000 or split_counts != Counter(train=36000, validation=2000, test=2000):
        raise SystemExit("unexpected Stage 1 row counts")
    if sidecar_ids != sequence_ids or cross_split_clusters:
        raise SystemExit("metadata join or ID50 split audit failed")
    return audit


def build_structure_index(source: Path, destination: Path) -> dict:
    keep = [
        "sequence_id", "sequence", "cold", "desiccation", "oxidative", "perchlorate",
        "radiation", "salt", "id50_cluster_id", "id50_cluster_size", "source_datasets",
        "prediction_id", "sequence_length", "prediction_status", "plddt_mean_0_1", "ptm",
        "prediction_seconds", "selection_batch", "split", "has_any_positive_label",
    ]
    output = []
    for row in rows(source):
        clean = {field: row.get(field, "") for field in keep}
        clean["structure_relative_path"] = f"structures/{row['prediction_id']}.pdb"
        clean["structure_in_repository"] = "false"
        output.append(clean)
    fields = keep + ["structure_relative_path", "structure_in_repository"]
    write_csv(destination, output, fields)
    return {
        "rows": len(output),
        "unique_sequence_ids": len({row["sequence_id"] for row in output}),
        "all_have_positive_label": all(row["has_any_positive_label"] == "1" for row in output),
        "source_sha256": sha256(source),
        "output_sha256": sha256(destination),
        "pdb_files_in_repository": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--structure-source", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "stage1_prediction" / "data"
    source_summary = build_source_inventory(data_dir)
    audit = build_dataset_audit(root, source_summary)
    payload = {"stage1": audit}
    if args.structure_source:
        destination = root / "stage2_design" / "structure_corpus" / "structure_corpus_index_10000.csv"
        payload["structure_corpus"] = build_structure_index(args.structure_source.resolve(), destination)
        (destination.parent / "structure_corpus_manifest.json").write_text(
            json.dumps(payload["structure_corpus"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
