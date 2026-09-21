#!/usr/bin/env python3
"""Build a leakage-controlled scientific-source holdout fold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contains_source(value: str, source_id: str) -> bool:
    return source_id in {item.strip() for item in value.split("|") if item.strip()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    return {
        task: dict(Counter(row.get(task, "unknown") or "unknown" for row in rows))
        for task in TASKS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "stage1_prediction" / "data",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print fold counts without writing derived CSV files",
    )
    args = parser.parse_args()

    split_rows: list[dict[str, str]] = []
    split_fields: list[str] | None = None
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = args.data_dir / name
        rows = read_csv(path)
        if split_fields is None:
            split_fields = list(rows[0].keys())
        split_rows.extend(rows)
    assert split_fields is not None

    sidecar_rows = read_csv(args.data_dir / "metadata_sidecar_40k.csv")
    sidecar_by_id = {row["sequence_id"]: row for row in sidecar_rows}
    if set(sidecar_by_id) != {row["sequence_id"] for row in split_rows}:
        raise SystemExit("sidecar and dataset sequence_id sets differ")

    anchors = {
        row["sequence_id"]
        for row in sidecar_rows
        if contains_source(row.get("scientific_source_ids", ""), args.source_id)
    }
    if not anchors:
        raise SystemExit(f"source not found: {args.source_id}")

    anchor_clusters = {
        sidecar_by_id[sequence_id].get("id50_cluster_id", "")
        for sequence_id in anchors
        if sidecar_by_id[sequence_id].get("id50_cluster_id", "")
    }
    anchor_relations = {
        sidecar_by_id[sequence_id].get("relation_component_id", "")
        for sequence_id in anchors
        if sidecar_by_id[sequence_id].get("relation_component_id", "")
    }

    challenge_ids: set[str] = set()
    membership: list[dict[str, str]] = []
    for row in sidecar_rows:
        sequence_id = row["sequence_id"]
        reasons: list[str] = []
        if sequence_id in anchors:
            reasons.append("held_out_source")
        if row.get("id50_cluster_id", "") in anchor_clusters:
            reasons.append("id50_component")
        if row.get("relation_component_id", "") in anchor_relations:
            reasons.append("relation_component")
        if reasons:
            challenge_ids.add(sequence_id)
            membership.append(
                {
                    "sequence_id": sequence_id,
                    "original_split": row["split"],
                    "scientific_source_ids": row.get("scientific_source_ids", ""),
                    "id50_cluster_id": row.get("id50_cluster_id", ""),
                    "relation_component_id": row.get("relation_component_id", ""),
                    "holdout_reason": "|".join(reasons),
                }
            )

    by_id = {row["sequence_id"]: row for row in split_rows}
    challenge = [by_id[sequence_id] for sequence_id in sorted(challenge_ids)]
    train = [row for row in split_rows if row["split"] == "train" and row["sequence_id"] not in challenge_ids]
    validation = [row for row in split_rows if row["split"] == "validation" and row["sequence_id"] not in challenge_ids]

    summary = {
        "protocol": "scientific_source_holdout_v1_20260918",
        "held_out_source_id": args.source_id,
        "anchor_sequences": len(anchors),
        "expanded_challenge_sequences": len(challenge),
        "remaining_train_sequences": len(train),
        "remaining_validation_sequences": len(validation),
        "challenge_original_split_counts": dict(Counter(row["split"] for row in challenge)),
        "challenge_label_state_counts": task_counts(challenge),
        "anchor_id50_clusters": len(anchor_clusters),
        "anchor_relation_components": len(anchor_relations),
        "unknown_policy": "unknown cells remain masked and are not converted to zero",
    }
    if args.report_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    validation_path = args.output_dir / "validation.csv"
    challenge_path = args.output_dir / "challenge.csv"
    membership_path = args.output_dir / "challenge_membership.csv"
    write_csv(train_path, train, split_fields)
    write_csv(validation_path, validation, split_fields)
    write_csv(challenge_path, challenge, split_fields)
    write_csv(
        membership_path,
        membership,
        [
            "sequence_id",
            "original_split",
            "scientific_source_ids",
            "id50_cluster_id",
            "relation_component_id",
            "holdout_reason",
        ],
    )

    manifest = {
        **summary,
        "files": {
            path.name: {"rows": len(rows), "sha256": sha256(path)}
            for path, rows in (
                (train_path, train),
                (validation_path, validation),
                (challenge_path, challenge),
                (membership_path, membership),
            )
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
