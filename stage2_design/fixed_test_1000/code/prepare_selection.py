#!/usr/bin/env python3
"""Freeze a non-overlapping 1,000-sequence challenge from the fixed test split."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / "AAA数据集" / "AAAA并集来源0_v3_20260910"
META = ROOT / "AAA数据集" / "第二阶段来源区片" / "数据。"
if not META.exists():
    META = ROOT / "AAA数据集" / "第二阶段来源区片的数据。"
SIDECAR = META / "MarsLikePro_4万条元数据侧表_v1_20260909.csv"
MPNN = ROOT / "mpnn数据集" / "train_with_esmfold2.csv"
TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
AA = set("ACDEFGHIKLMNPQRSTVWY")
SELECTION_SEED = "fixed-test-redesign-20260918"
TARGET = 1000
MAX_LENGTH = 500


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seq_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def main() -> None:
    inputs = OUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, str]]] = {
        split: read(DATA / f"{split}.csv") for split in ("train", "validation", "test")
    }
    sidecar_rows = read(SIDECAR)
    meta = {row["sequence_id"]: row for row in sidecar_rows}
    assert len(meta) == 40000
    mpnn_rows = read(MPNN)

    trainval_ids = {r["sequence_id"] for split in ("train", "validation") for r in split_rows[split]}
    trainval_hashes = {seq_hash(r["sequence"]) for split in ("train", "validation") for r in split_rows[split]}
    trainval_clusters = {r["id50_cluster_id"] for split in ("train", "validation") for r in split_rows[split]}
    trainval_relations = {
        meta[r["sequence_id"]]["relation_component_id"]
        for split in ("train", "validation")
        for r in split_rows[split]
        if meta[r["sequence_id"]]["relation_component_id"]
    }
    mpnn_ids = {r["sequence_id"] for r in mpnn_rows}
    mpnn_hashes = {seq_hash(r["sequence"]) for r in mpnn_rows}
    mpnn_clusters = {r["id50_cluster_id"] for r in mpnn_rows}

    audit: list[dict[str, object]] = []
    eligible: list[dict[str, str]] = []
    for row in split_rows["test"]:
        m = meta[row["sequence_id"]]
        sequence = row["sequence"].strip().upper()
        reasons: list[str] = []
        if not any(row[t] == "1" for t in TASKS):
            reasons.append("no_positive_label")
        if not sequence or set(sequence) - AA:
            reasons.append("noncanonical_sequence")
        if len(sequence) > MAX_LENGTH:
            reasons.append("length_above_500")
        if row["sequence_id"] in trainval_ids or seq_hash(sequence) in trainval_hashes:
            reasons.append("stage1_trainval_exact_overlap")
        if row["id50_cluster_id"] in trainval_clusters:
            reasons.append("stage1_trainval_id50_overlap")
        relation = m["relation_component_id"]
        if relation and relation in trainval_relations:
            reasons.append("stage1_trainval_relation_overlap")
        if row["sequence_id"] in mpnn_ids or seq_hash(sequence) in mpnn_hashes:
            reasons.append("mpnn_10k_exact_overlap")
        if row["id50_cluster_id"] in mpnn_clusters:
            reasons.append("mpnn_10k_id50_overlap")
        audit.append(
            {
                "sequence_id": row["sequence_id"],
                "length": len(sequence),
                "positive_label_count": sum(row[t] == "1" for t in TASKS),
                "eligible": int(not reasons),
                "exclusion_reasons": ";".join(reasons),
                "id50_cluster_id": row["id50_cluster_id"],
                "relation_component_id": relation,
            }
        )
        if not reasons:
            eligible.append(row)

    if len(eligible) < TARGET:
        raise RuntimeError(f"Only {len(eligible)} eligible sequences for target {TARGET}")
    eligible.sort(
        key=lambda r: hashlib.sha256(f"{SELECTION_SEED}|{r['sequence_id']}".encode()).hexdigest()
    )
    chosen = eligible[:TARGET]
    selected: list[dict[str, object]] = []
    for index, row in enumerate(chosen, 1):
        m = meta[row["sequence_id"]]
        selected.append(
            {
                "challenge_id": f"FT{index:04d}",
                "sequence_id": row["sequence_id"],
                "split": "test",
                "sequence": row["sequence"],
                "sequence_sha256": seq_hash(row["sequence"]),
                "sequence_length": len(row["sequence"]),
                "id50_cluster_id": row["id50_cluster_id"],
                "relation_component_id": m["relation_component_id"],
                "scientific_source_ids": m["scientific_source_ids"],
                "organism_names": m["organism_names"],
                "genus_names": m["genus_names"],
                "primary_pfam_accession": m["primary_pfam_accession"],
                **{task: row[task] for task in TASKS},
                "positive_label_count": sum(row[t] == "1" for t in TASKS),
                "structure_file": f"structures/FT{index:04d}.pdb",
                "selection_rank": index,
            }
        )

    write(inputs / "candidate_audit.csv", audit)
    write(inputs / "fixed_test_selected_1000.csv", selected)
    with (inputs / "fixed_test_selected_1000.fasta").open("w", encoding="ascii") as handle:
        for row in selected:
            handle.write(f">{row['challenge_id']}|{row['sequence_id']}\n{row['sequence']}\n")

    label_counts = {task: sum(row[task] == "1" for row in selected) for task in TASKS}
    length_bins = Counter(
        "1-100" if int(r["sequence_length"]) <= 100
        else "101-200" if int(r["sequence_length"]) <= 200
        else "201-300" if int(r["sequence_length"]) <= 300
        else "301-400" if int(r["sequence_length"]) <= 400
        else "401-500"
        for r in selected
    )
    protocol = {
        "protocol_version": "fixed_test_unseen_redesign_v1_20260918",
        "selection_seed": SELECTION_SEED,
        "source_split": "fixed test only",
        "target_rows": TARGET,
        "eligibility": [
            "at least one source-level positive label",
            "canonical amino-acid sequence",
            "length <= 500 aa",
            "no exact, registered ID50, or registered relation-component overlap with stage1 train/validation",
            "no exact or registered ID50 overlap with the existing 10k MPNN structure corpus",
        ],
        "selection": "ascending SHA256(selection_seed + '|' + sequence_id); prediction-blind",
        "candidate_test_rows": len(split_rows["test"]),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "label_positive_counts": label_counts,
        "length_bin_counts": dict(sorted(length_bins.items())),
        "selected_unique_ids": len({r["sequence_id"] for r in selected}),
        "selected_unique_sequences": len({r["sequence_sha256"] for r in selected}),
        "selected_unique_id50_clusters": len({r["id50_cluster_id"] for r in selected}),
        "selected_relation_components_known": len({r["relation_component_id"] for r in selected if r["relation_component_id"]}),
        "input_sha256": {
            split: file_hash(DATA / f"{split}.csv") for split in ("train", "validation", "test")
        },
        "sidecar_sha256": file_hash(SIDECAR),
        "mpnn_10k_sha256": file_hash(MPNN),
        "output_sha256": {
            "csv": file_hash(inputs / "fixed_test_selected_1000.csv"),
            "fasta": file_hash(inputs / "fixed_test_selected_1000.fasta"),
            "audit": file_hash(inputs / "candidate_audit.csv"),
        },
    }
    (inputs / "selection_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
