#!/usr/bin/env python3
"""Verify the public MarsPro release without modifying artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_FROM_MANIFEST = {"MANIFEST.sha256", "FILE_INVENTORY.csv"}
OBSOLETE_RESULT_NAMES = {
    "aggregate_statistics.json",
    "surface_and_structure_statistics.csv",
    "schemeB_tmalign_verification.json",
    "schemeB_tmalign_threshold_counts.csv",
    "schemeB_tmalign_paired_intervals.csv",
    "schemeB_tmalign_group_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    require(manifest.is_file(), "missing MANIFEST.sha256")
    recorded: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        artifact = ROOT / Path(relative)
        require(artifact.is_file(), f"manifest file missing: {relative}")
        require(sha256(artifact) == expected, f"manifest mismatch: {relative}")
        recorded.add(relative)

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FROM_MANIFEST
    }
    require(recorded == actual, "manifest file set differs from repository file set")


def verify_stage1() -> tuple[int, int]:
    data = ROOT / "stage1_prediction" / "data"
    public_data = data / "public_accession_only"
    expected_rows = {"train.csv": 36000, "validation.csv": 2000, "test.csv": 2000}
    split_ids: set[str] = set()
    for name, expected in expected_rows.items():
        rows = read_csv(public_data / name)
        require("sequence" not in rows[0], f"public accession-only table contains raw sequence: {name}")
        require(len(rows) == expected, f"row mismatch: {name}: {len(rows)} != {expected}")
        ids = {row["sequence_id"] for row in rows}
        require(len(ids) == expected, f"duplicate sequence_id in {name}")
        require(not split_ids.intersection(ids), f"sequence_id overlap involving {name}")
        split_ids.update(ids)

    sidecar = read_csv(public_data / "metadata_sidecar_40k.csv")
    require(len(sidecar) == 40000, "metadata sidecar row mismatch")
    require({row["sequence_id"] for row in sidecar} == split_ids, "sidecar sequence_id mismatch")

    audit = json.loads((data / "provenance" / "release_audit.json").read_text(encoding="utf-8"))
    require(audit.get("rows") == 40000, "release audit row mismatch")
    require(audit.get("sidecar_sequence_id_match") is True, "sidecar audit mismatch")
    require(audit.get("cross_split_id50_cluster_count") == 0, "cross-split ID50 leakage")
    require(audit.get("known_label_cells") == 50384, "known-label cell count mismatch")
    require(audit.get("unknown_label_cells") == 189616, "unknown-label cell count mismatch")

    model = ROOT / "stage1_prediction" / "model" / "best.pt"
    expected_model_hash = "9cac216f122bdccaebd277639062b103854e40d3e2b7b73b1482bfc78ba32073"
    require(sha256(model) == expected_model_hash, "Stage 1 checkpoint hash mismatch")

    locked_test = ROOT / "stage1_prediction" / "results" / "locked_test"
    probabilities = locked_test / "outputs" / "test_probabilities.csv"
    require(len(read_csv(probabilities)) == 2000, "locked-test prediction count mismatch")
    seal = json.loads((locked_test / "outputs" / "prediction_seal.json").read_text(encoding="utf-8"))
    expected_probability_hash = "fa9145250805db1cf60760bbeb596fe17300e6b8a90179636f25119b34522299"
    require(seal.get("row_count") == 2000, "locked-test seal row mismatch")
    require(seal.get("predictions_sha256", "").lower() == expected_probability_hash, "locked-test seal hash mismatch")
    require(sha256(probabilities) == expected_probability_hash, "locked-test probability hash mismatch")
    metrics = json.loads((locked_test / "evaluation" / "overall_metrics.json").read_text(encoding="utf-8"))
    require(abs(metrics.get("macro_auprc", 0.0) - 0.9752856420096873) <= 1e-12, "locked-test metric mismatch")
    require(metrics.get("unknown_counted_as_negative") is False, "locked-test unknown policy mismatch")

    inventory = read_csv(data / "provenance" / "source_inventory_40k.csv")
    require(len(inventory) == 18, "scientific source inventory count mismatch")
    require(all(row.get("license_public_redistribution_status", "").strip() for row in inventory), "blank redistribution status")
    # A non-empty status is not the same as a clearance. Until a source has an
    # explicit sequence-level clearance status, it remains conditional for a
    # public FASTA release. The dated audit documents the reason per source.
    cleared = {"confirmed_sequence_redistribution"}
    unresolved = sum(
        row["license_public_redistribution_status"] not in cleared
        for row in inventory
    )
    require(unresolved == 18, "unexpected redistribution-audit count")
    return len(split_ids), unresolved


def verify_stage2() -> tuple[int, int]:
    model = ROOT / "stage2_design" / "model" / "best_joint_lora.pt"
    expected_hash = "d5c161e53b417bd88f373e2f0fcc31d864768bf6169fe021513f60605dea37be"
    require(sha256(model) == expected_hash, "Stage 2 adapter hash mismatch")

    corpus = read_csv(ROOT / "stage2_design" / "structure_corpus" / "structure_corpus_index_10000.csv")
    require(len(corpus) == 10000, "structure-corpus index row mismatch")
    require(len({row["sequence_id"] for row in corpus}) == 10000, "structure-corpus duplicate sequence_id")
    require(all(row["has_any_positive_label"] == "1" for row in corpus), "structure corpus contains non-positive reference")
    require(all(row["structure_in_repository"].lower() == "false" for row in corpus), "structure index claims bundled PDB")

    forbidden_structures = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
    ]
    require(not forbidden_structures, "PDB/CIF files must not be included")

    matched_qc = json.loads((ROOT / "stage2_design" / "matched_seed_798" / "results" / "06_最终QC.json").read_text(encoding="utf-8"))
    require(matched_qc.get("design_rows") == 4788, "matched-seed design count mismatch")
    require(matched_qc.get("paired_rows") == 2394, "matched-seed pair count mismatch")
    require(matched_qc.get("backbones") == 798, "matched-seed backbone count mismatch")
    require(matched_qc.get("seeds") == [42, 43, 44], "matched-seed values mismatch")
    require(matched_qc.get("all_status_ok") is True, "matched-seed QC failed")

    fixed_root = ROOT / "stage2_design" / "fixed_test_1000"
    selected = read_csv(fixed_root / "inputs" / "fixed_test_selected_1000.csv")
    require(len(selected) == 1000, "fixed-test selection row mismatch")
    require(len({row["sequence_id"] for row in selected}) == 1000, "fixed-test duplicate sequence_id")
    require(len({row["id50_cluster_id"] for row in selected}) == 1000, "fixed-test ID50 count mismatch")

    verification = json.loads((fixed_root / "results" / "verification.json").read_text(encoding="utf-8"))
    require(verification.get("test_set_reused_for_training_or_selection") is False, "fixed test was reused")
    require(verification.get("reference_backbones") == 1000, "fixed-test verification backbone mismatch")
    require(verification.get("paired_seed_records") == 3000, "fixed-test seed-pair mismatch")

    aggregate = {row["metric"]: row for row in read_csv(fixed_root / "results" / "04_综合指标汇总.csv")}
    score = aggregate["aggregate_delta"]
    require(abs(float(score["proteinmpnn_mean"]) - 0.6185384743875927) <= 1e-12, "fixed-test baseline score mismatch")
    require(abs(float(score["marslike_mpnn_mean"]) - 0.672510831753413) <= 1e-12, "fixed-test Marslike score mismatch")
    require(int(score["improved_backbones"]) == 731, "fixed-test improved count mismatch")
    return len(corpus), matched_qc["paired_rows"]


def verify_cleanliness() -> None:
    obsolete = [path for path in ROOT.rglob("*") if path.is_file() and path.name in OBSOLETE_RESULT_NAMES]
    require(not obsolete, "obsolete result files remain in public repository")


def main() -> None:
    verify_manifest()
    sequence_count, unresolved_licenses = verify_stage1()
    structure_count, pair_count = verify_stage2()
    verify_cleanliness()
    print("MarsPro public-release verification: PASS")
    print(f"Stage 1: {sequence_count:,} sequences; locked-test macro AUPRC 0.975286")
    print(f"Stage 2: {structure_count:,} structure-index rows; {pair_count:,} matched-seed pairs")
    print("Fixed-test redesign: 1,000 backbones; aggregate score improved in 731")
    print("Bundled PDB/CIF structures: 0")
    print(f"Redistribution status still unverified for {unresolved_licenses}/18 scientific sources")


if __name__ == "__main__":
    main()
