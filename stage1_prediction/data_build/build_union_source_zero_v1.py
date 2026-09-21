from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"D:\火星蛋白")
INPUT = ROOT / "AAA数据集" / "AAAA严格unknown版本_无伪0_v2_20260910"
FULL = ROOT / "AAA数据集" / "严格unknown版本"
SIDECAR = ROOT / "AAA数据集" / "第二阶段来源区片的数据。" / "MarsLikePro_4万条元数据侧表_v1_20260909.csv"
RULES = ROOT / "审查" / "cleanroom_benchmark" / "definitions" / "source_zero_union_rules_v1_20260910.csv"
CONTROLS = ROOT / "审查" / "cleanroom_benchmark" / "definitions" / "source_control_registry_v1_20260910.csv"
OUTPUT = ROOT / "AAA数据集" / "AAAA严格unknown版本_并集来源0_v3_20260910"
AUDIT = ROOT / "AAA数据集" / "AAAA严格unknown版本_并集来源0_v3_20260910_审计"
STAGING = OUTPUT.with_name(OUTPUT.name + "_building")
AUDIT_STAGING = AUDIT.with_name(AUDIT.name + "_building")
TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
SPLITS = ["train", "validation", "test"]
VERSION = "source_zero_union_v1_20260910"


OLD_ZERO_BASIS = {
    "cold": "original_classifier_zero|temperature_class_nonpsychrophilic_control",
    "desiccation": "original_classifier_zero",
    "oxidative": "original_classifier_zero|non_antioxidant_reference",
    "perchlorate": "original_classifier_zero",
    "radiation": "original_classifier_zero|non_dna_repair_reference",
    "salt": "original_classifier_zero|nonhalophile_source_control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_tokens(value: str) -> set[str]:
    return {token.strip() for token in str(value or "").split("|") if token.strip()}


def main() -> None:
    for path in (OUTPUT, AUDIT, STAGING, AUDIT_STAGING):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    STAGING.mkdir(parents=True)
    AUDIT_STAGING.mkdir(parents=True)

    rules = read_csv(RULES)
    controls = [row for row in read_csv(CONTROLS) if row["decision_status"] == "approved"]
    if {row["task"] for row in rules} != set(TASKS):
        raise ValueError("Union rule table does not cover all six tasks")

    controls_by_organism: dict[str, list[dict[str, str]]] = defaultdict(list)
    for control in controls:
        if control["task"] not in TASKS:
            raise ValueError(f"Unknown task in source control registry: {control['task']}")
        controls_by_organism[control["organism_name_exact"]].append(control)

    sidecar_rows = read_csv(SIDECAR)
    sidecar = {row["sequence_id"]: row for row in sidecar_rows}
    if len(sidecar_rows) != 40000 or len(sidecar) != 40000:
        raise ValueError("Metadata sidecar is not a unique 40,000-row table")

    full: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        for row in read_csv(FULL / f"{split}.csv"):
            if row["sequence_id"] in full:
                raise ValueError(f"Duplicate full provenance row: {row['sequence_id']}")
            full[row["sequence_id"]] = row
    if len(full) != 40000:
        raise ValueError("Full provenance input does not contain 40,000 unique sequences")

    changes: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    source_matches = Counter()
    counts = Counter()
    output_fields: list[str] | None = None
    output_rows_by_split: dict[str, list[dict[str, object]]] = {}

    for split in SPLITS:
        source_rows = read_csv(INPUT / f"{split}.csv")
        output_rows: list[dict[str, object]] = []
        for source in source_rows:
            sequence_id = source["sequence_id"]
            if sequence_id not in sidecar or sequence_id not in full:
                raise KeyError(f"Missing joined metadata for {sequence_id}")
            meta = sidecar[sequence_id]
            provenance = full[sequence_id]
            organisms = split_tokens(meta.get("organism_names", ""))
            matched_controls = [control for organism in organisms for control in controls_by_organism.get(organism, [])]
            by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
            for control in matched_controls:
                by_task[control["task"]].append(control)
                source_matches[(control["control_id"], "metadata_match")] += 1

            row: dict[str, object] = dict(source)
            for task in TASKS:
                old_label = source[task]
                old_origin = source[f"{task}_label_origin"]
                candidates = by_task.get(task, [])
                row[f"{task}_label_before_union"] = old_label
                row[f"{task}_zero_source_ids"] = ""
                row[f"{task}_zero_source_families"] = ""
                row[f"{task}_zero_basis"] = ""
                row[f"{task}_zero_condition_scope"] = ""
                row[f"{task}_zero_rule_version"] = VERSION

                if old_label == "0":
                    if old_origin != "existing_source_control_zero":
                        raise ValueError(f"Unexpected old zero origin: {sequence_id}/{task}/{old_origin}")
                    row[f"{task}_zero_basis"] = OLD_ZERO_BASIS[task]
                    row[f"{task}_zero_source_families"] = provenance.get(f"{task}_original_source_families", "")
                    counts[(split, task, "old_zero_retained")] += 1
                elif old_label == "unknown" and candidates:
                    row[task] = "0"
                    row[f"{task}_label_origin"] = "phenotype_sensitive_source_zero"
                    row[f"{task}_sample_weight"] = "1.0"
                    row[f"{task}_zero_source_ids"] = "|".join(sorted({item["control_id"] for item in candidates}))
                    row[f"{task}_zero_basis"] = "|".join(sorted({item["zero_basis"] for item in candidates}))
                    row[f"{task}_zero_condition_scope"] = "|".join(sorted({item["condition_scope"] for item in candidates}))
                    changes.append({
                        "sequence_id": sequence_id,
                        "split": split,
                        "task": task,
                        "old_label": old_label,
                        "new_label": 0,
                        "old_origin": old_origin,
                        "new_origin": "phenotype_sensitive_source_zero",
                        "source_control_ids": row[f"{task}_zero_source_ids"],
                        "organism_names": meta.get("organism_names", ""),
                        "condition_scope": row[f"{task}_zero_condition_scope"],
                        "rule_version": VERSION,
                    })
                    counts[(split, task, "new_source_zero")] += 1
                    for item in candidates:
                        source_matches[(item["control_id"], "label_added")] += 1
                elif old_label == "1" and candidates:
                    conflicts.append({
                        "sequence_id": sequence_id,
                        "split": split,
                        "task": task,
                        "retained_label": 1,
                        "source_control_ids": "|".join(sorted({item["control_id"] for item in candidates})),
                        "organism_names": meta.get("organism_names", ""),
                        "reason": "positive_precedence_source_context_conflict",
                        "rule_version": VERSION,
                    })
                    counts[(split, task, "positive_precedence_conflict")] += 1

                final_label = str(row[task])
                if final_label not in {"0", "1", "unknown"}:
                    raise ValueError(f"Invalid output label: {sequence_id}/{task}/{final_label}")
                if final_label == "unknown" and float(row[f"{task}_sample_weight"]) != 0:
                    raise ValueError(f"Unmasked unknown: {sequence_id}/{task}")
                if final_label == "0" and float(row[f"{task}_sample_weight"]) != 1:
                    raise ValueError(f"Zero does not have unit weight: {sequence_id}/{task}")
                counts[(split, task, final_label)] += 1
            output_rows.append(row)

        if output_fields is None:
            output_fields = list(output_rows[0])
        write_csv(STAGING / f"{split}.csv", output_rows, output_fields)
        output_rows_by_split[split] = output_rows

    if sum(len(rows) for rows in output_rows_by_split.values()) != 40000:
        raise ValueError("Output does not contain exactly 40,000 rows")

    summary: list[dict[str, object]] = []
    for task in TASKS:
        summary.append({
            "task": task,
            "positive": sum(counts[(split, task, "1")] for split in SPLITS),
            "old_zero_retained": sum(counts[(split, task, "old_zero_retained")] for split in SPLITS),
            "new_source_zero": sum(counts[(split, task, "new_source_zero")] for split in SPLITS),
            "zero_total": sum(counts[(split, task, "0")] for split in SPLITS),
            "unknown": sum(counts[(split, task, "unknown")] for split in SPLITS),
            "positive_precedence_conflict": sum(counts[(split, task, "positive_precedence_conflict")] for split in SPLITS),
            "rule_version": VERSION,
        })
    write_csv(AUDIT_STAGING / "label_summary.csv", summary, list(summary[0]))
    if changes:
        write_csv(AUDIT_STAGING / "label_changes.csv", changes, list(changes[0]))
    if conflicts:
        write_csv(AUDIT_STAGING / "positive_precedence_conflicts.csv", conflicts, list(conflicts[0]))

    match_rows = []
    for control in controls:
        match_rows.append({
            "control_id": control["control_id"],
            "task": control["task"],
            "organism_name_exact": control["organism_name_exact"],
            "metadata_matches": source_matches[(control["control_id"], "metadata_match")],
            "labels_added": source_matches[(control["control_id"], "label_added")],
            "doi": control["doi"],
            "condition_scope": control["condition_scope"],
            "rule_version": VERSION,
        })
    write_csv(AUDIT_STAGING / "source_control_match_summary.csv", match_rows, list(match_rows[0]))
    shutil.copy2(RULES, AUDIT_STAGING / RULES.name)
    shutil.copy2(CONTROLS, AUDIT_STAGING / CONTROLS.name)

    input_hashes = {str(path): sha256_file(path) for path in [RULES, CONTROLS, SIDECAR]}
    for split in SPLITS:
        input_hashes[str(INPUT / f"{split}.csv")] = sha256_file(INPUT / f"{split}.csv")
        input_hashes[str(FULL / f"{split}.csv")] = sha256_file(FULL / f"{split}.csv")

    manifest = {
        "version": VERSION,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "semantics": {
            "one": "retained positive source or evidence-supported broad candidate membership",
            "zero": "union of frozen original classifier zero and approved phenotype-sensitive source comparator",
            "unknown": "no approved positive or zero source evidence; masked from loss",
            "precedence": "positive is never overwritten by a source-level zero; conflict is audited",
            "pseudozero": "not used",
        },
        "rows": {split: len(output_rows_by_split[split]) for split in SPLITS},
        "summary": summary,
        "approved_source_controls": len(controls),
        "new_zero_label_slots": len(changes),
        "new_zero_unique_sequences": len({str(row["sequence_id"]) for row in changes}),
        "positive_precedence_conflicts": len(conflicts),
        "input_sha256": input_hashes,
        "output_sha256": {split: sha256_file(STAGING / f"{split}.csv") for split in SPLITS},
        "qc": {
            "all_rows_preserved": True,
            "split_membership_preserved": True,
            "sequence_content_preserved": True,
            "unknown_weight_zero": True,
            "zero_weight_one": True,
            "pseudozero_origins_remaining": 0,
        },
    }
    (AUDIT_STAGING / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_by_task = {row["task"]: row for row in summary}
    lines = [
        "# MarsLikePro 4万条并集来源0版本",
        "",
        f"版本：`{VERSION}`",
        "",
        "- 保留全部原始分类器0，并加入经论文和菌株名核对的来源敏感对照0。",
        "- 新来源0只从原unknown中产生；既有1不被覆盖，冲突进入审计表。",
        "- 不使用伪0；unknown权重为0，正例和两类来源0权重为1。",
        "- 序列、ID50簇及36,000/2,000/2,000固定划分保持不变。",
        "",
        "| 任务 | 正例 | 原0保留 | 新来源0 | 0合计 | unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary_by_task[task]
        lines.append(
            f"| {task} | {item['positive']:,} | {item['old_zero_retained']:,} | "
            f"{item['new_source_zero']:,} | {item['zero_total']:,} | {item['unknown']:,} |"
        )
    lines.extend([
        "",
        "## 文件",
        "",
        "- `train.csv`：训练集。",
        "- `validation.csv`：固定验证集。",
        "- `test.csv`：固定测试集。",
        "- 相邻审计目录保存逐标签变更、来源对照登记、冲突和SHA-256。",
    ])
    (STAGING / "总结.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final conservation check against the immediate input.
    for split in SPLITS:
        before = read_csv(INPUT / f"{split}.csv")
        after = read_csv(STAGING / f"{split}.csv")
        if len(before) != len(after):
            raise ValueError(f"Row count changed in {split}")
        for old, new in zip(before, after):
            for field in ("sequence_id", "split", "sequence", "id50_cluster_id"):
                if old[field] != new[field]:
                    raise ValueError(f"Frozen field changed: {split}/{old['sequence_id']}/{field}")
            for task in TASKS:
                if old[task] == "1" and new[task] != "1":
                    raise ValueError(f"Positive was overwritten: {split}/{old['sequence_id']}/{task}")
                if old[task] == "0" and new[task] != "0":
                    raise ValueError(f"Original zero was overwritten: {split}/{old['sequence_id']}/{task}")
                if new[f"{task}_label_origin"] == "matched_unlabeled_pseudozero_train_only":
                    raise ValueError(f"Pseudozero origin remains: {split}/{old['sequence_id']}/{task}")

    STAGING.rename(OUTPUT)
    AUDIT_STAGING.rename(AUDIT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
