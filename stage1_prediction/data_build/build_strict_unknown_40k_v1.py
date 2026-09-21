from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"D:\火星蛋白")
INPUT = ROOT / "AAA数据集" / "AAAESM模型训练4w"
SIDECAR = (
    ROOT
    / "AAA数据集"
    / "第二阶段来源区片的数据。"
    / "MarsLikePro_4万条元数据侧表_v1_20260909.csv"
)
EXACT_MASTER = (
    ROOT
    / "审查"
    / "cleanroom_benchmark"
    / "staging"
    / "merge_three_user_datasets_v1_20260909"
    / "merged_exact_master.csv.gz"
)
SOURCE_RECORDS = (
    ROOT
    / "审查"
    / "cleanroom_benchmark"
    / "releases"
    / "issues_four_source_membership_doc_v1_20260908"
    / "provenance"
    / "source_records.csv.gz"
)
OUTPUT = ROOT / "AAA数据集" / "严格unknown版本"
STAGING = OUTPUT.with_name(OUTPUT.name + "_building")
TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
SPLITS = ["train", "validation", "test"]
VERSION = "strict_unknown_40k_v2_20260910"
SOURCE_TASK_MAP = {
    "cold_adaptation": "cold",
    "oxidative_stress_adaptation": "oxidative",
    "radiation_associated_dna_repair": "radiation",
    "salt_adaptation": "salt",
}
ALLOWED_SOURCE_BASES = {
    "documented_temperature_class_to_cold_vs_other",
    "explicit_source_label",
    "filename_label",
    "explicit_header_label",
    "legacy_50_list_membership_current_sequence",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def tokens(value: object) -> set[str]:
    return {token for token in str(value or "").split("|") if token and token != "nan"}


def read_sidecar() -> dict[str, dict[str, str]]:
    with SIDECAR.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["sequence_id"]: row for row in rows}
    if len(rows) != 40000 or len(lookup) != 40000:
        raise ValueError("The metadata sidecar is not a unique 40,000-row table")
    return lookup


def read_exact_master() -> dict[str, dict[str, str]]:
    with gzip.open(EXACT_MASTER, "rt", encoding="utf-8", newline="") as handle:
        return {row["sequence_sha256"]: row for row in csv.DictReader(handle)}


def read_source_classes() -> dict[str, dict[str, list[dict[str, str]]]]:
    lookup: dict[str, dict[str, list[dict[str, str]]]] = {}
    with gzip.open(SOURCE_RECORDS, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            task = SOURCE_TASK_MAP.get(row["task"])
            if not task:
                continue
            if row["role"] != "source_labeled" or row["label"] not in {"0", "1"}:
                continue
            if row["label_basis"] not in ALLOWED_SOURCE_BASES:
                continue
            by_task = lookup.setdefault(row["sequence_sha256"], {})
            by_task.setdefault(task, []).append(row)
    return lookup


def decide(
    task: str,
    base: dict[str, str],
    meta: dict[str, str],
    exact: dict[str, str] | None,
    source_rows: list[dict[str, str]],
) -> dict[str, object]:
    catalog = int(base[task])
    support = tokens(meta.get(f"{task}_support_classes", ""))
    valid = truthy(meta.get(f"{task}_individual_evidence_valid", ""))
    observed_sources = exact.get(f"{task}_observed_datasets", "") if exact else ""
    positive_sources = exact.get(f"{task}_positive_datasets", "") if exact else ""

    source_values = {int(row["label"]) for row in source_rows}
    source_families = sorted({row["source_family"] for row in source_rows})
    source_bases = sorted({row["label_basis"] for row in source_rows})

    # Original source classes take precedence over broad collection membership.
    # Conflicts are never resolved by voting or source preference.
    direct_positive = valid and "positive" in support
    if source_values == {0, 1}:
        status = "unknown_conflicting_original_source_classes"
        basis = "conflicting_original_source_classes_no_majority_vote"
        value: str | int = "unknown"
        mask = 0
    elif source_values == {1}:
        status = "positive_original_source_class"
        basis = "original_source_defined_positive"
        value: str | int = 1
        mask = 1
    elif source_values == {0}:
        status = "negative_original_source_control"
        basis = "original_source_defined_control_not_biological_negative"
        value = 0
        mask = 1
    elif direct_positive:
        status = "positive_direct_evidence"
        basis = "v3_valid_positive"
        value = 1
        mask = 1
    elif catalog == 1:
        status = "positive_broad_membership"
        basis = "matched_positive_source_catalog"
        value = 1
        mask = 1
    else:
        status = "unknown_after_frozen_source_crosscheck"
        basis = "no_positive_match_in_frozen_sources_not_a_biological_negative"
        value = "unknown"
        mask = 0

    scoped_negative = "none"
    if "scoped_negative" in support:
        scoped_negative = "validated_scoped_negative"
    elif "weak_scoped_negative" in support:
        scoped_negative = "weak_scoped_negative"

    return {
        "value": value,
        "mask": mask,
        "catalog": catalog,
        "status": status,
        "basis": basis,
        "observed_sources": observed_sources,
        "positive_sources": positive_sources,
        "scoped_negative": scoped_negative,
        "v3_support_classes": "|".join(sorted(support)),
        "original_source_record_count": len(source_rows),
        "original_source_families": "|".join(source_families),
        "original_source_label_bases": "|".join(source_bases),
    }


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    sidecar = read_sidecar()
    exact_master = read_exact_master()
    source_classes = read_source_classes()
    decisions = Counter()
    split_counts: dict[str, dict[str, Counter]] = {}
    input_hashes = {str(path): sha256_file(path) for path in [SIDECAR, EXACT_MASTER, SOURCE_RECORDS]}

    audit_path = STAGING / "label_audit.csv.gz"
    with gzip.open(audit_path, "wt", encoding="utf-8", newline="") as audit_handle:
        audit_fields = [
            "sequence_id", "split", "task", "catalog_label", "strict_value", "label_mask",
            "status", "basis", "observed_source_datasets", "positive_source_datasets",
            "scoped_negative_status", "v3_support_classes", "original_source_record_count",
            "original_source_families", "original_source_label_bases",
        ]
        audit_writer = csv.DictWriter(audit_handle, fieldnames=audit_fields, lineterminator="\n")
        audit_writer.writeheader()

        for split in SPLITS:
            source = INPUT / f"{split}.csv"
            input_hashes[str(source)] = sha256_file(source)
            split_counts[split] = {task: Counter() for task in TASKS}
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                base_rows = list(reader)
                base_fields = list(reader.fieldnames or [])

            extra_fields: list[str] = []
            for task in TASKS:
                extra_fields.extend([
                    f"{task}_mask", f"{task}_catalog_label", f"{task}_status", f"{task}_basis",
                    f"{task}_observed_source_datasets", f"{task}_positive_source_datasets",
                    f"{task}_scoped_negative_status", f"{task}_v3_support_classes",
                    f"{task}_original_source_record_count", f"{task}_original_source_families",
                    f"{task}_original_source_label_bases",
                ])
            fields = base_fields + [field for field in extra_fields if field not in base_fields]

            with (STAGING / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for base in base_rows:
                    sequence_id = base["sequence_id"]
                    if sequence_id not in sidecar:
                        raise KeyError(f"Missing sidecar row: {sequence_id}")
                    meta = sidecar[sequence_id]
                    digest = sequence_hash(base["sequence"])
                    exact = exact_master.get(digest)
                    if exact is None:
                        raise KeyError(f"Missing frozen exact-source row: {sequence_id} {digest}")
                    row = dict(base)
                    for task in TASKS:
                        source_rows = source_classes.get(digest, {}).get(task, [])
                        item = decide(task, base, meta, exact, source_rows)
                        row[task] = item["value"]
                        row[f"{task}_mask"] = item["mask"]
                        row[f"{task}_catalog_label"] = item["catalog"]
                        row[f"{task}_status"] = item["status"]
                        row[f"{task}_basis"] = item["basis"]
                        row[f"{task}_observed_source_datasets"] = item["observed_sources"]
                        row[f"{task}_positive_source_datasets"] = item["positive_sources"]
                        row[f"{task}_scoped_negative_status"] = item["scoped_negative"]
                        row[f"{task}_v3_support_classes"] = item["v3_support_classes"]
                        row[f"{task}_original_source_record_count"] = item["original_source_record_count"]
                        row[f"{task}_original_source_families"] = item["original_source_families"]
                        row[f"{task}_original_source_label_bases"] = item["original_source_label_bases"]
                        decisions[(task, item["status"])] += 1
                        split_counts[split][task][str(item["value"])] += 1
                        audit_writer.writerow({
                            "sequence_id": sequence_id,
                            "split": split,
                            "task": task,
                            "catalog_label": item["catalog"],
                            "strict_value": item["value"],
                            "label_mask": item["mask"],
                            "status": item["status"],
                            "basis": item["basis"],
                            "observed_source_datasets": item["observed_sources"],
                            "positive_source_datasets": item["positive_sources"],
                            "scoped_negative_status": item["scoped_negative"],
                            "v3_support_classes": item["v3_support_classes"],
                            "original_source_record_count": item["original_source_record_count"],
                            "original_source_families": item["original_source_families"],
                            "original_source_label_bases": item["original_source_label_bases"],
                        })
                    writer.writerow(row)

    summary_rows = []
    for task in TASKS:
        task_total = sum(count for (name, _), count in decisions.items() if name == task)
        positive = sum(count for (name, state), count in decisions.items() if name == task and state.startswith("positive"))
        negative = decisions[(task, "negative_original_source_control")]
        conflict = decisions[(task, "unknown_conflicting_original_source_classes")]
        unknown = decisions[(task, "unknown_after_frozen_source_crosscheck")]
        summary_rows.append({
            "task": task,
            "sequence_task_rows": task_total,
            "positive": positive,
            "source_defined_control": negative,
            "unknown_no_match": unknown,
            "unknown_conflict": conflict,
            "known_fraction": f"{(positive + negative) / task_total:.8f}",
            "positive_original_source_class": decisions[(task, "positive_original_source_class")],
            "positive_broad_membership": decisions[(task, "positive_broad_membership")],
            "positive_direct_evidence": decisions[(task, "positive_direct_evidence")],
        })
    with (STAGING / "label_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "version": VERSION,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "rows": {"train": 36000, "validation": 2000, "test": 2000},
        "tasks": TASKS,
        "input_sha256": input_hashes,
        "rules": {
            "one": "original source positive, broad positive membership, or valid v3 direct positive",
            "zero": "original source-defined comparison/control for the same task; not a biological negative",
            "unknown": "no positive match after frozen-source crosscheck; absence is not a biological negative",
            "conflict": "conflicting original same-task source classes are unknown; no voting",
            "scoped_negative": "preserved in a separate field and never promoted to a whole-task zero",
            "mask": "1 only for known top-level labels; 0 for unknown",
        },
        "summary": summary_rows,
        "split_task_counts": {
            split: {task: dict(counts) for task, counts in tasks.items()}
            for split, tasks in split_counts.items()
        },
    }
    (STAGING / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = f"""# MarsLikePro 4万条严格 unknown 版本

版本：`{VERSION}`

本版保留原始 40,000 条序列和 train/validation/test 划分，不覆盖原数据。六个顶层任务中：

- `1`：同任务原始来源正类、其他来源中交叉命中的宽阳性，或有效 v3 直接阳性。
- `0`：该任务原始数据明确提供的比较/对照类。它可用于来源任务监督，但不是实验功能阴性。
- `unknown`：交叉回查现有冻结来源后未找到阳性支持。它不是功能阴性。
- `*_mask=0`：该项不得进入普通 BCE 损失或二分类评价。

`*_catalog_label` 保留原来的 0/1 宽来源标签。`*_original_source_*` 记录恢复的原始正类/对照依据。`*_scoped_negative_status` 保留限定子任务阴性，但不把它上升为整个任务的 0。

本次“查”的范围是项目已冻结的三套来源、精确序列合并主表和 v3 证据。未对 193,482 个原始零单元格分别实时搜索整个互联网；“未命中”的科学状态因此严格保持为 unknown。
"""
    (STAGING / "README.md").write_text(readme, encoding="utf-8")

    generated = sorted(path for path in STAGING.iterdir() if path.is_file())
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in generated]
    (STAGING / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

    # Validate conservation and label domain before publishing the directory.
    expected_rows = {"train": 36000, "validation": 2000, "test": 2000}
    for split, expected in expected_rows.items():
        with (STAGING / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected}, got {len(rows)}")
        for row in rows:
            for task in TASKS:
                if row[task] not in {"0", "1", "unknown"}:
                    raise ValueError(f"{split}/{row['sequence_id']}/{task}: bad value")
                if row[task] == "unknown" and row[f"{task}_mask"] != "0":
                    raise ValueError(f"{split}/{row['sequence_id']}/{task}: unknown is not masked")

    STAGING.rename(OUTPUT)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
