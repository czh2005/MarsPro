from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DATA_ROOT = Path(r"D:\火星蛋白\数据集")
STAGING = Path(r"D:\火星蛋白\审查\cleanroom_benchmark\staging\merge_three_user_datasets_v1_20260909")
OUTPUT_DIR = DATA_ROOT / "合并数据集_六标签"
SEED = 20260909
VERSION = "merge_three_user_datasets_six_labels_v2_20260909"

TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
FRACTIONS = {"train": 0.90, "validation": 0.05, "test": 0.05}
ALLOWED_LABELS = {"0", "1"}

DATASETS = {
    "original": {
        "directory": DATA_ROOT / "原数据",
        "labels": {
            "cold": "cold_adaptation_membership",
            "desiccation": "desiccation_adaptation_membership",
            "oxidative": "oxidative_stress_membership",
            "perchlorate": "perchlorate_related_membership",
            "radiation": "radiation_dna_repair_membership",
            "salt": "salt_adaptation_membership",
        },
    },
    "new": {
        "directory": DATA_ROOT / "新benchmark",
        "labels": {
            "cold": "cold",
            "desiccation": "desiccation",
            "oxidative": "oxidative",
            "perchlorate": "perchlorate",
            "radiation": "radiation",
            "salt": "salt",
        },
    },
    "gao": {
        "directory": DATA_ROOT / "高钰峰",
        "labels": {
            "cold": "cold_adaptation_membership",
            "oxidative": "oxidative_stress_membership",
            "radiation": "radiation_dna_repair_membership",
            "salt": "salt_adaptation_membership",
        },
        "source_scope_defaults": {
            "desiccation": "0",
            "perchlorate": "0",
        },
    },
}


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def stable_token(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def deterministic_gzip_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    packed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0)
    text = io.TextIOWrapper(packed, encoding="utf-8", newline="")
    return raw, packed, text


def normalize_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def read_inputs() -> tuple[dict[str, dict], dict[str, int]]:
    combined: dict[str, dict] = {}
    input_counts: dict[str, int] = {}
    for dataset_name, spec in DATASETS.items():
        count = 0
        seen_in_dataset: set[str] = set()
        for split in ("train", "validation", "test"):
            path = spec["directory"] / f"{split}.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    count += 1
                    sequence = normalize_sequence(row["sequence"])
                    if not sequence:
                        raise ValueError(f"{path}: empty sequence")
                    sequence_hash = sequence_sha256(sequence)
                    if sequence_hash in seen_in_dataset:
                        raise ValueError(f"{dataset_name}: duplicate sequence across splits: {sequence_hash}")
                    seen_in_dataset.add(sequence_hash)
                    if "sequence_hash" in row and row["sequence_hash"] and row["sequence_hash"] != sequence_hash:
                        raise ValueError(f"{path}: sequence hash mismatch for {row['sequence_id']}")

                    record = combined.setdefault(
                        sequence_hash,
                        {
                            "sequence_sha256": sequence_hash,
                            "sequence": sequence,
                            "source_datasets": set(),
                            "source_ids": defaultdict(list),
                            "original_splits": {},
                            "label_values": {task: {} for task in TASKS},
                        },
                    )
                    if record["sequence"] != sequence:
                        raise ValueError(f"SHA-256 collision: {sequence_hash}")
                    record["source_datasets"].add(dataset_name)
                    record["source_ids"][dataset_name].append(row["sequence_id"])
                    record["original_splits"][dataset_name] = split
                    for task, column in spec["labels"].items():
                        value = str(row[column]).strip()
                        if value not in ALLOWED_LABELS:
                            raise ValueError(f"{path}: invalid {column}={value!r}")
                        record["label_values"][task][dataset_name] = value
                    for task, value in spec.get("source_scope_defaults", {}).items():
                        record["label_values"][task][dataset_name] = value
        input_counts[dataset_name] = count
    return combined, input_counts


def merged_label(values_by_dataset: dict[str, str]) -> int:
    if any(value == "1" for value in values_by_dataset.values()):
        return 1
    if values_by_dataset:
        return 0
    return -1


def prepare() -> None:
    combined, input_counts = read_inputs()
    STAGING.mkdir(parents=True, exist_ok=True)
    exact_path = STAGING / "merged_exact_master.csv.gz"
    fasta_path = STAGING / "merged_exact_sequences.fasta"
    columns = [
        "sequence_sha256",
        "sequence",
        "sequence_length",
        "source_datasets",
        "original_splits_json",
        *TASKS,
        "label_coverage",
        "label_disagreement_count",
    ]
    for task in TASKS:
        columns.extend([f"{task}_observed_datasets", f"{task}_positive_datasets", f"{task}_disagreement"])

    raw, packed, text = deterministic_gzip_writer(exact_path)
    try:
        writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        with fasta_path.open("w", encoding="ascii", newline="\n") as fasta:
            for sequence_hash in sorted(combined):
                record = combined[sequence_hash]
                output = {
                    "sequence_sha256": sequence_hash,
                    "sequence": record["sequence"],
                    "sequence_length": len(record["sequence"]),
                    "source_datasets": "|".join(sorted(record["source_datasets"])),
                    "original_splits_json": json.dumps(record["original_splits"], sort_keys=True, separators=(",", ":")),
                }
                disagreements = 0
                coverage = 0
                for task in TASKS:
                    values = record["label_values"][task]
                    label = merged_label(values)
                    disagreement = set(values.values()) == {"0", "1"}
                    output[task] = label
                    output[f"{task}_observed_datasets"] = "|".join(sorted(values))
                    output[f"{task}_positive_datasets"] = "|".join(sorted(name for name, value in values.items() if value == "1"))
                    output[f"{task}_disagreement"] = int(disagreement)
                    coverage += int(label != -1)
                    disagreements += int(disagreement)
                output["label_coverage"] = coverage
                output["label_disagreement_count"] = disagreements
                writer.writerow(output)
                fasta.write(f">{sequence_hash}\n{record['sequence']}\n")
    finally:
        text.flush()
        text.close()
        packed.close()
        raw.close()

    sequence_sets = {}
    for dataset_name, spec in DATASETS.items():
        values = set()
        for split in ("train", "validation", "test"):
            with (spec["directory"] / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                values.update(sequence_sha256(normalize_sequence(row["sequence"])) for row in csv.DictReader(handle))
        sequence_sets[dataset_name] = values

    summary = {
        "version": VERSION,
        "input_counts": input_counts,
        "exact_union_sequence_count": len(combined),
        "pairwise_exact_overlap": {
            f"{left}__{right}": len(sequence_sets[left] & sequence_sets[right])
            for left in DATASETS
            for right in DATASETS
            if left < right
        },
        "three_way_exact_overlap": len(set.intersection(*sequence_sets.values())),
        "label_merge_rule": "1 if any observed source-membership label is 1; otherwise 0 in the frozen combined source catalog",
        "unknown_is_negative": False,
        "gao_missing_task_resolution": {
            "desiccation": 0,
            "perchlorate": 0,
            "basis": "all seven frozen Gao source families have task scopes limited to cold, oxidative, radiation/DNA repair, or salt; zero means contrast nonmembership, not functional negative",
        },
        "old_split_reused": False,
        "fasta_sha256": hashlib.sha256(fasta_path.read_bytes()).hexdigest(),
    }
    (STAGING / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_exact_master() -> dict[str, dict[str, str]]:
    with gzip.open(STAGING / "merged_exact_master.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        return {row["sequence_sha256"]: row for row in csv.DictReader(handle)}


def parse_clusters(cluster_tsv: Path, expected_hashes: set[str]) -> dict[str, list[str]]:
    members_by_mmseqs_rep: dict[str, list[str]] = defaultdict(list)
    with cluster_tsv.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Malformed cluster row: {line[:100]!r}")
            representative, member = parts[:2]
            members_by_mmseqs_rep[representative].append(member)
    observed = {member for members in members_by_mmseqs_rep.values() for member in members}
    if observed != expected_hashes:
        missing = len(expected_hashes - observed)
        extra = len(observed - expected_hashes)
        raise ValueError(f"Cluster coverage mismatch: missing={missing}, extra={extra}")
    return members_by_mmseqs_rep


def allocation_counts(size: int) -> dict[str, int]:
    raw = {split: size * fraction for split, fraction in FRACTIONS.items()}
    counts = {split: math.floor(value) for split, value in raw.items()}
    remaining = size - sum(counts.values())
    order = sorted(FRACTIONS, key=lambda split: (-(raw[split] - counts[split]), stable_token(SEED, size, split)))
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def assign_splits(rows: list[dict]) -> None:
    by_signature: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        signature = "".join(str(row[task]) for task in TASKS)
        by_signature[signature].append(row)
    for signature, group in sorted(by_signature.items()):
        ordered = sorted(group, key=lambda row: stable_token(SEED, signature, row["sequence_id"]))
        counts = allocation_counts(len(ordered))
        start = 0
        for split in ("train", "validation", "test"):
            stop = start + counts[split]
            for row in ordered[start:stop]:
                row["split"] = split
            start = stop


def finalize(cluster_tsv: Path) -> None:
    exact = read_exact_master()
    prepare_summary = json.loads((STAGING / "prepare_summary.json").read_text(encoding="utf-8"))
    clusters = parse_clusters(cluster_tsv, set(exact))
    rows: list[dict] = []
    mixed_cluster_counts = Counter()
    for members in clusters.values():
        members = sorted(set(members))
        representative_hash = min(members)
        representative = exact[representative_hash]
        cluster_token = stable_token("MERGED_ID50", *members)[:20]
        row = {
            "sequence_id": f"MERGED50_{representative_hash[:16]}",
            "sequence": representative["sequence"],
            "cold": int(representative["cold"]),
            "desiccation": int(representative["desiccation"]),
            "oxidative": int(representative["oxidative"]),
            "perchlorate": int(representative["perchlorate"]),
            "radiation": int(representative["radiation"]),
            "salt": int(representative["salt"]),
            "split": "",
            "id50_cluster_id": f"MERGED_ID50_{cluster_token}",
            "id50_cluster_size": len(members),
            "source_datasets": representative["source_datasets"],
            "label_coverage": int(representative["label_coverage"]),
        }
        for task in TASKS:
            observed = {exact[member][task] for member in members}
            if len(observed) > 1:
                mixed_cluster_counts[task] += 1
        rows.append(row)

    assign_splits(rows)
    rows.sort(key=lambda row: row["sequence_id"])
    unresolved_labels = sum(row[task] == -1 for row in rows for task in TASKS)
    if unresolved_labels:
        raise ValueError(f"Unresolved labels remain after source-scope audit: {unresolved_labels}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "MarsLikePro_三数据集合并_ID50_六标签.csv"
    columns = [
        "sequence_id",
        "sequence",
        *TASKS,
        "split",
        "id50_cluster_id",
        "id50_cluster_size",
        "source_datasets",
        "label_coverage",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    split_columns = [column for column in columns if column != "split"]
    split_outputs = {}
    for split_name in ("train", "validation", "test"):
        split_path = OUTPUT_DIR / f"{split_name}.csv"
        split_rows = [
            {column: row[column] for column in split_columns}
            for row in rows
            if row["split"] == split_name
        ]
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=split_columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(split_rows)
        split_outputs[split_name] = {
            "file": split_path.name,
            "row_count": len(split_rows),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        }

    split_counts = Counter(row["split"] for row in rows)
    label_summary = {}
    for task in TASKS:
        label_summary[task] = {
            "positive": sum(row[task] == 1 for row in rows),
            "contrast_zero": sum(row[task] == 0 for row in rows),
            "unknown": sum(row[task] == -1 for row in rows),
        }
    summary = {
        "version": VERSION,
        "input_exact_union_sequence_count": len(exact),
        "global_id50_representative_count": len(rows),
        "global_id50_removed_sequence_count": len(exact) - len(rows),
        "split_counts": dict(split_counts),
        "split_outputs": split_outputs,
        "split_rule": "deterministic label-signature-stratified 90/5/5 after global ID50 connected-component clustering",
        "labels": label_summary,
        "mixed_label_id50_cluster_counts": dict(mixed_cluster_counts),
        "unknown_label_value": None,
        "unknown_is_negative": False,
        "all_exported_labels_are_binary": all(row[task] in {0, 1} for row in rows for task in TASKS),
        "gao_missing_task_resolution": {
            "desiccation": 0,
            "perchlorate": 0,
            "basis": "frozen source-scope audit across seven Gao source families",
        },
        "representative_rule": "lexicographically smallest sequence SHA-256 per ID50 component, independent of label",
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "cluster_tsv_sha256": hashlib.sha256(cluster_tsv.read_bytes()).hexdigest(),
    }
    (OUTPUT_DIR / "合并说明.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# MarsLikePro 三套数据合并说明",
        "",
        f"- 三套输入合计记录数：{sum(prepare_summary['input_counts'].values()):,}",
        f"- 精确序列去重后：{len(exact):,}",
        f"- 全局 ID50 后代表序列：{len(rows):,}",
        f"- 训练/验证/测试：{split_counts['train']:,} / {split_counts['validation']:,} / {split_counts['test']:,}",
        "- 已分别导出为 train.csv、validation.csv 和 test.csv；分表中不再保留 split 列。",
        "",
        "## 标签规则",
        "",
        "- 同一精确序列只要任一输入来源标为 1，联合来源归属标签取 1。",
        "- 有观测且全部为 0 时取 0；0 是对照非归属，不是功能阴性。",
        "- 高钰峰数据的七个来源家族均不属于干燥或高氯酸盐来源范围，因此这两列按联合来源目录标为 0。",
        "- 最终导出只包含 0/1，不包含 -1。",
        "- 同一蛋白可同时属于多个任务。",
        "",
        "## 切分规则",
        "",
        "- 不复用三套输入的旧 split。",
        "- 在全部精确序列上重新做 50% identity、双向 coverage 0.8 的全局聚类。",
        "- 每个 ID50 连通分量选择一个与标签无关的确定性代表，再按六标签签名做 90/5/5。",
        "- 输出是来源归属型宽候选数据，不是逐蛋白直接耐受证明。",
        "",
    ]
    (OUTPUT_DIR / "合并说明.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--cluster-tsv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        finalize(args.cluster_tsv)


if __name__ == "__main__":
    main()
