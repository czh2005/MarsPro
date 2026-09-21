from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"D:\火星蛋白")
INPUT = ROOT / "AAA数据集" / "严格unknown版本_v2_来源恢复"
OUTPUT = ROOT / "AAA数据集" / "严格unknown版本"
CACHE = ROOT / "AAA数据集" / "严格unknown版本_UniProt查询缓存_v1"
TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
SPLITS = ["train", "validation", "test"]
VERSION = "strict_unknown_40k_uniprot_v1_20260910"
RULE_VERSION = "uniprot_explicit_annotation_rules_v1_20260910"
UNIPROT_ENDPOINT = "https://rest.uniprot.org/uniprotkb/stream"
FIELDS = [
    "accession", "id", "reviewed", "protein_name", "gene_names", "organism_name",
    "organism_id", "annotation_score", "cc_function", "cc_catalytic_activity",
    "cc_pathway", "cc_induction", "cc_similarity", "keyword", "go_p", "go_f",
    "xref_interpro", "xref_pfam", "length", "sequence",
]
CHUNK_SIZE = 100
WORKERS = 3


PATTERNS = {
    "cold": [
        r"\bcold[- ]shock\b", r"\bresponse to cold\b", r"\bcold adaptation\b",
        r"\bcold acclim", r"\blow[- ]temperature response\b", r"\bantifreeze\b",
        r"\bice[- ]binding\b", r"\bice recrystallization\b", r"\bpsychrophil",
    ],
    "desiccation": [
        r"\bdesiccation(?: stress| tolerance| resistance)?\b",
        r"\bresponse to dehydration\b", r"\bdehydration stress\b", r"\bwater deprivation\b",
        r"\banhydrobiosis\b", r"\bhydrophilin\b", r"\blate embryogenesis abundant\b",
        r"\bLEA protein\b",
    ],
    "oxidative": [
        r"\boxidative stress\b", r"\bresponse to oxidative stress\b",
        r"\bhydrogen peroxide\b", r"\breactive oxygen species\b", r"\bantioxidant\b",
        r"\bsuperoxide dismutase\b", r"\bcatalase\b", r"\bperoxiredoxin\b",
        r"\bglutathione peroxidase\b", r"\bthioredoxin peroxidase\b",
    ],
    "perchlorate": [
        r"\bperchlorate\b", r"\bperchlorate reductase\b", r"\bperchlorate reduction\b",
    ],
    "radiation": [
        r"\bradiation resistance\b", r"\bresponse to radiation\b", r"\bionizing radiation\b",
        r"\bDNA repair\b", r"\bDNA damage response\b", r"\bDNA damage checkpoint\b",
        r"\bphotolyase\b", r"\bphotoreactivation\b", r"\bnucleotide[- ]excision repair\b",
        r"\bbase[- ]excision repair\b", r"\bmismatch repair\b",
        r"\bnon[- ]homologous end joining\b", r"\bhomologous recombination\b",
    ],
    "salt": [
        r"\bsalt stress\b", r"\bresponse to salt\b", r"\bsalt tolerance\b",
        r"\bhigh salinity\b", r"\bhalophil", r"\bosmotic stress\b",
        r"\bhyperosmotic\b", r"\bosmoregulation\b", r"\bcompatible solute\b",
        r"\bectoine\b", r"\bglycine betaine\b",
    ],
}
COMPILED = {task: [re.compile(pattern, re.I) for pattern in patterns] for task, patterns in PATTERNS.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_md5(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    return hashlib.md5(normalized.encode("ascii")).hexdigest()


def read_inputs() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]], dict[str, str]]:
    splits: dict[str, list[dict[str, str]]] = {}
    by_md5: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    for split in SPLITS:
        path = INPUT / f"{split}.csv"
        hashes[str(path)] = file_sha256(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        splits[split] = rows
        for row in rows:
            md5 = sequence_md5(row["sequence"])
            if md5 in by_md5 and by_md5[md5]["sequence"] != row["sequence"]:
                raise ValueError(f"MD5 collision: {md5}")
            by_md5[md5] = row
    if sum(map(len, splits.values())) != 40000 or len(by_md5) != 40000:
        raise ValueError("Expected 40,000 unique sequences")
    return splits, by_md5, hashes


def fetch_chunk(index: int, checksums: list[str]) -> dict[str, object]:
    raw_dir = CACHE / "raw_tsv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"chunk_{index:04d}.tsv"
    meta_path = raw_dir / f"chunk_{index:04d}.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("request_checksums") == checksums and meta.get("sha256") == file_sha256(path):
            return meta

    query = "(" + " OR ".join(f"checksum:{checksum}" for checksum in checksums) + ")"
    params = {"query": query, "format": "tsv", "fields": ",".join(FIELDS)}
    url = UNIPROT_ENDPOINT + "?" + urllib.parse.urlencode(params)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = ""
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MarsLikePro-benchmark/1.0"})
            with opener.open(request, timeout=120) as response:
                payload = response.read()
                release = response.headers.get("x-uniprot-release", "")
            if not payload.startswith(b"Entry\t"):
                raise ValueError("Unexpected UniProt TSV header")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            row_count = max(0, payload.count(b"\n") - 1)
            meta = {
                "chunk": index,
                "request_checksums": checksums,
                "request_count": len(checksums),
                "response_rows": row_count,
                "release": release,
                "sha256": file_sha256(path),
                "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return meta
        except Exception as exc:  # bounded retry for public API/network failures
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 5:
                break
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"chunk {index} failed after retries: {last_error}")


def fetch_all(checksums: list[str]) -> list[dict[str, object]]:
    chunks = [checksums[i:i + CHUNK_SIZE] for i in range(0, len(checksums), CHUNK_SIZE)]
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_chunk, index, chunk): index for index, chunk in enumerate(chunks)}
        for completed, future in enumerate(as_completed(futures), 1):
            meta = future.result()
            results.append(meta)
            if completed % 20 == 0 or completed == len(chunks):
                print(f"FETCH {completed}/{len(chunks)} chunks", flush=True)
    return sorted(results, key=lambda item: int(item["chunk"]))


def annotation_text(row: dict[str, str]) -> str:
    fields = [
        "Protein names", "Gene Names", "Function [CC]", "Catalytic activity", "Pathway",
        "Induction", "Sequence similarities", "Keywords", "Gene Ontology (biological process)",
        "Gene Ontology (molecular function)", "InterPro", "Pfam",
    ]
    return " | ".join(row.get(field, "") for field in fields if row.get(field, ""))


def parse_annotations(valid_md5: set[str]) -> tuple[dict[str, list[dict[str, str]]], str]:
    by_md5: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    releases: set[str] = set()
    for meta_path in sorted((CACHE / "raw_tsv").glob("chunk_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        releases.add(str(meta.get("release", "")))
        path = meta_path.with_suffix(".tsv")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sequence = row.get("Sequence", "")
                if not sequence:
                    continue
                md5 = sequence_md5(sequence)
                if md5 not in valid_md5:
                    raise ValueError(f"UniProt returned an unrequested sequence checksum: {md5}")
                key = (md5, row.get("Entry", ""))
                if key in seen:
                    continue
                seen.add(key)
                by_md5[md5].append(row)
    release_values = sorted(value for value in releases if value)
    if len(release_values) != 1:
        raise ValueError(f"Expected one UniProt release, found {release_values}")
    return by_md5, release_values[0]


def task_matches(entries: list[dict[str, str]], task: str) -> dict[str, object]:
    matched = []
    terms = set()
    for entry in entries:
        text = annotation_text(entry)
        found = sorted({match.group(0) for pattern in COMPILED[task] for match in pattern.finditer(text)})
        if found:
            matched.append(entry)
            terms.update(found)
    accessions = sorted({entry.get("Entry", "") for entry in matched if entry.get("Entry", "")})
    reviewed = sorted({
        entry.get("Entry", "") for entry in matched
        if entry.get("Reviewed", "").strip().lower() == "reviewed" and entry.get("Entry", "")
    })
    confidence = "reviewed_explicit_annotation" if reviewed else "unreviewed_explicit_annotation" if accessions else "none"
    return {
        "matched": bool(matched),
        "accessions": "|".join(accessions),
        "reviewed_accessions": "|".join(reviewed),
        "terms": "|".join(sorted(terms, key=str.lower)),
        "confidence": confidence,
    }


def build_outputs(
    splits: dict[str, list[dict[str, str]]],
    by_md5: dict[str, list[dict[str, str]]],
    release: str,
    input_hashes: dict[str, str],
    fetch_meta: list[dict[str, object]],
) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    staging = OUTPUT.with_name(OUTPUT.name + "_building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    counts = {task: Counter() for task in TASKS}
    coverage = Counter()
    changes: list[dict[str, object]] = []
    annotation_rows: list[dict[str, str]] = []
    for md5, entries in by_md5.items():
        coverage["matched_sequences"] += 1
        coverage["entries"] += len(entries)
        coverage["reviewed_entries"] += sum(entry.get("Reviewed", "").lower() == "reviewed" for entry in entries)
        for entry in entries:
            annotation_rows.append({"sequence_md5": md5, **entry})

    annotation_fields = ["sequence_md5"] + (list(annotation_rows[0])[1:] if annotation_rows else [])
    with gzip.open(staging / "uniprot_annotations.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotation_rows)

    for split in SPLITS:
        rows = splits[split]
        base_fields = list(rows[0])
        sequence_fields = ["uniprot_release", "uniprot_exact_match", "uniprot_entry_count", "uniprot_reviewed_entry_count", "uniprot_accessions"]
        task_fields = []
        for task in TASKS:
            task_fields.extend([
                f"{task}_pre_uniprot_value", f"{task}_uniprot_match", f"{task}_uniprot_confidence",
                f"{task}_uniprot_accessions", f"{task}_uniprot_reviewed_accessions",
                f"{task}_uniprot_matched_terms", f"{task}_uniprot_rule_version",
            ])
        fields = base_fields + [field for field in sequence_fields + task_fields if field not in base_fields]
        with (staging / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for source_row in rows:
                row = dict(source_row)
                md5 = sequence_md5(row["sequence"])
                entries = by_md5.get(md5, [])
                row["uniprot_release"] = release
                row["uniprot_exact_match"] = int(bool(entries))
                row["uniprot_entry_count"] = len(entries)
                row["uniprot_reviewed_entry_count"] = sum(entry.get("Reviewed", "").lower() == "reviewed" for entry in entries)
                row["uniprot_accessions"] = "|".join(sorted({entry.get("Entry", "") for entry in entries if entry.get("Entry", "")}))
                for task in TASKS:
                    before = row[task]
                    match = task_matches(entries, task)
                    row[f"{task}_pre_uniprot_value"] = before
                    row[f"{task}_uniprot_match"] = int(match["matched"])
                    row[f"{task}_uniprot_confidence"] = match["confidence"]
                    row[f"{task}_uniprot_accessions"] = match["accessions"]
                    row[f"{task}_uniprot_reviewed_accessions"] = match["reviewed_accessions"]
                    row[f"{task}_uniprot_matched_terms"] = match["terms"]
                    row[f"{task}_uniprot_rule_version"] = RULE_VERSION
                    outcome = "unchanged"
                    if match["matched"] and before == "unknown":
                        row[task] = "1"
                        row[f"{task}_mask"] = "1"
                        row[f"{task}_status"] = "positive_uniprot_explicit_annotation"
                        row[f"{task}_basis"] = match["confidence"]
                        outcome = "unknown_to_positive"
                    elif match["matched"] and before == "0":
                        row[task] = "unknown"
                        row[f"{task}_mask"] = "0"
                        row[f"{task}_status"] = "unknown_conflict_source_control_vs_uniprot_positive"
                        row[f"{task}_basis"] = "conflict_requires_review"
                        outcome = "source_control_to_conflict_unknown"
                    elif match["matched"] and before == "1":
                        outcome = "positive_confirmed_by_uniprot"
                    counts[task][str(row[task])] += 1
                    counts[task][outcome] += 1
                    if match["matched"]:
                        changes.append({
                            "sequence_id": row["sequence_id"], "split": split, "task": task,
                            "before": before, "after": row[task], "outcome": outcome,
                            "confidence": match["confidence"], "accessions": match["accessions"],
                            "reviewed_accessions": match["reviewed_accessions"], "matched_terms": match["terms"],
                        })
                writer.writerow(row)

    change_fields = ["sequence_id", "split", "task", "before", "after", "outcome", "confidence", "accessions", "reviewed_accessions", "matched_terms"]
    with gzip.open(staging / "uniprot_label_matches_and_changes.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=change_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)

    summary_rows = []
    for task in TASKS:
        item = counts[task]
        summary_rows.append({
            "task": task, "positive": item["1"], "source_control": item["0"], "unknown": item["unknown"],
            "unknown_to_positive": item["unknown_to_positive"],
            "source_control_to_conflict_unknown": item["source_control_to_conflict_unknown"],
            "positive_confirmed_by_uniprot": item["positive_confirmed_by_uniprot"],
        })
    with (staging / "label_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "version": VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "uniprot_endpoint": UNIPROT_ENDPOINT,
        "uniprot_release": release,
        "query": {"method": "exact canonical sequence MD5 checksum", "chunks": len(fetch_meta), "chunk_size": CHUNK_SIZE, "workers": WORKERS},
        "input_sha256": input_hashes,
        "coverage": dict(coverage),
        "summary": summary_rows,
        "rules": {
            "positive": "explicit task term in exact-sequence UniProtKB annotation fields",
            "negative": "UniProt absence never creates zero; original source controls remain zero",
            "unknown": "no explicit positive annotation and no original same-task source class",
            "conflict": "original source control plus UniProt positive becomes unknown for manual review",
            "organism_excluded": "organism name and habitat text are not searched for task terms",
        },
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# MarsLikePro 严格 unknown + UniProt 增强版

版本：`{VERSION}`；UniProt release：`{release}`。

对固定 40,000 条序列全部计算 MD5，通过 UniProtKB REST API 按精确序列查询。仅在蛋白名称、功能、催化活性、通路、诱导、关键词、GO、InterPro/Pfam 等注释中匹配预注册任务词。物种名和栖息地不用于自动标签。

- unknown + 明确 UniProt 任务注释 -> `1`。
- 原始来源对照 + UniProt 阳性 -> `unknown`（冲突待审）。
- 无匹配注释 -> 保持 unknown，绝不因“未注释”改为 0。
- reviewed/Swiss-Prot 和 unreviewed/TrEMBL 匹配分开记录。

六任务词规则固定在构建脚本 `PATTERNS`中，所有命中与变更见 `uniprot_label_matches_and_changes.csv.gz`。
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    generated = sorted(path for path in staging.rglob("*") if path.is_file())
    checksum_lines = [f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}" for path in generated]
    (staging / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

    for split, expected in [("train", 36000), ("validation", 2000), ("test", 2000)]:
        with (staging / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected}, got {len(rows)}")
        for row in rows:
            for task in TASKS:
                if row[task] not in {"0", "1", "unknown"}:
                    raise ValueError(f"Bad label {split}/{row['sequence_id']}/{task}")
                if (row[task] == "unknown") != (row[f"{task}_mask"] == "0"):
                    raise ValueError(f"Mask mismatch {split}/{row['sequence_id']}/{task}")
    staging.rename(OUTPUT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    splits, by_md5, input_hashes = read_inputs()
    checksums = sorted(by_md5)
    fetch_meta = fetch_all(checksums)
    annotations, release = parse_annotations(set(checksums))
    build_outputs(splits, annotations, release, input_hashes, fetch_meta)


if __name__ == "__main__":
    main()
