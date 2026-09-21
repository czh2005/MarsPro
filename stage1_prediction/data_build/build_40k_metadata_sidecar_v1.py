from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\火星蛋白")
INPUT_DIR = ROOT / "AAA数据集" / "AAAESM模型训练4w"
DEFAULT_OUTPUT_DIR = ROOT / "AAA数据集" / "第二阶段来源区片的数据。"
CLEANROOM = ROOT / "审查" / "cleanroom_benchmark"
BROAD_DIR = CLEANROOM / "releases" / "issues_four_source_membership_doc_v1_20260908" / "provenance"
V3_DIR = CLEANROOM / "intermediate" / "label_definition_remap" / "v3_20260907"

TASKS = {
    "cold": "cold_adaptation",
    "desiccation": "desiccation_adaptation",
    "oxidative": "oxidative_stress",
    "perchlorate": "perchlorate_related",
    "radiation": "radiation_dna_repair",
    "salt": "salt_adaptation",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_values(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(split_values(item))
        return result
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            return split_values(json.loads(text))
        except json.JSONDecodeError:
            pass
    for delimiter in ("|", ";"):
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text]


def joined(values) -> str:
    unique = sorted({str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() != "nan"})
    return "|".join(unique)


def add_values(target: set[str], value) -> None:
    target.update(split_values(value))


def evidence_state(classes: set[str]) -> str:
    if not classes:
        return "none"
    positive = bool(classes & {"positive", "weak_positive"})
    negative = bool(classes & {"scoped_negative", "weak_scoped_negative"})
    if positive and negative:
        return "mixed_condition_or_subtarget"
    if "positive" in classes:
        return "verified_positive"
    if "scoped_negative" in classes:
        return "verified_scoped_negative"
    if "weak_positive" in classes:
        return "weak_positive_only"
    if "weak_scoped_negative" in classes:
        return "weak_scoped_negative_only"
    if classes == {"unknown"}:
        return "unknown_only"
    if classes <= {"unknown", "context_or_mismatch"}:
        return "unknown_or_target_mismatch"
    return "other"


def load_training_release() -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    hashes = {}
    expected = {"train": 36000, "validation": 2000, "test": 2000}
    for split, expected_rows in expected.items():
        path = INPUT_DIR / f"{split}.csv"
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != expected_rows:
            raise ValueError(f"{path.name}: expected {expected_rows} rows, found {len(frame)}")
        if "split" in frame.columns:
            frame = frame.drop(columns=["split"])
        frame.insert(1, "split", split)
        frames.append(frame)
        hashes[str(path)] = file_sha256(path)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != 40000:
        raise ValueError(f"Expected 40000 rows, found {len(data)}")
    if data["sequence_id"].duplicated().any():
        raise ValueError("sequence_id is not unique")
    normalized = data["sequence"].astype(str).str.strip().str.upper()
    if normalized.duplicated().any():
        raise ValueError("sequence is not unique in the 40k release")
    data["sequence_sha256"] = normalized.map(sha256_text)
    data["sequence_length"] = normalized.str.len()
    return data, hashes


def build_source_title_map() -> tuple[dict[str, str], pd.DataFrame]:
    registry_paths = [CLEANROOM / "source_registry_seed.csv", BROAD_DIR / "source_registry.csv"]
    rows = []
    title_map: dict[str, str] = {}
    for path in registry_paths:
        reg = pd.read_csv(path, low_memory=False).fillna("")
        for _, row in reg.iterrows():
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            title = str(row.get("title", row.get("source_name", row.get("source_family", "")))).strip()
            if title:
                title_map[source_id] = title
            rows.append({
                "source_id": source_id,
                "title": title,
                "top_label": str(row.get("top_label", row.get("task_scope", ""))).strip(),
                "source_type": str(row.get("source_type", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                "access_status": str(row.get("access_status", "")).strip(),
                "validation_status": str(row.get("validation_status", "")).strip(),
                "license_public_redistribution_status": str(row.get("license_public_redistribution_status", "")).strip(),
                "source_label_meaning": str(row.get("source_label_meaning", "")).strip(),
            })
    registry = pd.DataFrame(rows).drop_duplicates(subset=["source_id"], keep="last")
    return title_map, registry


def aggregate_v3(v3: pd.DataFrame) -> dict[str, dict]:
    per_sequence: dict[str, dict] = {}
    for sequence_hash, group in v3.groupby("sequence_sha256", dropna=False):
        sequence_hash = str(sequence_hash)
        if not sequence_hash or sequence_hash == "nan":
            continue
        item: dict = {
            "v3_evidence_matched": True,
            "v3_sequence_task_condition_rows": int(len(group)),
            "v3_source_ids": joined(group["source_id"].dropna()),
            "v3_paper_or_study_ids": joined(group["paper_or_study_id"].dropna()),
            "v3_study_lineage_count": int(group["study_lineage_id"].dropna().astype(str).nunique()),
            "v3_evidence_tasks": joined(group["task_id"].dropna()),
            "v3_any_individual_evidence_valid": bool(group["individual_evidence_valid"].map(bool_value).any()),
            "v3_any_strict_eligible": bool(group["strict_eligible"].map(bool_value).any()),
            "v3_any_release_eligible": bool(group["release_eligible"].map(bool_value).any()),
            "v3_evaluation_eligibility": joined(group["evaluation_eligibility"].dropna()),
        }
        for short, task_id in TASKS.items():
            task_group = group[group["task_id"].astype(str) == task_id]
            classes = set(task_group["mapped_support_class"].dropna().astype(str))
            item[f"{short}_evidence_rows"] = int(len(task_group))
            item[f"{short}_support_classes"] = joined(classes)
            item[f"{short}_evidence_state"] = evidence_state(classes)
            item[f"{short}_evidence_study_count"] = int(task_group["study_lineage_id"].dropna().astype(str).nunique())
            item[f"{short}_evidence_conditions"] = joined(task_group["condition_id"].dropna())
            item[f"{short}_evidence_subtargets"] = joined(task_group["subtarget_id"].dropna())
            item[f"{short}_individual_evidence_valid"] = bool(task_group["individual_evidence_valid"].map(bool_value).any())
            item[f"{short}_strict_eligible"] = bool(task_group["strict_eligible"].map(bool_value).any())
            item[f"{short}_release_eligible"] = bool(task_group["release_eligible"].map(bool_value).any())
        per_sequence[sequence_hash] = item
    return per_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data, input_hashes = load_training_release()
    title_map, registry = build_source_title_map()

    broad_path = BROAD_DIR / "exact_sequence_membership_catalog.parquet"
    tier_path = CLEANROOM / "intermediate" / "tiered_source_total" / "exact_sequences.parquet"
    occ_path = CLEANROOM / "intermediate" / "tiered_source_total" / "sequence_source_occurrences.parquet"
    tax_path = CLEANROOM / "intermediate" / "source_diversity" / "v5" / "available_sequence_taxonomy_links.csv"
    legacy_path = CLEANROOM / "legacy" / "marslike_multilabel_primary_balanced_masked_id90_split.csv"
    v3_path = V3_DIR / "sequence_target_status_v3.csv"

    broad = pd.read_parquet(broad_path).drop_duplicates("sequence_sha256", keep="first").set_index("sequence_sha256")
    tier = pd.read_parquet(tier_path).drop_duplicates("sequence_sha256", keep="first").set_index("sequence_sha256")
    occurrences = pd.read_parquet(occ_path)
    taxonomy = pd.read_csv(tax_path, low_memory=False)
    legacy = pd.read_csv(legacy_path, low_memory=False)
    legacy["computed_sha256"] = legacy["sequence"].astype(str).str.strip().str.upper().map(sha256_text)
    legacy = legacy.drop_duplicates("computed_sha256", keep="first").set_index("computed_sha256")
    v3 = pd.read_csv(v3_path, low_memory=False)
    v3_map = aggregate_v3(v3)

    occ_map: dict[str, dict[str, set[str] | int]] = defaultdict(lambda: {
        "source_ids": set(), "registry_scopes": set(), "task_scopes": set(), "endpoint_count": 0
    })
    for _, row in occurrences.iterrows():
        key = str(row["sequence_sha256"])
        add_values(occ_map[key]["source_ids"], row.get("source_id"))
        add_values(occ_map[key]["registry_scopes"], row.get("registry_scope"))
        add_values(occ_map[key]["task_scopes"], row.get("task_scope_json"))
        occ_map[key]["endpoint_count"] = max(int(occ_map[key]["endpoint_count"]), int(row.get("source_endpoint_count", 0)))

    tax_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: {
        "organisms": set(), "genera": set(), "taxids": set(), "source_ids": set(), "link_basis": set()
    })
    for _, row in taxonomy.iterrows():
        key = str(row["sequence_sha256"])
        add_values(tax_map[key]["organisms"], row.get("organism_name"))
        add_values(tax_map[key]["genera"], row.get("genus_name"))
        add_values(tax_map[key]["taxids"], row.get("taxid"))
        add_values(tax_map[key]["source_ids"], row.get("source_id"))
        add_values(tax_map[key]["link_basis"], row.get("link_basis"))

    rows = []
    for _, base in data.iterrows():
        seq_hash = base["sequence_sha256"]
        broad_row = broad.loc[seq_hash] if seq_hash in broad.index else None
        tier_row = tier.loc[seq_hash] if seq_hash in tier.index else None
        legacy_row = legacy.loc[seq_hash] if seq_hash in legacy.index else None
        occ = occ_map.get(seq_hash)
        tax = tax_map.get(seq_hash)

        match_sources = []
        if broad_row is not None:
            match_sources.append("broad_membership_catalog")
        if tier_row is not None:
            match_sources.append("cleanroom_tiered_total")
        if legacy_row is not None:
            match_sources.append("legacy_69144")

        source_ids: set[str] = set()
        registry_scopes: set[str] = set()
        task_scopes: set[str] = set()
        organisms: set[str] = set()
        genera: set[str] = set()
        taxids: set[str] = set()
        endpoint_counts = []

        if broad_row is not None:
            add_values(source_ids, broad_row.get("source_ids_json"))
            add_values(registry_scopes, broad_row.get("registry_scopes_json"))
            add_values(task_scopes, broad_row.get("task_scopes_json"))
            add_values(organisms, broad_row.get("organism_names_json"))
            add_values(genera, broad_row.get("genus_names_json"))
            add_values(taxids, broad_row.get("taxids_json"))
            endpoint_counts.append(int(broad_row.get("source_endpoint_count", 0)))
        if tier_row is not None:
            add_values(source_ids, tier_row.get("source_ids_json"))
            add_values(registry_scopes, tier_row.get("registry_scopes_json"))
            add_values(task_scopes, tier_row.get("task_scopes_json"))
            endpoint_counts.append(int(tier_row.get("source_endpoint_count", 0)))
        if occ:
            source_ids.update(occ["source_ids"])
            registry_scopes.update(occ["registry_scopes"])
            task_scopes.update(occ["task_scopes"])
            endpoint_counts.append(int(occ["endpoint_count"]))
        if tax:
            organisms.update(tax["organisms"])
            genera.update(tax["genera"])
            taxids.update(tax["taxids"])

        legacy_sources = ""
        legacy_labels = ""
        legacy_taxonomy = ""
        legacy_uniprot = ""
        if legacy_row is not None:
            legacy_sources = str(legacy_row.get("source_datasets", ""))
            legacy_labels = str(legacy_row.get("source_labels", ""))
            legacy_taxonomy = str(legacy_row.get("taxonomy_groups", ""))
            legacy_uniprot = str(legacy_row.get("uniprot_accessions", ""))

        if source_ids and (organisms or genera or taxids):
            resolution = "scientific_source_and_taxonomy"
        elif source_ids:
            resolution = "scientific_source_only"
        elif legacy_sources:
            resolution = "legacy_source_only"
        else:
            resolution = "coarse_import_only"
        review_reasons = []
        if not source_ids:
            review_reasons.append("no_canonical_scientific_source_id")
        if not (organisms or genera or taxids or legacy_taxonomy):
            review_reasons.append("taxonomy_unresolved")

        result = {
            "sequence_id": base["sequence_id"],
            "split": base["split"],
            "sequence_sha256": seq_hash,
            "sequence_length": int(base["sequence_length"]),
            "id50_cluster_id": base["id50_cluster_id"],
            "id50_cluster_size": int(base["id50_cluster_size"]),
            "coarse_import_buckets": base["source_datasets"],
            "label_coverage": int(base["label_coverage"]),
        }
        for short in TASKS:
            result[short] = int(base[short])
        result.update({
            "metadata_match_sources": joined(match_sources),
            "metadata_match_count": len(match_sources),
            "metadata_resolution_status": resolution,
            "metadata_review_needed": bool(review_reasons),
            "metadata_review_reason": joined(review_reasons),
            "scientific_source_ids": joined(source_ids),
            "scientific_source_count": len(source_ids),
            "scientific_source_titles": joined(title_map.get(source_id, source_id) for source_id in source_ids),
            "registry_scopes": joined(registry_scopes),
            "task_scopes": joined(task_scopes),
            "source_endpoint_count_max": max(endpoint_counts) if endpoint_counts else 0,
            "legacy_source_datasets": legacy_sources,
            "legacy_source_labels": legacy_labels,
            "legacy_uniprot_accessions": legacy_uniprot,
            "organism_names": joined(organisms),
            "genus_names": joined(genera),
            "taxids": joined(taxids),
            "legacy_taxonomy_groups": legacy_taxonomy,
            "taxonomy_resolved": bool(organisms or genera or taxids or legacy_taxonomy),
            "taxonomy_link_basis": joined(tax["link_basis"]) if tax else "",
            "relation_component_id": broad_row.get("relation_component_id", "") if broad_row is not None else "",
            "relation_component_size": broad_row.get("relation_component_size", "") if broad_row is not None else "",
            "primary_pfam_accession": broad_row.get("primary_pfam_accession", "") if broad_row is not None else "",
            "architecture_id": broad_row.get("architecture_id", "") if broad_row is not None else "",
            "has_pfam_annotation": bool_value(broad_row.get("has_pfam_annotation")) if broad_row is not None else False,
            "source_component_block_id": broad_row.get("source_component_block_id", "") if broad_row is not None else "",
            "source_component_split": broad_row.get("source_component_split", "") if broad_row is not None else "",
            "organism_component_block_id": broad_row.get("organism_component_block_id", "") if broad_row is not None else "",
            "organism_component_split": broad_row.get("organism_component_split", "") if broad_row is not None else "",
            "reserved_direct_challenge": bool_value(broad_row.get("reserved_direct_challenge")) if broad_row is not None else False,
            "membership_semantics": "source_or_proteome_membership_not_direct_tolerance_proof",
            "zero_semantics": "not_member_of_this_task_source_scope_not_biological_negative",
        })

        ev = v3_map.get(seq_hash, {})
        result.update({
            "v3_evidence_matched": ev.get("v3_evidence_matched", False),
            "v3_sequence_task_condition_rows": ev.get("v3_sequence_task_condition_rows", 0),
            "v3_source_ids": ev.get("v3_source_ids", ""),
            "v3_paper_or_study_ids": ev.get("v3_paper_or_study_ids", ""),
            "v3_study_lineage_count": ev.get("v3_study_lineage_count", 0),
            "v3_evidence_tasks": ev.get("v3_evidence_tasks", ""),
            "v3_any_individual_evidence_valid": ev.get("v3_any_individual_evidence_valid", False),
            "v3_any_strict_eligible": ev.get("v3_any_strict_eligible", False),
            "v3_any_release_eligible": ev.get("v3_any_release_eligible", False),
            "v3_evaluation_eligibility": ev.get("v3_evaluation_eligibility", ""),
        })
        for short in TASKS:
            for suffix, default in (
                ("evidence_rows", 0), ("support_classes", ""), ("evidence_state", "none"),
                ("evidence_study_count", 0), ("evidence_conditions", ""), ("evidence_subtargets", ""),
                ("individual_evidence_valid", False), ("strict_eligible", False), ("release_eligible", False),
            ):
                key = f"{short}_{suffix}"
                result[key] = ev.get(key, default)
        result["six_task_evidence_states"] = joined(
            f"{short}={result[f'{short}_evidence_state']}" for short in TASKS
        )
        rows.append(result)

    sidecar = pd.DataFrame(rows)
    if len(sidecar) != 40000 or sidecar["sequence_id"].nunique() != 40000:
        raise ValueError("Output cardinality validation failed")
    if (sidecar["metadata_match_count"] == 0).any():
        raise ValueError("At least one 40k sequence is unmatched by all metadata systems")

    basename = "MarsLikePro_4万条元数据侧表_v1_20260909"
    csv_path = output_dir / f"{basename}.csv"
    sidecar.to_csv(csv_path, index=False, encoding="utf-8-sig")

    workbook_columns = [
        "sequence_id", "split", "sequence_sha256", "sequence_length", "id50_cluster_id", "coarse_import_buckets",
        "cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt",
        "scientific_source_ids", "scientific_source_titles", "legacy_source_datasets",
        "organism_names", "genus_names", "primary_pfam_accession",
        "v3_evidence_matched", "v3_any_individual_evidence_valid", "v3_any_strict_eligible",
        "six_task_evidence_states",
    ]
    workbook_view_path = output_dir / "Excel人工阅读字段视图.csv"
    sidecar[workbook_columns].to_csv(workbook_view_path, index=False, encoding="utf-8-sig")

    field_rows = []
    chinese = {
        "sequence_id": "序列ID", "split": "数据划分", "sequence_sha256": "序列SHA-256",
        "sequence_length": "序列长度", "id50_cluster_id": "ID50簇ID", "id50_cluster_size": "ID50簇大小",
        "coarse_import_buckets": "导入批次", "label_coverage": "正标签数量",
        "metadata_match_sources": "元数据匹配系统", "metadata_resolution_status": "元数据解析状态",
        "scientific_source_ids": "科学来源ID", "scientific_source_titles": "科学来源名称",
        "organism_names": "物种名称", "genus_names": "属名", "taxids": "TaxID",
        "primary_pfam_accession": "主要Pfam", "relation_component_id": "关系组件ID",
        "v3_evidence_matched": "是否匹配v3证据", "v3_sequence_task_condition_rows": "v3序列-任务-条件记录数",
    }
    for column in sidecar.columns:
        if column in TASKS:
            meaning = "宽候选来源归属标签；1表示属于该任务来源范围，0不是生物学阴性"
            source = "当前4万训练发布"
        elif column.endswith("_support_classes"):
            meaning = "该任务下保留的全部证据支持类别；不同条件并存时不强行折叠"
            source = "v3 sequence_target_status"
        elif column.endswith("_evidence_state"):
            meaning = "供人工浏览的证据状态摘要，不替代原始条件化记录"
            source = "v3规则汇总"
        elif column.startswith(tuple(f"{x}_" for x in TASKS)):
            meaning = "对应任务的条件化证据汇总字段"
            source = "v3 sequence_target_status"
        elif column == "six_task_evidence_states":
            meaning = "六个任务的证据状态紧凑汇总；完整条件化字段见同一行其他证据列"
            source = "v3规则汇总"
        elif column.startswith("v3_"):
            meaning = "六类标签重映射v3的序列级证据汇总"
            source = "v3 sequence_target_status"
        elif column.startswith("legacy_"):
            meaning = "旧69,144数据中按精确序列匹配得到的历史元数据，仅作溯源"
            source = "legacy 69144"
        elif column in {"metadata_review_needed", "metadata_review_reason"}:
            meaning = "元数据缺口提示；不等于标签无效"
            source = "本构建规则"
        elif column in {"membership_semantics", "zero_semantics"}:
            meaning = "宽标签语义边界"
            source = "项目规则"
        else:
            meaning = chinese.get(column, "序列级来源、分类、组件或证据元数据")
            source = "当前4万/宽来源目录/clean-room总库"
        field_rows.append({
            "field": column,
            "中文名称": chinese.get(column, column),
            "data_type": str(sidecar[column].dtype),
            "说明": meaning,
            "主要来源": source,
        })
    dictionary = pd.DataFrame(field_rows)
    dictionary_path = output_dir / "字段字典.csv"
    dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")

    matched_registry_ids = set()
    for value in sidecar["scientific_source_ids"]:
        matched_registry_ids.update(split_values(value))
    source_registry_out = registry[registry["source_id"].isin(matched_registry_ids)].copy()
    unresolved = sorted(matched_registry_ids - set(source_registry_out["source_id"]))
    if unresolved:
        source_registry_out = pd.concat([source_registry_out, pd.DataFrame({"source_id": unresolved})], ignore_index=True)
    source_registry_out = source_registry_out.fillna("").sort_values("source_id")
    registry_path = output_dir / "来源登记表.csv"
    source_registry_out.to_csv(registry_path, index=False, encoding="utf-8-sig")

    coverage_by_split = []
    for split, group in sidecar.groupby("split", sort=False):
        coverage_by_split.append({
            "split": split,
            "rows": int(len(group)),
            "scientific_source_id_rows": int((group["scientific_source_count"] > 0).sum()),
            "taxonomy_resolved_rows": int(group["taxonomy_resolved"].sum()),
            "v3_evidence_matched_rows": int(group["v3_evidence_matched"].sum()),
            "v3_individual_valid_rows": int(group["v3_any_individual_evidence_valid"].sum()),
            "v3_strict_rows": int(group["v3_any_strict_eligible"].sum()),
        })
    source_combinations = sidecar["coarse_import_buckets"].value_counts().rename_axis("coarse_import_buckets").reset_index(name="rows").to_dict("records")
    summary = {
        "version": "metadata_sidecar_v1_20260909",
        "generated_at": datetime.now().astimezone().isoformat(),
        "row_count": int(len(sidecar)),
        "unique_sequence_id_count": int(sidecar["sequence_id"].nunique()),
        "unique_sequence_sha256_count": int(sidecar["sequence_sha256"].nunique()),
        "split_counts": sidecar["split"].value_counts().to_dict(),
        "metadata_system_coverage": {
            "broad_membership_catalog": int(sidecar["metadata_match_sources"].str.contains("broad_membership_catalog", regex=False).sum()),
            "cleanroom_tiered_total": int(sidecar["metadata_match_sources"].str.contains("cleanroom_tiered_total", regex=False).sum()),
            "legacy_69144": int(sidecar["metadata_match_sources"].str.contains("legacy_69144", regex=False).sum()),
            "union": int((sidecar["metadata_match_count"] > 0).sum()),
        },
        "scientific_source_id_rows": int((sidecar["scientific_source_count"] > 0).sum()),
        "taxonomy_resolved_rows": int(sidecar["taxonomy_resolved"].sum()),
        "pfam_annotated_rows": int(sidecar["has_pfam_annotation"].sum()),
        "v3_evidence_matched_rows": int(sidecar["v3_evidence_matched"].sum()),
        "v3_individual_valid_rows": int(sidecar["v3_any_individual_evidence_valid"].sum()),
        "v3_strict_rows": int(sidecar["v3_any_strict_eligible"].sum()),
        "reserved_direct_challenge_rows": int(sidecar["reserved_direct_challenge"].sum()),
        "coverage_by_split": coverage_by_split,
        "coarse_import_bucket_counts": source_combinations,
        "input_sha256": input_hashes,
        "semantic_guards": [
            "six binary labels are broad source/proteome membership labels",
            "zero is not a biological negative",
            "unknown evidence is not converted to zero",
            "v3 evidence fields are supplementary and do not overwrite the six training labels",
        ],
    }
    summary_path = output_dir / "构建摘要.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# MarsLikePro 4万条元数据侧表 v1

本目录是当前 40,000 条六标签数据的序列级元数据侧表。主键为 `sequence_id`，并提供 `sequence_sha256` 供精确序列核对。原始 `train.csv`、`validation.csv`、`test.csv` 未修改。

## 主要文件

- `{csv_path.name}`：40,000 行主侧表，一条序列一行。
- `{basename}.xlsx`：便于人工筛选和浏览的 Excel 版本，主表保留 22 个核心字段。
- `{workbook_view_path.name}`：Excel 主表所用的核心字段 CSV；完整 110 字段仍在主侧表 CSV。
- `{dictionary_path.name}`：字段字典。
- `{registry_path.name}`：侧表中出现的规范来源登记。
- `{summary_path.name}`：覆盖率、输入哈希和验证统计。

## 标签边界

六个 0/1 标签表示宽候选来源或蛋白组归属。`1` 不是逐蛋白直接耐受实验结论，`0` 也不是生物学阴性。v3 证据字段独立保存直接、弱、限定负例和未知状态，不覆盖宽标签。

## 使用建议

建模时用 `sequence_id` 与原三份数据连接。来源、物种、组件、Pfam 和证据字段可用于来源去偏、分组抽样和误差分析；不要把 `metadata_review_needed` 或缺失分类信息直接当负类。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    hash_targets = [csv_path, workbook_view_path, dictionary_path, registry_path, summary_path, output_dir / "README.md"]
    workbook_path = output_dir / f"{basename}.xlsx"
    if workbook_path.exists():
        hash_targets.append(workbook_path)
    output_hashes = {
        path.name: file_sha256(path)
        for path in hash_targets
    }
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in sorted(output_hashes.items())) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
