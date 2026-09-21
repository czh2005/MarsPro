#!/usr/bin/env python3
"""Audit database coverage and explicit negative evidence for the fixed 40k set."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
TASK_ROOTS = {
    "cold": {"GO:0009409"},
    "desiccation": {"GO:0009269", "GO:0009414"},
    "oxidative": {"GO:0006979"},
    "perchlorate": set(),
    # Broad "response to radiation" also contains ordinary light responses.
    # Restrict this task to ionizing-radiation response and DNA repair.
    "radiation": {"GO:0010212", "GO:0006281"},
    "salt": {"GO:0009651"},
}
NOT_QUALIFIERS = ",".join(
    [
        "NOT|part_of",
        "NOT|located_in",
        "NOT|is_active_in",
        "NOT|involved_in",
        "NOT|enables",
        "NOT|contributes_to",
        "NOT|colocalizes_with",
        "NOT|acts_upstream_of_or_within_negative_effect",
        "NOT|acts_upstream_of_or_within",
        "NOT|acts_upstream_of",
    ]
)
UNIPROT_FIELDS = [
    "accession",
    "reviewed",
    "xref_refseq",
    "xref_kegg",
    "xref_eggnog",
    "xref_pdb",
    "xref_alphafolddb",
    "xref_string",
    "rhea",
    "xref_interpro",
    "xref_pfam",
]
MANUAL_EXPERIMENTAL_GO = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_bytes(url: str, accept: str, retries: int = 4) -> bytes:
    op = opener()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": accept, "User-Agent": "MarsLikePro-database-audit/1.0"},
            )
            with op.open(req, timeout=90) as response:
                return response.read()
        except Exception as exc:  # network errors are retried and recorded by caller
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {url}: {last}")


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def cache_name(prefix: str, values: list[str], suffix: str) -> str:
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.{suffix}"


def fetch_uniprot_crossrefs(batch: list[str], cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / cache_name("uniprot_xref", batch, "tsv.gz")
    if not path.exists():
        query = "(" + " OR ".join(f"accession:{x}" for x in batch) + ")"
        params = urllib.parse.urlencode(
            {"query": query, "format": "tsv", "fields": ",".join(UNIPROT_FIELDS)}
        )
        raw = fetch_bytes(f"https://rest.uniprot.org/uniprotkb/stream?{params}", "text/tab-separated-values")
        with gzip.open(path, "wb") as handle:
            handle.write(raw)
    return pd.read_csv(path, sep="\t", compression="gzip", dtype=str).fillna("")


def fetch_quickgo_not(batch: list[str], cache_dir: Path) -> list[dict]:
    path = cache_dir / cache_name("quickgo_not", batch, "json.gz")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    results: list[dict] = []
    page = 1
    while True:
        params = urllib.parse.urlencode(
            {
                "geneProductId": ",".join(f"UniProtKB:{x}" for x in batch),
                "qualifier": NOT_QUALIFIERS,
                "limit": 100,
                "page": page,
            }
        )
        payload = json.loads(
            fetch_bytes(
                f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?{params}",
                "application/json",
            )
        )
        current = payload.get("results", [])
        results.extend(current)
        if len(results) >= int(payload.get("numberOfHits", 0)) or not current:
            break
        page += 1
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False)
    return results


def fetch_go_ancestors(go_id: str, cache_dir: Path) -> dict:
    path = cache_dir / f"{go_id.replace(':', '_')}.json"
    if not path.exists():
        raw = fetch_bytes(
            f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{urllib.parse.quote(go_id)}/ancestors",
            "application/json",
        )
        path.write_bytes(raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not rows:
        return {"go_id": go_id, "go_name": "", "ancestors": {go_id}}
    row = rows[0]
    return {
        "go_id": go_id,
        "go_name": row.get("name", ""),
        "ancestors": set(row.get("ancestors", [])) | {go_id},
    }


def semicolon_present(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    quickgo_cache = cache_dir / "quickgo_not"
    xref_cache = cache_dir / "uniprot_xrefs"
    go_cache = cache_dir / "go_ancestors"
    for path in [quickgo_cache, xref_cache, go_cache]:
        path.mkdir(parents=True, exist_ok=True)

    frames = []
    keep = ["sequence_id", "sequence", "id50_cluster_id", "uniprot_accessions"] + TASKS
    for split in ["train", "validation", "test"]:
        frame = pd.read_csv(args.input_dir / f"{split}.csv", usecols=keep, dtype=str).fillna("")
        frame.insert(1, "split", split)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != 40000 or data["sequence_id"].nunique() != 40000:
        raise ValueError("expected exactly 40,000 unique sequence_id values")

    accession_to_sequences: dict[str, set[str]] = {}
    for sequence_id, raw in data[["sequence_id", "uniprot_accessions"]].itertuples(index=False):
        for accession in [x.strip() for x in str(raw).split("|") if x.strip()]:
            accession_to_sequences.setdefault(accession, set()).add(sequence_id)
    accessions = sorted(accession_to_sequences)
    accession_batches = chunks(accessions, 100)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        xref_frames = list(pool.map(lambda b: fetch_uniprot_crossrefs(b, xref_cache), accession_batches))
    xrefs = pd.concat(xref_frames, ignore_index=True).drop_duplicates("Entry")
    xrefs.to_csv(args.output_dir / "database_crossrefs_by_uniprot_accession.csv.gz", index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        not_lists = list(pool.map(lambda b: fetch_quickgo_not(b, quickgo_cache), accession_batches))
    not_rows = [row for group in not_lists for row in group]
    not_df = pd.DataFrame(not_rows)
    if not not_df.empty:
        not_df = not_df.drop_duplicates("id")
    not_df.to_csv(args.output_dir / "quickgo_not_annotations.csv.gz", index=False)

    go_ids = sorted(set(not_df.get("goId", pd.Series(dtype=str)).dropna().astype(str)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        go_info = list(pool.map(lambda x: fetch_go_ancestors(x, go_cache), go_ids))
    go_map = {row["go_id"]: row for row in go_info}

    sequence_lookup = data.set_index("sequence_id").to_dict("index")
    candidates = []
    for row in not_rows:
        accession = str(row.get("geneProductId", "")).replace("UniProtKB:", "")
        info = go_map.get(str(row.get("goId", "")), {"go_name": "", "ancestors": set()})
        for task, roots in TASK_ROOTS.items():
            matched_roots = sorted(roots & info["ancestors"])
            if not matched_roots:
                continue
            for sequence_id in sorted(accession_to_sequences.get(accession, [])):
                sample = sequence_lookup[sequence_id]
                candidates.append(
                    {
                        "sequence_id": sequence_id,
                        "split": sample["split"],
                        "id50_cluster_id": sample["id50_cluster_id"],
                        "task": task,
                        "current_label": sample[task],
                        "uniprot_accession": accession,
                        "qualifier": row.get("qualifier", ""),
                        "go_id": row.get("goId", ""),
                        "go_name": info["go_name"],
                        "matched_task_root": "|".join(matched_roots),
                        "go_evidence": row.get("goEvidence", ""),
                        "evidence_code": row.get("evidenceCode", ""),
                        "reference": row.get("reference", ""),
                        "assigned_by": row.get("assignedBy", ""),
                        "annotation_date": row.get("date", ""),
                        "candidate_strength": (
                            "manual_experimental_not"
                            if row.get("goEvidence", "") in MANUAL_EXPERIMENTAL_GO
                            else "curated_or_inferred_not_needs_review"
                        ),
                        "proposed_action": "manual_review_do_not_auto_change_label",
                    }
                )
    candidate_df = pd.DataFrame(candidates)
    candidate_df.to_csv(args.output_dir / "quickgo_task_scoped_zero_candidates.csv", index=False)

    db_columns = {
        "RefSeq": "RefSeq",
        "KEGG": "KEGG",
        "eggNOG": "eggNOG",
        "PDB": "PDB",
        "AlphaFoldDB": "AlphaFoldDB",
        "STRING": "STRING",
        "Rhea": "Rhea ID",
        "InterPro": "InterPro",
        "Pfam": "Pfam",
    }
    coverage = []
    for database, column in db_columns.items():
        matched_accessions = int(semicolon_present(xrefs[column]).sum())
        matched_sequences = set()
        for accession in xrefs.loc[semicolon_present(xrefs[column]), "Entry"]:
            matched_sequences.update(accession_to_sequences.get(accession, set()))
        coverage.append(
            {
                "database": database,
                "matched_uniprot_accessions": matched_accessions,
                "matched_unique_sequences": len(matched_sequences),
                "can_directly_create_zero": "no",
                "primary_use": "identifier_or_mechanism_support",
            }
        )
    coverage.append(
        {
            "database": "GOA/QuickGO NOT",
            "matched_uniprot_accessions": int(
                not_df.get("geneProductId", pd.Series(dtype=str)).nunique()
            ),
            "matched_unique_sequences": int(candidate_df.get("sequence_id", pd.Series(dtype=str)).nunique()),
            "can_directly_create_zero": "candidate_only_requires_scope_review",
            "primary_use": "explicit_negative_function_assertion",
        }
    )
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(args.output_dir / "database_coverage_summary.csv", index=False)

    label_summary = pd.read_csv(args.input_dir / "label_summary.csv")
    existing_controls = int(label_summary["source_control"].sum())
    summary = {
        "version": "multidatabase_zero_evidence_audit_v1_20260910",
        "input_directory": str(args.input_dir.resolve()),
        "input_unique_sequences": 40000,
        "uniprot_accessions_queried": len(accessions),
        "quickgo_not_annotation_rows": len(not_df),
        "task_scoped_not_candidate_rows": len(candidate_df),
        "task_scoped_not_candidate_sequences": int(
            candidate_df.get("sequence_id", pd.Series(dtype=str)).nunique()
        ),
        "manual_experimental_task_scoped_candidate_rows": int(
            (candidate_df.get("candidate_strength", pd.Series(dtype=str)) == "manual_experimental_not").sum()
        ),
        "existing_source_control_label_cells": existing_controls,
        "automatic_label_changes": 0,
        "note": "Database absence never creates zero; QuickGO NOT candidates require target and condition review.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
