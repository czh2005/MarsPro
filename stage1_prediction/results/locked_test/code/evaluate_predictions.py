#!/usr/bin/env python3
"""Evaluate MarsLikePro predictions with unknown masking and provenance strata."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)

LABELS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]


def state(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).str.strip().str.lower().replace({"": "unknown", "nan": "unknown"})


def tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item.strip() for item in re.split(r"[;|]", str(value)) if item.strip()}


def seen_status(value: object, training_values: set[str]) -> str:
    current = tokens(value)
    if not current:
        return "unknown"
    return "seen" if current & training_values else "unseen"


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scores = [matthews_corrcoef(y, p >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return result


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    result = {
        "count": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "positive_prevalence": float(y.mean()) if len(y) else None,
        "threshold": threshold,
    }
    if not len(y):
        return result
    predicted = p >= threshold
    result.update({
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "positive_recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "negative_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": ece(y, p),
        "mean_probability": float(p.mean()),
    })
    if len(np.unique(y)) == 2:
        ap = float(average_precision_score(y, p))
        prevalence = float(y.mean())
        result.update({
            "auprc": ap,
            "auroc": float(roc_auc_score(y, p)),
            "random_ap": prevalence,
            "ap_minus_random": ap - prevalence,
            "ap_lift_over_random": ap / prevalence if prevalence else None,
        })
    return result


def load_predictions(path: Path, tasks: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".npz":
        values = np.load(path, allow_pickle=False)
        probabilities = values["probabilities"]
        if probabilities.ndim == 1:
            probabilities = probabilities[:, None]
        if probabilities.shape[1] != len(tasks):
            raise ValueError("prediction width does not match tasks")
        frame = pd.DataFrame({"sequence_id": values["sequence_ids"].astype(str)})
        for index, task in enumerate(tasks):
            frame[f"p_{task}"] = probabilities[:, index]
        return frame
    frame = pd.read_csv(path, dtype={"sequence_id": str})
    required = {"sequence_id", *[f"p_{task}" for task in tasks]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    return frame[["sequence_id", *[f"p_{task}" for task in tasks]]]


def calibration_rows(task: str, y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        rows.append({
            "task": task, "bin": index, "lower": lower, "upper": upper,
            "count": int(selected.sum()),
            "mean_probability": float(p[selected].mean()) if selected.any() else None,
            "observed_positive_rate": float(y[selected].mean()) if selected.any() else None,
        })
    return rows


def grouped_bootstrap(task: str, split: str, method: str, y: np.ndarray, p: np.ndarray,
                      groups: np.ndarray, threshold: float, grouping: str,
                      repeats: int = 1000, seed: int = 42) -> dict:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {"auprc": [], "auroc": [], "f1": [], "mcc": []}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        sy, sp = y[selected], p[selected]
        if len(np.unique(sy)) < 2:
            continue
        predicted = sp >= threshold
        values["auprc"].append(float(average_precision_score(sy, sp)))
        values["auroc"].append(float(roc_auc_score(sy, sp)))
        values["f1"].append(float(f1_score(sy, predicted, zero_division=0)))
        values["mcc"].append(float(matthews_corrcoef(sy, predicted)))
    result = {"method": method, "split": split, "task": task, "grouping": grouping,
              "group_count": len(unique), "requested_repeats": repeats,
              "valid_repeats": len(values["auprc"])}
    for metric, observed in values.items():
        if observed:
            result[f"{metric}_p025"] = float(np.quantile(observed, 0.025))
            result[f"{metric}_median"] = float(np.quantile(observed, 0.5))
            result[f"{metric}_p975"] = float(np.quantile(observed, 0.975))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=LABELS, default=LABELS)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--min-stratum-count", type=int, default=10)
    args = parser.parse_args()

    root = args.package_root.resolve()
    labels = pd.read_csv(root / "data" / "v3" / f"{args.split}.csv", dtype=str, low_memory=False)
    metadata_columns = ["sequence_id", "scientific_source_ids", "organism_names", "genus_names",
                        "primary_pfam_accession", "relation_component_id", "relation_component_size"]
    metadata = pd.read_csv(args.metadata, usecols=metadata_columns, dtype=str, low_memory=False)
    train_ids = set(pd.read_csv(root / "data" / "v3" / "train.csv", usecols=["sequence_id"], dtype=str)["sequence_id"])
    train_metadata = metadata[metadata["sequence_id"].isin(train_ids)]
    predictions = load_predictions(args.predictions, args.tasks)
    if predictions["sequence_id"].duplicated().any():
        raise ValueError("duplicate sequence_id in predictions")
    frame = labels.merge(predictions, on="sequence_id", how="inner", validate="one_to_one")
    frame = frame.merge(metadata, on="sequence_id", how="left", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError(f"prediction coverage is {len(frame)}/{len(labels)}")
    frame["length_bin"] = pd.cut(frame["sequence"].str.len(), [0, 100, 200, 400, 800, np.inf],
                                 labels=["30-100", "101-200", "201-400", "401-800", "801-1024"])
    for field, output_field in [("scientific_source_ids", "source_seen"), ("organism_names", "organism_seen"),
                                ("genus_names", "genus_seen"), ("primary_pfam_accession", "pfam_seen")]:
        training_values = set().union(*(tokens(value) for value in train_metadata[field]))
        frame[output_field] = frame[field].map(lambda value: seen_status(value, training_values))
    freeze_report = json.loads((root / "reports" / "v3_freeze_and_internal_split.json").read_text(encoding="utf-8"))
    leaked_relations = set(freeze_report["outer_split_overlap"]["relation_component_id"]["group_ids"])
    frame["outer_relation_component_overlap"] = frame["relation_component_id"].isin(leaked_relations)

    if args.thresholds:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    elif args.split == "validation":
        thresholds = {}
    else:
        raise ValueError("test evaluation requires frozen validation thresholds")

    per_label, calibration, zero_rows, stratum_rows, relation_strict_rows, uncertainty_rows = [], [], [], [], [], []
    known_arrays, probability_arrays, prediction_arrays = [], [], []
    for task in args.tasks:
        task_state = state(frame[task])
        known = task_state.isin(["0", "1"]).to_numpy()
        y = task_state[known].astype(int).to_numpy()
        p = frame.loc[known, f"p_{task}"].astype(float).to_numpy()
        threshold = float(thresholds.get(task, best_threshold(y, p)))
        thresholds[task] = threshold
        result = binary_metrics(y, p, threshold)
        result.update({"method": args.method, "split": args.split, "task": task,
                       "total_count": len(frame), "known_coverage": float(known.mean())})
        per_label.append(result)
        calibration.extend(calibration_rows(task, y, p))
        known_arrays.append(y)
        probability_arrays.append(p)
        prediction_arrays.append((p >= threshold).astype(int))

        strict_known = known & ~frame["outer_relation_component_overlap"].to_numpy()
        strict_y = task_state[strict_known].astype(int).to_numpy()
        strict_p = frame.loc[strict_known, f"p_{task}"].astype(float).to_numpy()
        strict_result = binary_metrics(strict_y, strict_p, threshold)
        strict_result.update({"method": args.method, "split": args.split, "task": task,
                              "excluded_relation_overlap_sequences": int((known & ~strict_known).sum())})
        relation_strict_rows.append(strict_result)

        known_frame = frame.loc[known].copy()
        relation_groups = known_frame["relation_component_id"].fillna("").astype(str)
        relation_groups = np.where(relation_groups != "", relation_groups,
                                   "SEQ|" + known_frame["sequence_id"].astype(str))
        source_groups = known_frame["scientific_source_ids"].fillna("").astype(str)
        missing_source_groups = np.char.add(
            "MISSING|", np.asarray(relation_groups, dtype=str)
        )
        source_groups = np.where(
            source_groups != "", "SOURCE|" + source_groups, missing_source_groups
        )
        uncertainty_rows.append(grouped_bootstrap(task, args.split, args.method, y, p,
                                                  np.asarray(relation_groups), threshold,
                                                  "relation_component", seed=42 + LABELS.index(task)))
        uncertainty_rows.append(grouped_bootstrap(task, args.split, args.method, y, p,
                                                  np.asarray(source_groups), threshold,
                                                  "scientific_source_or_relation_fallback",
                                                  seed=142 + LABELS.index(task)))

        negatives = frame[task_state.eq("0")].copy()
        origin_field = f"{task}_label_origin"
        for origin, chunk in negatives.groupby(origin_field, dropna=False):
            values = chunk[f"p_{task}"].astype(float).to_numpy()
            zero_rows.append({
                "method": args.method, "split": args.split, "task": task,
                "zero_origin": str(origin) if pd.notna(origin) else "missing",
                "count": len(chunk), "mean_probability": float(values.mean()),
                "false_positive_rate": float((values >= threshold).mean()),
                "specificity": float((values < threshold).mean()),
            })

        for field in ["source_seen", "organism_seen", "genus_seen", "pfam_seen", "scientific_source_ids",
                      "organism_names", "genus_names", "primary_pfam_accession", "length_bin"]:
            subset = frame.loc[known, [field, task, f"p_{task}"]].copy()
            subset[field] = subset[field].astype(object).fillna("unknown").astype(str)
            for value, chunk in subset.groupby(field):
                if len(chunk) < args.min_stratum_count:
                    continue
                sy = chunk[task].astype(int).to_numpy()
                sp = chunk[f"p_{task}"].astype(float).to_numpy()
                metrics = binary_metrics(sy, sp, threshold)
                metrics.update({"method": args.method, "split": args.split, "task": task,
                                "stratum_type": field, "stratum": value})
                stratum_rows.append(metrics)

    flat_y = np.concatenate(known_arrays)
    flat_p = np.concatenate(probability_arrays)
    flat_predicted = np.concatenate(prediction_arrays)
    micro = binary_metrics(flat_y, flat_p, threshold=0.5)
    micro.update({
        "f1": float(f1_score(flat_y, flat_predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(flat_y, flat_predicted)),
        "positive_recall": float(recall_score(flat_y, flat_predicted, pos_label=1, zero_division=0)),
        "negative_recall": float(recall_score(flat_y, flat_predicted, pos_label=0, zero_division=0)),
    })
    numeric = [row for row in per_label if "auprc" in row]
    overall = {
        "method": args.method, "split": args.split,
        "macro_auprc": float(np.mean([row["auprc"] for row in numeric])),
        "macro_auroc": float(np.mean([row["auroc"] for row in numeric])),
        "macro_f1": float(np.mean([row["f1"] for row in numeric])),
        "macro_mcc": float(np.mean([row["mcc"] for row in numeric])),
        "macro_ap_minus_random": float(np.mean([row["ap_minus_random"] for row in numeric])),
        "micro": micro, "thresholds": thresholds,
        "unknown_counted_as_negative": False,
    }

    positive_count = pd.DataFrame({task: state(frame[task]).eq("1").astype(int) for task in args.tasks}).sum(axis=1)
    frame["known_positive_count_group"] = positive_count.map(lambda value: "0" if value == 0 else "1" if value == 1 else "2+")
    combo_rows = []
    for group, indices in frame.groupby("known_positive_count_group").groups.items():
        ys, ps, preds = [], [], []
        for task in args.tasks:
            task_state = state(frame.loc[indices, task])
            known = task_state.isin(["0", "1"]).to_numpy()
            if not known.any():
                continue
            y = task_state[known].astype(int).to_numpy()
            p = frame.loc[indices, f"p_{task}"].to_numpy(dtype=float)[known]
            ys.append(y); ps.append(p); preds.append((p >= thresholds[task]).astype(int))
        if ys:
            y = np.concatenate(ys); p = np.concatenate(ps); predicted = np.concatenate(preds)
            row = binary_metrics(y, p, 0.5)
            row.update({"method": args.method, "split": args.split, "known_positive_group": group,
                        "sequence_count": len(indices), "f1": float(f1_score(y, predicted, zero_division=0)),
                        "mcc": float(matthews_corrcoef(y, predicted))})
            combo_rows.append(row)

    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(per_label).to_csv(args.output / "per_label_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(args.output / "calibration_bins.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(args.output / "zero_origin_metrics.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(args.output / "stratified_metrics.csv", index=False)
    pd.DataFrame(relation_strict_rows).to_csv(args.output / "relation_component_strict_metrics.csv", index=False)
    pd.DataFrame(uncertainty_rows).to_csv(args.output / "uncertainty_intervals.csv", index=False)
    pd.DataFrame(combo_rows).to_csv(args.output / "multilabel_group_metrics.csv", index=False)
    frame.to_csv(args.output / "predictions_enriched.csv", index=False)
    (args.output / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    with pd.ExcelWriter(args.output / "evaluation_tables.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(per_label).to_excel(writer, sheet_name="per_label", index=False)
        pd.DataFrame(zero_rows).to_excel(writer, sheet_name="zero_origin", index=False)
        pd.DataFrame(combo_rows).to_excel(writer, sheet_name="multilabel", index=False)
        pd.DataFrame(stratum_rows).to_excel(writer, sheet_name="strata", index=False)
        pd.DataFrame(relation_strict_rows).to_excel(writer, sheet_name="relation_strict", index=False)
        pd.DataFrame(uncertainty_rows).to_excel(writer, sheet_name="uncertainty", index=False)
    print(json.dumps(overall, ensure_ascii=False))


if __name__ == "__main__":
    main()
