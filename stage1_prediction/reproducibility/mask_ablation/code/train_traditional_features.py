#!/usr/bin/env python3
"""Train auditable sequence-feature baselines on known labels only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

LABELS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
GROUPS = [set("GAVLMI"), set("FYW"), set("KRH"), set("DE"), set("STCPNQ")]
GROUP_INDEX = {aa: index for index, group in enumerate(GROUPS) for aa in group}
PROPERTIES = {
    "hydrophobic": set("AVILMFWY"),
    "charged": set("DEKRH"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "aromatic": set("FWY"),
    "small": set("AGSTCP"),
    "proline": set("P"),
    "glycine": set("G"),
    "cysteine": set("C"),
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["sequence_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate sequence_id in {path}")
    return rows


def parse_targets(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    masks = np.zeros_like(labels)
    for row_index, row in enumerate(rows):
        for task_index, task in enumerate(LABELS):
            value = str(row[task]).strip().lower()
            if value in {"0", "1"}:
                labels[row_index, task_index] = float(value)
                masks[row_index, task_index] = 1.0
            elif value not in {"", "unknown", "nan", "none"}:
                raise ValueError(f"unsupported state {value!r} for {task}")
    return labels, masks.astype(bool)


def sequence_features(sequence: str, max_gap: int) -> np.ndarray:
    sequence = "".join(aa for aa in sequence.upper() if aa in AA_INDEX)
    length = len(sequence)
    if not length:
        raise ValueError("empty canonical sequence")
    aa_indices = np.fromiter((AA_INDEX[aa] for aa in sequence), dtype=np.int16)
    counts = np.bincount(aa_indices, minlength=len(AA)).astype(np.float32)
    aac = counts / length

    dpc = np.zeros((len(AA), len(AA)), dtype=np.float32)
    if length > 1:
        np.add.at(dpc, (aa_indices[:-1], aa_indices[1:]), 1.0)
        dpc /= length - 1

    grouped = np.fromiter((GROUP_INDEX[aa] for aa in sequence), dtype=np.int16)
    cksaagp = []
    for gap in range(max_gap + 1):
        pairs = np.zeros((len(GROUPS), len(GROUPS)), dtype=np.float32)
        offset = gap + 1
        if length > offset:
            np.add.at(pairs, (grouped[:-offset], grouped[offset:]), 1.0)
            pairs /= length - offset
        cksaagp.append(pairs.ravel())

    fractions = np.asarray([
        sum(aa in residues for aa in sequence) / length
        for residues in PROPERTIES.values()
    ], dtype=np.float32)
    scalar = np.asarray([length, np.log1p(length)], dtype=np.float32)
    return np.concatenate([scalar, aac, dpc.ravel(), *cksaagp, fractions])


def build_matrix(rows: list[dict], max_gap: int) -> np.ndarray:
    return np.vstack([sequence_features(row["sequence"], max_gap) for row in rows]).astype(np.float32)


def best_mcc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scores = [matthews_corrcoef(y, p >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(labels: np.ndarray, masks: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    output = []
    for index, task in enumerate(LABELS):
        y = labels[masks[:, index], index].astype(int)
        p = probabilities[masks[:, index], index]
        threshold = best_mcc_threshold(y, p)
        output.append({
            "task": task,
            "count": int(len(y)),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "random_ap": float(y.mean()),
            "auprc": float(average_precision_score(y, p)),
            "auroc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, p >= threshold)),
            "mcc": float(matthews_corrcoef(y, p >= threshold)),
            "threshold": threshold,
        })
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["root"])
    run = root / "runs" / config["run_name"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.time()

    train_path = root / "data/v3/train.csv"
    validation_path = root / "data/v3/validation.csv"
    train_rows, validation_rows = read_rows(train_path), read_rows(validation_path)
    train_labels, train_masks = parse_targets(train_rows)
    validation_labels, validation_masks = parse_targets(validation_rows)
    train_x = build_matrix(train_rows, int(config["cksaagp_max_gap"]))
    validation_x = build_matrix(validation_rows, int(config["cksaagp_max_gap"]))

    models, probabilities = {}, np.zeros_like(validation_labels)
    for index, task in enumerate(LABELS):
        known = train_masks[:, index]
        if config["estimator"] == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=config["n_estimators"], learning_rate=config["learning_rate"],
                num_leaves=config["num_leaves"], max_depth=-1, subsample=0.9,
                colsample_bytree=0.8, random_state=config["seed"],
                n_jobs=config["n_jobs"], verbosity=-1)
        elif config["estimator"] == "random_forest":
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"], max_features="sqrt",
                min_samples_leaf=2, class_weight=None, random_state=config["seed"],
                n_jobs=config["n_jobs"])
        else:
            raise ValueError(f"unsupported estimator: {config['estimator']}")
        model.fit(train_x[known], train_labels[known, index].astype(int))
        probabilities[:, index] = model.predict_proba(validation_x)[:, 1]
        models[task] = model

    scored = metrics(validation_labels, validation_masks, probabilities)
    with (run / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    with (run / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sequence_id", *[f"p_{task}" for task in LABELS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, prediction in zip(validation_rows, probabilities):
            writer.writerow({"sequence_id": row["sequence_id"], **{
                f"p_{task}": float(prediction[index]) for index, task in enumerate(LABELS)}})
    joblib.dump(models, run / "models.joblib", compress=3)
    summary = {
        "status": "complete",
        "estimator": config["estimator"],
        "feature_count": int(train_x.shape[1]),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "macro_validation_auprc": float(np.mean([row["auprc"] for row in scored])),
        "elapsed_seconds": time.time() - started,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "test_evaluated": False,
    }
    (run / "complete.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
