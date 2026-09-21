#!/usr/bin/env python3
"""Generate the locked ten-run configuration set."""
from __future__ import annotations

import json
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/sddn/yyf_work/chenzhenghang/mars-tool"
BATCH = "mask_causal_traditional_20260912_141236"


def write(name: str, value: dict) -> None:
    path = LOCAL_ROOT / "configs" / f"{name}.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prott5(name: str, strategy: str, seed: int, gpus: int, missing_fraction: float | None = None) -> dict:
    config = {
        "protocol_version": "mask_causal_ablation_v1",
        "root": REMOTE_ROOT,
        "model_path": f"{REMOTE_ROOT}/models/prott5",
        "run_name": f"{BATCH}_{name}",
        "mode": "masked_bce",
        "mask_strategy": strategy,
        "target_task": "shared",
        "seed": seed,
        "epochs": 5,
        "microbatch_per_gpu": 1,
        "gradient_accumulation": 8 if gpus == 2 else 16,
        "effective_global_batch_size": 16,
        "adapter_lr": 0.0001,
        "head_lr": 0.0003,
        "weight_decay": 0.01,
        "class_balance": "none",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "precision": "float16",
        "num_workers": 2,
        "unknown_training_policy": "explicit_by_mask_strategy",
        "evaluation_policy": "validation_known_labels_only",
        "test_evaluated": False,
    }
    if missing_fraction is not None:
        config["artificial_missing_fraction"] = missing_fraction
    return config


def traditional(name: str, estimator: str) -> dict:
    return {
        "protocol_version": "traditional_sequence_features_v1",
        "root": REMOTE_ROOT,
        "run_name": f"{BATCH}_{name}",
        "estimator": estimator,
        "seed": 42,
        "cksaagp_max_gap": 5,
        "n_estimators": 400 if estimator == "lightgbm" else 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 12,
        "features": ["length", "AAC", "DPC", "CKSAAGP_gap0_to_5", "physicochemical_fractions"],
        "unknown_training_policy": "exclude_from_each_task_loss",
        "evaluation_policy": "validation_known_labels_only",
        "test_evaluated": False,
    }


experiments = [
    ("task_aware_seed42", "task_aware", 42, 2, None),
    ("task_aware_seed43", "task_aware", 43, 2, None),
    ("task_aware_seed44", "task_aware", 44, 2, None),
    ("random_matched_seed42", "random_matched", 42, 2, None),
    ("random_matched_seed43", "random_matched", 43, 2, None),
    ("random_matched_seed44", "random_matched", 44, 2, None),
    ("shuffled_rows_seed42", "shuffled_rows", 42, 2, None),
    ("artificial_missing25_seed42", "artificial_missing", 42, 1, 0.25),
]
for values in experiments:
    write(values[0], prott5(*values))
write("features_lightgbm_seed42", traditional("features_lightgbm_seed42", "lightgbm"))
write("features_rf_seed42", traditional("features_rf_seed42", "random_forest"))

registry = {
    "version": "mask_causal_traditional_registry_v1",
    "batch": BATCH,
    "run_count": 10,
    "deep_training_runs": 8,
    "cpu_baseline_runs": 2,
    "test_evaluated": False,
    "core_comparison_seeds": [42, 43, 44],
    "note": "Random and shuffled masks are negative controls and may select originally unknown positions as zero-coded control targets.",
    "experiments": [values[0] for values in experiments] + ["features_lightgbm_seed42", "features_rf_seed42"],
}
(LOCAL_ROOT / "configs" / "experiment_registry.json").write_text(
    json.dumps(registry, indent=2), encoding="utf-8")
