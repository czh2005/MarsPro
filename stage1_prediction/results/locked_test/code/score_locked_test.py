#!/usr/bin/env python3
"""Inference-only locked test runner with checkpoint and sentinel verification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    root = Path("/sddn/yyf_work/chenzhenghang/mars-tool")
    run = root / "runs/shared_masked_bce_20260910_234438"
    output = work / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    started = time.time()

    protocol = json.loads((work / "protocol.json").read_text(encoding="utf-8"))
    expected = protocol["expected_sha256"]
    assets = {
        "checkpoint": run / "best.pt",
        "model_config": run / "config.json",
        "training_code": work / "code/train_prott5_lora.py",
        "validation_sentinel_json": work / "inputs/validation_sentinel.json",
    }
    observed = {name: sha256(path) for name, path in assets.items()}
    mismatches = {name: [expected[name], value] for name, value in observed.items() if value != expected[name]}
    if mismatches:
        raise RuntimeError(f"asset hash mismatch: {mismatches}")

    sys.path.insert(0, str(work / "code"))
    from train_prott5_lora import Collator, LABELS, Predictor, predict, set_seed

    set_seed(42)
    torch.set_num_threads(2)
    cfg = json.loads(assets["model_config"].read_text())
    model = Predictor(cfg).to("cuda")
    checkpoint = torch.load(assets["checkpoint"], map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    required = {name for name, parameter in parameters.items() if parameter.requires_grad}
    supplied = set(checkpoint["trainable_state"])
    if required != supplied:
        raise RuntimeError({"missing": sorted(required - supplied), "extra": sorted(supplied - required)})
    with torch.no_grad():
        for name in sorted(required):
            state = checkpoint["trainable_state"][name]
            if parameters[name].shape != state.shape:
                raise RuntimeError(f"shape mismatch: {name}")
            parameters[name].copy_(state.to(parameters[name].device, dtype=parameters[name].dtype))
    if abs(checkpoint["validation_macro_auprc"] - 0.9770867003912059) >= 1e-10:
        raise RuntimeError("checkpoint validation identity mismatch")

    tokenizer = T5Tokenizer.from_pretrained(cfg["model_path"], local_files_only=True, do_lower_case=False)
    collator = Collator(tokenizer)
    sentinel_records = json.loads(assets["validation_sentinel_json"].read_text())
    sentinel = [
        {"index": index, **record, "labels": [0.0] * 6, "masks": [0.0] * 6}
        for index, record in enumerate(sentinel_records)
    ]
    actual = predict(model, sentinel, collator, 1, torch.device("cuda"))
    sentinel_error = float(
        np.max(np.abs(actual - np.asarray([record["expected"] for record in sentinel_records])))
    )
    if sentinel_error > 0.002:
        raise RuntimeError(f"validation sentinel mismatch: {sentinel_error}")

    input_path = work / "inputs/inference_sequences.csv"
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            rows.append(
                {
                    "index": index,
                    "sequence_id": record["sequence_id"],
                    "sequence": record["sequence"],
                    "labels": [0.0] * len(LABELS),
                    "masks": [0.0] * len(LABELS),
                }
            )
    if len(rows) != protocol["expected_test_rows"]:
        raise RuntimeError(f"input row count mismatch: {len(rows)}")
    destination = output / "test_probabilities.csv"
    if destination.exists():
        raise RuntimeError("refuse to overwrite existing sealed predictions")

    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id"] + [f"p_{label}" for label in LABELS])
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            probabilities = predict(model, batch, collator, 1, torch.device("cuda"))
            if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
                raise RuntimeError("invalid probability encountered")
            writer.writerows(
                [[record["sequence_id"], *values.tolist()] for record, values in zip(batch, probabilities)]
            )
            handle.flush()
            write_json(
                status_path,
                {
                    "state": "running",
                    "completed": min(offset + 32, len(rows)),
                    "total": len(rows),
                    "elapsed_seconds": time.time() - started,
                },
            )

    identity = {
        "state": "predictions_sealed",
        "row_count": len(rows),
        "predictions_sha256": sha256(destination),
        "blind_input_sha256": sha256(input_path),
        "checkpoint_sha256": observed["checkpoint"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_macro_ap": checkpoint["validation_macro_auprc"],
        "sentinel_count": len(sentinel_records),
        "sentinel_max_abs_error": sentinel_error,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "prediction_seal.json", identity)
    write_json(status_path, identity)
    print(json.dumps(identity), flush=True)


if __name__ == "__main__":
    main()
