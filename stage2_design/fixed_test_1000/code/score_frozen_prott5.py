"""Score sequence shards with the frozen stage-1 ProtT5-LoRA checkpoint."""
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


parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--shards", type=int, required=True)
parser.add_argument("--model-path", type=Path, required=True)
args = parser.parse_args()

work = args.work
sys.path.insert(0, str(work / "code"))
from train_prott5_lora import (  # noqa: E402
    Collator,
    LABELS,
    Predictor,
    atomic_json,
    predict,
    read_rows,
    set_seed,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


started = time.time()
output = work / "outputs"
output.mkdir(exist_ok=True)
status_path = output / f"shard_{args.shard}_status.json"

try:
    config_path = work / "model" / "config.json"
    checkpoint_path = work / "model" / "best.pt"
    config = json.loads(config_path.read_text())
    config["model_path"] = str(args.model_path)
    set_seed(42)
    torch.set_num_threads(2)

    model = Predictor(config).to("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    supplied = set(checkpoint["trainable_state"])
    assert expected == supplied, {
        "missing": sorted(expected - supplied),
        "extra": sorted(supplied - expected),
    }
    with torch.no_grad():
        for name in sorted(expected):
            state = checkpoint["trainable_state"][name]
            assert parameters[name].shape == state.shape, name
            parameters[name].copy_(state.to(parameters[name].device, dtype=parameters[name].dtype))

    tokenizer = T5Tokenizer.from_pretrained(
        config["model_path"], local_files_only=True, do_lower_case=False
    )
    collator = Collator(tokenizer)

    sentinel = json.loads((work / "inputs" / "validation_sentinel.json").read_text())
    sentinel_rows = [
        {"index": i, **row, "labels": [0.0] * 6, "masks": [0.0] * 6}
        for i, row in enumerate(sentinel)
    ]
    sentinel_predictions = predict(
        model, sentinel_rows, collator, 1, torch.device("cuda")
    )
    sentinel_error = float(
        np.max(
            np.abs(
                sentinel_predictions
                - np.asarray([row["expected"] for row in sentinel], dtype=float)
            )
        )
    )
    assert sentinel_error <= 0.002, f"Validation sentinel mismatch: {sentinel_error}"

    input_path = work / "inputs" / "inference_sequences.csv"
    identity = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "training_code_sha256": sha256(work / "code" / "train_prott5_lora.py"),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_macro_ap": checkpoint["validation_macro_auprc"],
        "trainable_tensors_loaded": len(expected),
        "sentinel_count": len(sentinel),
        "sentinel_max_abs_error": sentinel_error,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shard": args.shard,
        "shards": args.shards,
        "input_sha256": sha256(input_path),
        "model_path": str(args.model_path),
    }
    atomic_json(output / f"shard_{args.shard}_identity.json", identity)

    rows = read_rows(input_path, "shared", "masked_bce", False)[args.shard :: args.shards]
    destination = output / f"shard_{args.shard}_predictions.csv"
    assert not destination.exists(), "Refuse to overwrite an existing prediction shard"
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id"] + ["p_" + task for task in LABELS])
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            probabilities = predict(model, batch, collator, 1, torch.device("cuda"))
            assert np.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()
            writer.writerows(
                [row["sequence_id"], *scores.tolist()]
                for row, scores in zip(batch, probabilities)
            )
            handle.flush()
            atomic_json(
                status_path,
                {
                    "state": "running",
                    "completed": min(offset + 32, len(rows)),
                    "total": len(rows),
                    "seconds": time.time() - started,
                },
            )

    atomic_json(
        status_path,
        {
            "state": "complete",
            "completed": len(rows),
            "total": len(rows),
            "seconds": time.time() - started,
            "predictions_sha256": sha256(destination),
        },
    )
    print(json.dumps({"shard": args.shard, "complete": len(rows)}), flush=True)
except Exception as error:
    atomic_json(
        status_path,
        {"state": "failed", "error": repr(error), "seconds": time.time() - started},
    )
    raise
