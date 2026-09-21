#!/usr/bin/env python3
"""Train shared or task-specific ProtT5-LoRA with explicit unknown handling."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import T5EncoderModel, T5Tokenizer

LABELS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def parse_state(value: object) -> tuple[float, float]:
    state = str(value).strip().lower()
    if state == "1":
        return 1.0, 1.0
    if state == "0":
        return 0.0, 1.0
    if state in {"unknown", "", "nan", "none"}:
        return 0.0, 0.0
    raise ValueError(f"unsupported label state: {value}")


def read_rows(path: Path, target_task: str, mode: str, training: bool) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            sequence = "".join(residue if residue in AA else "X" for residue in row["sequence"].upper())
            if not sequence or len(sequence) > 1024:
                raise ValueError(f"unsupported sequence {row['sequence_id']}")
            tasks = LABELS if target_task == "shared" else [target_task]
            labels, masks = zip(*(parse_state(row[task]) for task in tasks))
            labels, masks = list(labels), list(masks)
            if mode == "bce_zero_fill":
                masks = [1.0] * len(masks)
            if target_task != "shared" and training and not masks[0]:
                continue
            rows.append({"index": index, "sequence_id": row["sequence_id"], "sequence": sequence,
                         "labels": labels, "masks": masks})
    if len({row["sequence_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate sequence_id in {path}")
    return rows


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict]) -> dict:
        encoded = self.tokenizer([" ".join(row["sequence"]) for row in rows], padding=True,
                                 truncation=False, return_tensors="pt", return_special_tokens_mask=True)
        residue_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
        expected = torch.tensor([len(row["sequence"]) for row in rows])
        if not torch.equal(residue_mask.sum(1), expected):
            raise ValueError("tokenizer changed residue coverage")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "residue_mask": residue_mask,
            "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
            "masks": torch.tensor([row["masks"] for row in rows], dtype=torch.float32),
        }


class Predictor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dtype = torch.float16 if config["precision"] == "float16" else torch.float32
        encoder = T5EncoderModel.from_pretrained(config["model_path"], local_files_only=True, torch_dtype=dtype)
        encoder.config.use_cache = False
        self.encoder = get_peft_model(encoder, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=["q", "v"], bias="none"))
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.encoder.enable_input_require_grads()
        outputs = len(LABELS) if config["target_task"] == "shared" else 1
        self.head = nn.Sequential(nn.LayerNorm(encoder.config.d_model), nn.Linear(encoder.config.d_model, 256),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(256, outputs))

    def encode(self, batch: dict) -> torch.Tensor:
        hidden = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        mask = batch["residue_mask"].unsqueeze(-1)
        return (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encode(batch))


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def known_metrics(rows: list[dict], probabilities: np.ndarray, tasks: list[str], thresholds=None) -> tuple[list[dict], dict]:
    results, selected = [], {}
    labels = np.asarray([row["labels"] for row in rows])
    masks = np.asarray([row["masks"] for row in rows]).astype(bool)
    for index, task in enumerate(tasks):
        y, p = labels[masks[:, index], index].astype(int), probabilities[masks[:, index], index]
        if len(np.unique(y)) < 2:
            results.append({"task": task, "count": len(y), "status": "single_class"})
            continue
        if thresholds is None:
            candidates = np.unique(np.r_[0.0, p, 1.0])
            threshold = float(candidates[int(np.argmax([matthews_corrcoef(y, p >= t) for t in candidates]))])
        else:
            threshold = float(thresholds[task])
        selected[task] = threshold
        prevalence = float(y.mean())
        auprc = float(average_precision_score(y, p))
        results.append({"task": task, "count": int(len(y)), "positive": int(y.sum()),
                        "negative": int(len(y) - y.sum()),
                        "total_count": int(len(rows)),
                        "known_coverage": float(len(y) / len(rows)),
                        "positive_prevalence": prevalence,
                        "random_ap": prevalence,
                        "auprc": auprc,
                        "ap_minus_random": float(auprc - prevalence),
                        "ap_lift_over_random": float(auprc / prevalence) if prevalence else None,
                        "auroc": float(roc_auc_score(y, p)),
                        "f1": float(f1_score(y, p >= threshold)),
                        "positive_recall": float(recall_score(y, p >= threshold, pos_label=1)),
                        "negative_recall": float(recall_score(y, p >= threshold, pos_label=0)),
                        "mcc": float(matthews_corrcoef(y, p >= threshold)),
                        "brier": float(brier_score_loss(y, p)),
                        "ece_10bin": expected_calibration_error(y, p),
                        "threshold": threshold, "evaluation": "known_labels_only"})
    return results, selected


def normalized_multitask_loss(matrix: torch.Tensor, masks: torch.Tensor, world: int) -> torch.Tensor:
    """Average within each task, then average tasks over the distributed microbatch."""
    local_sums = (matrix * masks).sum(0)
    global_counts = masks.sum(0).detach()
    if world > 1:
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    valid = global_counts > 0
    if not torch.any(valid):
        return local_sums.sum() * 0.0
    # DDP averages gradients across ranks, so compensate to recover global task means.
    return world * (local_sums[valid] / global_counts[valid]).mean()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return result


def load_training_ids(root: Path, config: dict) -> set[str] | None:
    role = config.get("training_internal_role")
    path = config.get("training_split_path")
    if role is None and path is None:
        return None
    if not role or not path:
        raise ValueError("training_internal_role and training_split_path must be set together")
    selected = set()
    with (root / path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["internal_role"] == role:
                selected.add(row["sequence_id"])
    if not selected:
        raise ValueError(f"no sequence IDs found for internal role {role}")
    return selected


def predict(model: nn.Module, rows: list[dict], collator: Collator, batch_size: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = move(collator(rows[start:start + batch_size]), device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    started_at = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["mode"] not in {"bce_zero_fill", "masked_bce"}:
        raise ValueError("mode must be bce_zero_fill or masked_bce")
    if config["target_task"] not in {"shared", *LABELS}:
        raise ValueError("invalid target_task")
    root = Path(config["root"])
    if Path(config["model_path"]).resolve().is_relative_to(root.resolve()) is False:
        raise ValueError("model_path must be inside deployment root")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if world > 1 else 0
    set_seed(config["seed"] + rank)
    stamp = [datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else ""]
    if world > 1:
        dist.broadcast_object_list(stamp, src=0)
    run = root / "runs" / f'{config["run_name"]}_{stamp[0]}'
    if rank == 0:
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "config.json", config)
    if world > 1:
        dist.barrier()

    data_root = root / "data" / "v3"
    train_rows = read_rows(data_root / "train.csv", config["target_task"], config["mode"], training=True)
    training_ids = load_training_ids(root, config)
    if training_ids is not None:
        train_rows = [row for row in train_rows if row["sequence_id"] in training_ids]
        if not train_rows:
            raise ValueError("training split filter removed every row")
    validation_rows = read_rows(data_root / "validation.csv", config["target_task"], "masked_bce", training=False)
    tasks = LABELS if config["target_task"] == "shared" else [config["target_task"]]
    tokenizer = T5Tokenizer.from_pretrained(config["model_path"], local_files_only=True, do_lower_case=False)
    collator = Collator(tokenizer)
    dataset = SequenceDataset(train_rows)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=config["seed"], drop_last=False) if world > 1 else None
    loader = DataLoader(dataset, batch_size=config["microbatch_per_gpu"], sampler=sampler,
                        shuffle=sampler is None, collate_fn=collator, num_workers=config["num_workers"], pin_memory=True)
    model = Predictor(config).to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else model
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": config["adapter_lr"]},
        {"params": model.head.parameters(), "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
    labels = np.asarray([row["labels"] for row in train_rows])
    masks = np.asarray([row["masks"] for row in train_rows])
    positives = (labels * masks).sum(0)
    negatives = masks.sum(0) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("a training task has only one known class")
    balance_mode = config.get("class_balance", "none")
    if balance_mode == "none":
        pos_weight = torch.ones(len(tasks), device=device, dtype=torch.float32)
    elif balance_mode == "capped_pos_weight":
        cap = float(config.get("positive_weight_cap", 10.0))
        pos_weight = torch.tensor(np.minimum(negatives / positives, cap), device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported class_balance: {balance_mode}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    accumulation = config["gradient_accumulation"]
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    warmup = max(1, int(total_updates * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, max(0.0, (total_updates - step) / max(1, total_updates - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "float16", init_scale=128.0)
    best, best_thresholds = -1.0, None
    if rank == 0:
        atomic_json(run / "training_manifest.json", {
            "status": "running",
            "training_rows": len(train_rows),
            "training_internal_role": config.get("training_internal_role", "full_train"),
            "known_positions_per_task": dict(zip(tasks, masks.sum(0).astype(int).tolist())),
            "positive_positions_per_task": dict(zip(tasks, positives.astype(int).tolist())),
            "negative_positions_per_task": dict(zip(tasks, negatives.astype(int).tolist())),
            "epochs": config["epochs"],
            "updates_per_epoch": updates_per_epoch,
            "total_updates_planned": total_updates,
            "world_size": world,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "test_evaluated": False,
        })

    for epoch in range(config["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(loader):
            batch = move(raw_batch, device)
            boundary = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            sync = ddp.no_sync() if world > 1 and not boundary else torch.enable_grad()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=config["precision"] == "float16"):
                    logits = ddp(batch)
                    matrix = loss_fn(logits, batch["labels"])
                    loss = normalized_multitask_loss(matrix, batch["masks"], world)
                    loss = loss / accumulation
                scaler.scale(loss).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        if world > 1:
            dist.barrier()
        if rank == 0:
            probabilities = predict(model, validation_rows, collator, config["microbatch_per_gpu"], device)
            scored, thresholds = known_metrics(validation_rows, probabilities, tasks)
            macro = float(np.mean([row["auprc"] for row in scored if "auprc" in row]))
            atomic_json(run / f"metrics/validation_epoch_{epoch + 1}.json",
                        {"epoch": epoch + 1, "macro_auprc": macro, "tasks": scored})
            if macro > best:
                best, best_thresholds = macro, thresholds
                trainable = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
                torch.save({"epoch": epoch + 1, "validation_macro_auprc": macro,
                            "thresholds": thresholds, "trainable_state": trainable}, run / "best.pt")
                np.savez_compressed(run / "validation_predictions.npz",
                                    sequence_ids=np.asarray([row["sequence_id"] for row in validation_rows]),
                                    probabilities=probabilities)
            print(json.dumps({"epoch": epoch + 1, "validation_macro_auprc": macro}), flush=True)
        if world > 1:
            dist.barrier()

    if rank == 0 and args.evaluate_test:
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in checkpoint["trainable_state"]:
                    parameter.copy_(checkpoint["trainable_state"][name].to(device))
        test_rows = read_rows(data_root / "test.csv", config["target_task"], "masked_bce", training=False)
        probabilities = predict(model, test_rows, collator, config["microbatch_per_gpu"], device)
        scored, _ = known_metrics(test_rows, probabilities, tasks, best_thresholds)
        atomic_json(run / "metrics/test_once.json", {"tasks": scored, "thresholds_from_validation": best_thresholds})
        frame = {"sequence_id": [row["sequence_id"] for row in test_rows]}
        for index, task in enumerate(tasks):
            frame[f"p_{task}"] = probabilities[:, index].tolist()
        import pandas as pd
        pd.DataFrame(frame).to_csv(run / "test_predictions.csv", index=False)
    if rank == 0:
        elapsed = time.time() - started_at
        atomic_json(run / "complete.json", {
            "status": "complete",
            "best_validation_macro_auprc": best,
            "best_thresholds": best_thresholds,
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * world / 3600.0,
            "training_rows": len(train_rows),
            "total_updates": total_updates,
            "test_evaluated": bool(args.evaluate_test),
        })
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
