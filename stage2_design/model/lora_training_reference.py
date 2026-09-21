"""Minimal single-label LoRA validation for the copied ProteinMPNN code."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class LoRALinear(nn.Module):
    """Frozen linear layer plus a low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        residual = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * residual


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float):
    targets = []
    for layer_index, layer in enumerate(model.decoder_layers):
        for name in ("W1", "W2", "W3"):
            original = getattr(layer, name)
            wrapped = LoRALinear(original, rank, alpha, dropout)
            setattr(layer, name, wrapped)
            targets.append(f"decoder_layers.{layer_index}.{name}")
    original = model.W_out
    model.W_out = LoRALinear(original, rank, alpha, dropout)
    targets.append("W_out")
    return targets


def read_positive_rows(csv_path: Path, label: str, limit: int | None):
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get(label) == "1"]
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_record(row, dataset_root: Path, parse_pdb):
    path = dataset_root / row["structure_file"]
    parsed = parse_pdb(str(path))
    if len(parsed) != 1:
        raise ValueError(f"expected one parsed biounit: {path}, got {len(parsed)}")
    record = parsed[0]
    target = row["sequence"].strip().upper()
    if len(target) != len(record["seq"]):
        raise ValueError(f"sequence/structure length mismatch for {path}: {len(target)} != {len(record['seq'])}")
    chain_letters = sorted(
        key.removeprefix("seq_chain_")
        for key in record
        if key.startswith("seq_chain_")
    )
    cursor = 0
    for letter in chain_letters:
        key = f"seq_chain_{letter}"
        length = len(record[key])
        record[key] = target[cursor : cursor + length]
        cursor += length
    record["seq"] = target
    record["source_structure_file"] = str(path)
    return record, chain_letters


def make_batch(record, chain_letters, device, tied_featurize):
    chain_dict = {record["name"]: (chain_letters, [])}
    return tied_featurize([record], device, chain_dict)


def nll_from_log_probs(S, log_probs, mask):
    loss = nn.functional.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)), S.reshape(-1), reduction="none"
    ).reshape_as(S)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def run_epoch(model, samples, device, tied_featurize, train, optimizer=None):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0.0
    for record, chain_letters in samples:
        features = make_batch(record, chain_letters, device, tied_featurize)
        X, S, mask, _, chain_M, _, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = features
        train_mask = mask * chain_M
        randn = torch.randn(chain_M.shape, device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        log_probs = model(X, S, mask, chain_M, residue_idx, features[5], randn)
        loss = nll_from_log_probs(S, log_probs, train_mask)
        if train:
            loss.backward()
            optimizer.step()
        token_count = float(train_mask.sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * token_count
        total_tokens += token_count
    return total_loss / max(total_tokens, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-copy", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="cold")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=8)
    parser.add_argument("--max-val", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo_copy))
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.0,
        augment_eps=0.0,
    )
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    targets = inject_lora(model, args.rank, args.alpha, args.dropout)
    for name, parameter in model.named_parameters():
        if ".lora_A" not in name and ".lora_B" not in name:
            parameter.requires_grad = False
    base_snapshot = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if not (".lora_A" in name or ".lora_B" in name)
    }
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)

    train_rows = read_positive_rows(args.train_csv, args.label, args.max_train)
    val_rows = read_positive_rows(args.validation_csv, args.label, args.max_val)
    train_samples = [load_record(row, args.dataset_root, parse_PDB) for row in train_rows]
    val_samples = [load_record(row, args.dataset_root, parse_PDB) for row in val_rows]

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_samples, device, tied_featurize, True, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_samples, device, tied_featurize, False)
        history.append({"epoch": epoch, "train_nll": train_loss, "validation_nll": val_loss})
        print(f"epoch={epoch} train_nll={train_loss:.6f} validation_nll={val_loss:.6f}", flush=True)

    base_unchanged = True
    for name, before in base_snapshot.items():
        base_unchanged &= torch.equal(before, dict(model.named_parameters())[name].detach().cpu())
    lora_norm = float(sum(parameter.detach().norm().cpu() for parameter in trainable))
    adapter_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_A" in name or ".lora_B" in name
    }
    torch.save(
        {
            "label": args.label,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "targets": targets,
            "adapter_state_dict": adapter_state,
            "history": history,
            "base_parameters_unchanged_outside_wrapped_layers": base_unchanged,
            "lora_parameter_norm_sum": lora_norm,
        },
        args.output / "cold_lora_smoke.pt",
    )
    metadata = {
        "label": args.label,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "targets": targets,
        "history": history,
        "base_parameters_unchanged_outside_wrapped_layers": base_unchanged,
        "lora_parameter_norm_sum": lora_norm,
    }
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
