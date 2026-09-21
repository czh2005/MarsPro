from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class LoRALinear(nn.Module):
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
        residual = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + self.scaling * residual


def inject_lora(model, rank: int, alpha: float, dropout: float):
    targets = []
    for layer_index, layer in enumerate(model.decoder_layers):
        for name in ("W1", "W2", "W3"):
            setattr(layer, name, LoRALinear(getattr(layer, name), rank, alpha, dropout))
            targets.append(f"decoder_layers.{layer_index}.{name}")
    model.W_out = LoRALinear(model.W_out, rank, alpha, dropout)
    targets.append("W_out")
    return targets


def make_model(ProteinMPNN, base_checkpoint: Path, device, lora_checkpoint: Path | None):
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
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(base["model_state_dict"], strict=True)
    load_report = None
    if lora_checkpoint is not None:
        adapter = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
        cfg = adapter["config"]
        targets = inject_lora(model, cfg["rank"], cfg["alpha"], cfg["dropout"])
        load_report = model.load_state_dict(adapter["model_state_dict"], strict=False)
        if load_report.unexpected_keys:
            raise RuntimeError(f"unexpected LoRA keys: {load_report.unexpected_keys}")
        loaded = {key for key in adapter["model_state_dict"] if ".lora_" in key}
        if len(loaded) != 20 or targets != cfg["target_modules"]:
            raise RuntimeError(f"LoRA identity mismatch: {len(loaded)} tensors, targets={targets}")
    model.to(device).eval()
    return model, load_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-copy", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_copy))
    from protein_mpnn_utils import ProteinMPNN, _S_to_seq, parse_PDB, tied_featurize

    device = torch.device("cuda:0")
    base_model, _ = make_model(ProteinMPNN, args.base_checkpoint, device, None)
    joint_model, lora_report = make_model(
        ProteinMPNN, args.base_checkpoint, device, args.lora_checkpoint
    )

    with args.test_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    omit_aas = np.array([aa in "X" for aa in ALPHABET]).astype(np.float32)
    bias_aas = np.zeros(len(ALPHABET), dtype=np.float32)
    results = {"base": [], "joint_lora": []}
    fasta = {"base": [], "joint_lora": []}
    started = time.time()

    with torch.inference_mode():
        for index, row in enumerate(rows):
            pdb_path = args.dataset_root / row["structure_file"]
            parsed = parse_PDB(str(pdb_path))
            if len(parsed) != 1:
                raise ValueError(f"expected one parsed structure for {pdb_path}, got {len(parsed)}")
            record = parsed[0]
            chains = sorted(
                key.removeprefix("seq_chain_")
                for key in record
                if key.startswith("seq_chain_")
            )
            chain_dict = {record["name"]: (chains, [])}
            features = tied_featurize([copy.deepcopy(record)], device, chain_dict)
            (
                X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _,
                chain_M_pos, omit_AA_mask, residue_idx, _, _, pssm_coef,
                pssm_bias, pssm_log_odds_all, bias_by_res_all, _
            ) = features
            pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()
            sample_seed = args.seed * 1_000_000 + index

            for method, model in (("base", base_model), ("joint_lora", joint_model)):
                set_seed(sample_seed)
                noise = torch.randn(chain_M.shape, device=device)
                sample = model.sample(
                    X, noise, S, chain_M, chain_encoding_all, residue_idx,
                    mask=mask,
                    temperature=args.temperature,
                    omit_AAs_np=omit_aas,
                    bias_AAs_np=bias_aas,
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=0.0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=pssm_log_odds_mask,
                    pssm_bias_flag=False,
                    bias_by_res=bias_by_res_all,
                )
                sequence = _S_to_seq(sample["S"][0], chain_M[0]).replace("/", "")
                if len(sequence) != int(mask[0].sum().item()):
                    raise ValueError(f"length mismatch for {row['sequence_id']} {method}")
                item = {
                    "sequence_id": row["sequence_id"],
                    "backbone_id": row.get("prediction_id", Path(row["structure_file"]).stem),
                    "structure_file": row["structure_file"],
                    "method": method,
                    "replicate_seed": args.seed,
                    "sample_seed": sample_seed,
                    "temperature": args.temperature,
                    "generated_length": len(sequence),
                    "sequence": sequence,
                }
                results[method].append(item)
                fasta[method].append(
                    f">{item['backbone_id']}|{method}|seed={args.seed}|sample_seed={sample_seed}\n{sequence}\n"
                )
            if (index + 1) % 25 == 0:
                print(f"completed {index + 1}/{len(rows)}", flush=True)

    fields = list(results["base"][0])
    for method in results:
        csv_path = args.output / f"{method}_seed{args.seed}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results[method])
        (args.output / f"{method}_seed{args.seed}.fasta").write_text(
            "".join(fasta[method]), encoding="ascii"
        )

    manifest = {
        "seed": args.seed,
        "temperature": args.temperature,
        "rows_per_method": len(rows),
        "elapsed_seconds": time.time() - started,
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "joint_lora_checkpoint_sha256": sha256(args.lora_checkpoint),
        "test_csv_sha256": sha256(args.test_csv),
        "lora_missing_key_count": len(lora_report.missing_keys),
        "lora_unexpected_keys": lora_report.unexpected_keys,
        "sampling_pairing": "same per-backbone derived seed for both methods",
    }
    (args.output / f"manifest_seed{args.seed}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
