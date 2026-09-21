#!/usr/bin/env python3
"""CUDA sequence novelty and matched-seed diversity for the 4,788 designs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: index + 1 for index, aa in enumerate(ALPHABET)}


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    identifiers, sequences, current = [], [], []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                identifiers.append(line[1:].split()[0])
                current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return identifiers, sequences


def pdb_sequence(path: Path) -> str:
    structure = PDBParser(QUIET=True).get_structure("reference", str(path))
    model = next(structure.get_models())
    return "".join(seq1(residue.resname) for residue in model.get_residues() if residue.id[0] == " " and "CA" in residue)


def one_hot(sequences: list[str], max_length: int, device: torch.device) -> torch.Tensor:
    tensor = torch.zeros((len(sequences), max_length, 21), dtype=torch.float32, device=device)
    indices = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        indices[row, : len(sequence)] = torch.tensor([VOCAB.get(aa, 0) for aa in sequence], device=device)
    tensor.scatter_(2, indices.unsqueeze(-1), 1.0)
    tensor[:, :, 0] = 0.0
    return tensor


def positional_identity(left: str, right: str) -> float:
    denominator = min(len(left), len(right))
    return sum(a == b for a, b in zip(left, right)) / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    for method, filename in (("base", "base.csv"), ("marslike", "joint_lora.csv")):
        frame = pd.read_csv(root / "inputs" / filename)
        frame["method"] = method
        frame["backbone_id"] = frame["backbone_id"].astype(int).map(lambda value: f"{value:04d}")
        frame["replicate_seed"] = frame["replicate_seed"].astype(int)
        frame["prediction_id"] = frame.apply(
            lambda row: f"{row['backbone_id']}_s{row['replicate_seed']}", axis=1
        )
        frames.append(frame)
    designs = pd.concat(frames, ignore_index=True)
    if len(designs) != 4788 or designs.duplicated(["method", "prediction_id"]).any():
        raise SystemExit("design_input_qc_failed")

    train_ids, train_sequences = read_fasta(root / "inputs" / "sequences_train.fasta")
    parent_sequences = {
        backbone_id: pdb_sequence(root / "inputs" / "native" / f"{backbone_id}.pdb")
        for backbone_id in sorted(designs["backbone_id"].unique())
    }
    all_sequences = designs["sequence"].tolist()
    max_length = max(max(map(len, all_sequences)), max(map(len, train_sequences)))
    device = torch.device("cuda:0")
    design_tensor = one_hot(all_sequences, max_length, device).reshape(len(all_sequences), -1)
    train_tensor = one_hot(train_sequences, max_length, device).reshape(len(train_sequences), -1)
    with torch.no_grad():
        matches = (design_tensor @ train_tensor.T).cpu().numpy().astype(np.float64)
    design_lengths = designs["sequence"].str.len().to_numpy(dtype=np.float64)
    train_lengths = np.asarray([len(sequence) for sequence in train_sequences], dtype=np.float64)
    identities = matches / np.minimum(design_lengths[:, None], train_lengths[None, :])
    max_identity = identities.max(axis=1)

    train_sequence_indices: dict[str, list[int]] = {}
    for index, sequence in enumerate(train_sequences):
        train_sequence_indices.setdefault(sequence, []).append(index)
    excluded = identities.copy()
    excluded_counts = np.zeros(len(designs), dtype=int)
    for row_index, backbone_id in enumerate(designs["backbone_id"]):
        for train_index in train_sequence_indices.get(parent_sequences[backbone_id], []):
            excluded[row_index, train_index] = -1.0
            excluded_counts[row_index] += 1
    max_identity_excluding_parent = excluded.max(axis=1)

    novelty = designs[["method", "prediction_id", "backbone_id", "replicate_seed", "sequence_id"]].copy()
    novelty["max_train_identity_pct"] = max_identity * 100.0
    novelty["novelty_pct"] = 100.0 - novelty["max_train_identity_pct"]
    novelty["parent_train_matches_excluded"] = excluded_counts
    novelty["max_train_identity_excl_parent_pct"] = max_identity_excluding_parent * 100.0
    novelty["novelty_excl_parent_pct"] = 100.0 - novelty["max_train_identity_excl_parent_pct"]
    novelty.to_csv(output / "design_novelty.csv", index=False)

    diversity_rows = []
    for (method, backbone_id), group in designs.groupby(["method", "backbone_id"], sort=True):
        seed_sequences = dict(zip(group["replicate_seed"], group["sequence"]))
        pair_values = []
        for left_seed, right_seed in combinations(sorted(seed_sequences), 2):
            identity = positional_identity(seed_sequences[left_seed], seed_sequences[right_seed])
            pair_values.append(identity)
            diversity_rows.append(
                {
                    "method": method,
                    "backbone_id": backbone_id,
                    "seed_left": left_seed,
                    "seed_right": right_seed,
                    "seed_pair_identity_pct": identity * 100.0,
                    "seed_pair_diversity_pct": (1.0 - identity) * 100.0,
                }
            )
    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(output / "matched_seed_diversity.csv", index=False)

    qc = {
        "designs": len(designs),
        "train_sequences": len(train_sequences),
        "backbones": len(parent_sequences),
        "max_length": max_length,
        "noncanonical_designs": int(sum(any(aa not in ALPHABET for aa in seq) for seq in all_sequences)),
        "gpu": torch.cuda.get_device_name(0),
        "output_rows": len(novelty),
        "seed_pair_rows": len(diversity),
    }
    (output / "sequence_novelty_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2), flush=True)


if __name__ == "__main__":
    main()
