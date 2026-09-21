#!/usr/bin/env python3
"""Matched-seed structural and surface evaluation for Base vs Marslike-MPNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path

import freesasa
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


MAX_ASA = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        [129, 274, 195, 193, 167, 225, 223, 104, 224, 197, 201, 236, 224, 240, 159, 155, 172, 285, 263, 174],
    )
)
SURFACE_SETS = {"STNQDE": set("STNQDE"), "MC": set("MC")}
PARSER = PDBParser(QUIET=True)


def parse_structure(path: Path) -> tuple[str, np.ndarray, float, list[tuple[str, str, str]]]:
    structure = PARSER.get_structure("protein", str(path))
    model = next(structure.get_models())
    residues = [r for r in model.get_residues() if r.id[0] == " " and "CA" in r]
    sequence = "".join(seq1(r.resname) for r in residues)
    if not sequence or any(aa not in MAX_ASA for aa in sequence):
        raise ValueError("empty_or_noncanonical_sequence")
    coordinates = np.asarray([r["CA"].coord for r in residues], dtype=np.float64)
    mean_bfactor = float(np.mean([r["CA"].bfactor for r in residues]))
    residue_keys = [(r.parent.id.strip() or "A", str(r.id[1]), r.id[2].strip()) for r in residues]
    return sequence, coordinates, mean_bfactor, residue_keys


def surface_features(path: Path) -> dict[str, float]:
    sequence, _, _, residue_keys = parse_structure(path)
    fs_structure = freesasa.Structure(str(path), freesasa.Classifier.getStandardClassifier("protor"))
    parameters = freesasa.Parameters(
        {"algorithm": freesasa.LeeRichards, "probe-radius": 1.4, "n-slices": 20, "n-threads": 1}
    )
    result = freesasa.calc(fs_structure, parameters)
    residue_areas: dict[tuple[str, str], float] = {}
    for index in range(fs_structure.nAtoms()):
        key = (fs_structure.chainLabel(index).strip() or "A", fs_structure.residueNumber(index).strip())
        residue_areas[key] = residue_areas.get(key, 0.0) + result.atomArea(index)

    surface = []
    for aa, (chain, number, insertion) in zip(sequence, residue_keys):
        candidates = [(chain, number + insertion), (chain, number)]
        area = next((residue_areas[k] for k in candidates if k in residue_areas), None)
        if area is None:
            raise ValueError(f"missing_residue_sasa:{chain}:{number}{insertion}")
        if area / MAX_ASA[aa] >= 0.25:
            surface.append(aa)
    if not surface:
        raise ValueError("no_surface_residues")

    output = {
        "sequence": sequence,
        "length": len(sequence),
        "total_sasa": float(result.totalArea()),
        "n_surface_t25": len(surface),
    }
    for label, chars in SURFACE_SETS.items():
        output[f"surface_{label}_ratio"] = sum(aa in chars for aa in surface) / len(surface)
        output[f"sequence_{label}_ratio"] = sum(aa in chars for aa in sequence) / len(sequence)
    return output


def parse_usalign(stdout: str) -> tuple[int, float, float, float]:
    aligned = None
    rmsd = None
    sequence_identity = None
    tm_reference = None
    first_tm = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Aligned length="):
            aligned = int(stripped.split("Aligned length=")[1].split(",")[0].strip())
            rmsd = float(stripped.split("RMSD=")[1].split(",")[0].strip())
            sequence_identity = float(stripped.split("Seq_ID=")[1].split("=")[-1].strip()) * 100.0
        if "TM-score=" in stripped:
            value = float(stripped.split("TM-score=")[1].split()[0])
            first_tm = value if first_tm is None else first_tm
            if "Structure_2" in stripped:
                tm_reference = value
    if aligned is None or rmsd is None:
        raise ValueError("usalign_parse_error")
    return aligned, rmsd, sequence_identity, tm_reference if tm_reference is not None else first_tm


def parse_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    translation = np.zeros(3)
    rotation = np.zeros((3, 3))
    for index, line in enumerate(lines):
        if line.startswith("m") and "t[m]" in line:
            for offset in range(3):
                fields = lines[index + 1 + offset].split()
                row = int(fields[0])
                translation[row] = float(fields[1])
                rotation[row] = [float(fields[2]), float(fields[3]), float(fields[4])]
            return translation, rotation
    raise ValueError("usalign_matrix_parse_error")


def gdt(distances: np.ndarray) -> tuple[float, float, float, float, float]:
    percentages = [float(np.mean(distances <= cutoff) * 100.0) for cutoff in (1.0, 2.0, 4.0, 8.0)]
    return float(np.mean(percentages)), *percentages


def kabsch(predicted: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    predicted_centered = predicted - predicted.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    covariance = predicted_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ np.diag([1.0, 1.0, sign]) @ right_t
    aligned = predicted_centered @ rotation
    distances = np.linalg.norm(aligned - reference_centered, axis=1)
    return float(np.sqrt(np.mean(distances**2))), distances


def reference_worker(item: tuple[str, str]) -> dict[str, object]:
    backbone_id, path_string = item
    row: dict[str, object] = {"backbone_id": backbone_id, "reference_pdb": path_string, "status": "failed"}
    try:
        row.update(surface_features(Path(path_string)))
        row["status"] = "ok"
        row["error"] = ""
    except Exception as error:
        row["error"] = repr(error)
    return row


def design_worker(task: dict[str, object]) -> dict[str, object]:
    started = time.time()
    row = dict(task)
    row["status"] = "failed"
    matrix_path = None
    try:
        predicted_path = Path(str(task["predicted_pdb"]))
        reference_path = Path(str(task["reference_pdb"]))
        predicted_sequence, predicted_ca, mean_bfactor, _ = parse_structure(predicted_path)
        reference_sequence, reference_ca, _, _ = parse_structure(reference_path)
        if predicted_sequence != task["sequence"]:
            raise ValueError("predicted_sequence_round_trip_mismatch")
        if len(predicted_ca) != len(reference_ca):
            raise ValueError(f"reference_length_mismatch:{len(predicted_ca)}:{len(reference_ca)}")

        with tempfile.NamedTemporaryFile(prefix="usalign_", suffix=".txt", delete=False) as matrix:
            matrix_path = Path(matrix.name)
        completed = subprocess.run(
            [str(task["usalign"]), str(predicted_path), str(reference_path), "-m", str(matrix_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(f"usalign_exit_{completed.returncode}:{completed.stderr[:160]}")
        aligned_length, usalign_rmsd, sequence_identity, tm_score = parse_usalign(completed.stdout)
        translation, rotation = parse_matrix(matrix_path)
        transformed = predicted_ca @ rotation.T + translation
        positional_distances = np.linalg.norm(transformed - reference_ca, axis=1)
        gdt_ts, p1, p2, p4, p8 = gdt(positional_distances)
        full_rmsd, full_distances = kabsch(predicted_ca, reference_ca)
        full_gdt, _, _, _, _ = gdt(full_distances)

        features = surface_features(predicted_path)
        reference_features = task["reference_features"]
        recovery = sum(a == b for a, b in zip(predicted_sequence, reference_sequence)) / len(reference_sequence)
        record = json.loads(Path(str(task["record_json"])).read_text())
        plddt_record = float(record["plddt_mean"]) * 100.0
        if abs(plddt_record - mean_bfactor) > 0.2:
            raise ValueError(f"plddt_record_pdb_mismatch:{plddt_record}:{mean_bfactor}")

        row.update(
            {
                "reference_length": len(reference_sequence),
                "aligned_length": aligned_length,
                "usalign_rmsd": usalign_rmsd,
                "tm_score": tm_score,
                "usalign_sequence_identity_pct": sequence_identity,
                "gdt_ts": gdt_ts,
                "gdt_p1": p1,
                "gdt_p2": p2,
                "gdt_p4": p4,
                "gdt_p8": p8,
                "full_length_kabsch_rmsd": full_rmsd,
                "full_length_kabsch_gdt_ts": full_gdt,
                "mean_plddt": plddt_record,
                "ptm": float(record["ptm"]),
                "sequence_recovery_pct": recovery * 100.0,
                "surface_STNQDE_ratio": features["surface_STNQDE_ratio"],
                "surface_MC_ratio": features["surface_MC_ratio"],
                "reference_surface_STNQDE_ratio": reference_features["surface_STNQDE_ratio"],
                "reference_surface_MC_ratio": reference_features["surface_MC_ratio"],
                "surface_STNQDE_abs_distance": abs(
                    features["surface_STNQDE_ratio"] - reference_features["surface_STNQDE_ratio"]
                ),
                "surface_MC_abs_distance": abs(
                    features["surface_MC_ratio"] - reference_features["surface_MC_ratio"]
                ),
                "total_sasa": features["total_sasa"],
                "n_surface_t25": features["n_surface_t25"],
                "status": "ok",
                "error": "",
            }
        )
    except Exception as error:
        row["error"] = repr(error)
    finally:
        if matrix_path is not None:
            matrix_path.unlink(missing_ok=True)
    row["elapsed_seconds"] = time.time() - started
    row.pop("reference_features", None)
    row.pop("usalign", None)
    return row


def load_tasks(root: Path, reference_map: dict[str, dict[str, object]], usalign: Path) -> list[dict[str, object]]:
    tasks = []
    for method, csv_name, method_root in (
        ("base", "base.csv", "base"),
        ("marslike", "joint_lora.csv", "joint_lora"),
    ):
        table = pd.read_csv(root / "inputs" / csv_name)
        for item in table.to_dict("records"):
            backbone_id = f"{int(item['backbone_id']):04d}"
            seed = int(item["replicate_seed"])
            prediction_id = f"{backbone_id}_s{seed}"
            reference = reference_map[backbone_id]
            tasks.append(
                {
                    "method": method,
                    "prediction_id": prediction_id,
                    "backbone_id": backbone_id,
                    "replicate_seed": seed,
                    "sequence_id": item["sequence_id"],
                    "sequence": item["sequence"],
                    "sequence_length": len(item["sequence"]),
                    "predicted_pdb": str(root.parent / method_root / "structures" / f"{prediction_id}.pdb"),
                    "reference_pdb": str(root / "inputs" / "native" / f"{backbone_id}.pdb"),
                    "record_json": str(root.parent / method_root / "records" / f"{prediction_id}.json"),
                    "reference_features": reference,
                    "usalign": str(usalign),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    usalign = root / "bin" / "USalign"

    reference_items = [
        (path.stem, str(path)) for path in sorted((root / "inputs" / "native").glob("*.pdb"))
    ]
    with mp.Pool(args.workers) as pool:
        references = list(pool.imap_unordered(reference_worker, reference_items, chunksize=4))
    reference_frame = pd.DataFrame(references).sort_values("backbone_id")
    reference_frame.to_csv(result_dir / "reference_surface.csv", index=False)
    if len(reference_frame) != 798 or (reference_frame["status"] != "ok").any():
        raise SystemExit("reference_surface_qc_failed")
    reference_map = {row["backbone_id"]: row for row in references}

    tasks = load_tasks(root, reference_map, usalign)
    if len(tasks) != 4788:
        raise SystemExit(f"unexpected task count: {len(tasks)}")
    if args.limit:
        tasks = tasks[: args.limit]
    results = []
    started = time.time()
    with mp.Pool(args.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(design_worker, tasks, chunksize=2), 1):
            results.append(result)
            if result["status"] != "ok" and sum(row["status"] != "ok" for row in results) <= 3:
                print(json.dumps({"failed_example": result}, ensure_ascii=False), flush=True)
            if index % 250 == 0 or index == len(tasks):
                failed = sum(row["status"] != "ok" for row in results)
                print(json.dumps({"done": index, "total": len(tasks), "failed": failed, "seconds": time.time() - started}), flush=True)

    frame = pd.DataFrame(results).sort_values(["backbone_id", "replicate_seed", "method"])
    frame.to_csv(result_dir / "design_structure_surface_metrics.csv", index=False)
    summary = {
        "total": len(frame),
        "complete": int((frame["status"] == "ok").sum()),
        "failed": int((frame["status"] != "ok").sum()),
        "methods": frame.groupby("method")["status"].value_counts().unstack(fill_value=0).to_dict("index"),
        "elapsed_seconds": time.time() - started,
    }
    (result_dir / "structure_surface_qc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
