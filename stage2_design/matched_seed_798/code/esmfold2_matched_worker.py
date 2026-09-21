#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
import traceback
import types
from pathlib import Path


PARAMS = dict(num_loops=3, num_sampling_steps=200, num_diffusion_samples=1, seed=0)
DEPLOY = Path("/sddn/yyf_work/chenzhenghang/esmfold2")


def atom_json(path: Path, obj):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    root = args.root
    for name in ("structures", "confidence", "records", "logs", "validation"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with (root / "input.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = json.loads(args.jobs.read_text())

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ESMCFOLD_CCD_PATH=str(DEPLOY / "weights/ESMFold2/ccd.pkl"),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    import numpy as np
    import torch
    from biotite.sequence import ProteinSequence
    from biotite.structure.io import pdb, pdbx
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmc.modeling_esmc import ESMCModel
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from batch_compat import install

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.97)
    validation = install()
    atom_json(root / "validation" / f"{args.worker}-attention.json", validation)
    model, head_info = ESMFold2Model.from_pretrained(
        str(DEPLOY / "weights/ESMFold2"), load_esmc=False,
        local_files_only=True, output_loading_info=True,
    )
    esmc, lm_info = ESMCModel.from_pretrained(
        str(DEPLOY / "weights/ESMC-6B"), local_files_only=True, output_loading_info=True
    )
    for name, info in (("head", head_info), ("lm", lm_info)):
        unexpected = [
            key for key in info.get("unexpected_keys", [])
            if not (name == "lm" and key.startswith("lm_head."))
            and not key.endswith("._extra_state")
        ]
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or unexpected:
            raise RuntimeError(f"checkpoint mismatch for {name}: {info}")
    model = model.float().cuda().eval().requires_grad_(False)
    model._esmc = esmc.float().eval().requires_grad_(False)
    model._esmc_fp8 = False
    model.set_kernel_backend(None)
    model.set_chunk_size(32)
    original_compute = model._compute_lm_hidden_states

    def offloaded_compute(self, *call_args, **call_kwargs):
        self._esmc.cuda()
        try:
            return original_compute(*call_args, **call_kwargs)
        finally:
            self._esmc.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    model._compute_lm_hidden_states = types.MethodType(offloaded_compute, model)
    builder = ESMFold2InputBuilder(ccd_cache=DEPLOY / "weights/ESMFold2")
    print("MODEL_READY", args.worker, len(jobs), flush=True)

    for index in jobs:
        row = rows[index]
        key = f"{row['backbone_id']}_s{row['replicate_seed']}"
        record_path = root / "records" / f"{key}.json"
        if record_path.exists():
            try:
                old = json.loads(record_path.read_text())
                if old.get("status") == "complete" and (root / old["pdb_file"]).is_file():
                    continue
            except Exception:
                pass
        sequence = row["sequence"]
        rec = dict(
            prediction_id=key,
            backbone_id=row["backbone_id"],
            replicate_seed=int(row["replicate_seed"]),
            method=row["method"],
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            sequence_length=len(sequence),
            parameters=PARAMS,
            worker=args.worker,
            hostname=os.uname().nodename,
            status="running",
            started_at=time.time(),
        )
        atom_json(record_path, rec)
        print("START", key, len(sequence), flush=True)
        start = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            result = builder.fold(
                model,
                StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)]),
                complex_id=key,
                **PARAMS,
            )
            confidence = result.plddt.detach().cpu().numpy()
            coordinates = result.complex.atom_coordinates
            if not np.isfinite(coordinates).all() or not np.isfinite(confidence).all():
                raise RuntimeError("non-finite ESMFold2 output")
            cif_path = root / "structures" / f"{key}.cif"
            pdb_path = root / "structures" / f"{key}.pdb"
            cif_tmp = cif_path.with_suffix(".cif.tmp")
            pdb_tmp = pdb_path.with_suffix(".pdb.tmp")
            cif_tmp.write_text(result.complex.to_mmcif())
            atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cif_tmp)), model=1, extra_fields=["b_factor"])
            ca = atoms[atoms.atom_name == "CA"]
            recovered = "".join(ProteinSequence.convert_letter_3to1(name) for name in ca.res_name)
            if recovered != sequence:
                raise RuntimeError("sequence round-trip mismatch")
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(atoms)
            pdb_file.write(str(pdb_tmp))
            if not np.isfinite(pdb.PDBFile.read(str(pdb_tmp)).get_structure(model=1).coord).all():
                raise RuntimeError("PDB round-trip coordinates are non-finite")
            cif_tmp.replace(cif_path)
            pdb_tmp.replace(pdb_path)
            np.savez_compressed(
                root / "confidence" / f"{key}.npz",
                plddt=confidence,
                pae=result.pae.detach().cpu().numpy() if result.pae is not None else np.empty((0, 0)),
            )
            rec.update(
                status="complete",
                finished_at=time.time(),
                elapsed_seconds=time.time() - start,
                pdb_file=str(pdb_path.relative_to(root)),
                cif_file=str(cif_path.relative_to(root)),
                plddt_mean=float(confidence.mean()),
                ptm=float(result.ptm),
                peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
                pdb_sha256=hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
                cif_sha256=hashlib.sha256(cif_path.read_bytes()).hexdigest(),
                sequence_round_trip="pass",
            )
            del result, atoms, ca, coordinates, confidence, pdb_file
        except Exception as exc:
            traceback.print_exc()
            rec.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - start)
            model._esmc.cpu()
        atom_json(record_path, rec)
        gc.collect()
        torch.cuda.empty_cache()
        print("DONE", key, rec["status"], round(rec["elapsed_seconds"], 2), flush=True)

    atom_json(root / "logs" / f"{args.worker}-finished.json", dict(status="finished", worker=args.worker, time=time.time()))
    print("WORKER_FINISHED", args.worker, flush=True)


if __name__ == "__main__":
    main()
