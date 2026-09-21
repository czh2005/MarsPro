from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


OUT = Path(r"D:\火星蛋白\结果\回头打分新不重复918")
SOURCE = Path(r"D:\火星蛋白\mpnn数据集\补充种子")
OLD = Path(r"D:\火星蛋白\结果\AAAA对mpnn的结果打分")
LABELS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


frames = []
for method, filename in (
    ("base", "Base_ProteinMPNN_匹配种子42_43_44.csv"),
    ("marslike", "Marslike_MPNN_匹配种子42_43_44.csv"),
):
    frame = pd.read_csv(SOURCE / filename)
    frame["method"] = method
    frame["backbone_id"] = frame["backbone_id"].astype(int).map(lambda value: f"{value:04d}")
    frame["replicate_seed"] = frame["replicate_seed"].astype(int)
    frame["prediction_id"] = frame.apply(
        lambda row: f"{row['backbone_id']}_s{row['replicate_seed']}", axis=1
    )
    frame["scoring_id"] = frame["method"] + "_" + frame["prediction_id"]
    frames.append(frame)

mapping = pd.concat(frames, ignore_index=True)
if len(mapping) != 4788 or mapping["scoring_id"].duplicated().any():
    raise SystemExit("scoring_mapping_qc_failed")

inference = mapping[["scoring_id", "sequence"]].rename(columns={"scoring_id": "sequence_id"})
for label in LABELS:
    inference[label] = "unknown"
inference.to_csv(OUT / "inputs" / "inference_sequences.csv", index=False)
mapping.to_csv(OUT / "inputs" / "scoring_mapping.csv", index=False)

for source, destination in (
    (OLD / "inputs" / "validation_sentinel.json", OUT / "inputs" / "validation_sentinel.json"),
    (OLD / "inputs" / "frozen_protocol.json", OUT / "inputs" / "frozen_protocol_previous.json"),
    (OLD / "tools" / "score_sequences.py", OUT / "tools" / "score_sequences.py"),
    (OLD / "raw" / "model" / "train_prott5_lora.py", OUT / "tools" / "train_prott5_lora.py"),
):
    shutil.copy2(source, destination)

manifest = {
    "rows": len(mapping),
    "methods": mapping["method"].value_counts().to_dict(),
    "backbones": int(mapping["backbone_id"].nunique()),
    "seeds": sorted(mapping["replicate_seed"].unique().tolist()),
    "input_sha256": sha256(OUT / "inputs" / "inference_sequences.csv"),
    "mapping_sha256": sha256(OUT / "inputs" / "scoring_mapping.csv"),
}
(OUT / "inputs" / "scoring_input_qc.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
