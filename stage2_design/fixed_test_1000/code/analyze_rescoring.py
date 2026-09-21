"""Analyze three-seed ProteinMPNN versus Marslike-MPNN frozen-model rescoring."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
TASKS = ["cold", "desiccation", "oxidative", "perchlorate", "radiation", "salt"]
TASK_CN = dict(
    zip(TASKS, ["低温", "干燥", "氧化", "高氯酸盐相关", "辐射/DNA修复", "盐适应"])
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values, seed, repeats=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repeats, 200):
        n = min(200, repeats - start)
        draw = rng.integers(0, len(values), size=(n, len(values)))
        means.extend(values[draw].mean(axis=1))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


input_qc = json.loads((ROOT / "inputs" / "scoring_input_qc.json").read_text())
mapping = read_csv(ROOT / "inputs" / "design_sequence_mapping.csv")
predictions = {}
identity_rows = []
remote = ROOT / "remote_outputs"
for path in sorted(remote.glob("shard_*_predictions.csv")):
    shard = path.stem.split("_")[1]
    status = json.loads((remote / f"shard_{shard}_status.json").read_text())
    identity = json.loads((remote / f"shard_{shard}_identity.json").read_text())
    assert status["state"] == "complete"
    assert status["predictions_sha256"] == sha256(path)
    assert identity["sentinel_max_abs_error"] <= 0.002
    identity_rows.append(identity)
    for row in read_csv(path):
        assert row["sequence_id"] not in predictions
        predictions[row["sequence_id"]] = np.asarray(
            [float(row["p_" + task]) for task in TASKS], dtype=float
        )
assert len(identity_rows) > 0
assert len({row["checkpoint_sha256"] for row in identity_rows}) == 1
assert len(predictions) == input_qc["unique_sequences_for_inference"]

scored_rows = []
lookup = {}
for row in mapping:
    scores = predictions[row["score_sequence_id"]]
    item = {**row, **{"p_" + task: float(score) for task, score in zip(TASKS, scores)}}
    item["six_label_mean"] = float(scores.mean())
    scored_rows.append(item)
    lookup[(row["challenge_id"], row["method"], row["replicate_seed"])] = item

reference = {
    row["challenge_id"]: row for row in scored_rows if row["method"] == "reference"
}
paired = []
for challenge_id in sorted(reference):
    ref = reference[challenge_id]
    ref_scores = np.asarray([float(ref["p_" + task]) for task in TASKS])
    positive_tasks = [task for task in TASKS if ref[task] == "1"]
    assert positive_tasks
    for seed in sorted({int(row["replicate_seed"]) for row in scored_rows if row["method"] != "reference"}):
        base = lookup[(challenge_id, "proteinmpnn", str(seed))]
        mars = lookup[(challenge_id, "marslike_mpnn", str(seed))]
        base_scores = np.asarray([float(base["p_" + task]) for task in TASKS])
        mars_scores = np.asarray([float(mars["p_" + task]) for task in TASKS])
        base_positive = np.mean([float(base["p_" + task]) for task in positive_tasks])
        mars_positive = np.mean([float(mars["p_" + task]) for task in positive_tasks])
        paired.append(
            {
                "challenge_id": challenge_id,
                "reference_sequence_id": ref["reference_sequence_id"],
                "id50_cluster_id": ref["id50_cluster_id"],
                "replicate_seed": seed,
                "positive_tasks": ";".join(positive_tasks),
                "positive_task_count": len(positive_tasks),
                "reference_six_label_mean": ref["six_label_mean"],
                "proteinmpnn_six_label_mean": base["six_label_mean"],
                "marslike_mpnn_six_label_mean": mars["six_label_mean"],
                "aggregate_delta": mars["six_label_mean"] - base["six_label_mean"],
                "proteinmpnn_positive_label_mean": base_positive,
                "marslike_mpnn_positive_label_mean": mars_positive,
                "positive_label_delta": mars_positive - base_positive,
                "proteinmpnn_distance_to_reference": float(np.linalg.norm(base_scores - ref_scores)),
                "marslike_mpnn_distance_to_reference": float(np.linalg.norm(mars_scores - ref_scores)),
                "distance_delta": float(
                    np.linalg.norm(mars_scores - ref_scores)
                    - np.linalg.norm(base_scores - ref_scores)
                ),
                "proteinmpnn_sequence_sha256": base["sequence_sha256"],
                "marslike_mpnn_sequence_sha256": mars["sequence_sha256"],
            }
        )

paired_df = pd.DataFrame(paired)
backbone_df = (
    paired_df.groupby(["challenge_id", "reference_sequence_id", "id50_cluster_id"], as_index=False)
    .agg(
        aggregate_delta=("aggregate_delta", "mean"),
        positive_label_delta=("positive_label_delta", "mean"),
        distance_delta=("distance_delta", "mean"),
        proteinmpnn_six_label_mean=("proteinmpnn_six_label_mean", "mean"),
        marslike_mpnn_six_label_mean=("marslike_mpnn_six_label_mean", "mean"),
        proteinmpnn_positive_label_mean=("proteinmpnn_positive_label_mean", "mean"),
        marslike_mpnn_positive_label_mean=("marslike_mpnn_positive_label_mean", "mean"),
        proteinmpnn_distance_to_reference=("proteinmpnn_distance_to_reference", "mean"),
        marslike_mpnn_distance_to_reference=("marslike_mpnn_distance_to_reference", "mean"),
    )
)


def summarize(metric, base_column, mars_column, lower_is_better=False, seed=20260918):
    values = backbone_df[metric].to_numpy(float)
    improved = values < 0 if lower_is_better else values > 0
    non_ties = int(np.count_nonzero(values))
    successes = int(improved.sum())
    p_value = float(binomtest(successes, non_ties, p=0.5).pvalue) if non_ties else 1.0
    return {
        "metric": metric,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "backbones": len(values),
        "proteinmpnn_mean": float(backbone_df[base_column].mean()),
        "marslike_mpnn_mean": float(backbone_df[mars_column].mean()),
        "mean_delta": float(values.mean()),
        "relative_mean_change": float(
            values.mean() / backbone_df[base_column].mean()
        ),
        "median_delta": float(np.median(values)),
        "ci95_lower": bootstrap(values, seed)[0],
        "ci95_upper": bootstrap(values, seed)[1],
        "improved_backbones": successes,
        "improved_fraction": float(improved.mean()),
        "sign_test_p": p_value,
    }


summary_rows = [
    summarize(
        "aggregate_delta",
        "proteinmpnn_six_label_mean",
        "marslike_mpnn_six_label_mean",
        False,
        20260918,
    ),
    summarize(
        "positive_label_delta",
        "proteinmpnn_positive_label_mean",
        "marslike_mpnn_positive_label_mean",
        False,
        20260919,
    ),
    summarize(
        "distance_delta",
        "proteinmpnn_distance_to_reference",
        "marslike_mpnn_distance_to_reference",
        True,
        20260920,
    ),
]

per_task = []
for task_index, task in enumerate(TASKS):
    rows = []
    for challenge_id, ref in reference.items():
        if ref[task] != "1":
            continue
        seed_deltas = []
        for seed in sorted(paired_df["replicate_seed"].unique()):
            base = lookup[(challenge_id, "proteinmpnn", str(seed))]
            mars = lookup[(challenge_id, "marslike_mpnn", str(seed))]
            seed_deltas.append(float(mars["p_" + task]) - float(base["p_" + task]))
        rows.append((challenge_id, np.mean(seed_deltas)))
    values = np.asarray([value for _, value in rows], dtype=float)
    ci = bootstrap(values, 20261000 + task_index)
    per_task.append(
        {
            "task": task,
            "task_cn": TASK_CN[task],
            "positive_backbones": len(values),
            "proteinmpnn_mean": float(
                np.mean(
                    [
                        np.mean(
                            [
                                float(lookup[(cid, "proteinmpnn", str(seed))]["p_" + task])
                                for seed in sorted(paired_df["replicate_seed"].unique())
                            ]
                        )
                        for cid, _ in rows
                    ]
                )
            ),
            "marslike_mpnn_mean": float(
                np.mean(
                    [
                        np.mean(
                            [
                                float(lookup[(cid, "marslike_mpnn", str(seed))]["p_" + task])
                                for seed in sorted(paired_df["replicate_seed"].unique())
                            ]
                        )
                        for cid, _ in rows
                    ]
                )
            ),
            "mean_delta": float(values.mean()),
            "ci95_lower": ci[0],
            "ci95_upper": ci[1],
            "improved_backbones": int((values > 0).sum()),
            "improved_fraction": float((values > 0).mean()),
        }
    )

results = ROOT / "results"
results.mkdir(exist_ok=True)
write_csv(results / "01_逐序列六标签评分.csv", scored_rows)
paired_df.to_csv(results / "02_逐骨架逐种子配对结果.csv", index=False, encoding="utf-8-sig")
backbone_df.to_csv(results / "03_逐骨架三种子汇总.csv", index=False, encoding="utf-8-sig")
write_csv(results / "04_综合指标汇总.csv", summary_rows)
write_csv(results / "05_六标签正类分层结果.csv", per_task)

verification = {
    "selection_protocol": json.loads((ROOT / "inputs" / "selection_protocol.json").read_text()),
    "scoring_input_qc": input_qc,
    "checkpoint_sha256": identity_rows[0]["checkpoint_sha256"],
    "checkpoint_epoch": identity_rows[0]["checkpoint_epoch"],
    "checkpoint_validation_macro_ap": identity_rows[0]["checkpoint_validation_macro_ap"],
    "sentinel_max_abs_error": max(row["sentinel_max_abs_error"] for row in identity_rows),
    "prediction_shards": len(identity_rows),
    "unique_predictions": len(predictions),
    "reference_backbones": len(reference),
    "paired_seed_records": len(paired_df),
    "analysis_unit": "reference backbone after averaging three matched seeds",
    "test_set_reused_for_training_or_selection": False,
    "functional_ground_truth_for_designs": False,
}
(results / "verification.json").write_text(
    json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
)

with pd.ExcelWriter(ROOT / "固定测试集重设计与回评结果.xlsx", engine="openpyxl") as writer:
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="综合结果", index=False)
    pd.DataFrame(per_task).to_excel(writer, sheet_name="六标签正类", index=False)
    backbone_df.to_excel(writer, sheet_name="三种子骨架汇总", index=False)
    paired_df.to_excel(writer, sheet_name="逐种子配对", index=False)
    pd.DataFrame(scored_rows).drop(columns=["sequence"]).to_excel(
        writer, sheet_name="逐序列评分", index=False
    )
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="365F91")
        for column in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in column[:200]]
            width = min(max(max(map(len, values), default=0) + 2, 10), 42)
            worksheet.column_dimensions[column[0].column_letter].width = width

aggregate = summary_rows[0]
positive = summary_rows[1]
distance = summary_rows[2]
lines = [
    "# 固定测试集序列重设计与冻结模型回评",
    "",
    "## 实验设计",
    "",
    f"从固定测试集预注册选取 {len(reference)} 条阳性候选序列，每条来自不同 ID50 簇，并排除与训练/验证集的精确序列、ID50簇和已登记关系组件重叠。对每个预测骨架分别使用 ProteinMPNN 与 Marslike-MPNN，在种子42、43、44下进行匹配采样。随后使用冻结的一阶段 ProtT5共享LoRA+Masked-BCE模型评分。",
    "本轮骨架由 `esm.pretrained.esmfold_v1` 生成，1,000条全部完成且序列回写校验通过；该名称不写成ESMFold2。",
    "",
    "两个MPNN对同一骨架和同一种子使用相同采样随机数。统计先在每个骨架内平均三个种子，再以1,000个骨架为独立单位计算区间和符号检验。未读取结果后重选序列、阈值或种子。",
    "",
    "## 主要结果",
    "",
    "| 指标 | ProteinMPNN | Marslike-MPNN | 平均变化 | 95% bootstrap CI | 改善骨架 |",
    "|---|---:|---:|---:|---:|---:|",
    f"| 六标签等权综合均分 | {aggregate['proteinmpnn_mean']:.4f} | {aggregate['marslike_mpnn_mean']:.4f} | {aggregate['mean_delta']:+.4f} | [{aggregate['ci95_lower']:+.4f}, {aggregate['ci95_upper']:+.4f}] | {aggregate['improved_backbones']}/{aggregate['backbones']} ({aggregate['improved_fraction']:.2%}) |",
    f"| 原始正标签均分 | {positive['proteinmpnn_mean']:.4f} | {positive['marslike_mpnn_mean']:.4f} | {positive['mean_delta']:+.4f} | [{positive['ci95_lower']:+.4f}, {positive['ci95_upper']:+.4f}] | {positive['improved_backbones']}/{positive['backbones']} ({positive['improved_fraction']:.2%}) |",
    f"| 与原始序列六维评分距离（越低越好） | {distance['proteinmpnn_mean']:.4f} | {distance['marslike_mpnn_mean']:.4f} | {distance['mean_delta']:+.4f} | [{distance['ci95_lower']:+.4f}, {distance['ci95_upper']:+.4f}] | {distance['improved_backbones']}/{distance['backbones']} ({distance['improved_fraction']:.2%}) |",
    "",
    "六标签等权均分是内部模型综合评分，不是新的单标签分类器，也不代表六种胁迫功能同时成立。原始正标签均分只聚合每条参考序列已有的1标签，科学解释更直接。",
    "低温、氧化和盐适应的区间明确高于0；高氯酸盐相关变化很小且分数接近饱和。干燥和辐射/DNA修复的95%区间跨0，本轮不能宣称这两类稳定改善。",
    "",
    "## 六标签结果",
    "",
    "| 标签 | 正类骨架 | ProteinMPNN | Marslike-MPNN | 平均变化及95%CI | 改善比例 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in per_task:
    lines.append(
        f"| {row['task_cn']} | {row['positive_backbones']} | {row['proteinmpnn_mean']:.4f} | {row['marslike_mpnn_mean']:.4f} | {row['mean_delta']:+.4f} [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}] | {row['improved_fraction']:.2%} |"
    )
lines += [
    "",
    "## 结论边界",
    "",
    "这是训练未见固定测试簇上的独立计算回评，可检验 Marslike-MPNN 设计是否更符合冻结六标签模型学到的宽候选特征。1,000条参考序列和6,000条设计序列均无训练/验证集精确重复；设计序列未重新执行全库ID50搜索，独立性主要由参考亲本的ID50隔离保证。设计序列没有直接功能真值，因此该实验不能单独证明实际耐受性提高，也不计算设计蛋白的AP、AUROC、MCC或真实命中率。",
    "",
    "## 可复现文件",
    "",
    "- `inputs/selection_protocol.json`：固定测试集选择规则、计数和输入哈希。",
    "- `inputs/scoring_input_qc.json`：三种子成对生成和回评输入核验。",
    "- `results/verification.json`：冻结checkpoint、哨兵误差和评价边界。",
    "- `固定测试集重设计与回评结果.xlsx`：完整可阅读结果。",
    "- `tools/`：选择、折叠、MPNN生成、评分和分析脚本。",
]
(ROOT / "说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary": summary_rows, "per_task": per_task}, ensure_ascii=False, indent=2))
