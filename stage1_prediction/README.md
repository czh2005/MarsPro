# Stage 1: six-label sequence prediction

## Final inputs

`data/` contains the final `source_zero_union_v1_20260910` dataset only:

| Split | Rows |
|---|---:|
| train | 36,000 |
| validation | 2,000 |
| test | 2,000 |

Each task state is `1`, `0`, or `unknown`. The selected model uses `unknown_training_policy=zero_loss_mask`; only known positions contribute to BCE loss and metrics.

## Selected model

- Backbone: ProtT5 encoder
- Adaptation: shared LoRA, rank 8, alpha 16, dropout 0.05
- Output: six independent sigmoid probabilities
- Loss: task-aware Masked-BCE
- Seed: 42
- Epochs: 5

The frozen model is `model/best.pt`; its SHA-256 is `9cac216f122bdccaebd277639062b103854e40d3e2b7b73b1482bfc78ba32073`.

## Results

`results/best_model/` contains the selected method's validation tables and predictions. `results/ablations/` contains the final v5 readable cross-method comparison. `reproducibility/` contains the final mask-ablation, traditional-feature, and frozen-control scripts/configs.

`results/locked_test/` contains the completed blind test protocol, sealed probabilities for all 2,000 test sequences, complete evaluation tables, per-label and stratified metrics, uncertainty intervals, and the documented pre-prediction parser failure. The locked-test macro AUPRC is 0.975286 versus 0.977087 on validation.

## Training

Install dependencies from `../requirements/stage1.txt`, provide a local ProtT5 model path in a copied config, and run:

```bash
torchrun --standalone --nproc_per_node=2 code/train_prott5_lora.py --config config/shared_masked_bce.json
```

The archived config keeps original machine paths for provenance. Replace only filesystem and device paths; keep scientific parameters fixed for an exact protocol reproduction.
