# Scientific-source holdout protocol

This protocol is an additional generalization test. It does not replace or modify the fixed 36k/2k/2k split. It is a derived evaluation protocol, not a claim that every registered source already has a completed model result.

Create one derived fold at a time without editing the canonical split files:

```bash
python scripts/build_source_holdout.py \
  --source-id ISS_SA_HALOCLASS \
  --output-dir derived/source_holdout/ISS_SA_HALOCLASS
```

The command writes filtered training and validation tables, an expanded challenge table, row-level exclusion reasons, and a hash-bearing manifest. The original non-challenge test rows are not added to training or validation.

The command is deliberately source-ID driven. A source ID must be present in
`source_registry.csv` and in the sidecar's `scientific_source_ids` field; a
missing source ID is an error rather than an empty fold. The script expands the
held-out source to touching ID50 clusters and relationship components before
writing the derived fold, so source leakage is handled by the same exclusion
unit used in the manuscript.

## Unit of exclusion

For each registered held-out source, move every sequence carrying that `scientific_source_id` into a challenge pool. Expand the exclusion to all registered ID50 clusters and relationship components touching that pool. No sequence from those expanded groups may remain in training or validation.

## Model fitting

Retrain the selected shared ProtT5-LoRA + Masked-BCE configuration from the same base checkpoint, with the same seed set, epochs, optimizer, and validation rule. Do not supply source IDs, species, genus, Pfam, or holdout membership as model inputs.

## Evaluation

- Evaluate only known `1`/`0` cells; unknown remains excluded.
- Report per-task positive, zero, unknown, sequence, ID50-cluster, relationship-component, species/genus, and Pfam counts.
- For folds containing both observed classes, report AUPRC, AUROC, F1, MCC, recall, and calibration. For one-class folds, report positive recall/rank enrichment only; do not report degenerate binary metrics.
- Bootstrap by relationship component or ID50 cluster, not by repeated label cells.
- Compare with the length/amino-acid-composition baseline and homology transfer.

## Feasible first folds and current status

- cold: `ISS_CO_ESMPSYPRED` gives 309 positives, 1,615 zeros, and 12 unknown cells after group expansion; `ISS_CO_AAC` gives 243/1,149/6. Both support binary source-holdout evaluation.
- desiccation: `DE003` gives 2,492 positives and no observed zero. Use positive recall or retrieval, not AUROC/MCC.
- oxidative: `ISS_OX_AOP2020` gives 169 positives, 1,397 zeros, and 12 unknown cells. `OX002` gives 4,777/7/10 and is therefore primarily a positive-recall stress test.
- radiation: `RA006` gives 832 positives and no observed zero. Use positive recall or retrieval.
- salt: `ISS_SA_HALOCLASS` gives 1,788 positives, 2,307 zeros, and 12 unknown cells and supports binary source-holdout evaluation.

Perchlorate currently lacks a sufficiently independent second large source; a source-holdout result for that task should be marked not evaluable until another source is added.

The counts above were generated from the frozen 40,000-row release with
`--report-only`; see `source_holdout_candidate_counts.csv`. They are fold sizes,
not completed model results. Completed model metrics and one-class limitations
are tracked separately in `SOURCE_HOLDOUT_RESULTS_STATUS.md` so that a
candidate-count report cannot be mistaken for a trained holdout result.
