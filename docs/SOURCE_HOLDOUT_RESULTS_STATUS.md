# Source-holdout evaluation status

This file separates the reproducible holdout protocol from completed results.
The canonical 36,000/2,000/2,000 split is unchanged. Every derived source
fold must expand exclusions to registered ID50 clusters and relationship
components before model fitting.

## Completed folds reported in MarsPro

The manuscript reports four predeclared, single-seed source folds:

| Fold | Task | Known-label sequences | AP | Prevalence | AUROC | Interpretation |
|---|---|---:|---:|---:|---:|---|
| AAC-cold | cold | 1,392 | 0.327 | 0.175 | 0.691 | ranking above prevalence; threshold transfer is limited |
| ESM-PsyPred-cold | cold | 1,924 | 0.198 | 0.161 | 0.657 | ranking above prevalence; threshold transfer is limited |
| AOP2020-oxidative | oxidative | 1,566 | 0.226 | 0.108 | 0.736 | ranking above prevalence; threshold transfer is limited |
| HaloClass-salt | salt | 4,095 | 0.903 | 0.437 | 0.946 | ranking above prevalence; threshold transfer is limited |

These values are model results, not report-only candidate counts. Unknown
sequence-task cells are excluded from the metrics, and bootstrap units are
relationship components or ID50 clusters rather than repeated label cells.

## Not completed as binary source folds

- Desiccation sources `DE003` and radiation source `RA006` have no observed
  source-comparator zero in the current expanded pools. They support positive
  recall or retrieval analyses, not AUROC/MCC claims.
- `OX002` is similarly dominated by positive labels and is a positive-recall
  stress test rather than a balanced binary source fold.
- Perchlorate has no sufficiently independent second large source in the
  current registry; a source-holdout binary result is therefore not evaluable
  until another independent source is added.

## Reproduction

Use `scripts/build_source_holdout.py` with one registered source ID and a new
derived output directory. Use `--report-only` only for fold-size inspection.
Do not edit the canonical split files and do not treat one-class folds as
binary classification results.
