# Fixed Six Task Label Remap v3 Validation Checklist

## Deterministic Count Checks

| check | status |
|---|---|
| exact_total_unique_sequences_is_24651 | PASS |
| co001_sequence_target_rows_is_659 | PASS |
| co001_individual_development_usable_rows_is_616 | PASS |
| co001_primary_paper_endpoint_rows_is_30 | PASS |
| pe002_table_s3_endpoint_rows_is_79 | PASS |
| pe002_target_matched_unique_sequences_is_43 | PASS |
| ox002_screen_positive_unique_sequences_is_173 | PASS |
| v1_endpoint_rows_is_10631 | PASS |
| v1_sequence_target_condition_rows_is_9600 | PASS |
| v3_evidence_rows_is_10740 | PASS |
| v3_sequence_target_rows_is_9662 | PASS |
| six_class_rows_is_6 | PASS |
| formal_binary_candidate_count_is_0 | PASS |
| pe002_credible_negative_records_is_0 | PASS |

## Rule Checks

- Top-level task renaming is not used as label evidence.
- CO001 TH, IRI, morphology and application endpoints remain condition/subtarget separated.
- OX002 non-significant records remain unknown and are not used as negatives.
- PE002 Table S3 selected-gene rows are positive related signed-effect evidence only; PE002 all-gene numeric matrix remains direction-pending unknown.
- Nitrate-only PE002 rows are context, not perchlorate/chlorate target labels.
- Formal supervised binary benchmark candidate count remains zero for this version.

## Reproducibility Checks

- Re-run `python scripts/build_pe002_table_s3_adjudication_v1.py --root .` before v3 if PE002 inputs changed.
- Re-run `python scripts/build_fixed_six_task_label_remap_v3.py --root .` to reproduce this derivative release.
- Compare `fixed_six_task_label_remap_v3_outputs.sha256` between local and novlight.
