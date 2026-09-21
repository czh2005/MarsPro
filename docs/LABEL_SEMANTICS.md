# Three-state label semantics

## States

- `1`: the sequence belongs to a source, family, proteome, or study-defined collection associated with the task.
- `0`: a source-supported comparator or an original classifier zero exists for that task.
- `unknown`: the task was not established for that sequence. It is not converted to zero.

## Training and evaluation

- Known positive and known zero cells have unit sample weight.
- Unknown cells have zero loss weight under task-aware Masked-BCE.
- Metrics use known cells only; unknown cells never enter AUROC, AUPRC, F1, MCC, or threshold selection as negatives.
- Multi-label rows are retained. The six task labels are not collapsed into a single functional truth label.

## Zero construction

`source_zero_union_v1_20260910` takes the union of:

1. native zeros supplied by the original task-specific classification resources; and
2. reviewed source-comparator zeros added only where the prior state was unknown.

Positive-priority conflicts remain positive and are recorded rather than overwritten. No matched pseudo-zero samples are used in the final release. Row-level fields ending in `_zero_source_ids`, `_zero_basis`, `_zero_condition_scope`, and `_zero_rule_version` provide the audit trail.

The complete definitions and narrower evidence-layer mappings are under `stage1_prediction/data/provenance/`.
