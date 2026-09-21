# Evaluation Entry Protocol v3

## OX002_POSITIVE_UNKNOWN_RETRIEVAL_UNDER_oxidative_stress_v3

- Task: oxidative_stress.
- Subtarget: OX_PEROXIDE_HOST_FITNESS_CONTRIBUTION_V1.
- Input: OX002 canonical status has 4451 unique sequences; 173 known positives and 4278 unknown-background sequences after excluding cross-condition known positives.
- Labels: source-threshold screen positives are positives; non-significant or unreported endpoints are unknown, not negatives.
- Strict boundary: strict positive follow-up rows = 3; strict negative rows = 0.
- Baseline already valid for this v3 mapping: length+composition one-class retrieval, mean recall@100=0.0467, mean enrichment@100=2.0123.
- Metrics: recall@K and enrichment@K for held-out known positives. Do not report AUROC/MCC using unknown background as negatives.

## CO001_MD15100318_CONSTRUCT_CHALLENGE_UNDER_cold_adaptation_v3

- Task: cold_adaptation.
- Subtarget: COLD_ICE_INTERACTION_TH_ACTIVITY_V1.
- Input: one parent/mutant component from DOI:10.3390/md15100318, 9 unique mature-core sequences.
- Labels: 6 primary construct-qualified positives and 3 primary construct-qualified scoped negatives.
- Split: keep parent and all mutants in one challenge group.
- Boundary: not a formal broad binary benchmark; exact His-tagged constructs and release gates remain unresolved.

## PE002_TABLE_S3_POSITIVE_UNKNOWN_UNDER_perchlorate_related_v3

- Task: perchlorate_related.
- Subtarget: PE_PERCHLORATE_CHLORATE_CONDITION_FITNESS_EFFECT_V1.
- Input: Supplementary Table S3 contributes 45 condition-matched signed-effect records over 43 unique mapped sequences, 43 relation components and 35 primary Pfam families.
- Labels: selected genes in perchlorate/chlorate-containing groups are positive related signed fitness-effect evidence; effect polarity remains a separate column.
- Negatives: none. Nitrate-only groups are adjacent context, and the all-gene numeric matrix remains direction-pending unknown.
- Metrics: retrieval/enrichment or source-lineage-held-out positive recovery only; no AUROC/MCC using unknowns as negatives.

## Existing small challenges retained

- desiccation_adaptation: DE_HOST_SURVIVAL_V1 remains a small challenge.
- salt_adaptation: SA_STABILITY_V1 remains a small challenge.
