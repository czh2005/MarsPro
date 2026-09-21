# Compute provenance

The final artifacts were produced across three compute environments. Server addresses and credentials are intentionally excluded from this public release.

| Environment | Final role | Included in this repository |
|---|---|---|
| 4-card V100 workstation | early ProtT5 baselines, evidence/source-debias experiments, six-expert training | selected final scorer; final v5 comparison tables; evidence, source-debias, expert/fusion outcomes through the unified result table |
| 8-card V100 cluster reached through gateway | ESM-2/ProtT5 comparison, mask ablations, frozen controls, final shared Masked-BCE scorer, locked-test inference | final v5 result tables; mask-ablation/traditional-feature/frozen-control code and configs; selected scorer checkpoint; sealed 2,000-sequence locked-test predictions and evaluation |
| RTX 3090 workstation | Marslike-MPNN adapter recovery and matched-seed generation | released Marslike-MPNN adapter checkpoint, generation code, and matched seed manifests |
| V100 compute nodes | final ESMFold2 matched-seed refolding and frozen Stage 1 rescoring | 798-backbone/4,788-structure metrics and 1,000-fixed-test-cluster rescoring outputs; PDB/CIF payloads excluded |

The selected Stage 1 scorer and final Marslike-MPNN adapter checkpoints are included. Checkpoints for rejected alternatives are excluded because this is a final-method release rather than a full compute snapshot. Their final metrics, method identifiers, seeds, and replay configurations are retained where they support paper comparisons.

## Locked-test completion

The run `locked_test_shared_prott5_maskedbce_20260918_003820` completed after the initial inventory. GPU inference received only `sequence_id` and sequence; predictions were sealed before labels were joined. The first parser attempt exited before producing predictions because the blind input had no label columns. The amendment and failed-attempt log are retained under `stage1_prediction/results/locked_test/`.
