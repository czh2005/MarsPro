# Stage 2: domain-adapted inverse folding

This directory contains only current final cohorts. Superseded unmatched-seed and mixed-folding-pipeline tables are excluded from the public project.

## `structure_corpus/`

Index and provenance for 10,000 unique positive candidates. PDB/CIF payloads are excluded.

## `matched_seed_798/`

Final structural and surface comparison on 798 reference backbones with matched seeds 42, 43, and 44 for ProteinMPNN and Marslike-MPNN. The independent unit is the backbone; seeds are aggregated within backbone before uncertainty estimation.

## `fixed_test_1000/`

Prediction-blind redesign and frozen Stage 1 rescoring for 1,000 distinct fixed-test ID50 clusters. This cohort tests model-mediated consistency on references excluded from Stage 1 train/validation and from the registered 10k structure corpus.

The two cohorts answer different questions and are not pooled. See `../docs/REFOLDING_PROTOCOL.md` and `../docs/DATA_AND_CLAIMS.md`.
