# MarsLikePro

MarsLikePro is a prediction--design--rescoring framework for proteins associated with six Mars-relevant stress categories: cold, desiccation, oxidative stress, perchlorate-related conditions, radiation/DNA repair, and salt adaptation.

The Stage 1 labels encode broad source, family, or proteome membership. They do not assert that every sequence has experimentally demonstrated tolerance. Every sequence--task cell is explicitly represented as `1`, `0`, or `unknown`; unknown cells are excluded from Masked-BCE loss and known-cell evaluation.

## Repository layout

- `stage1_prediction/`: the 40,000-sequence three-state dataset, row-level provenance sidecar, source inventory, construction scripts, selected ProtT5-LoRA checkpoint, locked-test outputs, and ablations.
- `stage2_design/structure_corpus/`: the 10,000-positive sequence/structure index. PDB/CIF files are intentionally excluded.
- `stage2_design/matched_seed_798/`: final 798-backbone, three-matched-seed ProteinMPNN versus Marslike-MPNN comparison (4,788 refolded designs).
- `stage2_design/fixed_test_1000/`: prediction-blind challenge on 1,000 distinct fixed-test ID50 clusters with three matched generation seeds.
- `docs/`: claim boundaries, data provenance, licenses, refolding protocol, source-holdout protocol, and reproducibility instructions.
- `paper/`: the current anonymous IEEE manuscript, bilingual Overleaf sources,
  compiled PDFs, and editable figure sources.
- `scripts/build_source_holdout.py`: source-level challenge split builder with ID50 and relationship-component exclusion.
- `scripts/update_manifest.py` and `scripts/verify_release.py`: deterministic inventory and release checks.

## Current verified results

- Fixed split: 36,000 train / 2,000 validation / 2,000 locked test; no registered ID50 cluster crosses splits.
- Shared ProtT5-LoRA + task-aware Masked-BCE: validation macro AUPRC 0.977087; one-time locked-test macro AUPRC 0.975286.
- Matched-seed design comparison: 798 backbones, seeds 42/43/44, 2,394 paired comparisons, and 4,788 structures. STNQDE and Met/Cys absolute reference distances decrease by 16.67% and 17.30%; US-align RMSD is similar.
- Fixed-test redesign challenge: mean frozen six-output score increases from 0.61854 to 0.67251; 731/1,000 backbone-level comparisons improve.

These are source-association and computational-design results, not experimental proof of stress tolerance.

## Quick verification

```bash
python scripts/update_manifest.py
python scripts/verify_release.py
```

## External assets

Pretrained ProtT5, ProteinMPNN base weights, ESMFold/ESMFold2 weights, and the 10,000 PDB/CIF files are not redistributed. Their identifiers, hashes where available, inputs, parameters, and expected relative paths are documented so users can restore them locally.

## Licensing

Code is released under the MIT License. Author-created metadata and result tables are released under CC BY 4.0. Protein sequences and third-party source content retain their upstream terms; see `DATA_LICENSE.md` and `stage1_prediction/data/provenance/source_inventory_40k.csv` before redistribution.

## Repository visibility

This complete reproducibility bundle should remain **private** until every
sequence source marked `not_verified_for_redistribution` in the source
inventory has been cleared or removed from a public release. The code,
author-created metadata, derived metrics, paper source, and compact adapter
checkpoints may be published under their stated licenses. See
`PUBLIC_RELEASE_CHECKLIST.md` for the final visibility gate.
