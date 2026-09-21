# Refolding protocol

Two cohorts use different folding pipelines and must not be merged.

## 798-backbone matched-seed structural evaluation

- Designs: ProteinMPNN and Marslike-MPNN, seeds 42/43/44, matched per backbone.
- Total: 798 backbones, 2,394 method pairs, 4,788 design structures.
- Model interface: `transformers.models.esmfold2.modeling_esmfold2.ESMFold2Model` with `ESMCModel` loaded from a local `ESMC-6B` checkpoint.
- Parameters: `num_loops=3`, `num_sampling_steps=200`, `num_diffusion_samples=1`, `seed=0`.
- Precision: model and ESMC loaded in float32; inference mode; one diffusion sample.
- Chunk size: 32; ESMC moved to GPU only during language-model hidden-state computation.
- Validation: finite coordinates and confidence, sequence round-trip equality, PDB/mmCIF round-trip, per-output SHA-256, and zero failed final records.
- Implementation: `stage2_design/matched_seed_798/code/esmfold2_matched_worker.py`.

The exact upstream checkpoint files are excluded. Local folders were named `ESMFold2` and `ESMC-6B`; users must record their own upstream revision and weight hashes when regenerating structures.

## 1,000-fixed-test-cluster rescoring cohort

- Reference backbones were generated with `esm.pretrained.esmfold_v1()`.
- Seed: 0; chunk sizes attempted in order 64, 32, 16 after CUDA OOM.
- All 1,000 structures passed sequence round-trip and confidence validation.
- Implementation: `stage2_design/fixed_test_1000/code/fold_esmfold_v1.py`.

PDB/CIF files are intentionally excluded; input sequences, selection rules, generated sequences, metrics, and scripts are retained.
