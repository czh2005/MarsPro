# Reproducibility status

## Packaged

1. Final 40,000-sequence Stage 1 dataset, fixed splits, row-level provenance sidecar, source inventory, and label-construction scripts.
2. Selected shared ProtT5-LoRA + Masked-BCE checkpoint, training/evaluation code, configuration, validation predictions, ablations, and locked-test outputs.
3. Final Marslike-MPNN LoRA adapter checkpoint (`best_joint_lora.pt`) and matched generation scripts.
4. Ten-thousand-positive Stage 2 sequence/structure index without PDB/CIF payloads.
5. Final 798-backbone matched-seed design tables and analysis code.
6. Final 1,000-fixed-test-cluster selection, generated sequences, frozen rescoring outputs, and replay code.
7. SHA-256 manifest and deterministic release verifier.

## External prerequisites

- ProtT5 pretrained weights are not redistributed.
- ProteinMPNN base weights are not redistributed; expected SHA-256 is recorded in the generation manifests.
- ESMFold/ESMFold2 and ESMC-6B weights are not redistributed.
- PDB/CIF structures are not included, per release policy.
- US-align and FreeSASA should be installed from their upstream projects.

Machine-specific paths in provenance files describe original execution. Portable example configuration and relative repository paths should be used for replay.

## Replay levels

- Stage 1 metrics and locked-test evaluation: self-contained except for the ProtT5 base model.
- Marslike-MPNN sequence generation: self-contained except for the ProteinMPNN base model and upstream code.
- Stage 2 statistical replay: self-contained from released CSV/JSON tables.
- Structure regeneration: requires external ESMFold/ESMFold2 weights and recreates excluded PDB/CIF files.
