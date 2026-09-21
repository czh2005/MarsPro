# Shared ProtT5-LoRA + Masked-BCE model

This checkpoint is the frozen Stage 1 scorer used in the downstream Marslike-MPNN rescoring analysis.

- Tasks: six broad source/proteome-association outputs.
- Unknown policy: zero loss mask.
- Validation macro AUPRC: 0.977087.
- Locked-test macro AUPRC: 0.975286; test labels were joined only after probabilities were sealed.
- Test status: completed once after validation-only model and threshold selection.
- Intended use: candidate ranking and comparative computational evaluation.
- Not intended as: experimental proof of protein function, joint six-stress tolerance, or organism-level Mars viability.

The checkpoint requires the same ProtT5 backbone revision used during training. The pretrained backbone itself is not redistributed here.
