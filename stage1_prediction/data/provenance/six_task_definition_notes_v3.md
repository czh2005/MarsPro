# Fixed Six Task Definition Notes v3

The six names are retained as top-level task identifiers. They do not assign labels by themselves.
Every mapped record must pass through a narrow target, evidence layer, condition and endpoint rule.

| task_id | primary object | primary narrow targets | main exclusion |
|---|---|---|---|
| cold_adaptation | protein_sequence_under_defined_cold_related_evidence | CO_ICE_BINDING_V1;CO_SUBZERO_ACTIVITY_V1;CO_HOST_COLD_V1 | general cold-source metadata, expression association, or cell-application outcomes cannot replace a direct cold-related protein property |
| desiccation_adaptation | protein_sequence_with_desiccation_related_function_or_protection_evidence | DE_HOST_SURVIVAL_V1;DE_DIRECT_PROTECTION_V1 | general intrinsically disordered status or transcriptome presence alone is auxiliary, not a task label |
| oxidative_stress | protein_sequence_with_defined oxidative-condition functional evidence | OX_PEROXIDE_HOST_V1;OX_ORGANIC_HYDROPEROXIDE_V1;OX_PURIFIED_DETOX_V1 | renaming an oxidative source label or RA source alias does not create a target label |
| perchlorate_related | protein_sequence_with defined perchlorate/chlorate pathway or condition evidence | PE_RESPIRATION_V1;PE_CHLORITE_DISMUTASE_V1;PE_TOXICITY_RESPONSE_V1 | general pathway membership or condition annotation alone is not sufficient for a scoped label |
| radiation_dna_repair | protein_sequence_with radiation-condition survival or DNA-protection evidence | RA_IONIZING_SURVIVAL_V1;RA_DNA_PROTECTION_V1;RA_UV_SURVIVAL_V1 | RA001/RA005 contextual rows cannot be used as negatives without target-level assay support |
| salt_adaptation | protein_sequence_with defined high-salt activity, stability, or host contribution evidence | SA_STABILITY_V1;SA_ACTIVITY_RETENTION_V1;SA_HOST_PHENOTYPE_V1 | halophilic source annotation alone does not label a protein as salt-adapted |
